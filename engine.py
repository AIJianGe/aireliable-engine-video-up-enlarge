"""videoUpEnlarge 视频高清放大引擎推理核心。

继承 ``CustomEngine``（全自定义模式 / integrated 范式），实现 7 个
``_xxx_impl`` 方法，封装 ``UpEnlargeWrapper`` 的加载/推理/卸载流程。

8 端点契约：
    - /capabilities  返回扫描到的超分变体（2 档位，含 backend / ncnn_model）
    - /load          校验后端资源（可执行文件 + ncnn 权重）+ 配置设备（快速返回）
    - /unload        释放状态（ncnn 为进程外调用，无常驻显存）
    - /infer         解析 VideoUpEnlargeInputs → wrapper.run → SSE 推送
    - /status        返回引擎状态（已加载变体 / 设备 / 后端 / 可用变体）
    - /lora/load     v1 不支持 LoRA（预留接口）
    - /lora/unload   v1 不支持 LoRA（预留接口）

任务类型：video_upscale（视频空间分辨率提升，保留原音频）。

⚠️ 三处易踩空的契约（详见本引擎 __init__.py 与 ANALYSIS-TASKS/42-*）：
    1. 网关把视频放在 ``inputs["videos"][0]["path"]``，**不是**顶层 ``request.videos``
    2. 产物以 ``videos[].video_path`` + ``video_url`` 形式返回（T0.2 已核实）
    3. LoRA 不支持；音色管理（data/voice_library.db / data/voices/）零触碰
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    InferRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
)
from aireliable_engine.state_machine import RuntimeState

from .config import EngineConfig
from .model_scanner import ModelScanner
from .progress_reporter import ProgressReporter
from .schemas import CAPABILITY, TASK_TYPE, VideoUpEnlargeInputs, VideoUpEnlargeVariantInfo
from .upscale_wrapper import UpEnlargeProgressStage, UpEnlargeWrapper

logger = logging.getLogger(__name__)


class VideoUpEnlargeEngine(CustomEngine):
    """videoUpEnlarge 视频高清放大引擎。

    全自定义模式（integrated 范式），直接控制后端资源校验与推理流程。
    首期支持 2 个超分档位（均为 Real-ESRGAN ncnn-vulkan 后端）：
        - videoUpEnlarge/realesrgan-x4plus    （通用写实，默认）
        - videoUpEnlarge/realesr-animevideov3 （动漫 / 风格化视频）
    """

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )
        self.config: EngineConfig = config

        # ===== 核心组件 =====
        self.wrapper = UpEnlargeWrapper(config, logger_=logger)
        # 方案 B：注入 engine=self，启用平台 API 声明式检测
        self.scanner = ModelScanner(
            models_root=config.models_root,
            model_dir=config.model_dir,
            engine=self,
            logger=logger,
        )

        # 启动时扫描（同步 fallback，异步扫描在 lifespan 中触发）
        self._available_variants: list[VideoUpEnlargeVariantInfo] = []
        self._refresh_scan()

        # 输出根目录（供 BaseEngine.get_output_path 使用）
        self.output_root = config.output_root

        # 读取 manifest 中的 default_variant（供 BaseEngine.infer auto-load 使用）
        _manifest_path = Path(__file__).parent / "manifest.yaml"
        self.default_variant = self._read_default_variant_from_manifest(_manifest_path)

        # 任务类型（任务中心筛选语义）
        self.task_type = TASK_TYPE

    # ============ 扫描 ============

    def _refresh_scan(self) -> None:
        """刷新资源扫描结果（同步 fallback，用于 __init__ 阶段）。"""
        logger.info("刷新模型扫描（本地 fallback）: %s", self.config.model_dir)
        self._available_variants = self.scanner.scan_variants()

    async def _refresh_scan_async(self) -> None:
        """方案 B：异步刷新扫描结果（优先平台 API），lifespan startup 阶段调用。"""
        logger.info("异步刷新模型扫描（方案 B 平台 API）: %s", self.config.model_dir)
        try:
            self._available_variants = await self.scanner.scan_variants_async()
        except Exception as e:
            logger.warning("异步扫描失败，回退到本地扫描: %s", e)
            self._available_variants = self.scanner.scan_variants()

    # ============ 7 个 _xxx_impl 实现 ============

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """GET /capabilities：返回可用超分变体（2 档位）。"""
        variants: list[VariantInfo] = []
        for v in self._available_variants:
            variants.append(VideoUpEnlargeVariantInfo(
                name=v.name,
                base_model=v.base_model,
                precision=v.precision,
                loader=v.loader,
                vram_required=v.vram_required,
                available=v.available,
                backend=v.backend,
                ncnn_model=v.ncnn_model,
                scale_range=v.scale_range,
                supports_target_size=v.supports_target_size,
            ))

        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=[],  # v1 不支持 LoRA
            supported_sizes=["1920x1080", "2560x1440", "3840x2160"],
            default_variant=self.default_variant,
            # v0.4.1+ MCP 化：推理输入 JSON Schema
            input_schema=VideoUpEnlargeInputs.model_json_schema(),
            # 单能力引擎：capabilities 列表退化为 [capability]
            capabilities=[self.capability],
        )

    async def _load_impl(self, variant: Optional[str]) -> LoadResponse:
        """POST /load：校验后端资源（**只校验不下载**）+ 配置设备。

        流程：
            1. variant 路径解析（不存在抛 FileNotFoundError）
            2. 可执行文件定位（缺失时可对本地 zip 做一次离线解包）
            3. ncnn 权重完整性校验（缺失抛 RuntimeError + 下载指引）
            4. 设备解析（不留 "auto" 歧义）
        """
        # variant 缺省时回退（SDK LoadRequest 契约，v1.1 修复）：
        #   "variant 可为 None：表示不指定具体 variant，由引擎根据 manifest 的
        #    default_variant 或第一个 available variant 决定"
        # 历史行为：直接 raise，导致 POST /load {"variant": null} 返回 500。
        # 平台链路虽总会带 variant，但直连 API / WebUI 场景会命中此路径。
        if not variant:
            variant = self.default_variant or next(
                (v.name for v in self._available_variants), None
            )
        if not variant:
            raise RuntimeError(
                "未指定 variant，且 manifest 无 default_variant / 无可用 variant，无法加载"
            )

        variant_path = self.scanner.resolve_variant_path(variant)
        if variant_path is None:
            raise FileNotFoundError(
                f"variant 不存在或资源目录缺失: {variant} "
                f"（可用: {[v.name for v in self._available_variants]}；"
                f"期望目录: {self.config.model_dir}）"
            )

        vdef = self.scanner.get_variant_definition(variant)
        if not vdef:
            raise RuntimeError(f"variant 定义缺失（manifest 与 _KNOWN_VARIANTS 不一致?）: {variant}")

        device = self.config.resolve_device()
        start_time = time.time()
        try:
            await asyncio.to_thread(
                self.wrapper.load,
                variant,
                variant_path,
                device,
                vdef.get("ncnn_model", ""),
            )
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as e:
            logger.exception("后端资源配置失败: variant=%s", variant)
            raise RuntimeError(f"后端资源配置失败: {e}") from e

        load_time = time.time() - start_time
        logger.info(
            "加载就绪: variant=%s, backend=%s, ncnn_model=%s, device=%s, load_time=%.2fs",
            variant, self.wrapper.loaded_backend, self.wrapper.loaded_ncnn_model,
            device, load_time,
        )

        return LoadResponse(
            variant=variant,
            state="ready",
            vram_used="0MB",  # ncnn 为进程外调用，无常驻显存
            load_time=load_time,
        )

    async def _unload_impl(self) -> UnloadResponse:
        """POST /unload：释放后端配置。

        即使未加载（如 ERROR 状态下），也要确保状态机回到 IDLE，
        避免 ERROR 状态卡死无法恢复。
        """
        if self.state_machine.state == RuntimeState.ERROR:
            try:
                self.state_machine.reset()
                logger.info("unload()：状态机从 ERROR 重置为 IDLE")
            except Exception as e:
                logger.warning("unload()：reset 状态机失败: %s", e)

        if not self.wrapper.is_loaded:
            if self.state_machine.state != RuntimeState.IDLE:
                try:
                    self.state_machine.transition(RuntimeState.IDLE)
                except Exception as e:
                    logger.warning("unload()：状态转换到 IDLE 失败: %s", e)
            return UnloadResponse(state="idle", vram_freed="0MB")

        await asyncio.to_thread(self.wrapper.unload)

        if self.state_machine.state != RuntimeState.IDLE:
            try:
                self.state_machine.transition(RuntimeState.IDLE)
            except Exception as e:
                logger.warning("unload()：unload 后状态转换到 IDLE 失败: %s", e)

        logger.info("卸载完成（后端配置已清空）")
        return UnloadResponse(state="idle", vram_freed="0MB")

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """POST /infer：视频高清放大（SSE 流式推送进度）。

        流程：
            1. 生成 / 复用 task_id
            2. 解析 VideoUpEnlargeInputs
            3. 解析输入视频路径（inputs.videos[0].path 优先，多路兜底）
            4. 处理前自动加载（variant 变化时先切换）
            5. 上报 created / start / preparing
            6. 线程池执行 wrapper.run（30s 心跳保活）
            7. 上报 completed（含 videos[].video_path / video_url）

        错误处理：内部捕获异常，推送 error 帧后 return，**不重新抛出**
        （避免 BaseEngine.infer 把状态置为 ERROR）。
        """
        # ----- 0. task_id -----
        task_id = request.task_id or f"vuen_{uuid.uuid4().hex[:16]}"
        reporter = ProgressReporter(stream)

        # ----- 1. 解析输入 -----
        try:
            inputs = VideoUpEnlargeInputs(**(request.inputs or {}))
        except Exception as e:
            logger.error("VideoUpEnlargeInputs 解析失败: %s", e)
            reporter.emit_error(f"输入参数无效: {e}", "invalid_inputs")
            return

        reporter.set_task_id(task_id)
        await reporter.init_task_reporter()

        # ----- 2. 解析输入视频路径（网关链路优先 inputs.videos[]）-----
        video_path = inputs.resolve_input_video()
        if not video_path and request.videos:
            video_path = request.videos[0].path
        if not video_path:
            reporter.emit_error(
                "缺少输入视频：请通过 inputs.videos[0].path 传入"
                "（网关链路），或 inputs.video_path / request.videos（直连链路）",
                "missing_video",
            )
            return
        if not Path(video_path).is_file():
            reporter.emit_error(f"输入视频不存在: {video_path}", "video_not_found")
            return

        # ----- 3. 处理前自动加载 -----
        # 平台任务中心/网关直接传 variant（不显式调 /load）时，
        # 保证用户选择的档位真正生效，而不是沿用已加载的旧档位。
        requested_variant = (
            request.variant or self.wrapper.loaded_variant or self.default_variant
        )
        if requested_variant and self.wrapper.loaded_variant != requested_variant:
            logger.info(
                "处理前自动加载: 已加载=%s, 请求=%s → 切换",
                self.wrapper.loaded_variant, requested_variant,
            )
            reporter.emit(0.0, UpEnlargeProgressStage.MODEL_LOADING,
                          message=f"超分后端加载中（{requested_variant}）…")
            try:
                await self._load_impl(requested_variant)
            except Exception as e:
                logger.exception("处理前自动加载失败: variant=%s", requested_variant)
                reporter.emit_error(f"超分后端加载失败: {e}", "load_failed")
                return

        # ----- 4. 上报任务创建 + 开始 -----
        await reporter.report_created(
            request_summary={
                "video": Path(video_path).name,
                "scale": inputs.scale,
                "target_width": inputs.target_width,
                "target_height": inputs.target_height,
                "backend": self.wrapper.loaded_backend,
                "ncnn_model": self.wrapper.loaded_ncnn_model,
            },
            variant=request.variant or self.wrapper.loaded_variant or "",
            task_type=self.task_type,
        )

        reporter.emit_start(task_id, Path(video_path).name)
        reporter.emit_preparing()

        # ----- 5. 构造后端入参 -----
        infer_inputs: dict[str, Any] = {
            "video_path": video_path,
            "scale": float(inputs.scale),
            "target_width": inputs.target_width,
            "target_height": inputs.target_height,
            "tile_size": inputs.tile_size,
            "tta": inputs.tta,
            "threads": inputs.threads,
            "gpu_id": inputs.gpu_id,
            "frame_format": inputs.frame_format,
            "crf": inputs.crf,
            "dn": inputs.dn,
            "keep_temp": inputs.keep_temp,
        }

        # 进度回调（工作线程调用 → put_threadsafe 线程安全）
        def _progress(progress: float, stage: str, detail: str) -> None:
            if stage == UpEnlargeProgressStage.PREPARING:
                # 抽帧阶段用专门的映射（0.05 → 0.35）
                if detail and "抽取视频帧" in detail:
                    reporter.emit_extracting(max(0.0, (progress - 0.05) / 0.30), detail)
                else:
                    reporter.emit(progress, stage, message=detail)
            elif stage == UpEnlargeProgressStage.PROCESSING:
                if detail and detail.startswith("超分处理"):
                    reporter.emit_processing(max(0.0, (progress - 0.40) / 0.50), detail)
                else:
                    reporter.emit(progress, stage, message=detail)
            elif stage == UpEnlargeProgressStage.MODEL_LOADING:
                reporter.emit_model_loading(detail or "校验超分后端资源…")
            elif stage == UpEnlargeProgressStage.MERGING:
                reporter.emit_merging(detail or "合帧并回填原音轨…")
            else:
                reporter.emit(progress, stage, message=detail)

        # 心跳任务：wrapper.run 是同步阻塞调用，期间定期推送心跳帧，
        # 重置 gateway 的 read timeout 计时（避免长视频下 502）。
        async def _heartbeat() -> None:
            while True:
                await asyncio.sleep(30)
                reporter.emit_heartbeat()

        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            result = await asyncio.to_thread(self.wrapper.run, infer_inputs, _progress)
        except Exception as e:
            logger.exception("视频放大失败")
            reporter.emit_error(f"视频高清放大失败: {e}", "infer_failed")
            return
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # ----- 6. 上报完成 -----
        # 产物契约（T0.2 核实）：网关 _normalize_video_result 读取
        # ``result.videos[].video_path / video_url``，故必须包一层 videos 数组。
        entry: dict[str, Any] = dict(result)
        final_result: dict[str, Any] = {
            "videos": [entry],
            "mode": "upscale",
            "workflow": "upscale",
            "variant": self.wrapper.loaded_variant,
            "backend": self.wrapper.loaded_backend,
            "ncnn_model": self.wrapper.loaded_ncnn_model,
            "device": self.wrapper._device,
            "elapsed": result.get("elapsed"),
            "input_video": Path(video_path).name,
        }

        logger.info(
            "视频放大完成: task_id=%s, video_url=%s, %dx%d, elapsed=%.1fs",
            task_id, entry.get("video_url"), entry.get("width", 0), entry.get("height", 0),
            float(result.get("elapsed") or 0.0),
        )

        await reporter.emit_completed(task_id, final_result)

    async def _status_impl(self) -> StatusResponse:
        """GET /status：返回引擎状态。"""
        raw_device = self.wrapper._device
        if raw_device.startswith("cuda") or raw_device.startswith("gpu"):
            device_name = "GPU (Vulkan/CUDA)"
        elif raw_device == "cpu":
            device_name = "CPU"
        else:
            device_name = "Vulkan 自动选择"

        return StatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=self.wrapper.loaded_variant,
            vram_used="0MB",  # ncnn 为进程外调用，显存占用由驱动侧持有
            loaded_loras=[],  # v1 不支持 LoRA
            last_used_at=self.state_machine.last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                "device": raw_device,
                "device_name": device_name,
                "is_loaded": self.wrapper.is_loaded,
                "backend": self.wrapper.loaded_backend,
                "ncnn_model": self.wrapper.loaded_ncnn_model,
                "task_type": self.task_type,
                "capability": CAPABILITY,
                "model_dir": str(self.config.model_dir),
                "ffmpeg": self.config.ffmpeg_path,
                "available_variants": [
                    {
                        "name": v.name,
                        "available": v.available,
                        "backend": v.backend,
                        "ncnn_model": v.ncnn_model,
                        "scale_range": v.scale_range,
                        "vram_required": v.vram_required,
                    }
                    for v in self._available_variants
                ],
            },
        )

    # ============ LoRA（v1 预留）============

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """POST /lora/load：videoUpEnlarge v1 不支持 LoRA（预留接口）。"""
        raise RuntimeError("videoUpEnlarge v1 不支持 LoRA（预留接口，后续版本扩展）")

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """POST /lora/unload：videoUpEnlarge v1 不支持 LoRA（预留接口）。"""
        raise RuntimeError("videoUpEnlarge v1 不支持 LoRA（预留接口，后续版本扩展）")
