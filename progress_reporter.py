"""videoUpEnlarge SSE 进度上报器。

把超分处理过程转换为 AIReliable 进度帧，并通过 SDK PlatformTaskReporter
同步任务状态到 gateway（v0.3.4+）。

进度阶段（与 upscale_wrapper.UpEnlargeProgressStage 对齐，单一真相源）：
    start          0.00          任务创建 / 开始
    preparing      0.05 - 0.35   探测源视频 + 抽帧（按已抽帧数推进）
    model_loading  0.36          后端资源就位校验 / 离线解包
    processing     0.40 - 0.90   超分（按已产出帧数推进）
    merging        0.92          合帧 + 回填源音轨
    completed      1.00          完成（含 video_path / video_url）
    error          -             失败

⚠️ 阶段名只使用上面这一组（不额外发明新阶段）：网关与前端对阶段文案有既有
渲染逻辑，新增未知阶段名只会退化为纯文本展示，收益低而风险高。抽帧进度因此
并入 ``preparing``，超分进度并入 ``processing``。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from aireliable_engine import (
    SSEStream,
    get_platform_task_reporter,
    make_error_frame,
    make_final_frame,
    make_frame,
    PlatformTaskReporter,
)

from .upscale_wrapper import UpEnlargeProgressStage

logger = logging.getLogger(__name__)


# 进度阶段（复用 wrapper 的定义，避免双源漂移）
ProgressStage = UpEnlargeProgressStage


class ProgressReporter:
    """videoUpEnlarge 进度上报器。

    封装 SSEStream，提供按阶段构造帧的便捷方法；
    v0.3.4+ 同时通过 SDK PlatformTaskReporter 上报任务状态到 gateway。
    """

    def __init__(
        self,
        stream: SSEStream,
        task_reporter: Optional[PlatformTaskReporter] = None,
    ):
        """初始化进度上报器。

        Args:
            stream: SSE 流
            task_reporter: 可选的 SDK 任务上报器；为 None 时仅推送 SSE
        """
        self.stream = stream
        self._start_time: float = time.time()
        self._task_reporter: Optional[PlatformTaskReporter] = task_reporter
        self._task_id: Optional[str] = None

    @property
    def elapsed(self) -> float:
        """已用时（秒）。"""
        return time.time() - self._start_time

    def set_task_id(self, task_id: str) -> None:
        """设置任务 ID（用于 SDK 上报）。"""
        self._task_id = task_id

    async def init_task_reporter(self) -> None:
        """v0.3.4+ 异步初始化 SDK 任务上报器。

        必须在 emit_* 方法之前调用（_infer_impl 入口处）。失败不阻断推理。
        """
        if self._task_reporter is None and self._task_id:
            try:
                self._task_reporter = await get_platform_task_reporter()
            except Exception as e:
                logger.warning("init_task_reporter 失败（不阻断推理）: %s", e)

    async def report_created(
        self,
        request_summary: dict,
        variant: str = "",
        task_type: str = "",
    ) -> None:
        """v0.3.4+ 上报任务创建到 gateway（status=queued）。"""
        if self._task_reporter and self._task_id:
            try:
                await self._task_reporter.report_created(
                    task_id=self._task_id,
                    request_summary=request_summary,
                    variant=variant,
                    task_type=task_type,
                )
            except Exception as e:
                logger.warning("report_created 失败（不阻断推理）: %s", e)

    # ===== SDK 上报（fire-and-forget）=====

    def _fire_running(self, stage: str) -> None:
        """fire-and-forget 上报任务进入 running 状态。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_running(self._task_id, stage=stage)

    def _fire_progress(self, progress: float, stage: str) -> None:
        """fire-and-forget 上报任务进度。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_progress(
                self._task_id, progress=progress, stage=stage,
            )

    def _fire_succeeded(self, result: dict) -> None:
        """fire-and-forget 上报任务成功。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_succeeded(self._task_id, result=result)

    def _fire_failed(self, error: dict) -> None:
        """fire-and-forget 上报任务失败。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_failed(self._task_id, error=error)

    # ===== 通用帧 =====

    def emit(self, progress: float, stage: str, **extra: Any) -> None:
        """发送一帧进度（线程安全）。

        超分的进度回调发生在工作线程中，因此统一使用 put_threadsafe，
        与 stream 绑定的主事件循环线程安全互通。
        """
        frame = make_frame(progress, stage, **extra)
        self.stream.put_threadsafe(frame)
        self._fire_progress(progress, stage)

    async def emit_async(self, progress: float, stage: str, **extra: Any) -> None:
        """发送一帧进度（协程内调用）。"""
        frame = make_frame(progress, stage, **extra)
        await self.stream.put(frame)
        self._fire_progress(progress, stage)

    def emit_error(self, message: str, error_type: str = "video_upscale_error") -> None:
        """发送错误帧。"""
        frame = make_error_frame(message, error_type)
        self.stream.put_threadsafe(frame)
        self._fire_failed({"type": error_type, "message": message})

    def emit_heartbeat(self) -> None:
        """发送心跳帧（重置 gateway httpx read timeout）。

        超分是分钟级长任务，期间可能出现长时间"无进度变化"的静默段
        （例如 ncnn 正在处理单张超大帧）。30s 周期推送心跳，
        避免 gateway read timeout → 502。
        """
        self.emit(
            0.65,
            ProgressStage.HEARTBEAT,
            message="超分处理中（心跳）",
            elapsed=self.elapsed,
        )

    # ===== 阶段便捷方法 =====

    def emit_start(self, task_id: str, description: str = "") -> None:
        """开始处理。"""
        self._task_id = task_id
        self.emit(
            0.0,
            ProgressStage.START,
            message="开始视频高清放大",
            task_id=task_id,
            description=description[:200],
        )
        self._fire_running(ProgressStage.START)

    def emit_preparing(self, message: str = "探测源视频（尺寸 / 帧率 / 音轨）…") -> None:
        """准备阶段：探测 + 抽帧（0.05 → 0.35）。"""
        self.emit(0.05, ProgressStage.PREPARING, message=message)

    def emit_extracting(self, progress: float, detail: str = "") -> None:
        """抽帧进度（归入 preparing 阶段，0.05 → 0.35）。"""
        ratio = max(0.0, min(1.0, progress))
        mapped = 0.05 + 0.30 * ratio
        self.emit(
            mapped,
            ProgressStage.PREPARING,
            message=detail or f"抽取视频帧 {int(ratio * 100)}%",
        )

    def emit_model_loading(self, message: str = "校验超分后端资源…") -> None:
        """后端资源就位校验 / 离线解包（0.36）。"""
        self.emit(0.36, ProgressStage.MODEL_LOADING, message=message)

    def emit_processing(self, progress: float, detail: str = "") -> None:
        """超分处理中（0.40 → 0.90）。"""
        ratio = max(0.0, min(1.0, progress))
        mapped = 0.40 + 0.50 * ratio
        self.emit(
            mapped,
            ProgressStage.PROCESSING,
            message=detail or f"超分处理 {int(ratio * 100)}%",
        )

    def emit_merging(self, message: str = "合帧并回填原音轨…") -> None:
        """合帧 + 回填源音轨（0.92）。"""
        self.emit(0.92, ProgressStage.MERGING, message=message)

    async def emit_completed(self, task_id: str, result: dict) -> None:
        """完成（progress=1.0，携带 result 供网关解析产物）。"""
        timing = {
            "total_seconds": self.elapsed,
        }
        final_payload = {
            "task_id": task_id,
            **result,
        }
        frame = make_final_frame(final_payload, timing)
        await self.stream.put(frame)
        self._fire_succeeded(final_payload)
