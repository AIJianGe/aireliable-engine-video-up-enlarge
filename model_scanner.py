"""videoUpEnlarge 模型/后端资源扫描器（2 档位 variant 版）。

扫描 ``models/video/upscale/realesrgan/`` 目录，识别可用的超分变体。

2 variant 设计（ANALYSIS-TASKS/42-* 分析文档 §5.3）：
    - videoUpEnlarge/realesrgan-x4plus    ：realesrgan-x4plus（通用写实，默认）
    - videoUpEnlarge/realesr-animevideov3 ：realesr-animevideov3（动漫/风格化视频）

v0.2.11+ 方案 B 约定：
    优先调用平台 API（``engine.get_variants()``）获取 variant 可用性，
    平台不可达时 fallback 到本地扫描。

本地 fallback 检查策略（project-rules §第二部分 5）：
    - 只检查标志文件存在性，**不校验大小**（下载一半时也不能让 variant 从 UI 消失，
      否则用户失去下载入口）
    - 完整性校验由平台 API（key_files 机制）或 ``wrapper.load()`` 兜底

⚠️ 本引擎的标志文件是"任一存在即可"（ANY 语义），与 VSR 的"全部必须存在"（ALL）
不同：因为 ncnn 便携包是一个 zip，用户在平台模型管理里下载到 zip 之后、
尚未在首次 load 时解包之前，就应该能在 UI 里看到 variant 并触发下载/更新。

模型目录结构（与 manifest.yaml model_files.target_dir 一致）：
    models/video/upscale/realesrgan/
    ├── realesrgan-ncnn-vulkan-20220424-windows.zip   ← 平台下载的便携包
    ├── realesrgan-ncnn-vulkan.exe                    ← 首次 load 离线解包后
    └── models/                                       ← 解包出的 ncnn 权重
        ├── realesrgan-x4plus.bin / .param
        └── realesr-animevideov3-x2|3|4.bin / .param
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import NCNN_BUNDLE_FILENAME, NCNN_EXE_NAME, NCNN_MODELS_SUBDIR
from .schemas import VideoUpEnlargeVariantInfo

logger = logging.getLogger(__name__)


# ===== 已知变体定义 =====
# ⚠️ "name" 必须与 manifest.yaml 的 models.variants[].name 逐字一致
#    （variant 扫描匹配契约，双处不一致 → UI 扫不到 → 无法触发下载）
_KNOWN_VARIANTS: list[dict] = [
    # ============================================================
    # 1. 通用写实（默认）
    # ============================================================
    {
        "name": "videoUpEnlarge/realesrgan-x4plus",
        "alias": "RealESRGAN-x4plus",
        "base_model": "Real-ESRGAN",
        "precision": "fp16",
        "loader": "ncnn-vulkan",
        "vram_required": "2GB",
        "backend": "realesrgan-ncnn",
        "ncnn_model": "realesrgan-x4plus",
        "scale_range": "2x / 3x / 4x",
        "supports_target_size": True,
        "description": "通用写实档：RealESRGAN_x4plus（照片/实拍/AI 生成写实视频）",
    },
    # ============================================================
    # 2. 动漫 / 风格化视频
    # ============================================================
    {
        "name": "videoUpEnlarge/realesr-animevideov3",
        "alias": "RealESRGAN-animevideo",
        "base_model": "Real-ESRGAN (AnimeVideo-v3)",
        "precision": "fp16",
        "loader": "ncnn-vulkan",
        "vram_required": "2GB",
        "backend": "realesrgan-ncnn",
        "ncnn_model": "realesr-animevideov3",
        "scale_range": "2x / 3x / 4x",
        "supports_target_size": True,
        "description": "动漫视频档：RealESRGAN AnimeVideo-v3（动漫/插画/风格化视频，官方视频专用小模型）",
    },
]

# 便携包内 ncnn 权重文件名（按真实发布包核对：模型位于 <bundle>/models/）
# realesr-animevideov3 按倍率拆分为 -x2 / -x3 / -x4 三个文件
_BUNDLE_MARKERS: list[str] = [
    NCNN_EXE_NAME,          # 已解包
    NCNN_BUNDLE_FILENAME,   # 仅下载了 zip，尚未解包
]


def _ncnn_weight_names(ncnn_model: str) -> list[str]:
    """返回某 ncnn 模型"可接受"的权重文件名列表（满足其一即视为就位）。

    - realesr-animevideov3：发布包按倍率拆分为 ``-x2`` / ``-x3`` / ``-x4``
    - 其余模型：单一 ``<model>.bin`` / ``<model>.param``

    Returns:
        不带扩展名的权重基名列表（调用方拼接 .bin / .param）
    """
    if ncnn_model == "realesr-animevideov3":
        return [f"{ncnn_model}-x2", f"{ncnn_model}-x3", f"{ncnn_model}-x4"]
    return [ncnn_model]


class ModelScanner:
    """超分资源扫描器（2 档位 variant 版）。

    v0.2.11+ 方案 B：优先调用平台 API，fallback 到本地扫描。
    """

    def __init__(
        self,
        models_root: Path,
        model_dir: Path,
        engine=None,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化扫描器。

        Args:
            models_root: 模型根目录（如 models/）
            model_dir: 本引擎模型目录（video/upscale/realesrgan/）
            engine: 可选的 BaseEngine 实例（方案 B 注入）。
                传入时优先调用平台 API 检测 variant；为 None 时走本地扫描。
            logger: 可选的日志器
        """
        self.models_root = models_root
        self.model_dir = model_dir
        self.engine = engine
        self._logger = logger or logging.getLogger(__name__)

    # ===== 扫描入口 =====

    async def scan_variants_async(self) -> list[VideoUpEnlargeVariantInfo]:
        """方案 B：异步扫描，优先调用平台 API。"""
        if self.engine is not None:
            try:
                platform_variants = await self.engine.get_variants()
            except Exception as e:
                self._logger.warning("平台 API 获取 variants 失败，回退本地扫描: %s", e)
                platform_variants = None
            if platform_variants is not None:
                return self._merge_with_known(platform_variants)

        return self._scan_locally()

    def scan_variants(self) -> list[VideoUpEnlargeVariantInfo]:
        """同步扫描接口（fallback 本地扫描）。"""
        return self._scan_locally()

    # ===== 平台 API 合并 =====

    def _merge_with_known(self, platform_variants: list[dict]) -> list[VideoUpEnlargeVariantInfo]:
        """合并平台 variant 数据 + _KNOWN_VARIANTS 引擎特有字段。"""
        known_lookup: dict[str, dict] = {v["name"]: v for v in _KNOWN_VARIANTS}

        result: list[VideoUpEnlargeVariantInfo] = []
        for pv in platform_variants:
            v_name = pv.get("name", "")
            kv = known_lookup.get(v_name, {})
            if not kv:
                # 平台声明了引擎不认识（或已下线）的 variant：跳过，避免误报可用
                self._logger.debug("忽略未知 variant（不在 _KNOWN_VARIANTS 中）: %s", v_name)
                continue

            available = bool(pv.get("available", False))
            result.append(VideoUpEnlargeVariantInfo(
                name=v_name,
                base_model=pv.get("base_model", kv.get("base_model", "")),
                precision=pv.get("precision", kv.get("precision", "fp16")),
                loader=pv.get("loader", kv.get("loader", "ncnn-vulkan")),
                vram_required=pv.get("vram_required", kv.get("vram_required", "2GB")),
                available=available,
                backend=kv.get("backend", "realesrgan-ncnn"),
                ncnn_model=kv.get("ncnn_model", ""),
                scale_range=kv.get("scale_range", "1.0-4.0"),
                supports_target_size=kv.get("supports_target_size", True),
            ))
            if available:
                self._logger.info("发现模型变体: %s (%s)", v_name, kv.get("alias", ""))

        if not any(v.available for v in result):
            self._warn_nothing_found()
        return result

    # ===== 本地扫描 =====

    def _scan_locally(self) -> list[VideoUpEnlargeVariantInfo]:
        """fallback 本地扫描（只检查标志文件存在性，不校验大小）。"""
        capabilities: list[VideoUpEnlargeVariantInfo] = []

        if not self.model_dir.exists():
            self._logger.warning("模型目录不存在: %s", self.model_dir)
            self._warn_nothing_found()
            return capabilities

        for variant_def in _KNOWN_VARIANTS:
            if not self._check_variant_ready(variant_def["name"]):
                self._logger.debug(
                    "变体 %s 尚未就位（缺 %s 或 %s）: %s",
                    variant_def["name"], NCNN_EXE_NAME, NCNN_BUNDLE_FILENAME, self.model_dir,
                )
                continue

            cap = VideoUpEnlargeVariantInfo(
                name=variant_def["name"],
                base_model=variant_def["base_model"],
                precision=variant_def["precision"],
                loader=variant_def["loader"],
                vram_required=variant_def["vram_required"],
                available=True,
                backend=variant_def["backend"],
                ncnn_model=variant_def["ncnn_model"],
                scale_range=variant_def["scale_range"],
                supports_target_size=variant_def["supports_target_size"],
            )
            capabilities.append(cap)
            self._logger.info(
                "发现模型变体: %s (%s, ncnn_model=%s)",
                cap.name, variant_def["alias"], variant_def["ncnn_model"],
            )

        if not capabilities:
            self._warn_nothing_found()

        return capabilities

    def _check_variant_ready(self, variant_name: str) -> bool:
        """检查 variant 资源是否就位（ANY 语义：exe 已解包 或 zip 已下载）。"""
        for marker in _BUNDLE_MARKERS:
            if (self.model_dir / marker).exists():
                return True
        return False

    # ===== 路径与定义解析 =====

    def resolve_variant_path(self, variant_name: str) -> Optional[Path]:
        """根据变体名解析后端资源目录。

        Returns:
            资源根目录（如 models/video/upscale/realesrgan），不存在返回 None
        """
        for variant_def in _KNOWN_VARIANTS:
            if variant_def["name"] == variant_name:
                return self.model_dir if self.model_dir.exists() else None
        return None

    def get_variant_definition(self, variant_name: str) -> Optional[dict]:
        """获取 variant 的完整定义（供 wrapper.load() 定位 ncnn 模型与校验文件）。"""
        for v in _KNOWN_VARIANTS:
            if v["name"] == variant_name:
                return v
        return None

    def expected_ncnn_weights(self, variant_name: str) -> list[str]:
        """返回该 variant 期望的 ncnn 权重基名（供错误信息与校验使用）。"""
        vdef = self.get_variant_definition(variant_name) or {}
        ncnn_model = vdef.get("ncnn_model") or ""
        if not ncnn_model:
            return []
        return _ncnn_weight_names(ncnn_model)

    def ncnn_models_dir(self) -> Path:
        """便携包内 ncnn 权重目录（解包后位于 <model_dir>/models/）。"""
        return self.model_dir / NCNN_MODELS_SUBDIR

    # ===== 诊断 =====

    def _warn_nothing_found(self) -> None:
        """扫描无结果时的统一告警（含具体路径 + 仓库 ID + 下载指引）。"""
        self._logger.warning(
            "未在 %s 中找到任何可用的 video-upscale 后端资源，"
            "请到「模型管理」界面下载 Real-ESRGAN ncnn-vulkan 便携包"
            "（release 资产：realesrgan-ncnn-vulkan-20220424-windows.zip，"
            "来源 GitHub xinntao/Real-ESRGAN v0.2.5.0，BSD-3-Clause）",
            self.model_dir,
        )
