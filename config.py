"""videoUpEnlarge 视频高清放大引擎配置。

从环境变量 + 命令行参数加载引擎运行所需的全部配置。

设计参考（同 VSR 引擎的双源真相消除策略）：
    - MODEL_SUBDIR 默认读环境变量 ``AIRELIABLE_MODEL_LAYOUT_BASE``
      （由调度层 EngineManager 启动子进程时同步 manifest.models.model_layout.base）
    - 优先级：环境变量 > 硬编码 fallback

命名与端口（ANALYSIS-TASKS/42-* 分析文档 §6.4）：
    - engine_id  : video_up_enlarge_98782fcd（目录 / manifest / config 三同名契约）
    - engine_name: VideoUpEnlarge（用户指定）
    - capability : video-upscale（网关硬性要求）
    - 端口       : 7013（7001-7008 已占用、7012 被 VSR 占用）
    - 模型布局   : video/upscale/realesrgan/

⚠️ 与 VSR 引擎的两处有意差异：
    1. 本引擎**不依赖 torch**（ncnn-vulkan 走 Vulkan），因此
       ``__post_init__`` 与 ``resolve_device()`` 都不 import torch 作为硬依赖
       （仅在可用时用于 auto 探测，import 失败静默降级）
    2. 设备语义映射到 ncnn 的 ``-g`` 参数（见 ``to_ncnn_gpu_arg``）
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ===== 常量 =====

ENGINE_ID = "video_up_enlarge_98782fcd"
ENGINE_NAME = "VideoUpEnlarge"
CAPABILITY = "video-upscale"
ENGINE_TYPE = "video"

# 默认端口（与 config/engines/video_up_enlarge_98782fcd.yaml 一致）
DEFAULT_PORT = 7013

# 模型目录布局（相对 MODELS_ROOT）
# 优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE（平台注入）
_DEFAULT_MODEL_SUBDIR = "video/upscale/realesrgan/"

# ===== Real-ESRGAN ncnn-vulkan 便携包（首期后端）=====
# 官方 Release 资产（BSD-3-Clause 可再分发；包内含 exe + 4 个 ncnn 模型）
NCNN_BUNDLE_FILENAME = "realesrgan-ncnn-vulkan-20220424-windows.zip"
NCNN_EXE_NAME = "realesrgan-ncnn-vulkan.exe"
NCNN_BUNDLE_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
)
# 便携包内模型子目录名（解包后模型位于 <bundle_root>/models/）
NCNN_MODELS_SUBDIR = "models"

# ncnn 支持的可选倍率（-s 参数取值域）
NCNN_SUPPORTED_SCALES = (2, 3, 4)


def _resolve_model_subdir() -> str:
    """解析模型子目录（优先读环境变量，双源真相消除）。"""
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_MODEL_SUBDIR


# 运行时常量
MODEL_SUBDIR = _resolve_model_subdir()


def resolve_ffmpeg(explicit: Optional[str] = None) -> Optional[str]:
    """查找可用的 ffmpeg 可执行文件。

    查找顺序（与 LTX-2.5 引擎 video_utils 的既有约定一致）：
        1. 显式传入（命令行 ``--ffmpeg``）
        2. 环境变量 ``AIRELIABLE_FFMPEG``
        3. ``imageio_ffmpeg.get_ffmpeg_exe()`` —— 随环境分发的完整版
           （系统 PATH 中的 ffmpeg 可能是裁剪版，缺 aac 等编码器）
        4. ``shutil.which("ffmpeg")`` —— 系统 PATH

    Returns:
        ffmpeg 可执行文件路径；全部失败返回 None
    """
    if explicit:
        return explicit
    env_val = os.environ.get("AIRELIABLE_FFMPEG")
    if env_val and Path(env_val).exists():
        return env_val
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def resolve_ffprobe(explicit: Optional[str] = None) -> Optional[str]:
    """查找可用的 ffprobe 可执行文件。

    查找顺序：
        1. 显式传入 / 环境变量 ``AIRELIABLE_FFPROBE``
        2. ffmpeg 同级目录下的 ffprobe（Windows 发行包布局）
        3. 系统 PATH

    注意：``imageio_ffmpeg`` 只分发 ffmpeg、不含 ffprobe，因此本函数可能返回
    None。此时探测环节自动降级为解析 ``ffmpeg -i`` 的 stderr（见
    ``upscale_wrapper._probe_via_ffmpeg``），不阻断流程。
    """
    if explicit:
        return explicit
    env_val = os.environ.get("AIRELIABLE_FFPROBE")
    if env_val and Path(env_val).exists():
        return env_val
    sibling = Path(resolve_ffmpeg() or "").parent
    if sibling.is_dir():
        for name in ("ffprobe.exe", "ffprobe"):
            cand = sibling / name
            if cand.exists():
                return str(cand)
    return shutil.which("ffprobe")


@dataclass
class EngineConfig:
    """videoUpEnlarge 引擎运行配置。"""

    # ===== 基础 =====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"

    # ===== 路径 =====
    models_root: Path = Path("models")
    data_root: Path = Path("data")
    project_root: Path = Path(".")
    output_root: Path = Path("output")
    # 抽帧 / 合帧临时目录根（默认落在项目盘，避免系统盘被数 GB 抽帧撑满）
    work_root: Optional[Path] = None

    # ===== 推导路径 =====
    model_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    # ===== 设备 =====
    # auto: 有 CUDA 则 cuda:0，否则交给 Vulkan 自动选择
    # cpu : 强制 CPU（ncnn -g -1）
    # cuda:N / gpu:N：指定 GPU
    device: str = "auto"

    # ===== 外部工具 =====
    ffmpeg_path: Optional[str] = None
    ffprobe_path: Optional[str] = None

    # ===== 处理参数默认值 =====
    # ncnn 线程配置 load:proc:save（默认 1:2:2，与官方一致）
    ncnn_threads: str = "1:2:2"
    # 抽帧图像格式（png 无损 / jpg 省盘省时，但有一次 JPEG 损失）
    frame_format: str = "png"
    # 输出视频 CRF（libx264）
    output_crf: int = 17
    # 保留临时抽帧目录（排障用）
    keep_temp: bool = False

    # ===== 日志 =====
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """推导子路径、统一绝对化并创建运行时目录。

        ⚠️ 必须绝对化（v1.1 修复）：
            本引擎会以 ``cwd=<realesrgan-ncnn-vulkan.exe 所在目录>`` 调用外部命令
            （见 ``upscale_wrapper._upscale_frames``，ncnn 需要以自身目录为 cwd）。
            此时任何相对路径都会解析到**引擎可执行文件目录**而非项目根。实测后果：
            ``-o .tmp/.../frames_out`` 指向不存在的目录 → ncnn 把 ``-o`` 当作
            输出**文件** → 无扩展名 → 报 "invalid outputpath extension type"，
            超分必然失败（返回码 4294967295）。
            在配置层一次性绝对化，从根上消除这一整类 cwd 依赖缺陷。
            （对照：抽帧 / 合帧用默认 cwd = 引擎进程 cwd = 项目根，因此不受影响，
             这也正是该缺陷隐蔽的原因。）
        """
        # 统一绝对化（对已是绝对路径的输入，resolve() 是幂等的）
        self.project_root = Path(self.project_root).resolve()
        self.models_root = Path(self.models_root).resolve()
        self.output_root = Path(self.output_root).resolve()

        # 模型目录（video/upscale/realesrgan/）
        self.model_dir = (self.models_root / MODEL_SUBDIR).resolve()

        # 日志目录（v0.2.15+ 统一到 logs/engines/）
        self.log_dir = (self.project_root / "logs" / "engines").resolve()

        # 临时工作目录根（与 output 同盘）
        if self.work_root is None:
            self.work_root = self.project_root / ".tmp" / "video_up_enlarge"
        self.work_root = Path(self.work_root).resolve()

        # 确保运行时目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.work_root.mkdir(parents=True, exist_ok=True)

    # ===== 设备解析 =====

    def resolve_device(self) -> str:
        """解析推理设备（在传给后端之前完成，不留 "auto" 歧义）。

        Returns:
            "cuda:0" / "cpu" / "auto"
            —— 其中 "auto" 表示"交给 Vulkan 自动选择"，是已解析的终态语义
            （ncnn 不传 -g 即为自动，不存在未决状态）

        说明：本引擎不依赖 torch。仅在 torch 恰好可用时用它做一次 CUDA 探测，
        用于把 "auto" 具体化为 cuda:0；torch 不可用（首期环境的常态）时保持
        "auto"，由 ncnn/Vulkan 自行挑选设备——这比强行回退 CPU 更正确。
        """
        device = (self.device or "auto").strip()
        if device == "cpu":
            return "cpu"
        if device.startswith(("cuda", "gpu")):
            return device
        # auto
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "auto"

    def to_ncnn_gpu_arg(self, device: Optional[str] = None) -> Optional[str]:
        """把设备语义映射为 ncnn ``-g`` 参数值。

        - "auto"        → None（不传 -g，Vulkan 自动选择）
        - "cpu"         → "-1"（ncnn-vulkan 约定的 CPU 回退）
        - "cuda:N"      → "N"
        - "gpu:N"       → "N"

        Returns:
            传给 ``-g`` 的字符串；None 表示不传该参数
        """
        dev = (device or self.resolve_device()).strip()
        if dev == "cpu":
            return "-1"
        if dev.startswith("cuda:"):
            return dev.split(":", 1)[1] or "0"
        if dev.startswith("gpu:"):
            return dev.split(":", 1)[1] or "0"
        if dev.isdigit():
            return dev
        return None

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "capability": self.capability,
            "engine_type": self.engine_type,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "model_dir": str(self.model_dir),
            "output_root": str(self.output_root),
            "work_root": str(self.work_root),
            "log_dir": str(self.log_dir),
            "device": self.device,
            "ffmpeg": self.ffmpeg_path,
            "ffprobe": self.ffprobe_path,
            "log_level": self.log_level,
        }


def parse_args(argv: Optional[list[str]] = None) -> EngineConfig:
    """解析命令行参数 + 环境变量，构造 EngineConfig。"""
    parser = argparse.ArgumentParser(description="videoUpEnlarge 视频高清放大引擎")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP 端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1）")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备：auto / cpu / cuda:0（默认 auto）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别（默认 INFO）")
    parser.add_argument("--models-root", type=str, default=None,
                        help="模型根目录（默认 models/）")
    parser.add_argument("--output-root", type=str, default=None,
                        help="产物根目录（默认 output/）")
    parser.add_argument("--work-root", type=str, default=None,
                        help="抽帧/合帧临时目录根（默认 .tmp/video_up_enlarge/）")
    parser.add_argument("--ffmpeg", type=str, default=None,
                        help="ffmpeg 可执行文件路径（默认自动探测）")
    parser.add_argument("--ffprobe", type=str, default=None,
                        help="ffprobe 可执行文件路径（默认自动探测）")
    parser.add_argument("--keep-temp", action="store_true",
                        help="保留抽帧临时目录（排障用）")

    args = parser.parse_args(argv)

    models_root = Path(args.models_root or os.environ.get("AIRELIABLE_MODELS_ROOT", "models"))
    output_root = Path(args.output_root or os.environ.get("AIRELIABLE_OUTPUT_ROOT", "output"))
    work_root = Path(args.work_root) if args.work_root else None

    return EngineConfig(
        port=args.port,
        host=args.host,
        device=args.device,
        log_level=args.log_level,
        models_root=models_root,
        output_root=output_root,
        work_root=work_root,
        ffmpeg_path=resolve_ffmpeg(args.ffmpeg),
        ffprobe_path=resolve_ffprobe(args.ffprobe),
        keep_temp=bool(args.keep_temp),
    )
