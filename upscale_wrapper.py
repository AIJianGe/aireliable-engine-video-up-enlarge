"""videoUpEnlarge 超分后端隔离层（首期：Real-ESRGAN ncnn-vulkan）。

本文件是"后端差异的唯一收敛点"：二期接入 SeedVR2 时只在此新增分支，
``engine.py`` / ``server.py`` / ``schemas.py`` 不需要改动。

首期后端特性：
    - 便携包 ``realesrgan-ncnn-vulkan.exe``（Vulkan），**不依赖 torch / CUDA**
    - 目录模式一次性处理整个抽帧目录（``-i frames_in -o frames_out``），
      避免"每帧拉起一次进程"的巨大开销
    - 官方参数：``-n`` 模型名 / ``-s`` 倍率 / ``-f`` 输出格式 /
      ``-t`` 分块 / ``-g`` GPU / ``-j`` 线程 / ``-x`` TTA / ``-m`` 模型目录

处理链路（视频 → 视频，保留原音轨）：
    1. 探测源视频（宽高 / 帧率 / 时长 / 帧数 / 音轨）
    2. ffmpeg 抽帧到临时目录（png 无损；可切 jpg）
    3. realesrgan-ncnn-vulkan 目录模式超分
    4. 校验产出帧数 → ffmpeg 合帧（源帧率）+ 回填源音轨
    5. 落 ``output_root``，同时返回绝对路径与 ``/output`` 相对 URL

三条硬约束：
    - **只校验不下载**：资源缺失抛 ``RuntimeError`` 并给出下载指引；唯一的
      "准备动作"是对**已经在本地**的便携包 zip 做一次离线解包（不联网）
    - **音轨必须保留**：抽出音轨 → 合帧时回填；源无音轨时在结果里显式标记
      ``has_audio=False``，不静默变成"无声但看起来正常"
    - **音色管理红线**：本文件不出现 ``voice_library`` / ``data/voices`` 任何逻辑
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .config import (
    NCNN_BUNDLE_FILENAME,
    NCNN_BUNDLE_URL,
    NCNN_EXE_NAME,
    NCNN_MODELS_SUBDIR,
    NCNN_SUPPORTED_SCALES,
    EngineConfig,
    resolve_ffmpeg,
    resolve_ffprobe,
)

logger = logging.getLogger(__name__)

# 进度回调签名：(progress 0-1, stage, detail)
ProgressCb = Callable[[float, str, str], None]

# 图像文件后缀（抽帧 / 超分产物识别）
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# 可直接 copy 进 mp4 容器的音频编码（其余一律重编码为 aac，避免 mux 失败）
_AUDIO_COPY_SAFE_CODECS = ("aac", "mp3", "ac3", "eac3", "alac")


class UpEnlargeProgressStage:
    """进度阶段名（单一真相源，progress_reporter 直接复用）。

    只使用网关与前端已有渲染逻辑覆盖的阶段名，不新增未知阶段。
    """

    START = "start"
    PREPARING = "preparing"
    MODEL_LOADING = "model_loading"
    PROCESSING = "processing"
    MERGING = "merging"
    COMPLETED = "completed"
    ERROR = "error"
    HEARTBEAT = "video_up_enlarge.heartbeat"


# ===== 数据模型 =====


@dataclass
class VideoInfo:
    """源视频探测结果。"""

    path: Path
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int
    has_audio: bool
    video_codec: str = ""
    audio_codec: str = ""
    pix_fmt: str = ""

    @property
    def est_frames(self) -> int:
        """估算总帧数（nb_frames 缺失时用 时长 × 帧率 推算）。"""
        if self.nb_frames > 0:
            return self.nb_frames
        if self.duration > 0 and self.fps > 0:
            return int(round(self.duration * self.fps))
        return 0


# ===== 子进程工具 =====


@dataclass
class _CmdResult:
    """子进程执行结果（统一为文本 + 可读错误）。"""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def tail(self, limit: int = 800) -> str:
        """错误摘要（stderr 优先，回退 stdout）。"""
        text = (self.stderr or self.stdout or "").strip()
        return text[-limit:]


def _no_window_kwargs() -> dict:
    """Windows 下隐藏子进程控制台窗口（避免抽帧期间弹黑框）。"""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def _run_cmd(cmd: list[str], cwd: Optional[str] = None) -> _CmdResult:
    """执行外部命令（不做超时限制：超分是分钟级长任务）。"""
    logger.debug("执行命令: %s", " ".join(str(c) for c in cmd))
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            capture_output=True,
            cwd=cwd,
            **_no_window_kwargs(),
        )
    except FileNotFoundError as e:
        return _CmdResult(-1, "", f"可执行文件不存在: {e}")
    except Exception as e:  # pragma: no cover - 兜底
        return _CmdResult(-1, "", f"{type(e).__name__}: {e}")
    return _CmdResult(
        proc.returncode,
        proc.stdout.decode("utf-8", "replace") if proc.stdout else "",
        proc.stderr.decode("utf-8", "replace") if proc.stderr else "",
    )


def _parse_fraction(value: Optional[str]) -> float:
    """解析 ffprobe 的分数形式（如 "30000/1001"）。"""
    if not value:
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_even(value: int) -> int:
    """yuv420p 要求偶数分辨率。"""
    value = max(2, int(value))
    return value if value % 2 == 0 else value + 1


def _png_size(path: Path) -> Optional[tuple[int, int]]:
    """从 PNG 头读取宽高（不引入 Pillow 依赖）。"""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
        if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        width, height = struct.unpack(">II", head[16:24])
        return int(width), int(height)
    except Exception:
        return None


def _jpeg_size(path: Path) -> Optional[tuple[int, int]]:
    """从 JPEG 的 SOF 段读取宽高（纯字节扫描，不引入 Pillow 依赖）。"""
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                marker = f.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None
                code = marker[1]
                # SOF0..SOF15（跳过 DHT=0xC4 / JPG=0xC8 / DAC=0xCC）
                if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                    f.read(3)  # 段长度(2) + 采样精度(1)
                    data = f.read(4)
                    if len(data) < 4:
                        return None
                    height, width = struct.unpack(">HH", data)
                    return int(width), int(height)
                seg_len_raw = f.read(2)
                if len(seg_len_raw) < 2:
                    return None
                seg_len = struct.unpack(">H", seg_len_raw)[0]
                if seg_len < 2:
                    return None
                f.seek(seg_len - 2, 1)
    except Exception:
        return None


def _image_size(path: Path) -> Optional[tuple[int, int]]:
    """读取图像宽高（PNG / JPEG；其他格式返回 None）。"""
    suffix = path.suffix.lower()
    if suffix == ".png":
        return _png_size(path)
    if suffix in (".jpg", ".jpeg"):
        return _jpeg_size(path)
    return None


def _count_images(directory: Path) -> int:
    """统计目录下的图像文件数（跳过隐藏/临时文件）。"""
    if not directory.is_dir():
        return 0
    count = 0
    try:
        for name in os.listdir(directory):
            if name.startswith("."):
                continue
            if name.lower().endswith(_IMAGE_SUFFIXES):
                count += 1
    except OSError:
        return 0
    return count


def _first_image_suffix(directory: Path) -> Optional[str]:
    """返回目录内首个图像文件的后缀（用于确定合帧的输入模式）。"""
    if not directory.is_dir():
        return None
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        lower = name.lower()
        if lower.endswith(_IMAGE_SUFFIXES):
            return Path(name).suffix
    return None


class _DirProgressPoller(threading.Thread):
    """目录文件数进度轮询器。

    ncnn 的目录模式不输出结构化进度，因此用"产出文件数 / 期望帧数"估算进度。
    ffmpeg 抽帧同理。轮询在线程中执行，通过 ``progress_cb`` 上报（回调内部使用
    ``put_threadsafe``，与主事件循环线程安全互通）。
    """

    def __init__(
        self,
        watch_dir: Path,
        total: int,
        report: Callable[[float, int], None],
        interval: float = 0.5,
    ):
        """
        Args:
            watch_dir: 被监视的目录（产出帧落在这里）
            total: 期望帧数（<=0 时只上报计数，不做百分比估算）
            report: ``report(ratio, count)`` 形式的回调
            interval: 轮询间隔（秒）
        """
        super().__init__(daemon=True, name="vuen-progress-poller")
        self._dir = watch_dir
        self._total = int(total or 0)
        self._report = report
        self._interval = interval
        self._stop = threading.Event()

    def run(self) -> None:  # pragma: no cover - 线程体
        while not self._stop.is_set():
            try:
                count = _count_images(self._dir)
                if self._total > 0:
                    ratio = min(1.0, count / float(self._total))
                    self._report(ratio, count)
            except Exception:
                pass
            self._stop.wait(self._interval)

    def stop(self) -> None:
        """终止轮询。"""
        self._stop.set()


# ===== 主类 =====


class UpEnlargeWrapper:
    """视频超分包装层（加载 / 推理 / 卸载）。"""

    def __init__(self, config: EngineConfig, logger_: Optional[logging.Logger] = None):
        """初始化包装层。

        Args:
            config: 引擎配置（提供 models/output/work_root 与 ffmpeg 路径）
            logger_: 可选日志器
        """
        self.config = config
        self._logger = logger_ or logger

        # 运行时状态
        self._loaded_variant: Optional[str] = None
        self._loaded_backend: str = ""
        self._loaded_ncnn_model: str = ""
        self._device: str = "auto"

        # 后端资源路径
        self._exe: Optional[Path] = None
        self._models_dir: Optional[Path] = None

        # 外部工具
        self._ffmpeg: Optional[str] = None
        self._ffprobe: Optional[str] = None
        self._ffmpeg_major: Optional[int] = None

    # ===== 状态属性 =====

    @property
    def is_loaded(self) -> bool:
        """后端资源是否已就位（exe + 权重均已解析）。"""
        return bool(self._exe and self._models_dir and self._loaded_variant)

    @property
    def loaded_variant(self) -> Optional[str]:
        return self._loaded_variant

    @property
    def loaded_backend(self) -> str:
        return self._loaded_backend

    @property
    def loaded_ncnn_model(self) -> str:
        return self._loaded_ncnn_model

    # ===== 加载 / 卸载 =====

    def load(
        self,
        variant: str,
        variant_path: Path,
        device: str = "auto",
        ncnn_model: str = "",
        **opts: Any,
    ) -> None:
        """校验后端资源并配置本次会话（**只校验不下载**）。

        Args:
            variant: variant 名（如 videoUpEnlarge/realesrgan-x4plus）
            variant_path: 后端资源根目录（model_dir）
            device: 设备（auto / cpu / cuda:N）
            ncnn_model: ncnn ``-n`` 模型名

        Raises:
            FileNotFoundError: variant 不合法或资源目录不存在
            RuntimeError: 资源缺失（含下载指引）/ ffmpeg 不可用 / 解包失败
        """
        if not variant:
            raise RuntimeError("未指定 variant，无法加载")
        if not ncnn_model:
            raise RuntimeError(f"variant {variant} 缺少 ncnn_model 定义（manifest 与 scanner 不一致？）")

        bundle_root = Path(variant_path)
        if not bundle_root.exists():
            raise FileNotFoundError(
                f"后端资源目录不存在: {bundle_root}"
                f"（请到「模型管理」界面下载 Real-ESRGAN ncnn-vulkan 便携包）"
            )

        # 1) 外部工具（ffmpeg 是硬依赖：抽帧/合帧都靠它）
        self._ffmpeg = resolve_ffmpeg(self.config.ffmpeg_path)
        self._ffprobe = resolve_ffprobe(self.config.ffprobe_path)
        if not self._ffmpeg:
            raise RuntimeError(
                "未找到 ffmpeg。请在引擎环境安装 imageio-ffmpeg"
                "（`uv sync` 已包含），或设置环境变量 AIRELIABLE_FFMPEG 指向 ffmpeg 可执行文件。"
            )

        # 2) 可执行文件（缺失时对本地 zip 做一次离线解包；**不联网下载**）
        self._exe = self._resolve_executable(bundle_root)

        # 3) ncnn 权重校验（只校验，不下载）
        self._models_dir = self._validate_weights(bundle_root, ncnn_model)

        # 4) 设备解析（不留 "auto" 歧义）
        self._device = device if (device and device != "auto") else self.config.resolve_device()

        self._loaded_variant = variant
        self._loaded_backend = "realesrgan-ncnn"
        self._loaded_ncnn_model = ncnn_model

        self._logger.info(
            "后端就绪: variant=%s, backend=ncnn-vulkan, model=%s, device=%s, "
            "exe=%s, models=%s, ffmpeg=%s",
            variant, ncnn_model, self._device, self._exe, self._models_dir, self._ffmpeg,
        )

    def unload(self) -> None:
        """释放后端配置（ncnn 为进程外调用，无显存常驻，仅清状态）。"""
        self._exe = None
        self._models_dir = None
        self._loaded_variant = None
        self._loaded_backend = ""
        self._loaded_ncnn_model = ""
        self._logger.info("后端已卸载（ncnn-vulkan 无常驻显存）")

    # ===== 资源定位 =====

    def _resolve_executable(self, bundle_root: Path) -> Path:
        """定位 ``realesrgan-ncnn-vulkan.exe``。

        查找顺序：
            1. ``<bundle_root>/realesrgan-ncnn-vulkan.exe``
            2. ``<bundle_root>/*/realesrgan-ncnn-vulkan.exe``（zip 保留顶层目录的情况）
            3. 若存在 ``<bundle_root>/<便携包文件名>.zip`` → **离线解包**后重试 1/2
            4. 仍找不到 → RuntimeError（含下载指引）

        Raises:
            RuntimeError: 可执行文件缺失或解包失败
        """
        found = self._find_exe(bundle_root)
        if found:
            return found

        zip_path = bundle_root / NCNN_BUNDLE_FILENAME
        if not zip_path.exists():
            # 兼容用户手工放置的其他 zip 名
            zips = sorted(bundle_root.glob("realesrgan*ncnn*vulkan*.zip"))
            zip_path = zips[0] if zips else zip_path

        if zip_path.exists():
            self._logger.info(
                "检测到本地便携包 %s，执行一次性离线解包（不联网）", zip_path.name
            )
            self._extract_bundle(zip_path, bundle_root)
            found = self._find_exe(bundle_root)
            if found:
                return found
            raise RuntimeError(
                f"便携包解包后仍未找到 {NCNN_EXE_NAME}: {bundle_root}。"
                f"请确认 zip 内容完整（官方资产应为 {NCNN_BUNDLE_FILENAME}）。"
            )

        raise RuntimeError(
            f"未找到超分可执行文件 {NCNN_EXE_NAME}（也可能缺便携包 zip）。\n"
            f"  期望路径: {bundle_root / NCNN_EXE_NAME}\n"
            f"  缺失文件: {NCNN_EXE_NAME}\n"
            f"  下载来源: {NCNN_BUNDLE_URL}\n"
            f"            （GitHub xinntao/Real-ESRGAN v0.2.5.0，BSD-3-Clause）\n"
            f"  放置目录: {bundle_root}\n"
            f"  操作方式: 打开「模型管理」界面下载本引擎模型（下载 zip 后引擎首次 /load 会自动离线解包）"
        )

    @staticmethod
    def _find_exe(bundle_root: Path) -> Optional[Path]:
        """在资源根目录（含一层子目录）中查找可执行文件。"""
        direct = bundle_root / NCNN_EXE_NAME
        if direct.is_file():
            return direct
        for sub in bundle_root.iterdir() if bundle_root.is_dir() else []:
            if sub.is_dir():
                cand = sub / NCNN_EXE_NAME
                if cand.is_file():
                    return cand
        return None

    def _extract_bundle(self, zip_path: Path, bundle_root: Path) -> None:
        """把便携包 zip 解包到 ``bundle_root``（**离线操作，不联网**）。

        处理两种打包布局：
            - 文件在 zip 根目录
            - 文件统一位于单层顶层目录下（如 ``realesrgan-ncnn-vulkan-20220424-windows/``）
              → 剥离该层，保证 ``<bundle_root>/realesrgan-ncnn-vulkan.exe`` 可直接命中

        Raises:
            RuntimeError: zip 损坏 / 解包失败 / 含非法路径
        """
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                if not names:
                    raise RuntimeError(f"便携包内容为空: {zip_path}")

                # 检测单层顶层目录
                tops = {n.split("/", 1)[0] for n in names}
                strip_prefix = ""
                if len(tops) == 1:
                    only_top = next(iter(tops))
                    if all("/" in n for n in names):
                        strip_prefix = only_top + "/"

                for name in names:
                    rel = name[len(strip_prefix):] if strip_prefix and name.startswith(strip_prefix) else name
                    if not rel or rel.startswith(("/", "\\")):
                        continue
                    target = (bundle_root / rel).resolve()
                    # 防 zip slip（路径穿越）
                    if bundle_root.resolve() not in target.parents and target != bundle_root.resolve():
                        raise RuntimeError(f"便携包含非法路径（拒绝解包）: {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)

            self._logger.info("便携包解包完成: %s -> %s", zip_path.name, bundle_root)
        except zipfile.BadZipFile as e:
            raise RuntimeError(
                f"便携包损坏，无法解包: {zip_path}（{e}）。"
                f"请在「模型管理」界面重新下载 {NCNN_BUNDLE_FILENAME}"
            ) from e
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"便携包解包失败: {zip_path}（{type(e).__name__}: {e}）") from e

    def _validate_weights(self, bundle_root: Path, ncnn_model: str) -> Path:
        """校验 ncnn 权重文件是否就位（**只校验不下载**）。

        Raises:
            RuntimeError: 权重缺失（含缺失清单 + 本地路径 + 下载来源）
        """
        candidates = self._weight_basenames(ncnn_model)
        search_dirs = [bundle_root / NCNN_MODELS_SUBDIR, bundle_root]
        search_dirs.extend(
            d for d in bundle_root.iterdir() if d.is_dir() and d.name != NCNN_MODELS_SUBDIR
        ) if bundle_root.is_dir() else None

        for directory in search_dirs:
            if not directory or not directory.is_dir():
                continue
            for base in candidates:
                if (directory / f"{base}.param").is_file() and (directory / f"{base}.bin").is_file():
                    return directory

        tried = ", ".join(str(d) for d in search_dirs if d)
        raise RuntimeError(
            f"未找到 ncnn 权重文件（模型 {ncnn_model}）。\n"
            f"  缺失文件: {', '.join(f'{b}.param/.bin' for b in candidates)}\n"
            f"  已搜索目录: {tried}\n"
            f"  下载来源: {NCNN_BUNDLE_URL}\n"
            f"            （GitHub xinntao/Real-ESRGAN v0.2.5.0，BSD-3-Clause）\n"
            f"  操作方式: 打开「模型管理」界面下载本引擎模型（便携包内置全部权重）"
        )

    @staticmethod
    def _weight_basenames(ncnn_model: str) -> list[str]:
        """返回某 ncnn 模型的权重基名候选（满足其一即可）。"""
        if ncnn_model == "realesr-animevideov3":
            return [f"{ncnn_model}-x{s}" for s in NCNN_SUPPORTED_SCALES] + [ncnn_model]
        return [ncnn_model]

    # ===== 主流程 =====

    def run(self, infer_inputs: dict[str, Any], progress_cb: ProgressCb) -> dict[str, Any]:
        """执行一次视频高清放大。

        Args:
            infer_inputs: 处理入参（video_path / scale / target_width / target_height /
                tile_size / tta / threads / gpu_id / frame_format / crf / dn / keep_temp）
            progress_cb: 进度回调 ``(progress, stage, detail)``（在工作线程中被调用）

        Returns:
            产物描述 dict（含 video_path / video_url / width / height / duration 等），
            由 ``engine.py`` 包装成网关可解析的 ``{"videos": [...]}``

        Raises:
            RuntimeError: 输入非法 / 后端未加载 / 抽帧或超分或合帧失败
        """
        if not self.is_loaded:
            raise RuntimeError("后端未加载：请先调用 /load（或由平台自动加载）")

        started = time.time()

        # ----- 1. 解析并校验输入 -----
        video_path = str(infer_inputs.get("video_path") or "").strip()
        if not video_path:
            raise RuntimeError("缺少输入视频路径（inputs.videos[0].path / inputs.video_path）")
        src = Path(video_path)
        if not src.is_file():
            raise RuntimeError(f"输入视频不存在: {src}")

        effective_scale, target_w, target_h = self._resolve_target(infer_inputs)
        tile = infer_inputs.get("tile_size")
        tta = bool(infer_inputs.get("tta"))
        threads = infer_inputs.get("threads") or self.config.ncnn_threads
        gpu_id = infer_inputs.get("gpu_id")
        fmt = (infer_inputs.get("frame_format") or self.config.frame_format or "png").lower()
        if fmt not in ("png", "jpg"):
            raise RuntimeError(f"不支持的抽帧格式: {fmt}（仅支持 png / jpg）")
        crf = infer_inputs.get("crf")
        crf = int(crf) if crf is not None else int(self.config.output_crf)
        dn = infer_inputs.get("dn")
        keep_temp = bool(infer_inputs.get("keep_temp") or self.config.keep_temp)

        # ----- 2. 探测源视频 -----
        progress_cb(0.05, UpEnlargeProgressStage.PREPARING, "探测源视频（尺寸 / 帧率 / 音轨）…")
        info = self.probe_video(src)
        if info.width <= 0 or info.height <= 0:
            raise RuntimeError(f"无法读取视频尺寸（文件损坏或格式不支持）: {src}")
        if info.fps <= 0:
            info.fps = 30.0
            self._logger.warning("源视频帧率探测失败，回退 30 fps: %s", src)

        out_w, out_h = self._compute_output_size(info, effective_scale, target_w, target_h)
        ncnn_scale, scale_fallback = self._pick_ncnn_scale(effective_scale, out_w / info.width)

        # 45-专项接缝修复：ncnn 未显式传 -t 时默认用很小的 tile（约 200px），
        # 768×448 这类输入会被切成多块独立超分，各块边界色温/白平衡不一致，
        # 拼接后产生规则网格接缝（画质方案 A 实测：竖缝 399/799/1199、横缝 399/799）。
        # 显式指定"整帧单 tile（≥ max 边长）"，让整帧一次超分、无接缝。
        # 本机 48GB 显存充裕，整帧单 tile 无显存压力。
        if not tile:
            tile = max(info.width, info.height)
            self._logger.info(
                "未指定 tile_size，默认使用整帧单 tile=%d（消除多 tile 接缝）", tile
            )

        total = info.est_frames

        self._logger.info(
            "开始放大: %s (%dx%d @%.3ffps, %d 帧, 音轨=%s) → %dx%d, "
            "ncnn_scale=%d%s, model=%s",
            src.name, info.width, info.height, info.fps, total,
            (info.audio_codec or "无"), out_w, out_h, ncnn_scale,
            "（倍率回退）" if scale_fallback else "", self._loaded_ncnn_model,
        )

        # ----- 3. 准备临时工作目录 -----
        work_root = self.config.work_root
        work_dir = Path(tempfile.mkdtemp(prefix="vuen_", dir=str(work_root) if work_root else None))
        frames_in = work_dir / "frames_in"
        frames_out = work_dir / "frames_out"
        frames_in.mkdir(parents=True, exist_ok=True)
        frames_out.mkdir(parents=True, exist_ok=True)

        try:
            # ----- 4. 抽帧 -----
            self._extract_frames(info, frames_in, fmt, total, progress_cb)
            frames_extracted = _count_images(frames_in)
            if frames_extracted <= 0:
                raise RuntimeError(f"抽帧结果为空（源视频可能无视频流或编码不受支持）: {src}")
            if total and frames_extracted != total:
                self._logger.warning(
                    "抽帧数(%d) 与探测帧数(%d) 不一致，以实际抽帧数为准", frames_extracted, total
                )
            total_frames = frames_extracted

            # 预估临时占用，提前给出可读提示（避免用户以为卡死）
            est_bytes = self._estimate_frames_bytes(frames_in, total_frames, out_w, out_h)
            self._logger.info(
                "抽帧完成: %d 帧 → %s（预估超分产出约 %.1f GB）",
                total_frames, frames_in, est_bytes / (1024 ** 3),
            )

            # ----- 5. 超分（scale<=1 时跳过，仅重编码）-----
            progress_cb(0.38, UpEnlargeProgressStage.MODEL_LOADING,
                        f"加载超分模型 {self._loaded_ncnn_model}…")
            if ncnn_scale >= 2:
                self._upscale_frames(
                    frames_in, frames_out, ncnn_scale, fmt, total_frames,
                    tile, tta, threads, gpu_id, dn, progress_cb,
                )
                frames_final = frames_out
                frames_produced = _count_images(frames_out)
                if frames_produced <= 0:
                    raise RuntimeError(
                        "超分未产出任何帧（可执行文件可能因缺少 Vulkan 运行时而失败）"
                    )
                if frames_produced != total_frames:
                    self._logger.warning(
                        "超分产出帧数(%d) 与输入帧数(%d) 不一致，按实际产出合帧",
                        frames_produced, total_frames,
                    )
            else:
                # scale == 1.0：不放大，仅走一次合帧（保留音轨 + 统一编码）
                self._logger.info("倍率 1.0：跳过超分，仅重编码并保留音轨")
                progress_cb(0.90, UpEnlargeProgressStage.PROCESSING, "跳过超分（倍率 1.0）")
                frames_final = frames_in

            # ----- 6. 合帧 + 回填音轨 -----
            progress_cb(0.92, UpEnlargeProgressStage.MERGING,
                        "合帧并回填原音轨…" if info.has_audio else "合帧…（源无音轨）")
            out_path = self._merge_frames(info, frames_final, out_w, out_h, crf, ncnn_scale)
            if not out_path.is_file() or out_path.stat().st_size <= 0:
                raise RuntimeError(f"合帧产物为空: {out_path}")

            # ----- 7. 组织返回（字段与网关 _normalize_video_result 契约对齐）-----
            url = self._to_output_url(out_path)
            elapsed = time.time() - started
            result: dict[str, Any] = {
                # 网关读取字段（videos[] 优先）
                "video_path": str(out_path),
                "video_url": url,
                "width": out_w,
                "height": out_h,
                "duration": info.duration,
                # 兼容/诊断字段
                "output_path": str(out_path),
                "fps": info.fps,
                "frame_count": _count_images(frames_final),
                "source_width": info.width,
                "source_height": info.height,
                "source_fps": info.fps,
                "source_duration": info.duration,
                "requested_scale": effective_scale,
                "ncnn_scale": ncnn_scale,
                "applied_scale": round(out_w / info.width, 4) if info.width else None,
                "has_audio": info.has_audio,
                "audio_codec": info.audio_codec,
                "video_codec": info.video_codec,
                "backend": self._loaded_backend,
                "ncnn_model": self._loaded_ncnn_model,
                "variant": self._loaded_variant,
                "device": self._device,
                "elapsed": round(elapsed, 2),
                "input_video": src.name,
            }
            self._logger.info(
                "放大完成: %s → %s (%dx%d, %.1fs)",
                src.name, out_path.name, out_w, out_h, elapsed,
            )
            return result

        finally:
            if keep_temp:
                self._logger.info("保留临时目录（keep_temp=true）: %s", work_dir)
            else:
                shutil.rmtree(work_dir, ignore_errors=True)

    # ===== 输入解析 =====

    @staticmethod
    def _resolve_target(infer_inputs: dict[str, Any]) -> tuple[float, Optional[int], Optional[int]]:
        """解析放大目标，返回 ``(scale, target_w, target_h)``。

        网关已做 scale / target_* 互斥校验，此处做双保险（引擎侧独立可直连调用）。
        """
        scale = infer_inputs.get("scale")
        scale_f = float(scale) if scale is not None else 2.0
        tw = infer_inputs.get("target_width")
        th = infer_inputs.get("target_height")
        tw = int(tw) if tw else None
        th = int(th) if th else None

        if (tw or th) and scale is not None and abs(scale_f - 2.0) > 1e-6:
            raise RuntimeError("scale 与 target_width/target_height 二选一，不能同时传")
        if not (1.0 - 1e-6 <= scale_f <= 4.0 + 1e-6):
            raise RuntimeError(f"scale {scale_f} 超出支持范围 1.0–4.0")
        return scale_f, tw, th

    @staticmethod
    def _compute_output_size(
        info: VideoInfo,
        scale: float,
        target_w: Optional[int],
        target_h: Optional[int],
    ) -> tuple[int, int]:
        """计算输出尺寸（保持宽高比，且保证偶数）。"""
        aspect = info.width / info.height if info.height else 1.0
        if target_w and target_h:
            return _to_even(target_w), _to_even(target_h)
        if target_w:
            return _to_even(target_w), _to_even(round(target_w / aspect))
        if target_h:
            return _to_even(round(target_h * aspect)), _to_even(target_h)
        return _to_even(round(info.width * scale)), _to_even(round(info.height * scale))

    def _pick_ncnn_scale(self, requested: float, target_ratio: float) -> tuple[int, bool]:
        """选择 ncnn ``-s`` 的整数倍率。

        ncnn 只接受 2/3/4（离散）；非整数倍率通过"取不小于目标的整数档 + 后置
        ffmpeg 缩放到精确尺寸"实现。同时规避"请求档位权重缺失"的情况
        （animevideov3 按倍率拆包，可能只下载了其中某个）。

        Returns:
            ``(ncnn_scale, fallback_used)``；``ncnn_scale <= 1`` 表示不需要超分
        """
        ratio = max(requested, target_ratio if target_ratio else requested)
        if ratio <= 1.0 + 1e-6:
            return 1, False

        wanted = next((s for s in NCNN_SUPPORTED_SCALES if s >= ratio - 1e-6), NCNN_SUPPORTED_SCALES[-1])

        if self._loaded_ncnn_model == "realesr-animevideov3" and self._models_dir:
            if self._has_animevideo_scale(wanted):
                return wanted, False
            available = [s for s in NCNN_SUPPORTED_SCALES if self._has_animevideo_scale(s)]
            if available:
                chosen = min(available, key=lambda s: abs(s - wanted))
                self._logger.warning(
                    "ncnn 倍率 %d 的权重缺失，回退到 %d（产物会按目标尺寸后置缩放）",
                    wanted, chosen,
                )
                return chosen, True
        return wanted, False

    def _has_animevideo_scale(self, scale: int) -> bool:
        """animevideov3 是否具备指定倍率的权重文件。"""
        if not self._models_dir:
            return False
        base = f"{self._loaded_ncnn_model}-x{scale}"
        return (self._models_dir / f"{base}.param").is_file() and (self._models_dir / f"{base}.bin").is_file()

    # ===== 探测 =====

    def probe_video(self, path: Path) -> VideoInfo:
        """探测视频元信息（优先 ffprobe，缺失时回退解析 ffmpeg -i 的 stderr）。"""
        if self._ffprobe:
            info = self._probe_via_ffprobe(path)
            if info:
                return info
            self._logger.warning("ffprobe 探测失败，回退 ffmpeg 解析: %s", path)
        return self._probe_via_ffmpeg(path)

    def _probe_via_ffprobe(self, path: Path) -> Optional[VideoInfo]:
        """用 ffprobe JSON 输出解析视频信息。"""
        res = _run_cmd([
            self._ffprobe or "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(path),
        ])
        if not res.ok:
            self._logger.warning("ffprobe 返回非零: %s", res.tail(300))
            return None
        try:
            data = json.loads(res.stdout or "{}")
        except json.JSONDecodeError:
            return None

        streams = data.get("streams") or []
        vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
        astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        if vstream is None:
            return None

        fmt = data.get("format") or {}
        fps = _parse_fraction(vstream.get("avg_frame_rate")) or _parse_fraction(vstream.get("r_frame_rate"))
        duration = 0.0
        for raw in (fmt.get("duration"), vstream.get("duration")):
            try:
                duration = float(raw)
                break
            except (TypeError, ValueError):
                continue

        nb_raw = vstream.get("nb_frames")
        try:
            nb_frames = int(nb_raw)
        except (TypeError, ValueError):
            nb_frames = 0

        return VideoInfo(
            path=path,
            width=int(vstream.get("width") or 0),
            height=int(vstream.get("height") or 0),
            fps=fps,
            duration=duration,
            nb_frames=nb_frames,
            has_audio=astream is not None,
            video_codec=str(vstream.get("codec_name") or ""),
            audio_codec=str((astream or {}).get("codec_name") or ""),
            pix_fmt=str(vstream.get("pix_fmt") or ""),
        )

    def _probe_via_ffmpeg(self, path: Path) -> VideoInfo:
        """回退方案：解析 ``ffmpeg -i <file>`` 的 stderr（fast，不解码）。"""
        res = _run_cmd([self._ffmpeg or "ffmpeg", "-hide_banner", "-i", str(path)])
        text = f"{res.stdout}\n{res.stderr}"

        duration = 0.0
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            duration = h * 3600 + mi * 60 + s

        width = height = 0
        fps = 0.0
        vcodec = ""
        pix_fmt = ""
        video_line = next((ln for ln in text.splitlines() if "Video:" in ln), "")
        if video_line:
            mres = re.search(r"(\d{2,5})x(\d{2,5})", video_line)
            if mres:
                width, height = int(mres.group(1)), int(mres.group(2))
            mfps = re.search(r"([\d.]+)\s*fps", video_line)
            if mfps:
                try:
                    fps = float(mfps.group(1))
                except ValueError:
                    fps = 0.0
            mcodec = re.search(r"Video:\s*([A-Za-z0-9_]+)", video_line)
            if mcodec:
                vcodec = mcodec.group(1)
            mpix = re.search(r"Video:.*?,\s*(yuv[\w\d]+|rgb[\w\d]*|gray\w*)", video_line)
            if mpix:
                pix_fmt = mpix.group(1)

        audio_line = next((ln for ln in text.splitlines() if "Audio:" in ln), "")
        acodec = ""
        if audio_line:
            mac = re.search(r"Audio:\s*([A-Za-z0-9_]+)", audio_line)
            if mac:
                acodec = mac.group(1)

        return VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            nb_frames=0,
            has_audio=bool(audio_line),
            video_codec=vcodec,
            audio_codec=acodec,
            pix_fmt=pix_fmt,
        )

    # ===== 抽帧 / 超分 / 合帧 =====

    def _extract_frames(
        self,
        info: VideoInfo,
        frames_in: Path,
        fmt: str,
        total: int,
        progress_cb: ProgressCb,
    ) -> None:
        """ffmpeg 抽帧到 ``frames_in``（进度按已产出帧数估算）。"""
        # 绝对化：避免未来任何 cwd 变化导致相对路径错位（v1.1 加固）
        pattern = str(Path(frames_in).resolve() / f"%08d.{fmt}")
        cmd = [self._ffmpeg or "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(Path(info.path).resolve())]
        # 抽帧质量：png 无损（qscale 仅对 jpg 生效）
        cmd += ["-qscale:v", "2" if fmt == "jpg" else "1", "-qmin", "1", "-qmax", "1"]
        cmd += [*self._fps_mode_args(), "-start_number", "1", pattern]

        def report(ratio: float, count: int) -> None:
            progress_cb(
                0.05 + 0.30 * ratio,
                UpEnlargeProgressStage.PREPARING,
                f"抽取视频帧 {count}/{total}" if total else f"抽取视频帧 {count}",
            )

        poller = _DirProgressPoller(frames_in, total, progress_cb, report)
        poller.start()
        try:
            res = _run_cmd(cmd)
        finally:
            poller.stop()
        if not res.ok:
            raise RuntimeError(f"抽帧失败（ffmpeg 返回 {res.returncode}）: {res.tail()}")

    def _fps_mode_args(self) -> list[str]:
        """返回与当前 ffmpeg 版本兼容的帧率处理参数（**必须是"标志+取值"成对**）。

        ffmpeg 5.1+ 用 ``-fps_mode passthrough`` 取代 ``-vsync 0``；
        旧版本（含较早的 imageio-ffmpeg 分发包）只认 ``-vsync 0``。

        ⚠️ 不可只返回标志名（历史缺陷，v1.1 修复）：
            ffmpeg 会把紧随其后的下一个参数当作该标志的取值。若只传 ``-fps_mode``，
            则 ``-start_number`` 被当成取值吞掉，紧随的 ``"1"`` 被误认为输出文件名，
            实测报 "Unable to choose an output format for '1'"，抽帧产出 **0 帧**，
            整条放大链路直接失败（4294967274 / 0xFFFFFFEA）。
        """
        if self._ffmpeg_major is None:
            self._ffmpeg_major = self._detect_ffmpeg_major()
        if self._ffmpeg_major >= 5:
            return ["-fps_mode", "passthrough"]
        return ["-vsync", "0"]

    def _detect_ffmpeg_major(self) -> int:
        """解析 ffmpeg 主版本号（失败按 4 处理，走 -vsync 兼容分支）。"""
        res = _run_cmd([self._ffmpeg or "ffmpeg", "-version"])
        m = re.search(r"ffmpeg version\s+(\d+)", res.stdout or res.stderr or "")
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        self._logger.warning("无法解析 ffmpeg 版本，按 4.x 兼容分支处理")
        return 4

    def _upscale_frames(
        self,
        frames_in: Path,
        frames_out: Path,
        ncnn_scale: int,
        fmt: str,
        total: int,
        tile: Optional[int],
        tta: bool,
        threads: Optional[str],
        gpu_id: Optional[str],
        dn: Optional[int],
        progress_cb: ProgressCb,
    ) -> None:
        """调用 realesrgan-ncnn-vulkan 目录模式超分。"""
        # 输出目录必须存在且为空（ncnn 不会自动建目录，残留旧帧会污染合帧）
        if frames_out.exists():
            shutil.rmtree(frames_out, ignore_errors=True)
        frames_out.mkdir(parents=True, exist_ok=True)

        assert self._exe is not None and self._models_dir is not None
        # ⚠️ 必须传绝对路径（v1.1 修复）：本函数以 cwd=<exe 所在目录> 调用 ncnn，
        # 相对路径会被解析到 exe 目录而非项目根，导致 -o 指向不存在的目录，
        # ncnn 遂把 -o 当作输出文件并报 "invalid outputpath extension type"。
        in_abs = Path(frames_in).resolve()
        out_abs = Path(frames_out).resolve()
        # 兜底：确保输出目录存在（ncnn 不会自动创建）
        out_abs.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self._exe),
            "-i", str(in_abs),
            "-o", str(out_abs),
            "-n", self._loaded_ncnn_model,
            "-s", str(ncnn_scale),
            "-f", fmt,
            "-m", str(Path(self._models_dir).resolve()),
        ]
        if tile:
            cmd += ["-t", str(int(tile))]
        if tta:
            cmd += ["-x"]
        if threads:
            cmd += ["-j", str(threads)]
        gpu_arg = gpu_id or self.config.to_ncnn_gpu_arg(self._device)
        if gpu_arg:
            cmd += ["-g", str(gpu_arg)]
        # -dn 仅部分 ncnn 构建支持，因此只在显式传入时携带
        if dn is not None:
            cmd += ["-dn", str(int(dn))]

        def report(ratio: float, count: int) -> None:
            progress_cb(
                0.40 + 0.50 * ratio,
                UpEnlargeProgressStage.PROCESSING,
                f"超分处理 {count}/{total}",
            )

        poller = _DirProgressPoller(frames_out, total, progress_cb, report)
        poller.start()
        try:
            res = _run_cmd(cmd, cwd=str(self._exe.parent))
        finally:
            poller.stop()
        if not res.ok:
            raise RuntimeError(
                f"超分失败（{NCNN_EXE_NAME} 返回 {res.returncode}）: {res.tail()}"
            )

    def _merge_frames(
        self,
        info: VideoInfo,
        frames_dir: Path,
        out_w: int,
        out_h: int,
        crf: int,
        ncnn_scale: int,
    ) -> Path:
        """合帧 + 回填源音轨，落到 ``output_root``。"""
        suffix = _first_image_suffix(frames_dir) or ".png"
        # 绝对化：产物路径与帧序列都用绝对路径（v1.1 加固）
        pattern = str(Path(frames_dir).resolve() / f"%08d{suffix}")

        # 需要精确缩放？ncnn 只支持整数倍，非整数倍率（如 2.5x）在此补齐
        need_scale = False
        sample = self._first_image(frames_dir)
        if sample is not None:
            size = _image_size(sample)
            if size is not None:
                need_scale = (size[0] != out_w or size[1] != out_h)
            else:
                # 读不到实际尺寸（罕见格式）时按"ncnn 整数倍"假设判断
                need_scale = (info.width * ncnn_scale != out_w) or (info.height * ncnn_scale != out_h)

        out_dir = self.config.output_root / "video_upscale"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{self._stamp()}_{info.path.stem}_x{ncnn_scale if ncnn_scale > 1 else 1}.mp4"
        out_path = out_dir / out_name

        cmd = [self._ffmpeg or "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-framerate", f"{info.fps:.6f}", "-start_number", "1", "-i", pattern]
        if info.has_audio:
            cmd += ["-i", str(Path(info.path).resolve())]
        cmd += ["-map", "0:v:0"]
        if info.has_audio:
            cmd += ["-map", "1:a:0"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)),
                "-pix_fmt", "yuv420p"]
        if need_scale:
            cmd += ["-vf", f"scale={out_w}:{out_h}:flags=lanczos"]
        if info.has_audio:
            if (info.audio_codec or "").lower() in _AUDIO_COPY_SAFE_CODECS:
                cmd += ["-c:a", "copy"]
            else:
                # opus/vorbis 等无法直接 mux 进 mp4，重编码为 aac
                cmd += ["-c:a", "aac", "-b:a", "192k"]
            cmd += ["-shortest"]
        cmd += ["-r", f"{info.fps:.6f}", "-movflags", "+faststart", str(out_path)]

        res = _run_cmd(cmd)
        if not res.ok:
            raise RuntimeError(f"合帧失败（ffmpeg 返回 {res.returncode}）: {res.tail()}")
        return out_path

    @staticmethod
    def _first_image(directory: Path) -> Optional[Path]:
        """返回目录内排序后的第一张图像（用于探测实际分辨率）。"""
        if not directory.is_dir():
            return None
        for name in sorted(os.listdir(directory)):
            if name.startswith("."):
                continue
            if name.lower().endswith(_IMAGE_SUFFIXES):
                return directory / name
        return None

    # ===== 辅助 =====

    @staticmethod
    def _estimate_frames_bytes(frames_in: Path, total: int, out_w: int, out_h: int) -> float:
        """预估超分产出总字节（采样单帧大小 × 目标像素比 × 帧数）。"""
        sample = UpEnlargeWrapper._first_image(frames_in)
        if sample is None or total <= 0:
            return 0.0
        try:
            per_frame = sample.stat().st_size
        except OSError:
            return 0.0
        size = _image_size(sample)
        ratio = 1.0
        if size and size[0] and size[1]:
            ratio = (out_w * out_h) / float(size[0] * size[1])
        return per_frame * ratio * total

    def _to_output_url(self, path: Path) -> str:
        """把产物绝对路径转换为 ``/output`` 相对 URL（引擎静态挂载口径）。"""
        try:
            rel = Path(path).resolve().relative_to(self.config.output_root.resolve())
            return f"/output/{rel.as_posix()}"
        except Exception:
            # 不在 output_root 下时退化为文件名（网关仍可从 video_path 推导）
            return f"/output/video_upscale/{Path(path).name}"

    @staticmethod
    def _stamp() -> str:
        """时间戳 + 短 uuid（避免并发任务产物重名）。"""
        return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
