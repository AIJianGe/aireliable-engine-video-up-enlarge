"""videoUpEnlarge 视频高清放大引擎 Pydantic 数据模型。

扩展 SDK 的 VariantInfo / InferRequest.inputs，定义视频超分专属字段。

variant = 超分模型档位（首期 2 个，均为 Real-ESRGAN ncnn-vulkan 后端）：
    - videoUpEnlarge/realesrgan-x4plus      ：通用写实（默认）
    - videoUpEnlarge/realesr-animevideov3   ：动漫/插画/风格化视频

任务类型（task_type）：
    - video_upscale：视频空间分辨率提升（保留原音频）

infer 入参（放入 InferRequest.inputs，由网关 / WebUI 透传）：
    - videos      ：``[{"path": "<绝对路径>"}]`` —— **网关链路的主入口**
                    （gateway/src/gateway/api/video_endpoints.py:441-444 构造）
    - video       ：``{"path": ...}`` 或 ``{"url": ...}``（WebUI 直连兜底）
    - video_path  ：绝对路径字符串（WebUI 直连 / 旧链路兜底）
    - scale       ：放大倍率（1.0–4.0，与 target_* 互斥；网关未传时兜底 2.0）
    - target_width / target_height：目标尺寸（与 scale 互斥）
    - 其余为可选处理参数（tile / tta / threads / frame_format / crf / dn）
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aireliable_engine import VariantInfo

# ===== 能力声明 =====
CAPABILITY = "video-upscale"
ENGINE_TYPE = "video"

# ===== 任务类型（必须与 gateway video_endpoints.py 的 task_type 逐字一致）=====
TASK_TYPE = "video_upscale"

# ===== 倍率取值域（与前端 app.js 中 video-upscale 的能力声明一致）=====
SCALE_MIN = 1.0
SCALE_MAX = 4.0

# 输入视频格式（用于校验与错误提示）
SUPPORTED_INPUT_SUFFIXES = (
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".ts", ".mpg", ".mpeg",
)

# 抽帧格式取值域
FRAME_FORMATS = ("png", "jpg")


class VideoUpEnlargeVariantInfo(VariantInfo):
    """videoUpEnlarge 变体能力声明（扩展 SDK VariantInfo）。

    在标准字段之外补充视频超分特有信息：
        - backend            ：后端标识（首期固定 realesrgan-ncnn，二期可为 seedvr2）
        - ncnn_model         ：ncnn ``-n`` 参数对应的模型名（仅 ncnn 后端有意义）
        - scale_range        ：支持的倍率描述
        - supports_target_size：是否支持指定目标宽高（而非仅倍率）
    """

    model_config = ConfigDict(extra="allow")

    backend: str = Field(
        "realesrgan-ncnn",
        description="超分后端标识：realesrgan-ncnn（首期）/ seedvr2（二期）",
    )
    ncnn_model: str = Field(
        "",
        description="Real-ESRGAN ncnn 模型名（对应 CLI 的 -n 参数；非 ncnn 后端为空）",
    )
    scale_range: str = Field(
        "1.0-4.0",
        description="支持的放大倍率范围（ncnn 后端按 2/3/4 离散执行，其余走后置缩放）",
    )
    supports_target_size: bool = Field(
        True,
        description="是否支持指定目标宽高（target_width / target_height）",
    )


class VideoUpEnlargeInputs(BaseModel):
    """视频放大推理输入参数（放入 InferRequest.inputs）。

    所有字段均有默认值，网关 / WebUI 按需覆盖。
    ``extra="allow"`` 保证网关透传的未知字段不会导致解析失败。
    """

    model_config = ConfigDict(extra="allow")

    # ===== 输入视频（三种形状都接受，见模块 docstring）=====
    videos: list[dict[str, Any]] = Field(
        default_factory=list,
        description="输入视频列表（网关链路：inputs.videos[].path）",
    )
    video: Optional[dict[str, Any]] = Field(
        None,
        description="输入视频对象（WebUI 直连兜底：{path|url}）",
    )
    video_path: Optional[str] = Field(
        None,
        description="输入视频绝对路径（WebUI 直连 / 旧链路兜底）",
    )

    # ===== 放大目标（scale 与 target_* 互斥）=====
    scale: float = Field(
        2.0,
        description=f"放大倍率（{SCALE_MIN}–{SCALE_MAX}），与 target_width/height 互斥",
    )
    target_width: Optional[int] = Field(
        None, ge=1, description="目标宽度（与 scale 互斥）"
    )
    target_height: Optional[int] = Field(
        None, ge=1, description="目标高度（与 scale 互斥）"
    )

    # ===== 通用 =====
    prompt: Optional[str] = Field(None, description="提示词（本后端忽略，占位兼容）")
    workflow: str = Field("upscale", description="工作流标识（网关固定传 upscale）")

    # ===== 后端处理参数（可选，均有安全默认值）=====
    tile_size: Optional[int] = Field(
        None,
        description="ncnn 分块尺寸（-t，>=32；None/0 = 自动）。显存紧张时调小",
    )
    tta: bool = Field(False, description="启用 TTA 测试时增强（-x，更慢但更清晰）")
    threads: Optional[str] = Field(
        None, description="ncnn 线程配置 load:proc:save（-j，默认 1:2:2）"
    )
    gpu_id: Optional[str] = Field(
        None, description="ncnn GPU 设备（-g，默认由 device 配置推导）"
    )
    frame_format: Optional[str] = Field(
        None, description=f"抽帧图像格式：{' / '.join(FRAME_FORMATS)}（默认 png 无损）"
    )
    crf: Optional[int] = Field(
        None, ge=0, le=40, description="输出视频 CRF（libx264，默认 17）"
    )
    dn: Optional[int] = Field(
        None,
        description="降噪强度（-dn，0–10；仅部分 ncnn 构建支持，默认不传该参数）",
    )
    keep_temp: Optional[bool] = Field(
        None, description="保留抽帧临时目录（排障用）"
    )

    @model_validator(mode="after")
    def _validate_scale_and_target(self) -> "VideoUpEnlargeInputs":
        """scale 与 target_* 互斥 + scale 取值域校验（网关已挡，此处双保险）。"""
        has_target = bool(self.target_width or self.target_height)
        if has_target and self.scale and float(self.scale) != 2.0:
            # 仅在用户"显式"给了非默认 scale 时判定冲突：
            # 网关未传 scale 时会兜底写入 2.0，不能据此报错。
            raise ValueError(
                "scale 与 target_width/target_height 二选一，不能同时传"
            )
        if self.scale is not None and not (
            SCALE_MIN - 1e-6 <= float(self.scale) <= SCALE_MAX + 1e-6
        ):
            raise ValueError(
                f"scale {self.scale} 超出支持范围 {SCALE_MIN}–{SCALE_MAX}"
            )
        return self

    def resolve_input_video(self) -> Optional[str]:
        """解析输入视频路径（按优先级兜底，覆盖三种入参形状）。

        Returns:
            视频绝对路径字符串；三种形状都拿不到时返回 None
        """
        # 1) 网关链路：inputs.videos[].path
        if isinstance(self.videos, list):
            for item in self.videos:
                if isinstance(item, dict):
                    p = item.get("path") or item.get("file")
                    if p:
                        return str(p)
                elif isinstance(item, str) and item:
                    return item
        # 2) inputs.video.path
        if isinstance(self.video, dict):
            p = self.video.get("path")
            if p:
                return str(p)
        # 3) inputs.video_path
        if self.video_path:
            return str(self.video_path)
        return None
