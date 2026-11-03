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

任务 71 P0-B：通用帧构造 / 任务上报 / 心跳 / 完成帧全部由 SDK
``BaseProgressReporter`` 提供，本文件只保留阶段常量与语义化方法。
"""

from __future__ import annotations

from aireliable_engine.progress import BaseProgressReporter

from .upscale_wrapper import UpEnlargeProgressStage


# 进度阶段（复用 wrapper 的定义，避免双源漂移）
ProgressStage = UpEnlargeProgressStage


class ProgressReporter(BaseProgressReporter):
    """videoUpEnlarge 进度上报器（阶段语义 + SDK 通用帧）。"""

    error_type = "video_upscale_error"
    heartbeat_progress = 0.65
    heartbeat_stage = UpEnlargeProgressStage.HEARTBEAT
    heartbeat_message = "超分处理中（心跳）"

    # ===== 阶段便捷方法 =====

    def emit_start(self, task_id: str, description: str = "") -> None:
        """开始处理。"""
        self.set_task_id(task_id)
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
        await super().emit_completed(
            {"task_id": task_id, **result},
            {"total_seconds": self.elapsed},
        )
