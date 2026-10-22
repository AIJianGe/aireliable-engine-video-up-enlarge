"""videoUpEnlarge 视频高清放大引擎（首期后端：Real-ESRGAN ncnn-vulkan）。

设计要点：
    - paradigm: integrated（继承 CustomEngine，实现 7 个 _xxx_impl 方法）
    - capability: video-upscale（网关 POST /v1/videos/upscale 硬性校验）
    - task_type : video_upscale（任务中心分类 / heavy_task 串行槽）
    - 首期后端不走 torch / CUDA，靠 Vulkan，因此引擎环境极轻

⚠️ 本引擎的三处易踩空契约（详见 ANALYSIS-TASKS/42-*）：
    1. 输入视频在 ``inputs["videos"][0]["path"]``，**不是**顶层 ``request.videos``
       ——网关 ``video_endpoints.py`` 构造 inputs 时把视频塞进 inputs.videos，
       且调用回调时未透传 ``videos=`` 参数（与 LTX/H3 引擎形状不同）
    2. 产物必须同时给出 ``video_path``（绝对路径）与 ``video_url``（/output 相对），
       供网关 ``_normalize_video_result`` 解析（只给 video_path 时网关会尝试
       自行推导 URL，路径不在 output_root 下则会推导失败）
    3. 音色管理红线：本引擎**不得**读写 ``data/voice_library.db`` 与 ``data/voices/``

Turing（SM 7.5）等老卡不受影响：ncnn-vulkan 路径不依赖 bf16 / FP8 /
FlashAttention，也不依赖 torch 版本。
"""

__version__ = "0.1.0"
