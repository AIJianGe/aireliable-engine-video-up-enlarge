"""videoUpEnlarge 视频高清放大引擎 HTTP 服务。

8 个标准端点 / CORS / 日志端点 / 静态挂载由 SDK 骨架（``build_engine_app``）
提供；本文件只保留引擎特有路由（``/info`` + 根路径）。

启动：
    uv run --no-sync --project envs/video-upenlarge \\
        python -m engines.video_up_enlarge_98782fcd.server --port 7013

端点：
    GET  /health  /capabilities  /status          （骨架）
    POST /load  /unload  /infer(SSE)  /lora/*     （骨架；LoRA 为 v1 预留）
    GET  /output/*       静态产物文件（放大后的视频）
    GET|POST /log_level  （骨架）
    GET  /info           引擎信息（恒可用）
    GET  /ui/*           WebUI（webui/index.html 存在时）
    GET  /               有 WebUI → 307 /ui/；否则返回引擎信息

⚠️ 本引擎**不含**原生 WebUI 时（分析文档 §9 D6：先不需要，二期按需补）
   不挂载 /ui，也不做 307 重定向，根路径返回引擎信息 JSON。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# ===== sys.path 注入（保证任意 cwd 下均可启动）=====
# 以 `python -m engines.<id>.server` 启动时项目根已在 sys.path；
# 但若被以 `python server.py` 或绝对路径方式拉起，包相对导入会失败。
# 这里显式注入项目根 + 引擎根目录，兼容两种启动方式。
_ENGINE_ROOT = Path(__file__).resolve().parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))
_PROJECT_ROOT = _ENGINE_ROOT.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from aireliable_engine import build_engine_app, run_engine_server

from .config import EngineConfig, parse_args
from .engine import VideoUpEnlargeEngine

logger = logging.getLogger(__name__)

_WEBUI_DIR = _ENGINE_ROOT / "webui"
_HAS_WEBUI = (_WEBUI_DIR / "index.html").is_file()


# ===== 全局引擎实例 =====
# 由 SDK 骨架的 startup_hook 绑定，供本文件的路由使用
_engine: Optional[VideoUpEnlargeEngine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> VideoUpEnlargeEngine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== Lifespan 钩子（由 SDK 骨架调用）=====

async def _on_startup(engine: VideoUpEnlargeEngine) -> None:
    """启动钩子：绑定全局实例 + 异步刷新 variant 扫描。"""
    global _engine

    _engine = engine
    # 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    await engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d, default_variant=%s",
        engine.state_machine.state.value,
        len(engine._available_variants),
        engine.default_variant,
    )


async def _on_shutdown(engine: VideoUpEnlargeEngine) -> None:
    """关闭钩子：卸载后端并清空全局引用。"""
    global _engine, _config

    logger.info("===== videoUpEnlarge 视频高清放大引擎关闭 =====")
    try:
        if engine.wrapper.is_loaded:
            try:
                await engine._unload_impl()
            except Exception as e:
                logger.warning("关闭时卸载后端失败: %s", e)
    finally:
        _engine = None
        _config = None


# ===== 引擎特有路由：/info + 根路径 =====

router = APIRouter()


def _engine_info() -> dict:
    """引擎信息（``/info`` 与无 WebUI 时的根路径共用同一份 JSON）。"""
    cfg = get_config()
    eng = get_engine()
    note = (
        "WebUI: GET /ui/；引擎信息: GET /info"
        if _HAS_WEBUI
        else "本引擎未提供 WebUI（webui/ 目录缺失），请通过平台任务中心或 API 调用"
    )
    return {
        "engine_id": cfg.engine_id,
        "engine_name": cfg.engine_name,
        "capability": cfg.capability,
        "type": cfg.engine_type,
        "version": "0.1.0",
        "backend": eng.wrapper.loaded_backend or "realesrgan-ncnn",
        "variants": [v.name for v in eng._available_variants],
        "endpoints": [
            "GET /health", "GET /capabilities", "POST /load", "POST /unload",
            "POST /infer", "GET /status", "POST /lora/load", "POST /lora/unload",
            "GET /output/*", "GET|POST /log_level",
        ]
        + (["GET /ui/", "GET /info"] if _HAS_WEBUI else ["GET /"]),
        "webui": ("/ui/" if _HAS_WEBUI else None),
        "note": note,
    }


@router.get("/info", include_in_schema=False)
async def info() -> dict:
    """引擎信息（恒可用）。"""
    return _engine_info()


@router.get("/", include_in_schema=False)
async def root():
    """根路径：有 WebUI 则跳转，否则返回引擎信息。"""
    if _HAS_WEBUI:
        # 307 保留 method 语义不变（脚本探测根路径时不被改写）
        return RedirectResponse(url="/ui/", status_code=307)
    return _engine_info()


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config

    if config is None:
        config = parse_args()
    _config = config

    static_dirs = {"/output": config.output_root}
    if _HAS_WEBUI:
        static_dirs["/ui"] = _WEBUI_DIR

    return build_engine_app(
        VideoUpEnlargeEngine,
        config=config,
        title="VideoUpEnlarge Engine",
        description=(
            "AIReliable 平台的 videoUpEnlarge 视频高清放大引擎"
            "（首期后端：Real-ESRGAN ncnn-vulkan，保留原音频）"
        ),
        version="0.1.0",
        static_dirs=static_dirs,
        redirect_root=None,
        extra_routers=[router],
        startup_hook=_on_startup,
        shutdown_hook=_on_shutdown,
    )


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global app

    config = parse_args()
    app = create_app(config)
    run_engine_server(app, config=config)


if __name__ == "__main__":
    main()
