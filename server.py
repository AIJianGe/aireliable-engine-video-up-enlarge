"""videoUpEnlarge 视频高清放大引擎 HTTP 服务（FastAPI）。

实现 8 个标准 HTTP 端点 + 挂载输出目录。

启动：
    uv run --no-sync --project envs/video-upenlarge \\
        python -m engines.video_up_enlarge_98782fcd.server --port 7013

端点：
    GET  /health         健康检查
    GET  /capabilities   能力声明（可用超分档位）
    POST /load           校验后端资源 + 配置设备（**只校验不下载**）
    POST /unload         释放后端配置
    POST /infer          视频放大推理（SSE 流式进度）
    GET  /status         状态查询（含已加载变体 / 设备 / 后端）
    POST /lora/load      LoRA 加载（v1 预留，不支持）
    POST /lora/unload    LoRA 卸载（v1 预留，不支持）
    GET  /output/*       静态产物文件（放大后的视频）
    GET  /log_level      查询/调整日志级别（SDK 注册）
    GET  /               引擎信息（本引擎无原生 WebUI，按 D6 决策暂不提供）

⚠️ 本引擎**不含**原生 WebUI（分析文档 §9 D6：先不需要，二期按需补），
   因此不挂载 /ui，也不做 307 重定向（否则会 404）。
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from aireliable_engine import (
    ErrorResponse,
    HealthResponse,
    InferRequest,
    LoadRequest,
    LoraLoadRequest,
    LoraUnloadRequest,
    register_log_level_endpoint,
    setup_engine_logging,
)

from .config import EngineConfig, parse_args
from .engine import VideoUpEnlargeEngine

logger = logging.getLogger(__name__)


# ===== 全局引擎实例 =====
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


# ===== Lifespan 生命周期管理 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理：启动初始化引擎，关闭清理资源。"""
    global _engine, _config

    if _config is None:
        _config = parse_args()

    # 配置日志（v0.2.15+ 统一到 logs/engines/，依赖平台注入 AIRELIABLE_LOG_DIR）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info("===== videoUpEnlarge 视频高清放大引擎启动 =====")
    logger.info("配置: %s", _config.to_dict())

    _engine = VideoUpEnlargeEngine(_config)
    # 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    await _engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d, default_variant=%s",
        _engine.state_machine.state.value,
        len(_engine._available_variants),
        _engine.default_variant,
    )

    yield

    # 关闭清理
    logger.info("===== videoUpEnlarge 视频高清放大引擎关闭 =====")
    if _engine is not None and _engine.wrapper.is_loaded:
        try:
            await _engine._unload_impl()
        except Exception as e:
            logger.warning("关闭时卸载后端失败: %s", e)
    _engine = None


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config
    if config is not None:
        _config = config

    app = FastAPI(
        title="VideoUpEnlarge Engine",
        description=(
            "AIReliable 平台的 videoUpEnlarge 视频高清放大引擎"
            "（首期后端：Real-ESRGAN ncnn-vulkan，保留原音频）"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS（允许主控跨域调用）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],                  # v0.1 简化，生产环境应限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ===== 注册 8 个标准端点 =====
    _register_endpoints(app)

    # ===== 挂载输出目录（/output/* 提供产物下载）=====
    if _config is not None:
        app.mount(
            "/output",
            StaticFiles(directory=str(_config.output_root)),
            name="output",
        )
        logger.info("输出目录挂载: /output -> %s", _config.output_root)

    # ===== 挂载原生 WebUI（webui/index.html 存在时）=====
    webui_mounted = _mount_webui(app)

    # ===== 根路径（依赖是否挂了 WebUI）=====
    _register_root(app, webui_mounted)

    return app


def _mount_webui(app: FastAPI) -> bool:
    """挂载引擎原生 WebUI（``webui/index.html`` 存在时挂到 ``/ui``）。

    与 Qwen-Image / FoleyCrafter 同一范式：纯静态、无构建。

    Returns:
        True 表示已挂载；False 表示无 webui 目录（保持 headless 行为）
    """
    webui_dir = Path(__file__).resolve().parent / "webui"
    if webui_dir.is_dir() and (webui_dir / "index.html").is_file():
        app.mount("/ui", StaticFiles(directory=str(webui_dir), html=True), name="webui")
        logger.info("WebUI 挂载: /ui -> %s", webui_dir)
        return True
    logger.info("WebUI 未挂载（webui/index.html 不存在），保持 headless")
    return False


def _register_root(app: FastAPI, has_webui: bool) -> None:
    """注册根路径 + ``/info``。

    平台「引擎操作」页与「在新窗口打开」都打**基础 URL**（``http://127.0.0.1:<port>/``）：
        - 有 WebUI → 307 跳转到 ``/ui/``，用户直接看到操作界面
        - 无 WebUI → 返回引擎信息 JSON（等价于旧行为，便于脚本/排障发现能力）
    ``/info`` 恒返回同一份 JSON，两种情况下都可用于发现端点与变体。
    """

    def _engine_info() -> dict:
        cfg = get_config()
        eng = get_engine()
        note = (
            "WebUI: GET /ui/；引擎信息: GET /info"
            if has_webui
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
            + (["GET /ui/", "GET /info"] if has_webui else ["GET /"]),
            "webui": ("/ui/" if has_webui else None),
            "note": note,
        }

    @app.get("/info", include_in_schema=False)
    async def _info() -> dict:
        """引擎信息（恒可用）。"""
        return _engine_info()

    @app.get("/", include_in_schema=False)
    async def _root():
        """根路径：有 WebUI 则跳转，否则返回引擎信息。"""
        if has_webui:
            # 307 保留 method（脚本探测根路径时不被改写；与 Qwen-Image 一致）
            return RedirectResponse(url="/ui/", status_code=307)
        return _engine_info()


def _register_endpoints(app: FastAPI) -> None:
    """注册 8 个标准 HTTP 端点。"""

    # 1. /health
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """健康检查。"""
        return await get_engine().health()

    # 2. /capabilities
    @app.get("/capabilities")
    async def capabilities() -> dict:
        """能力声明（可用超分档位）。"""
        resp = await get_engine().capabilities()
        # 手动序列化 variants，确保 VideoUpEnlargeVariantInfo 的扩展字段
        # （backend / ncnn_model / scale_range / supports_target_size）被保留。
        # CapabilitiesResponse.variants 声明类型为 list[VariantInfo]，
        # Pydantic model_dump() 按声明类型序列化会丢弃子类扩展字段。
        result = resp.model_dump(exclude={"variants"})
        result["variants"] = [v.model_dump() for v in resp.variants]
        return result

    # 3. /load
    @app.post("/load")
    async def load(request: LoadRequest) -> dict:
        """校验后端资源 + 配置设备（**只校验不下载**）。

        请求体：``{"variant": "videoUpEnlarge/realesrgan-x4plus"}``
        资源缺失时返回 500 + 含"缺失文件 / 本地路径 / 下载来源 / 操作方式"的错误。
        """
        try:
            response = await get_engine().load(request)
            return response.model_dump()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 4. /unload
    @app.post("/unload")
    async def unload() -> dict:
        """释放后端配置。"""
        try:
            response = await get_engine().unload()
            return response.model_dump()
        except Exception as e:
            logger.exception("卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 5. /infer（SSE 流式）
    @app.post("/infer")
    async def infer(request: InferRequest):
        """视频高清放大推理（SSE 流式返回进度 + 最终视频 URL）。

        请求体参考 InferRequest：
            - variant: 档位名
            - inputs : VideoUpEnlargeInputs 字段
                       （videos[].path / scale / target_width / target_height / …）
            - task_id: 可选任务 ID
        """
        engine = get_engine()
        sse_stream = engine.infer(request)
        return StreamingResponse(
            sse_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",          # Nginx 不缓冲
            },
        )

    # 6. /status
    @app.get("/status")
    async def status() -> dict:
        """引擎状态查询（含已加载变体 / 设备 / 后端 / 可用变体）。"""
        return (await get_engine().status()).model_dump()

    # 7. /lora/load（v1 预留，不支持）
    @app.post("/lora/load")
    async def lora_load(request: LoraLoadRequest) -> dict:
        """加载 LoRA（本引擎 v1 不支持，预留接口）。"""
        try:
            response = await get_engine().load_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 8. /lora/unload（v1 预留，不支持）
    @app.post("/lora/unload")
    async def lora_unload(request: LoraUnloadRequest) -> dict:
        """卸载 LoRA（本引擎 v1 不支持，预留接口）。"""
        try:
            response = await get_engine().unload_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== 通用 =====

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP 异常统一格式。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error={
                "type": exc.__class__.__name__,
                "message": str(exc.detail),
                "code": exc.status_code,
            }).model_dump(),
        )

    # v0.2.5+：日志级别动态调整端点（GET /log_level + POST /log_level）
    register_log_level_endpoint(app)

    # 根路径与 /info 由 ``_register_root`` 注册（需先知道 WebUI 是否挂载，
    # 因此不能放在这里 —— 见 create_app 中的调用顺序）。


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global _config, app

    _config = parse_args()

    # 配置日志（lifespan 也会配置，这里先配一遍让启动日志可见）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info(
        "启动 videoUpEnlarge 引擎: host=%s, port=%d",
        _config.host, _config.port,
    )

    app = create_app(_config)

    import uvicorn
    uvicorn.run(
        app,
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
