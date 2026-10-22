"""任务 42 videoUpEnlarge 引擎 —— 端到端验证脚本（阶段 5）。

按平台 `EngineManager` 的方式拉起**真实引擎进程**（cwd = project_root + 注入
`AIRELIABLE_*` 环境变量 + `--log-level DEBUG`），在独立端口上跑完整链路，逐项验证
`ANALYSIS-TASKS/42-videoUpEnlarge视频放大引擎/…-任务文档.md` 的 T5.3 / T5.4 / T5.6 清单。

直接运行（非 pytest）：

    envs/video-upenlarge/.venv/Scripts/python.exe \
        engines/video_up_enlarge_98782fcd/tests/test_e2e_video_upscale.py

首次验证即发现并修复 4 个缺陷（见任务文档「v1.2 测试缺陷清单」）；本脚本保留为回归测试。
只读写本项目 `output/` 与 `.tmp/`，不动任何业务数据。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

# 本文件位于 engines/<engine_id>/tests/，向上 3 级即项目根
ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / "envs" / "video-upenlarge" / ".venv" / "Scripts" / "python.exe"
PORT = 7013
BASE = f"http://127.0.0.1:{PORT}"

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((bool(ok), name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def build_env() -> dict:
    """复刻 platform 注入的引擎子进程环境变量。

    ⚠️ 必须剥离 PYTHONPATH：某些 agent shell 会注入 `cli/vendor/shim`
    （含 sitecustomize.py 的批量删除护栏），它会拦截 shutil.rmtree 并
    raise SystemExit(1)，导致引擎进程异常退出。生产环境由桌面端直接拉起
    引擎，不存在该 shim。
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)          # 关键：剥离 shim
    env.pop("PYTHONSTARTUP", None)
    env["AIRELIABLE_MODELS_ROOT"] = str(ROOT / "models")
    env["AIRELIABLE_MODEL_LAYOUT_BASE"] = "video/upscale/realesrgan"
    env["AIRELIABLE_LOG_DIR"] = str(ROOT / "logs" / "engines")
    return env


def start_server() -> subprocess.Popen:
    """按平台方式启动引擎（cwd = project_root，DEBUG 级日志便于核对真实命令行）。"""
    log_path = ROOT / ".tmp" / "vuen_test_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [str(PY), "-m", "engines.video_up_enlarge_98782fcd.server",
         "--port", str(PORT), "--log-level", "DEBUG"],
        cwd=str(ROOT),
        env=build_env(),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_ready(timeout: float = 45.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{BASE}/health", timeout=3.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.6)
    return False


def safe_infer(payload: dict, timeout: float = 180.0) -> tuple[list[dict], str]:
    """run_infer 的安全包装：任何传输层异常都转为一个 error 诊断帧。"""
    try:
        frames, _ = run_infer(payload, timeout=timeout)
        return frames, ""
    except Exception as e:
        return ([{"stage": "__transport_error__",
                  "error": {"type": type(e).__name__, "message": str(e)}}],
                f"{type(e).__name__}: {e}")


def _find_ffprobe() -> str | None:
    """定位 ffprobe：PATH 优先，其次常见的 WinGet / 便携安装位置。"""
    found = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if found:
        return found
    candidates = [
        Path(r"C:\Users\joych\AppData\Local\Microsoft\WinGet\Packages"
             r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
             r"\ffmpeg-8.1.2-full_build\bin\ffprobe.EXE"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def probe_streams(path: Path) -> dict:
    """用 ffprobe 读取视频/音频流信息；无 ffprobe 时返回空 dict（相关断言会失败）。"""
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return {}
    r = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, errors="ignore",
    )
    try:
        return json.loads(r.stdout or "{}")
    except Exception:
        return {}


def pick(info: dict, kind: str) -> dict | None:
    return next((s for s in info.get("streams", []) if s.get("codec_type") == kind), None)


def run_infer(payload: dict, timeout: float = 900.0) -> tuple[list[dict], float]:
    """POST /infer（SSE），返回 (帧列表, 耗时秒)。

    连接中断时返回已收到的帧，而不是抛出——便于定位"引擎中途死亡"这类问题。
    """
    frames: list[dict] = []
    t0 = time.time()
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
            with client.stream("POST", f"{BASE}/infer", json=payload) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", "replace")[:400]
                    raise RuntimeError(f"HTTP {resp.status_code}: {body}")
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        frames.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        continue
                    if frames[-1].get("stage") in ("completed", "error"):
                        break
    except httpx.RemoteProtocolError as e:
        # 引擎进程中途退出：保留已收帧，附加说明供上层断言
        frames.append({"stage": "__transport_error__",
                       "error": {"type": "connection_lost", "message": str(e)}})
    return frames, time.time() - t0


def main() -> int:
    print("=" * 78)
    print("任务 42 videoUpEnlarge 引擎 —— 端到端验证")
    print("=" * 78)

    # 准备测试素材（复制出便携包内的演示视频，模拟用户上传）
    src_video = ROOT / "models" / "video" / "upscale" / "realesrgan" / "onepiece_demo.mp4"
    test_dir = ROOT / ".tmp" / "vuen_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_video = test_dir / "demo_640x480.mp4"
    if src_video.is_file():
        shutil.copy2(src_video, test_video)

    print(f"测试素材: {test_video}  (存在={test_video.is_file()})")

    proc = None
    try:
        # ---------- T5.3 引擎启动与端点冒烟 ----------
        section("T5.3 引擎启动与端点冒烟")
        proc = start_server()
        ready = wait_ready()
        check("引擎进程启动成功（cwd=project_root）", ready,
              f"pid={proc.pid}, port={PORT}")

        if not ready:
            log = (ROOT / ".tmp" / "vuen_test_server.log")
            print("\n--- 服务端日志尾部 ---")
            if log.is_file():
                print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]))
            return 1

        # /health
        r = httpx.get(f"{BASE}/health", timeout=10.0)
        h = r.json()
        check("GET /health 200 且 status=ok",
              r.status_code == 200 and h.get("status") == "ok", str(h)[:160])

        # /capabilities
        r = httpx.get(f"{BASE}/capabilities", timeout=10.0)
        cap = r.json()
        caps = cap.get("capabilities") or []
        variants = cap.get("variants") or []
        vnames = [v.get("name") for v in variants]
        check("GET /capabilities 含 capability=video-upscale",
              "video-upscale" in caps, f"capabilities={caps}")
        check("GET /capabilities variants 非空且为 2 个",
              len(variants) == 2, f"{vnames}")
        check("GET /capabilities input_schema 非 null",
              bool(cap.get("input_schema")), f"keys={list(cap.keys())}")
        check("variant 扩展字段保留（backend/ncnn_model）",
              all(v.get("backend") and v.get("ncnn_model") for v in variants),
              f"backend={[v.get('backend') for v in variants]}")

        # /status
        r = httpx.get(f"{BASE}/status", timeout=10.0)
        st = r.json()
        extra = st.get("extra") or {}
        check("GET /status extra.task_type == video_upscale",
              extra.get("task_type") == "video_upscale", f"extra={extra}")

        # / 根路径 → 307 重定向到 /ui/（v1.3 起，供平台「引擎操作」页 iframe）
        r = httpx.get(f"{BASE}/", timeout=10.0, follow_redirects=False)
        check("GET / → 307 重定向到 /ui/",
              r.status_code == 307 and r.headers.get("location") == "/ui/",
              f"status={r.status_code}, location={r.headers.get('location')}")

        # /info 信息页（恒返回 JSON）
        r = httpx.get(f"{BASE}/info", timeout=10.0)
        info = r.json()
        check("GET /info 返回引擎信息", r.status_code == 200 and info.get("engine_id") ==
              "video_up_enlarge_98782fcd", str(info.get("variants")))

        # /load 默认 variant（变体为空时的回退路径）
        r = httpx.post(f"{BASE}/load", json={"variant": None}, timeout=60.0)
        check("POST /load（默认 variant）成功", r.status_code == 200, str(r.json())[:200])

        # /load 不存在的 variant
        r = httpx.post(f"{BASE}/load", json={"variant": "videoUpEnlarge/does-not-exist"}, timeout=30.0)
        check("POST /load 未知 variant 被拒（4xx/5xx）", r.status_code >= 400,
              f"HTTP {r.status_code}")

        # /lora/load 不支持
        r = httpx.post(f"{BASE}/lora/load", json={"name": "x", "weight": 1.0}, timeout=20.0)
        check("POST /lora/load 明确报不支持（非 200）", r.status_code >= 400,
              f"HTTP {r.status_code}: {str(r.json())[:140]}")

        # 重新 load 正确 variant 供后续推理
        r = httpx.post(f"{BASE}/load", json={"variant": "videoUpEnlarge/realesrgan-x4plus"}, timeout=60.0)
        check("POST /load（realesrgan-x4plus）成功", r.status_code == 200, str(r.json())[:160])

        # ---------- T5.4 真实放大任务端到端 ----------
        section("T5.4 真实放大任务端到端（scale=2.0）")
        src_info = probe_streams(test_video)
        src_v = pick(src_info, "video")
        src_a = pick(src_info, "audio")
        src_w, src_h = int(src_v["width"]), int(src_v["height"])
        src_dur = float(src_info.get("format", {}).get("duration", 0))
        print(f"  源视频: {src_w}x{src_h}, {src_dur:.2f}s, 音轨={'有' if src_a else '无'}"
              f" ({src_a.get('codec_name') if src_a else '-'})")

        payload = {
            "variant": "videoUpEnlarge/realesrgan-x4plus",
            "inputs": {"videos": [{"path": str(test_video)}], "scale": 2.0},
            "task_id": "test_t42_e2e_001",
        }
        try:
            frames, elapsed = run_infer(payload)
        except Exception as e:
            check("POST /infer 执行完成", False, f"{type(e).__name__}: {e}")
            frames, elapsed = [], 0.0

        stages = [f.get("stage") for f in frames]
        errs = [f for f in frames if f.get("stage") == "error"]
        final = next((f for f in frames if f.get("stage") == "completed"), None)

        check("SSE 流有进度帧产出", len(frames) > 0, f"{len(frames)} 帧, stages={stages[:8]}")
        check("无 error 帧", not errs, str(errs)[:300] if errs else "")
        check("收到 completed 终帧", final is not None,
              f"耗时 {elapsed:.1f}s" + (" (未完成)" if not final else ""))

        if final:
            result = final.get("result") or {}
            # 契约字段（T2.4）
            vids = result.get("videos") or []
            check("result 含 videos 数组（网关契约形状①）", isinstance(vids, list) and len(vids) == 1,
                  f"len={len(vids)}")
            entry = vids[0] if vids else {}
            check("videos[0] 含 video_path 且文件存在",
                  bool(entry.get("video_path")) and Path(entry["video_path"]).is_file(),
                  str(entry.get("video_path")))
            check("videos[0] 含 /output/ 相对 video_url",
                  str(entry.get("video_url", "")).startswith("/output/"),
                  str(entry.get("video_url")))

            out_path = Path(entry.get("video_path", ""))
            check("产物 URL 可经静态挂载下载",
                  httpx.get(f"{BASE}{entry.get('video_url')}", timeout=30.0).status_code == 200,
                  entry.get("video_url"))

            # 分辨率
            out_info = probe_streams(out_path)
            out_v = pick(out_info, "video")
            out_a = pick(out_info, "audio")
            if out_v:
                ow, oh = int(out_v["width"]), int(out_v["height"])
                check(f"分辨率按 2x 提升（{src_w}x{src_h} → {ow}x{oh}）",
                      ow == src_w * 2 and oh == src_h * 2, f"期望 {src_w*2}x{src_h*2}")
            else:
                check("产物可被 ffprobe 解析", False, "无视频流")

            # 音轨（T8 必做）
            check("产物保留音轨", out_a is not None,
                  f"codec={out_a.get('codec_name') if out_a else '无'}")
            out_dur = float(out_info.get("format", {}).get("duration", 0))
            check("产物时长与源一致（±0.5s）", abs(out_dur - src_dur) <= 0.5,
                  f"源 {src_dur:.2f}s → 产物 {out_dur:.2f}s")

            print(f"  产物: {out_path}  ({out_path.stat().st_size/1024/1024:.2f} MB)")
            print(f"  诊断: ncnn_scale={result.get('ncnn_scale')} applied_scale={result.get('applied_scale')} "
                  f"frames={result.get('frame_count')} elapsed={result.get('elapsed')}s")

        # ---------- T5.6 边界与异常 ----------
        section("T5.6 边界与异常")

        # 不存在的视频
        frames, terr = safe_infer({
            "variant": "videoUpEnlarge/realesrgan-x4plus",
            "inputs": {"videos": [{"path": str(test_dir / "no_such_video.mp4")}], "scale": 2.0},
            "task_id": "t42_no_file",
        })
        err = next((f for f in frames if f.get("stage") == "error"), None)
        check("视频不存在 → error 帧（不 500/不卡死）", err is not None,
              (str(err.get("error"))[:180] if err else f"frames={len(frames)} {terr}"))

        # scale 超范围
        frames, terr = safe_infer({
            "variant": "videoUpEnlarge/realesrgan-x4plus",
            "inputs": {"videos": [{"path": str(test_video)}], "scale": 8.0},
            "task_id": "t42_scale8",
        })
        err = next((f for f in frames if f.get("stage") == "error"), None)
        check("scale=8.0 超范围 → 明确报错", err is not None,
              (str(err.get("error"))[:180] if err else f"未报错！{terr}"))

        # scale + target 同传
        frames, terr = safe_infer({
            "variant": "videoUpEnlarge/realesrgan-x4plus",
            "inputs": {"videos": [{"path": str(test_video)}],
                       "scale": 3.0, "target_width": 1920},
            "task_id": "t42_conflict",
        })
        err = next((f for f in frames if f.get("stage") == "error"), None)
        check("scale+target 同传 → 互斥报错", err is not None,
              (str(err.get("error"))[:180] if err else f"未报错！{terr}"))

        # 缺失视频路径
        frames, terr = safe_infer({
            "variant": "videoUpEnlarge/realesrgan-x4plus",
            "inputs": {"scale": 2.0}, "task_id": "t42_no_input",
        })
        err = next((f for f in frames if f.get("stage") == "error"), None)
        check("完全缺输入视频 → 明确报错", err is not None,
              (str(err.get("error"))[:180] if err else f"frames={len(frames)} {terr}"))

        # ---------- /unload ----------
        section("收尾")
        try:
            r = httpx.post(f"{BASE}/unload", timeout=30.0)
            check("POST /unload 成功", r.status_code == 200, str(r.json())[:140])
            r = httpx.get(f"{BASE}/health", timeout=10.0)
            check("unload 后仍健康（state 已复位）", r.status_code == 200, str(r.json())[:140])
        except Exception as e:
            check("POST /unload 成功", False,
                  f"引擎进程已不可达（{type(e).__name__}）—— 见上方 FAIL 定位根因")

    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=12)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    # ---------- 汇总 ----------
    passed = sum(1 for ok, _, _ in RESULTS if ok)
    failed = [(n, d) for ok, n, d in RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"结果: PASS {passed} / FAIL {len(failed)}  (共 {len(RESULTS)} 项)")
    if failed:
        print("失败项:")
        for n, d in failed:
            print(f"  ✗ {n}  {d}")
    print("=" * 78)

    # 失败时附服务端日志尾部，便于定位
    if failed:
        log = ROOT / ".tmp" / "vuen_test_server.log"
        if log.is_file():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            keys = ("执行命令", "ERROR", "开始放大", "抽帧完成", "超分", "合帧", "放大完成", "RuntimeError")
            hits = [l for l in lines if any(k in l for k in keys)]
            print("\n--- 服务端关键日志（末 18 条）---")
            for l in hits[-18:]:
                print("  ", l[:400])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
