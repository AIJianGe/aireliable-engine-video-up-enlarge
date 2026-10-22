"""videoUpEnlarge 引擎 —— WebUI 路由回归测试（v1.3）。

在 **7014** 端口另起一个引擎实例（不打扰 7013 上运行中的实例），验证 v1.3 新增的
WebUI 路由链路与"既有端点未受影响"：

    GET /            → 307 Location: /ui/
    GET / (跟随)      → 200 text/html（模拟浏览器 iframe）
    GET /ui/         → 200 text/html，含页面标题
    GET /ui/app.js   → 200
    GET /ui/style.css→ 200
    GET /info        → 200 JSON，含 webui="/ui/" 与端点声明
    GET /health /capabilities /status /docs → 行为不变
    POST /load {}    → 200（默认 variant 回退）

直接运行（非 pytest）：

    envs/video-upenlarge/.venv/Scripts/python.exe \
        engines/video_up_enlarge_98782fcd/tests/test_webui_routes.py

背景：平台「引擎操作」页对所有引擎一律 iframe `http://127.0.0.1:<端口>/`，
故引擎必须让根路径跳到自己的界面，否则该页只显示一段 JSON。
详见任务文档「v1.3 变更清单」与 `project-rules` §第六部分 12。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

# 本文件位于 engines/<engine_id>/tests/，向上 3 级即项目根
ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / "envs" / "video-upenlarge" / ".venv" / "Scripts" / "python.exe"
PORT = 7014
BASE = f"http://127.0.0.1:{PORT}"

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((bool(ok), name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def build_env() -> dict:
    """复刻 platform 注入的引擎子进程环境变量（并剥离 agent shell 的 shim）。"""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)          # 关键：剥离 shim（见 test_e2e 同名函数注释）
    env.pop("PYTHONSTARTUP", None)
    env["AIRELIABLE_MODELS_ROOT"] = str(ROOT / "models")
    env["AIRELIABLE_MODEL_LAYOUT_BASE"] = "video/upscale/realesrgan"
    env["AIRELIABLE_LOG_DIR"] = str(ROOT / "logs" / "engines")
    return env


def start_server() -> subprocess.Popen:
    log_path = ROOT / ".tmp" / "vuen_webui_routes_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    return subprocess.Popen(
        [str(PY), "-m", "engines.video_up_enlarge_98782fcd.server",
         "--port", str(PORT), "--log-level", "INFO"],
        cwd=str(ROOT), env=build_env(),
        stdout=log_f, stderr=subprocess.STDOUT, text=True,
    )


def main() -> int:
    proc = start_server()
    print(f"启动测试实例 :{PORT}（cwd=项目根，与平台 EngineManager 一致）")

    try:
        t0 = time.time()
        ready = False
        while time.time() - t0 < 60:
            if proc.poll() is not None:
                print(f"!! 子进程提前退出 rc={proc.returncode}")
                break
            try:
                if httpx.get(f"{BASE}/health", timeout=3.0).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1.0)
        if not ready:
            return 2
        print(f"引擎就绪（{time.time() - t0:.1f}s）")

        section("根路径重定向（平台「引擎操作」页 iframe 依赖此路由）")
        r = httpx.get(f"{BASE}/", timeout=10.0, follow_redirects=False)
        check("GET / → 307", r.status_code == 307, f"status={r.status_code}")
        check("Location == /ui/", r.headers.get("location") == "/ui/",
              f"location={r.headers.get('location')}")

        section("WebUI 静态资源")
        r = httpx.get(f"{BASE}/ui/", timeout=10.0)
        check("GET /ui/ → 200", r.status_code == 200, f"status={r.status_code}")
        check("Content-Type text/html", "text/html" in (r.headers.get("content-type") or ""),
              r.headers.get("content-type", ""))
        check("页面含标题 videoUpEnlarge", "videoUpEnlarge" in r.text, f"{len(r.text)} bytes")
        for asset in ("app.js", "style.css"):
            ra = httpx.get(f"{BASE}/ui/{asset}", timeout=10.0)
            check(f"GET /ui/{asset} → 200", ra.status_code == 200,
                  f"status={ra.status_code}, {len(ra.content)} bytes")

        section("跟随后端重定向（模拟浏览器 iframe）")
        r = httpx.get(f"{BASE}/", timeout=10.0, follow_redirects=True)
        check("GET / (跟随) → 200 HTML",
              r.status_code == 200 and "text/html" in (r.headers.get("content-type") or ""),
              f"status={r.status_code}, final={r.url}")

        section("/info 引擎信息（恒可用）")
        r = httpx.get(f"{BASE}/info", timeout=10.0)
        check("GET /info → 200", r.status_code == 200, f"status={r.status_code}")
        if r.status_code == 200:
            d = r.json()
            check("info.webui == /ui/", d.get("webui") == "/ui/", f"webui={d.get('webui')!r}")
            check("info 含 variants", len(d.get("variants") or []) == 2, str(d.get("variants")))
            check("info 含 /ui/ 端点声明", "GET /ui/" in (d.get("endpoints") or []), "")

        section("既有端点未受影响")
        r = httpx.get(f"{BASE}/health", timeout=10.0)
        check("GET /health → 200", r.status_code == 200, f"status={r.status_code}")
        r = httpx.get(f"{BASE}/capabilities", timeout=15.0)
        check("GET /capabilities → 200", r.status_code == 200, f"status={r.status_code}")
        r = httpx.get(f"{BASE}/status", timeout=15.0)
        check("GET /status → 200", r.status_code == 200, f"status={r.status_code}")
        r = httpx.get(f"{BASE}/docs", timeout=10.0)
        check("GET /docs → 200", r.status_code == 200, f"status={r.status_code}")
        r = httpx.post(f"{BASE}/load", json={}, timeout=60.0)
        check("POST /load {}（默认 variant 回退）→ 200", r.status_code == 200,
              f"status={r.status_code}, {r.text[:120]}")

    finally:
        try:
            import psutil
            parent = psutil.Process(proc.pid)
            for ch in parent.children(recursive=True):
                ch.kill()
            parent.kill()
        except Exception:
            proc.kill()

    total = len(RESULTS)
    passed = sum(1 for ok, _, _ in RESULTS if ok)
    print(f"\n===== 汇总：PASS {passed} / FAIL {total - passed}（共 {total}）=====")
    for ok, name, _ in RESULTS:
        if not ok:
            print(f"  FAIL: {name}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
