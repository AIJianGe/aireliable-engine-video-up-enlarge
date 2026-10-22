/* videoUpEnlarge 引擎 WebUI 前端逻辑。
 *
 * 设计要点：
 *   - 与引擎**同源**（页面由引擎在 /ui/ 提供），引擎接口用**根绝对路径**
 *     （`/capabilities` 而非 `capabilities`）：页面 URL 是 `/ui/`，
 *     相对路径会被解析成 `/ui/capabilities` → 404。
 *   - 输入视频有两种通道：
 *       ① 上传：POST {gateway}/v1/files/video（multipart，SHA256 去重）→ 本地路径
 *       ② 直填：本地绝对路径（引擎与桌面端同机，最省事）
 *   - 推理走 POST /infer，响应是 SSE（text/event-stream），用 fetch + 手写解析
 *     （EventSource 不支持 POST）。
 *   - 帧格式（SDK aireliable_engine.sse）：
 *       progress 帧 {progress, stage, message?, ...}
 *       final  帧 {progress:1.0, stage:"completed", result:{videos:[…]}, timing}
 *       error  帧 {progress:0.0, stage:"error", error:{type,message}}
 */

(function () {
    'use strict';

    var STAGE_TEXT = {
        start: '开始',
        preparing: '抽取视频帧',
        model_loading: '加载超分后端',
        processing: '超分处理',
        merging: '合帧并回填音轨',
        heartbeat: '超分处理中（心跳）',
        completed: '完成',
        error: '失败'
    };

    var ALLOWED_EXT = ['.mp4', '.mov', '.mkv', '.avi', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.mpg', '.mpeg'];

    var state = {
        variants: [],
        defaultVariant: null,
        videoPath: null,
        videoLabel: null,
        gatewayOk: false,
        running: false,
        abort: null
    };

    /* ---------- DOM ---------- */

    function $(id) { return document.getElementById(id); }

    var el = {
        stateDot: $('state-dot'),
        engineMeta: $('engine-meta'),
        btnUnload: $('btn-unload'),
        tabs: document.querySelectorAll('.tab'),
        panes: document.querySelectorAll('.tab-pane'),
        dropzone: $('dropzone'),
        fileInput: $('file-input'),
        btnPick: $('btn-pick'),
        uploadHint: $('upload-hint'),
        pathInput: $('path-input'),
        picked: $('picked'),
        pickedPath: $('picked-path'),
        pickedMeta: $('picked-meta'),
        variantSelect: $('variant-select'),
        variantHint: $('variant-hint'),
        modeSeg: $('mode-seg'),
        scaleSeg: $('scale-seg'),
        fieldScale: $('field-scale'),
        fieldTarget: $('field-target'),
        targetWidth: $('target-width'),
        targetHeight: $('target-height'),
        optTile: $('opt-tile'),
        optCrf: $('opt-crf'),
        optFrameFormat: $('opt-frame-format'),
        optThreads: $('opt-threads'),
        optTta: $('opt-tta'),
        optKeepTemp: $('opt-keep-temp'),
        optGateway: $('opt-gateway'),
        btnRun: $('btn-run'),
        btnStop: $('btn-stop'),
        bar: $('bar'),
        progressPct: $('progress-pct'),
        progressStage: $('progress-stage'),
        result: $('result'),
        resultVideo: $('result-video'),
        resultMeta: $('result-meta'),
        btnDownload: $('btn-download'),
        btnAgain: $('btn-again'),
        log: $('log')
    };

    /* ---------- 工具 ---------- */

    function log(text, kind) {
        var line = document.createElement('div');
        if (kind) line.className = kind;
        var t = new Date().toTimeString().slice(0, 8);
        line.textContent = '[' + t + '] ' + text;
        el.log.appendChild(line);
        el.log.scrollTop = el.log.scrollHeight;
    }

    function setProgress(p, stage) {
        var pct = Math.max(0, Math.min(100, Math.round((p || 0) * 100)));
        el.bar.style.width = pct + '%';
        el.progressPct.textContent = pct + '%';
        if (stage) el.progressStage.textContent = STAGE_TEXT[stage] || stage;
    }

    function fmtSize(bytes) {
        if (!bytes && bytes !== 0) return '—';
        var u = ['B', 'KB', 'MB', 'GB'], i = 0, v = bytes;
        while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
        return (i === 0 ? v : v.toFixed(2)) + ' ' + u[i];
    }

    function fmtDuration(sec) {
        if (!sec && sec !== 0) return '—';
        return Number(sec).toFixed(2) + ' s';
    }

    function extOf(name) {
        var i = String(name || '').lastIndexOf('.');
        return i < 0 ? '' : String(name).slice(i).toLowerCase();
    }

    function setState(kind, title) {
        el.stateDot.className = 'dot' + (kind ? ' ' + kind : '');
        if (title) el.stateDot.title = title;
    }

    function setBusy(on) {
        state.running = on;
        el.btnRun.disabled = on;
        el.btnStop.disabled = !on;
    }

    /* ---------- 引擎信息 ---------- */

    async function loadCapabilities() {
        var r = await fetch('/capabilities');
        if (!r.ok) throw new Error('GET /capabilities → ' + r.status);
        var d = await r.json();
        state.variants = d.variants || [];
        state.defaultVariant = d.default_variant || null;

        el.variantSelect.innerHTML = '';
        state.variants.forEach(function (v) {
            var o = document.createElement('option');
            o.value = v.name;
            o.textContent = v.name;
            if (v.name === state.defaultVariant) o.selected = true;
            el.variantSelect.appendChild(o);
        });
        if (!state.variants.length) {
            var o2 = document.createElement('option');
            o2.textContent = '（无可用 variant）';
            el.variantSelect.appendChild(o2);
        }
        updateVariantHint();
        el.engineMeta.textContent = d.engine_name + ' · ' + (d.capabilities || []).length
            + ' 能力 · ' + state.variants.length + ' 档位';
    }

    function currentVariant() {
        return state.variants.filter(function (v) { return v.name === el.variantSelect.value; })[0];
    }

    function updateVariantHint() {
        var v = currentVariant();
        if (!v) { el.variantHint.textContent = '—'; return; }
        var parts = [];
        parts.push('后端 ' + (v.backend || '—'));
        if (v.ncnn_model) parts.push('ncnn 模型 ' + v.ncnn_model);
        if (v.scale_range) parts.push('倍率 ' + v.scale_range);
        parts.push(v.available ? '文件就绪' : '⚠ 文件缺失');
        el.variantHint.textContent = parts.join(' · ');
    }

    async function loadStatus() {
        try {
            var r = await fetch('/status');
            if (!r.ok) throw new Error('GET /status → ' + r.status);
            var d = await r.json();
            var s = String(d.state || '').toLowerCase();
            if (s === 'ready') setState('ready', '引擎就绪');
            else if (s === 'error') setState('error', '引擎异常');
            else if (s === 'inferring' || s === 'loading') setState('busy', '忙碌：' + s);
            else setState('', '引擎空闲');
            var extra = d.extra || {};
            var bits = [];
            if (extra.loaded_variant) bits.push('已加载 ' + extra.loaded_variant);
            if (extra.device) bits.push('设备 ' + extra.device);
            if (extra.backend) bits.push(extra.backend);
            if (bits.length) log('引擎状态：' + bits.join(' · '));
        } catch (e) {
            setState('error', '状态读取失败');
            log('读取 /status 失败：' + e.message, 'e');
        }
    }

    /* ---------- 输入视频 ---------- */

    function normalizePath(p) {
        return String(p || '').trim().replace(/^["']|["']$/g, '');
    }

    function setVideo(path, label, meta) {
        state.videoPath = path || null;
        state.videoLabel = label || null;
        if (!path) { el.picked.classList.add('hidden'); return; }
        el.picked.classList.remove('hidden');
        el.pickedPath.textContent = path;
        el.pickedMeta.textContent = meta || '—';
    }

    async function probeGateway() {
        var gw = el.optGateway.value.trim().replace(/\/+$/, '') || 'http://127.0.0.1:8000';
        try {
            var ctl = new AbortController();
            var t = setTimeout(function () { ctl.abort(); }, 3000);
            var r = await fetch(gw + '/v1/models', { signal: ctl.signal });
            clearTimeout(t);
            state.gatewayOk = r.ok;
        } catch (e) {
            state.gatewayOk = false;
        }
        el.uploadHint.innerHTML = state.gatewayOk
            ? '上传经网关 <code>/v1/files/video</code> 换取本地路径（同内容按 SHA256 去重）。'
            : '<b>网关未就绪</b>（' + gw + '）——请改用「本地路径」页签直接填绝对路径，或用「引擎管理」页启动平台。';
        return state.gatewayOk;
    }

    async function uploadFile(file) {
        var ext = extOf(file.name);
        if (ALLOWED_EXT.indexOf(ext) < 0) {
            log('不支持的扩展名 ' + (ext || '(无)') + '，仍会尝试上传', 'w');
        }
        var gw = el.optGateway.value.trim().replace(/\/+$/, '') || 'http://127.0.0.1:8000';
        log('上传中：' + file.name + '（' + fmtSize(file.size) + '）…');
        var fd = new FormData();
        fd.append('file', file, file.name);
        var r = await fetch(gw + '/v1/files/video', { method: 'POST', body: fd });
        var body = await r.json().catch(function () { return {}; });
        if (!r.ok) {
            throw new Error('上传失败 HTTP ' + r.status + '：' + (body.detail || JSON.stringify(body)));
        }
        log('上传完成：' + body.path + (body.cached ? '（命中缓存）' : ''), 'o');
        setVideo(body.path, file.name, fmtSize(body.size) + ' · sha256 ' + String(body.sha256 || '').slice(0, 12));
    }

    /* ---------- SSE ---------- */

    function parseFrame(text) {
        // 取第一行 data: 之后的内容（引擎每帧只有一个 data 行）
        var out = null;
        text.split('\n').forEach(function (line) {
            if (line.indexOf('data:') === 0) {
                var payload = line.slice(5).trim();
                if (!payload) return;
                try { out = JSON.parse(payload); } catch (e) { /* 非 JSON 帧忽略 */ }
            }
        });
        return out;
    }

    /* ---------- 主流程 ---------- */

    function collectInputs() {
        var mode = document.querySelector('.seg.active[data-mode]').dataset.mode;
        var inputs = { workflow: 'upscale' };

        if (mode === 'target') {
            var w = parseInt(el.targetWidth.value, 10);
            var h = parseInt(el.targetHeight.value, 10);
            if (w > 0) inputs.target_width = w;
            if (h > 0) inputs.target_height = h;
            if (!w && !h) throw new Error('按目标分辨率放大时，至少填写宽或高');
        } else {
            inputs.scale = parseFloat(document.querySelector('.seg.active[data-scale]').dataset.scale);
        }

        if (el.optTile.value) inputs.tile_size = parseInt(el.optTile.value, 10);
        if (el.optCrf.value !== '') inputs.crf = parseInt(el.optCrf.value, 10);
        if (el.optFrameFormat.value) inputs.frame_format = el.optFrameFormat.value;
        if (el.optThreads.value.trim()) inputs.threads = el.optThreads.value.trim();
        if (el.optTta.checked) inputs.tta = true;
        if (el.optKeepTemp.checked) inputs.keep_temp = true;

        return inputs;
    }

    async function run() {
        if (state.running) return;
        if (!state.videoPath) { log('请先选择视频（上传文件或填本地路径）', 'e'); return; }

        var variant = el.variantSelect.value;
        var inputs;
        try {
            inputs = collectInputs();
        } catch (e) {
            log(e.message, 'e');
            return;
        }

        setBusy(true);
        el.result.classList.add('hidden');
        el.log.innerHTML = '';
        setProgress(0, 'start');

        var taskId = 'webui_' + Math.random().toString(16).slice(2, 18);
        log('任务 ' + taskId + '：variant=' + variant + '，视频=' + state.videoPath);
        log('参数：' + JSON.stringify(inputs));

        try {
            // ---- 1. 先 /load，把"资源缺失"这类错误在推理前暴露出来（只校验不下载）----
            setProgress(0, 'model_loading');
            log('校验超分后端资源（POST /load）…');
            var lr = await fetch('/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ variant: variant })
            });
            var lbody = await lr.json().catch(function () { return {}; });
            if (!lr.ok) {
                throw new Error('加载失败 HTTP ' + lr.status + '：' + ((lbody.error && lbody.error.message) || JSON.stringify(lbody)));
            }
            log('后端就绪：variant=' + lbody.variant + '，显存 ' + (lbody.vram_used || '—')
                + (lbody.load_time != null ? '，耗时 ' + Number(lbody.load_time).toFixed(1) + 's' : ''), 'o');

            // ---- 2. 推理（SSE）----
            var ctl = new AbortController();
            state.abort = ctl;
            log('提交推理（POST /infer，SSE 流式进度）…');

            var resp = await fetch('/infer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ variant: variant, inputs: inputs, task_id: taskId }),
                signal: ctl.signal
            });
            if (!resp.ok) {
                var eb = await resp.json().catch(function () { return {}; });
                throw new Error('推理请求被拒 HTTP ' + resp.status + '：' + ((eb.error && eb.error.message) || JSON.stringify(eb)));
            }
            if (!resp.body) throw new Error('响应无流式 body（浏览器不支持 ReadableStream？）');

            var reader = resp.body.getReader();
            var decoder = new TextDecoder('utf-8');
            var buf = '';
            var finalFrame = null;
            var lastStage = null;

            while (true) {
                var chunk = await reader.read();
                if (chunk.done) break;
                buf += decoder.decode(chunk.value, { stream: true });

                var idx;
                while ((idx = buf.indexOf('\n\n')) >= 0) {
                    var raw = buf.slice(0, idx);
                    buf = buf.slice(idx + 2);
                    var frame = parseFrame(raw);
                    if (!frame) continue;

                    if (frame.stage === 'error') {
                        var msg = (frame.error && frame.error.message) || '未知错误';
                        throw new Error(msg + (frame.error && frame.error.type ? '（' + frame.error.type + '）' : ''));
                    }
                    if (frame.stage === 'completed') { finalFrame = frame; continue; }

                    if (frame.stage !== lastStage) {
                        lastStage = frame.stage;
                        log('阶段：' + (STAGE_TEXT[frame.stage] || frame.stage));
                    }
                    if (frame.message && frame.stage !== 'heartbeat') log('  ' + frame.message);
                    if (typeof frame.progress === 'number') setProgress(frame.progress, frame.stage);
                }
            }

            // ---- 3. 收尾 ----
            if (finalFrame) {
                setProgress(1, 'completed');
                var payload = finalFrame.result || {};
                var entry = (payload.videos || [])[0] || {};
                renderResult(entry, payload, finalFrame.timing || {});
                log('完成：' + (entry.video_url || '') + '（' + (payload.elapsed != null ? Number(payload.elapsed).toFixed(1) + 's' : '—') + '）', 'o');
                loadStatus();
            } else {
                log('流已结束但未收到 completed 帧——请查看引擎日志', 'w');
                setProgress(0, 'error');
            }
        } catch (e) {
            if (e && e.name === 'AbortError') {
                log('已断开前端连接。注意：服务端线程池任务仍会跑完（引擎无取消端点），如需硬停请在「引擎管理」页重启引擎。', 'w');
                setProgress(0, 'error');
            } else {
                log('失败：' + (e && e.message ? e.message : e), 'e');
                setProgress(0, 'error');
            }
        } finally {
            state.abort = null;
            setBusy(false);
        }
    }

    function renderResult(entry, payload, timing) {
        var url = entry.video_url || '';
        if (!url) { log('完成帧未携带 video_url', 'w'); return; }
        var abs = new URL(url, location.origin).href;
        el.resultVideo.src = abs;
        el.btnDownload.href = abs;
        el.btnDownload.setAttribute('download', String(url).split('/').pop() || 'upscaled.mp4');

        var pairs = [
            ['分辨率', (entry.source_width || '?') + '×' + (entry.source_height || '?') + ' → <b>' + (entry.width || '?') + '×' + (entry.height || '?') + '</b>'],
            ['实际倍率', entry.applied_scale != null ? entry.applied_scale + '×' : '—'],
            ['时长', fmtDuration(entry.duration)],
            ['原音轨', entry.has_audio ? ('保留（' + (entry.audio_codec || '—') + '）') : '源无音轨'],
            ['帧数', entry.frame_count != null ? entry.frame_count : '—'],
            ['耗时', (payload.elapsed != null ? Number(payload.elapsed).toFixed(1) + ' s' : (timing.total_seconds != null ? Number(timing.total_seconds).toFixed(1) + ' s' : '—'))],
            ['后端', (payload.backend || '—') + (payload.ncnn_model ? ' / ' + payload.ncnn_model : '')],
            ['设备', payload.device || '—']
        ];
        el.resultMeta.innerHTML = pairs.map(function (p) {
            return '<span>' + p[0] + ' <b>' + p[1] + '</b></span>';
        }).join('');
        el.result.classList.remove('hidden');
    }

    /* ---------- 事件绑定 ---------- */

    function switchTab(name) {
        el.tabs.forEach(function (t) { t.classList.toggle('active', t.dataset.tab === name); });
        el.panes.forEach(function (p) { p.classList.toggle('hidden', p.dataset.pane !== name); });
    }

    function bind() {
        el.tabs.forEach(function (t) {
            t.addEventListener('click', function () { switchTab(t.dataset.tab); });
        });

        el.dropzone.addEventListener('click', function () { el.fileInput.click(); });
        el.btnPick.addEventListener('click', function (ev) { ev.stopPropagation(); el.fileInput.click(); });

        ['dragenter', 'dragover'].forEach(function (n) {
            el.dropzone.addEventListener(n, function (e) { e.preventDefault(); el.dropzone.classList.add('over'); });
        });
        ['dragleave', 'drop'].forEach(function (n) {
            el.dropzone.addEventListener(n, function (e) { e.preventDefault(); el.dropzone.classList.remove('over'); });
        });
        el.dropzone.addEventListener('drop', function (e) {
            var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (f) doUpload(f);
        });

        el.fileInput.addEventListener('change', function () {
            if (el.fileInput.files && el.fileInput.files[0]) doUpload(el.fileInput.files[0]);
        });

        el.pathInput.addEventListener('input', function () {
            var p = normalizePath(el.pathInput.value);
            if (p) setVideo(p, null, '本地路径（未上传）');
            else setVideo(null);
        });

        document.querySelectorAll('.seg[data-mode]').forEach(function (b) {
            b.addEventListener('click', function () {
                document.querySelectorAll('.seg[data-mode]').forEach(function (x) { x.classList.remove('active'); });
                b.classList.add('active');
                var target = b.dataset.mode === 'target';
                el.fieldScale.classList.toggle('hidden', target);
                el.fieldTarget.classList.toggle('hidden', !target);
            });
        });

        el.scaleSeg.querySelectorAll('.seg').forEach(function (b) {
            b.addEventListener('click', function () {
                el.scaleSeg.querySelectorAll('.seg').forEach(function (x) { x.classList.remove('active'); });
                b.classList.add('active');
            });
        });

        el.variantSelect.addEventListener('change', updateVariantHint);
        el.optGateway.addEventListener('change', probeGateway);

        el.btnRun.addEventListener('click', run);

        el.btnStop.addEventListener('click', function () {
            if (state.abort) state.abort.abort();
        });

        el.btnAgain.addEventListener('click', function () {
            el.result.classList.add('hidden');
            el.resultVideo.removeAttribute('src');
            setProgress(0, null);
            el.progressStage.textContent = '待开始';
        });

        el.btnUnload.addEventListener('click', async function () {
            try {
                var r = await fetch('/unload', { method: 'POST' });
                var d = await r.json().catch(function () { return {}; });
                if (!r.ok) throw new Error('HTTP ' + r.status);
                log('已卸载后端：' + (d.state || 'idle') + '，释放 ' + (d.vram_freed || '0MB'), 'o');
                loadStatus();
            } catch (e) {
                log('卸载失败：' + e.message, 'e');
            }
        });
    }

    async function doUpload(file) {
        if (!state.gatewayOk && !(await probeGateway())) {
            log('网关不可用，无法上传。请切到「本地路径」页签直接填绝对路径。', 'e');
            switchTab('path');
            return;
        }
        try {
            await uploadFile(file);
        } catch (e) {
            log(e.message, 'e');
        }
    }

    /* ---------- 启动 ---------- */

    (async function init() {
        bind();
        try {
            await loadCapabilities();
        } catch (e) {
            log('读取 /capabilities 失败：' + e.message, 'e');
            setState('error', '能力读取失败');
        }
        loadStatus();
        probeGateway();
    })();
})();
