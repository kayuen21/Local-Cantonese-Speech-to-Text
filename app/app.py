import collections
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx

# --- self-heal: always run inside the project's .venv -----------------------
# Launched with a different interpreter (e.g. `uv run app.py`, which uses uv's
# managed Python 3.11), the process imports a CPU-only torch and speaker
# diarization dies with "AssertionError: Torch not compiled with CUDA enabled".
# Re-exec ourselves with the project venv so the CUDA torch wheels and the
# cuBLAS/cuDNN DLLs are always the ones installed for this project.
def _reexec_in_project_venv() -> None:
    here = Path(__file__).resolve()
    venv_python = here.parent / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return
    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except OSError:
        pass
    if os.environ.get("STT_IN_VENV") == "1":  # already re-exec'd — never loop
        return
    os.environ["STT_IN_VENV"] = "1"
    print(f"[app] wrong interpreter: {sys.executable}", file=sys.stderr)
    print(f"[app] re-launching inside project venv: {venv_python}", file=sys.stderr)
    # os.execv on Windows joins argv with spaces and does NOT quote — the
    # project path contains spaces, so quote the path arguments ourselves
    # (the MSVC runtime strips the quotes when it builds the child's argv).
    os.execv(str(venv_python), [f'"{venv_python}"', f'"{here}"', *sys.argv[1:]])


_reexec_in_project_venv()

from flask import (Flask, Response, jsonify, render_template, request,
                   send_file, stream_with_context)

from engine import (JobCancelled, WhisperEngine, assign_speakers, chat_stream,
                    diarize, ensure_deps, fit_transcript, labeled_srt, labeled_text,
                    labeled_vtt, load_nvidia_dlls, proofread)

load_nvidia_dlls()
ensure_deps()

# --- hard guard: never allow a second server instance on port 5000 ----------
# (two instances caused stale templates/code being served to the browser)
def _port_in_use(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()

if _port_in_use(5000):
    print("[app] port 5000 is already in use — opening it in your browser instead", file=sys.stderr)
    try:
        import webbrowser
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:
        pass
    sys.exit(0)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

ALLOWED_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma", ".mp4", ".mkv", ".webm", ".mov", ".amr", ".3gp", ".aiff", ".caf"}

def _load_config() -> dict:
    cfg = BASE_DIR / "config.json"
    try:
        if cfg.is_file():
            return json.loads(cfg.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — a broken config must not stop the server
        print("[app] config.json could not be parsed — using defaults", file=sys.stderr)
    return {}


CONFIG = _load_config()

# HuggingFace token for the gated pyannote diarization model — read from the
# local config file (never from chat memory), or the HF_TOKEN env var.
HF_TOKEN = CONFIG.get("hf_token") or os.environ.get("HF_TOKEN", "")

# Embedded LLM server (llama-server) — runs locally from the bundle. All overridable in config.json.
LLM_URL = CONFIG.get("llm_url", "http://127.0.0.1:8080")
CHAT_MODEL = CONFIG.get("chat_model", "")  # auto-detected from models/*.gguf if empty
PROOFREAD_MODEL = CONFIG.get("proofread_model", CHAT_MODEL)
CHAT_NUM_CTX = int(CONFIG.get("chat_num_ctx", 4096))
CHAT_NUM_PREDICT = int(CONFIG.get("chat_num_predict", 1024))
# How much of the transcript fits in that context. Chinese runs roughly one
# token per character; ~0.7 leaves room for the system prompt, the chat
# history and the reply itself.
CHAT_CHAR_BUDGET = max(1500, int(CHAT_NUM_CTX * 0.7))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024**3  # 2 GB local limit
app.config["JSON_AS_ASCII"] = False

ENGINE = WhisperEngine()

# --------------------------------------------------------------------------
# Job queue: one GPU at a time, jobs processed serially by a worker thread
# --------------------------------------------------------------------------

JOBS = {}
JOBS_LOCK = threading.Lock()
QUEUE = collections.deque()
QUEUE_COND = threading.Condition()


def _snapshot(job, include_text=True):
    keys = ["status", "progress", "error", "language", "language_probability",
            "duration", "segments_count", "filename", "model", "speakers",
            "diarize", "diarize_skipped"]
    snap = {k: job.get(k) for k in keys}
    if include_text:
        snap["text"] = job.get("text", "")
        snap["corrected_text"] = job.get("corrected_text")
    snap["segments"] = job.get("segments", []) if job.get("status") in ("done", "cancelled") else []
    return snap


def _finish_cancelled(job) -> None:
    """Mark a job the user aborted. Keep whatever transcript we already have."""
    with JOBS_LOCK:
        job["status"] = "cancelled"
        job["progress"] = 0
        job["error"] = None
        text, srt, vtt = job.get("text", ""), job.get("srt", ""), job.get("vtt", "")
    if not text:
        return
    RESULT_DIR.joinpath(f"{job['id']}.txt").write_text(text, encoding="utf-8")
    if srt:
        RESULT_DIR.joinpath(f"{job['id']}.srt").write_text(srt, encoding="utf-8")
    if vtt:
        RESULT_DIR.joinpath(f"{job['id']}.vtt").write_text(vtt, encoding="utf-8")


def _run_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        cancel = job["cancel"]
        job["status"] = "loading"
        job["progress"] = 0
        audio_path, model, language = job["path"], job["model"], job["language"]
    try:
        if cancel.is_set():  # aborted while still sitting in the queue
            _finish_cancelled(job)
            return

        # 1. Start speaker diarization (pyannote) concurrently in background if requested
        diarize_thread = None
        diarize_result = {"turns": None, "err": None}

        if job["diarize"] and HF_TOKEN:
            def _diarize_worker():
                try:
                    turns, derr = diarize(audio_path, HF_TOKEN)
                    diarize_result["turns"] = turns
                    diarize_result["err"] = derr
                except Exception as exc:
                    diarize_result["err"] = str(exc)

            diarize_thread = threading.Thread(target=_diarize_worker, daemon=True)
            diarize_thread.start()

        # 2. Transcribe audio with Whisper concurrently
        for update in ENGINE.transcribe(audio_path, model=model, language=language,
                                        should_cancel=cancel.is_set):
            with JOBS_LOCK:
                job["status"] = "transcribing" if update["phase"] == "transcribing" else "loading"
                job["progress"] = update.get("progress", 0)
                job["language"] = update.get("language")
                job["language_probability"] = update.get("language_probability")
                job["duration"] = update.get("duration")
                job["segments_count"] = update.get("segments_count")
                job["segments"] = update.get("segments", [])
                job["text"] = update.get("text", "")
                if update["phase"] == "done":
                    job["srt"] = update.get("srt", "")
                    job["vtt"] = update.get("vtt", "")
                    break

        if cancel.is_set():
            _finish_cancelled(job)
            return

        # 3. Join diarization if it ran concurrently
        if diarize_thread:
            if diarize_thread.is_alive():
                with JOBS_LOCK:
                    job["status"] = "diarizing"
                    job["progress"] = 99.0
                diarize_thread.join()

            turns = diarize_result["turns"]
            derr = diarize_result["err"]
            if turns:
                segs = assign_speakers(job["segments"], turns)
                n_spk = len({s["speaker"] for s in segs if s.get("speaker")}) or None
                with JOBS_LOCK:
                    job["segments"] = segs
                    job["speakers"] = n_spk
                    job["segments_count"] = len(segs)
                    job["text"] = labeled_text(segs)
                    job["srt"] = labeled_srt(segs)
                    job["vtt"] = labeled_vtt(segs)
            else:
                with JOBS_LOCK:
                    job["diarize_skipped"] = derr or "分離失敗"

        # Persist results
        with JOBS_LOCK:
            RESULT_DIR.joinpath(f"{job_id}.txt").write_text(job["text"], encoding="utf-8")
            if job.get("srt"):
                RESULT_DIR.joinpath(f"{job_id}.srt").write_text(job["srt"], encoding="utf-8")
            if job.get("vtt"):
                RESULT_DIR.joinpath(f"{job_id}.vtt").write_text(job["vtt"], encoding="utf-8")

        if cancel.is_set():
            _finish_cancelled(job)
            return
        # optional second pass: local LLM proofreading via embedded server
        if job["proofread"]:
            with JOBS_LOCK:
                job["status"] = "proofreading"
                job["progress"] = 99.0
            ensure_llm_server()
            gguf_model = get_bundled_gguf_path()
            model_name = gguf_model.stem if gguf_model else (PROOFREAD_MODEL or "")
            corrected = proofread(job["text"], base_url=LLM_URL,
                                  model=model_name,
                                  should_cancel=cancel.is_set)
            with JOBS_LOCK:
                job["corrected_text"] = corrected  # None if LLM unavailable
        # always finish — diarization/proofread are both optional stages
        with JOBS_LOCK:
            job["status"] = "done"
            job["progress"] = 100.0
    except JobCancelled:
        print(f"[app] job {job_id} cancelled by user")
        _finish_cancelled(job)
    except Exception as exc:  # noqa: BLE001 — report any failure to the UI
        with JOBS_LOCK:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"


def _worker_loop() -> None:
    while True:
        with QUEUE_COND:
            while not QUEUE:
                QUEUE_COND.wait()
            job_id = QUEUE.popleft()
        _run_job(job_id)


threading.Thread(target=_worker_loop, daemon=True, name="whisper-worker").start()

# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "gpu": ENGINE.cuda_device_count() > 0,
        "cuda_devices": ENGINE.cuda_device_count(),
        "model_loaded": ENGINE._model_name,  # noqa: SLF001
        "chat_model": CHAT_MODEL,
        "chat_ctx": CHAT_NUM_CTX,
        "queued": len(QUEUE),
        "running": sum(1 for j in JOBS.values() if j.get("status") in ("loading", "transcribing")),
    })


@app.post("/api/transcribe")
def transcribe():
    f = request.files.get("audio")
    if f is None or not f.filename:
        return jsonify({"error": "no audio file provided"}), 400
    suffix = Path(f.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return jsonify({"error": f"unsupported file type '{suffix}'"}), 400

    model = request.form.get("model", "medium")
    if model not in ("medium", "large-v3"):
        model = "medium"
    language = request.form.get("language", "auto")
    if language not in ("auto", "zh", "yue"):
        language = "auto"
    proofread_on = request.form.get("proofread", "1") == "1"
    diarize_on = request.form.get("diarize", "0") == "1"

    job_id = uuid.uuid4().hex[:12]
    store_path = UPLOAD_DIR / f"{job_id}{suffix}"
    f.save(store_path)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id, "filename": f.filename, "path": str(store_path),
            "model": model, "language": language, "proofread": proofread_on,
            "diarize": diarize_on, "diarize_skipped": False,
            "status": "queued", "progress": 0, "error": None,
            "language_detected": None, "language_probability": None,
            "duration": None, "segments_count": 0, "segments": [],
            "speakers": None, "text": "", "srt": "", "vtt": "", "corrected_text": None,
            "cancel": threading.Event(),
            "created": time.time(),
        }
    with QUEUE_COND:
        QUEUE.append(job_id)
        QUEUE_COND.notify()

    return jsonify({"job_id": job_id}), 202


@app.get("/api/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        include_text = job["status"] in ("transcribing", "done", "cancelled")
        return jsonify(_snapshot(job, include_text=include_text))


@app.post("/api/cancel/<job_id>")
def cancel(job_id):
    """Abort a job. Cooperative: the worker checks the flag between transcription
    segments, between pipeline stages and between proofread chunks, so the status
    flips to 'cancelled' within a few seconds."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        if job["status"] in ("done", "error", "cancelled"):
            return jsonify({"ok": True, "status": job["status"]}), 200
        job["cancel"].set()
        job["status"] = "cancelling"
    # a job still parked in the queue never reaches the worker — retire it here
    removed = False
    with QUEUE_COND:
        if job_id in QUEUE:
            QUEUE.remove(job_id)
            QUEUE_COND.notify()
            removed = True
    if removed:
        _finish_cancelled(job)
    with JOBS_LOCK:
        return jsonify({"ok": True, "status": job["status"]}), 200


def _sse(payload) -> str:
    """One server-sent event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def api_chat():
    """Stream a local-LLM reply about a finished transcript.

    The browser sends the text it is currently showing (校對版 or 原文) plus the
    conversation so far; the answer streams back as server-sent events so the
    box fills in while the model writes instead of hanging for a minute.
    """
    body = request.get_json(silent=True) or {}
    transcript = (body.get("text") or "").strip()
    if not transcript:
        return jsonify({"error": "還沒有逐字稿可以對話"}), 400

    msgs = []
    for m in (body.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            msgs.append({"role": role, "content": content[:8000]})
    msgs = msgs[-12:]  # keep the tail of the conversation only
    if not msgs:
        return jsonify({"error": "沒有訊息"}), 400

    cfg = _load_config()
    tier_id = get_active_tier()
    tier_ctx = LLM_TIERS.get(tier_id, {}).get("ctx", 16384)
    active_ctx = int(cfg.get("chat_num_ctx") or tier_ctx)
    char_budget = max(2500, int(active_ctx * 0.7))
    fitted, truncated, omitted = fit_transcript(transcript, char_budget)
    ensure_llm_server()
    gguf_model = get_bundled_gguf_path()
    model_name = gguf_model.stem if gguf_model else (CHAT_MODEL or "")
    # Whisper and the LLM share one 8 GB GPU — hand the whole card to the LLM
    # for the duration of the chat (the next transcription reloads Whisper).
    ENGINE.unload()

    def gen():
        yield _sse({"meta": {"truncated": truncated, "omitted": omitted,
                             "chars": len(fitted), "model": model_name}})
        try:
            for delta in chat_stream(msgs, fitted, base_url=LLM_URL, model=model_name,
                                     num_ctx=active_ctx, num_predict=CHAT_NUM_PREDICT):
                yield _sse({"delta": delta})
            yield _sse({"done": True})
        except GeneratorExit:  # client pressed 停止 or closed the tab
            print("[app] chat stream closed by the client", file=sys.stderr)
            raise
        except Exception as exc:  # noqa: BLE001 — report the real reason to the UI
            print(f"[app] chat failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            yield _sse({"error": str(exc)})

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/download/<job_id>/<fmt>")
def download(job_id, fmt):
    if fmt not in ("txt", "srt", "vtt"):
        return jsonify({"error": "unknown format"}), 400
    # 1) prefer the persisted file on disk (survives restarts)
    path = RESULT_DIR / f"{job_id}.{fmt}"
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        in_memory = (job is not None and bool(job["text"])
                     and job["status"] in ("done", "cancelled"))
        stem = Path(job["filename"]).stem if job else job_id
    if not path.is_file():
        # 2) fall back to in-memory results (job done in THIS server session)
        if not in_memory:
            return jsonify({"error": "result file not found — please transcribe the audio again"}), 404
        body = job[fmt if fmt in ("srt", "vtt") else "text"]
    stem = re.sub(r'[\\/:*?"<>|\r\n]', "_", stem).strip() or job_id
    mime = {"txt": "text/plain; charset=utf-8",
            "srt": "application/x-subrip",
            "vtt": "text/vtt; charset=utf-8"}[fmt]
    # send_file handles non-ASCII filenames via RFC 5987 (filename*=UTF-8''…)
    from urllib.parse import quote
    if stem.isascii():
        disp = f'attachment; filename="{stem}.{fmt}"'
    else:
        disp = f'attachment; filename="transcript.{fmt}"; filename*=UTF-8\'\'{quote(stem + "." + fmt)}'
    if path.is_file():
        resp = send_file(path, mimetype=mime)
    else:
        resp = Response(body, mimetype=mime)
    resp.headers["Content-Disposition"] = disp
    return resp


@app.get("/api/jobs")
def jobs():
    with JOBS_LOCK:
        recent = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)[:10]
        return jsonify([_snapshot(j, include_text=False) for j in recent])


# --------------------------------------------------------------------------
# Embedded LLM server management (llama-server)
# --------------------------------------------------------------------------

LLM_DIR = BASE_DIR.parent / "llm"
LLM_EXE = LLM_DIR / "llama-server.exe"
MODELS_DIR = BASE_DIR.parent / "models"
_LLM_PROCESS = None
_CURRENT_RUNNING_MODEL = None
_DOWNLOAD_ABORT = threading.Event()

LLM_TIERS = {
    "weak": {
        "id": "weak",
        "name": "Qwen2.5-1.5B",
        "label": "輕量 1.5B",
        "desc": "超快響應、低硬體佔用 (~1.1 GB)",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_str": "1.1 GB",
        "size_bytes": 1117320736,
        "ctx": 8192,
    },
    "medium": {
        "id": "medium",
        "name": "Qwen2.5-3B",
        "label": "平衡 3B",
        "desc": "推薦首選、速度與質量均衡 (~2.0 GB)",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_str": "2.0 GB",
        "size_bytes": 2104932768,
        "ctx": 8192,
    },
    "strong": {
        "id": "strong",
        "name": "Qwen3-8B",
        "label": "強力 8B",
        "desc": "最強推理、深層總結 (~4.9 GB)",
        "filename": "qwen3-8b.gguf",
        "url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
        "size_str": "4.9 GB",
        "size_bytes": 5027783488,
        "ctx": 16384,
    },
}


def get_model_path(tier_id: str) -> Path | None:
    """Check if model for tier exists in models/ or on Desktop."""
    info = LLM_TIERS.get(tier_id)
    if not info:
        return None
    # 1. Look in models/
    p = MODELS_DIR / info["filename"]
    if p.is_file():
        return p
    # 2. Look on Desktop
    desktop = Path(os.environ.get("USERPROFILE", "C:/Users/kevin")) / "Desktop"
    dp = desktop / info["filename"]
    if dp.is_file():
        return dp
    return None


def get_active_tier() -> str:
    """Return currently active tier ID."""
    cfg = _load_config()
    tier = cfg.get("selected_tier")
    if tier in LLM_TIERS and get_model_path(tier) is not None:
        return tier
    # Default to first available downloaded tier
    for tid in ("weak", "medium", "strong"):
        if get_model_path(tid) is not None:
            return tid
    return "weak"


def get_bundled_gguf_path() -> Path | None:
    """Find the GGUF model path according to active tier, with fallbacks."""
    active = get_active_tier()
    p = get_model_path(active)
    if p is not None:
        return p
    # Fallback: check any .gguf in models/
    if MODELS_DIR.is_dir():
        ggufs = sorted(MODELS_DIR.glob("*.gguf"))
        if ggufs:
            return ggufs[0]
    return None


def _is_llm_running(timeout: float = 1.0) -> bool:
    import httpx
    try:
        r = httpx.get(f"{LLM_URL}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def stop_llm_server():
    """Stop the running embedded llama-server."""
    global _LLM_PROCESS, _CURRENT_RUNNING_MODEL
    if _LLM_PROCESS is not None:
        try:
            _LLM_PROCESS.kill()
        except Exception:
            pass
        _LLM_PROCESS = None
    _CURRENT_RUNNING_MODEL = None
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    time.sleep(0.5)


def ensure_llm_server():
    """Ensure the bundled llama-server is running on port 8080 with the active model."""
    global _LLM_PROCESS, _CURRENT_RUNNING_MODEL
    gguf_model = get_bundled_gguf_path()
    if not LLM_EXE.is_file():
        print("[llm] bundled llama-server.exe not found", file=sys.stderr)
        return False

    if gguf_model is None or not gguf_model.is_file():
        tier_id = get_active_tier()
        info = LLM_TIERS.get(tier_id)
        if info:
            print(f"[llm] model not found in models directory, auto-downloading {info['name']}...", file=sys.stderr)
            desktop = Path(os.environ.get("USERPROFILE", "C:/Users/kevin")) / "Desktop"
            df = desktop / info["filename"]
            target = MODELS_DIR / info["filename"]
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            if df.is_file():
                try:
                    shutil.copy2(str(df), str(target))
                    gguf_model = target
                except Exception as e:
                    print(f"[llm] copy from desktop failed: {e}", file=sys.stderr)
            if not gguf_model or not gguf_model.is_file():
                try:
                    import urllib.request
                    temp_p = MODELS_DIR / (info["filename"] + ".download")
                    urllib.request.urlretrieve(info["url"], str(temp_p))
                    temp_p.rename(target)
                    gguf_model = target
                except Exception as e:
                    print(f"[llm] auto-download failed: {e}", file=sys.stderr)

    if not gguf_model or not gguf_model.is_file():
        print(f"[llm] bundled llama-server or .gguf model not found", file=sys.stderr)
        return False

    running = _is_llm_running(timeout=0.5)
    if running and _CURRENT_RUNNING_MODEL == gguf_model.name:
        return True
    if running:
        print(f"[llm] model changed from {_CURRENT_RUNNING_MODEL} to {gguf_model.name}, restarting server", file=sys.stderr)
        stop_llm_server()

    try:
        tier_id = get_active_tier()
        ctx = LLM_TIERS.get(tier_id, {}).get("ctx", CHAT_NUM_CTX)
        cmd = [
            str(LLM_EXE),
            "-m", str(gguf_model),
            "--port", "8080",
            "--host", "127.0.0.1",
            "-c", str(ctx),
            "-ngl", "99",
            "-t", "8",
            "--reasoning-format", "none",
        ]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        _LLM_PROCESS = subprocess.Popen(cmd, cwd=str(LLM_DIR), creationflags=flags)
        _CURRENT_RUNNING_MODEL = gguf_model.name
        print(f"[llm] started embedded llama-server with {gguf_model.name} (PID {_LLM_PROCESS.pid})", file=sys.stderr)
        for _ in range(25):
            time.sleep(0.5)
            if _is_llm_running(timeout=0.5):
                print(f"[llm] embedded llama-server ({gguf_model.name}) is ready", file=sys.stderr)
                return True
        return False
    except Exception as exc:
        print(f"[llm] failed to start llama-server: {exc}", file=sys.stderr)
        return False


# --------------------------------------------------------------------------
# Model Selection and Download APIs
# --------------------------------------------------------------------------

@app.get("/api/models")
def api_models_list():
    active = get_active_tier()
    desktop = Path(os.environ.get("USERPROFILE", "C:/Users/kevin")) / "Desktop"
    tiers_list = []
    for tid, info in LLM_TIERS.items():
        is_dl = (MODELS_DIR / info["filename"]).is_file()
        has_desktop = (desktop / info["filename"]).is_file()
        tiers_list.append({
            "id": tid,
            "name": info["name"],
            "label": info["label"],
            "desc": info["desc"],
            "size_str": info["size_str"],
            "filename": info["filename"],
            "downloaded": is_dl or has_desktop,
            "in_models_dir": is_dl,
            "active": (tid == active),
        })
    return jsonify({
        "active_tier": active,
        "tiers": tiers_list,
    })


@app.post("/api/models/select")
def api_models_select():
    data = request.get_json(silent=True) or {}
    tier_id = data.get("tier")
    if tier_id not in LLM_TIERS:
        return jsonify({"error": "invalid tier"}), 400
    info = LLM_TIERS[tier_id]
    model_path = get_model_path(tier_id)
    if not model_path:
        return jsonify({"ok": False, "need_download": True, "tier": tier_id, "size_str": info["size_str"]})

    # If file was on Desktop and not yet in models/, copy it over now
    target = MODELS_DIR / info["filename"]
    if not target.is_file() and model_path.is_file():
        try:
            shutil.copy2(str(model_path), str(target))
        except Exception as e:
            print(f"[models] copy from desktop failed: {e}", file=sys.stderr)

    cfg = _load_config()
    cfg["selected_tier"] = tier_id
    cfg["chat_model"] = info["name"]
    cfg["chat_num_ctx"] = info["ctx"]
    cfg_file = BASE_DIR / "config.json"
    cfg_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    stop_llm_server()
    ensure_llm_server()
    return jsonify({"ok": True, "active_tier": tier_id, "downloaded": True, "model_name": info["name"]})


@app.post("/api/models/abort-download")
def api_models_abort_download():
    _DOWNLOAD_ABORT.set()
    return jsonify({"ok": True})


@app.post("/api/models/download")
def api_models_download():
    data = request.get_json(silent=True) or {}
    tier_id = data.get("tier")
    if tier_id not in LLM_TIERS:
        return jsonify({"error": "invalid tier"}), 400
    info = LLM_TIERS[tier_id]
    filename = info["filename"]
    url = info["url"]
    target = MODELS_DIR / filename
    temp_target = MODELS_DIR / (filename + ".download")

    _DOWNLOAD_ABORT.clear()

    def sse_stream():
        # 1. Fast check: is it already on Desktop?
        desktop = Path(os.environ.get("USERPROFILE", "C:/Users/kevin")) / "Desktop"
        desktop_file = desktop / filename
        if desktop_file.is_file():
            yield _sse({"status": "copying", "message": "從桌面複製現有模型中…"})
            try:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(desktop_file), str(target))
                cfg = _load_config()
                cfg["selected_tier"] = tier_id
                cfg["chat_model"] = info["name"]
                (BASE_DIR / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
                stop_llm_server()
                ensure_llm_server()
                yield _sse({"status": "done", "tier": tier_id, "completed": desktop_file.stat().st_size, "total": desktop_file.stat().st_size, "done": True})
                return
            except Exception as e:
                yield _sse({"status": "error", "message": f"複製失敗: {e}"})
                return

        # 2. Download from HuggingFace
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 CantoneseSTT"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length", 0)) or info["size_bytes"]
                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB chunks
                last_yield = 0
                with open(temp_target, "wb") as out_f:
                    while True:
                        if _DOWNLOAD_ABORT.is_set():
                            out_f.close()
                            if temp_target.is_file():
                                temp_target.unlink(missing_ok=True)
                            yield _sse({"status": "stopped", "message": "下載已取消"})
                            return
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_yield > 0.3:
                            last_yield = now
                            pct = (downloaded / total * 100) if total else 0
                            yield _sse({"status": "downloading", "completed": downloaded, "total": total, "percent": round(pct, 1)})

                # Download finished
                if temp_target.is_file():
                    if target.is_file():
                        target.unlink(missing_ok=True)
                    temp_target.rename(target)

            cfg = _load_config()
            cfg["selected_tier"] = tier_id
            cfg["chat_model"] = info["name"]
            (BASE_DIR / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            stop_llm_server()
            ensure_llm_server()
            yield _sse({"status": "done", "tier": tier_id, "completed": total, "total": total, "done": True})
        except Exception as exc:
            if temp_target.is_file():
                temp_target.unlink(missing_ok=True)
            yield _sse({"status": "error", "message": str(exc)})

    return Response(sse_stream(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


# --------------------------------------------------------------------------
# Setup helpers (HF token config, LLM status)
# --------------------------------------------------------------------------

@app.get("/api/setup/status")
def setup_status():
    global HF_TOKEN
    running = _is_llm_running(timeout=0.5)
    gguf_model = get_bundled_gguf_path()
    active_tier = get_active_tier()
    tier_info = LLM_TIERS.get(active_tier, {})
    model_name = tier_info.get("name") or (gguf_model.stem if gguf_model else "none")
    has_token = bool(HF_TOKEN)
    token_display = (HF_TOKEN[:4] + "…" + HF_TOKEN[-4:]) if len(HF_TOKEN) >= 8 else ("已設定" if has_token else "未設定")

    return jsonify({
        "ollama_running": running,
        "ollama_installed": LLM_EXE.is_file(),
        "has_ollama": LLM_EXE.is_file(),
        "installer_bundled": False,
        "has_model": gguf_model is not None,
        "model": model_name,
        "chat_model": model_name,
        "active_tier": active_tier,
        "models": [model_name] if gguf_model else [],
        "has_hf_token": has_token,
        "hf_token_display": token_display
    })


@app.post("/api/setup/save-token")
def setup_save_token():
    global HF_TOKEN
    data = request.get_json(silent=True) or {}
    token = (data.get("hf_token") or "").strip()
    cfg_file = BASE_DIR / "config.json"
    cfg = _load_config()
    cfg["hf_token"] = token
    cfg_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    HF_TOKEN = token
    token_display = (HF_TOKEN[:4] + "…" + HF_TOKEN[-4:]) if len(HF_TOKEN) >= 8 else ("已設定" if bool(token) else "未設定")
    return jsonify({"ok": True, "has_hf_token": bool(token), "hf_token_display": token_display})


@app.post("/api/setup/install-ollama")
def setup_install_ollama():
    return jsonify({"ok": True})


@app.post("/api/setup/start-ollama")
def setup_start_ollama():
    ok = ensure_llm_server()
    return jsonify({"ok": ok})


@app.post("/api/setup/pull-model")
def setup_pull_model():
    return jsonify({"ok": True, "done": True})


if __name__ == "__main__":
    import webbrowser
    url = "http://127.0.0.1:5000"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f" * Serving on {url}  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)