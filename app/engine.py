"""Shared Whisper transcription engine — used by both the CLI (transcribe.py)
and the web app (app.py).

Responsibilities:
  - Load NVIDIA CUDA DLLs on Windows (fixes cublas64_12.dll errors)
  - Self-heal: re-launch with the project venv if faster-whisper is missing
  - Lazy model loading (loaded once, reused across requests)
  - Streaming transcription: yields progress updates per segment
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Common Cantonese words to bias the model toward 粤语 when auto-detecting,
# plus explicit instructions to keep English loanwords in English (fixes
# errors like 啲bug→debug, call→確 in code-switched HK speech).
YUE_PROMPT = (
    "以下英文詞語請用英文原文輸出，不要音譯成中文：project, deadline, bug, fix, call, meeting, "
    "OT, happy hour, email, boss, app, wifi, file, update, server, database, code, team, "
    "schedule, report, check, confirm。"
    "廣東話粵語唔係唔得嘅，佢哋喺香港同廣州講嘢，點解噉樣做。"
)

SUPPORTED_MODELS = ("tiny", "base", "small", "medium", "large-v3")

# --------------------------------------------------------------------------
# LLM proofreading (local Ollama + Qwen3) — optional second pass over the
# transcript to fix whisper's Cantonese-character / punctuation errors.
# --------------------------------------------------------------------------

PROOFREAD_SYSTEM = """你是粵語逐字稿校對員。你的任務是「校對」，不是總結、不是改寫、不是整理格式、不是做筆記。
輸入是語音轉寫稿。你要逐字逐句把它原樣輸出，只修正以下問題：
- 錯別字/同音字：你地→你哋，佢地→佢哋，我地→我哋，既→嘅（語氣助詞時），個→嗰（指代「那」時）
- 明顯語病：確佢哋開會→叫佢哋開會（call 的意思）
- 標點：半形 , . ? 改為全形 ，。？；英文單字內的標點除外
嚴格規則（違反任何一條都算失敗）：
1) 輸出必須與輸入「等長、等段數、同順序」，一句對一句，逐字逐句對應
2) 保留粵語口語書寫（佢哋/嘅/唔/冇/咗/喺/嗰），絕不改成書面語（他們/的/不/沒有）
3) 保留英文借詞原文（project、deadline、call、bug、meeting、OT、happy hour 等）
4) 不得改動人名、地名、數字、時間、金額
5) 不得增刪語句、不得改寫原意、不得重組段落、不得合併或拆分句子
6) 嚴禁添加任何格式：禁止標題、禁止 ###、禁止 emoji、禁止粗體 **、禁止項目符號、禁止編號列表、禁止總結性語句、禁止任何解釋或注釋
7) 輸出繁體中文
8) 只輸出修正後的全文，前後不得有任何額外文字
9) 如句子以「說話人N：」開頭（N 是數字），必須完整保留該標籤，不得刪除、改動或移動

範例：
輸入：今日我地去咗茶餐廳,食咗雲吞麵。好正啊!
輸出：今日我哋去咗茶餐廳，食咗雲吞麵。好正啊！"""

_SUMMARY_MARKERS = ("###", "**", "🔍", "📊", "📈", "📉", "🎯", "✨", "✅", "❌", "📌", "📝", "💡", "🎉")


def _looks_like_summary(out: str, text: str) -> bool:
    """Heuristic: detect when the LLM summarized/restructured instead of
    correcting verbatim (markdown, emoji, bullets, length collapse/explosion)."""
    if not out or not out.strip():
        return True
    if len(out) < len(text) * 0.5:          # far shorter → summarized
        return True
    if any(m in out for m in _SUMMARY_MARKERS):
        return True
    in_lines = [l for l in text.splitlines() if l.strip()]
    out_lines = [l for l in out.splitlines() if l.strip()]
    if in_lines and out_lines:
        if sum(1 for l in out_lines if l.startswith(("-", "•", "*", "1.", "2."))) > max(2, len(out_lines) // 2):
            return True  # bulleted restructure
        if len(out_lines) > max(6, len(in_lines) * 1.6) or len(out_lines) < max(1, len(in_lines) * 0.4):
            return True  # line-count explosion or collapse
    return False


class JobCancelled(Exception):
    """Raised when the caller's should_cancel() reports the job was aborted."""


def proofread(text: str, base_url: str = "http://127.0.0.1:8080",
              model: str = "", chunk_chars: int = 1200,
              should_cancel=None) -> str | None:
    """Proofread a transcript with the local LLM model via embedded server.

    The text is split into small chunks (~chunk_chars each) and each chunk is
    sent in ONE call — long transcripts make LLMs summarize instead of
    correcting, so chunking keeps every line verbatim. Every line is
    proofread exactly once (no retries, no double passes), and speaker
    labels (「說話人N：」) are preserved by the system prompt. Call this AFTER
    speaker labels have been added.

    Returns the corrected text, the input unchanged on empty input, or None if
    the LLM server is unavailable. A chunk whose output looks like a summary keeps its
    original text.
    """
    if not text or not text.strip():
        return text
    lines = [l for l in text.splitlines() if l.strip()]
    chunks, cur, cur_len = [], [], 0
    for ln in lines:
        if cur and cur_len + len(ln) > chunk_chars:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += len(ln)
    if cur:
        chunks.append(cur)
    if not chunks:
        return text
    try:
        import httpx
    except ImportError:  # httpx missing → proofreading unavailable
        return None
    out = []
    for ch in chunks:
        if should_cancel and should_cancel():
            raise JobCancelled()
        src = "\n".join(ch)
        try:
            endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
            msgs = [{"role": "system", "content": PROOFREAD_SYSTEM},
                    {"role": "user", "content": src}]
            if "qwen3" in (model or "").lower():
                msgs.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
            payload = {
                "messages": msgs,
                "temperature": 0.0,
                "max_tokens": 4096,
            }
            if model:
                payload["model"] = model
            r = httpx.post(endpoint, json=payload, timeout=600)
            r.raise_for_status()
            res = r.json()
            choices = res.get("choices") or []
            msg = choices[0].get("message") if choices else {}
            content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
        except Exception as exc:  # noqa: BLE001 — LLM down / not started
            print(f"[engine] proofread skipped (LLM unavailable): {exc}", file=sys.stderr)
            return None
        if content and not _looks_like_summary(content, src):
            out.append(content)
        else:
            print("[engine] a proofread chunk looked like a summary — kept raw", file=sys.stderr)
            out.append(src)
    return "\n".join(out)

# --------------------------------------------------------------------------
# Transcript chat (local Ollama) — the meeting-notes box at the bottom of the
# page sends the finished transcript plus a user prompt to the local LLM.
# --------------------------------------------------------------------------

CHAT_SYSTEM = """你是一個本地助理，正在協助使用者處理一份粵語語音轉寫逐字稿。逐字稿全文附在下面，行首的「說話人N：」代表不同的發言者。

規則：
1) 答案只可以來自這份逐字稿。逐字稿沒有提到，就講「逐字稿未有提及」，
   絕對不可以虛構人名、數字、日期或決定。
2) 用繁體中文（港式書面語）回答，保留原本的英文借詞與專有名詞（project、deadline、OT 等）。
3) 當使用者要求「生成會議記錄」時，輸出結構化的會議記錄，至少包含：
   會議主題 / 出席者 / 討論重點 / 決議事項 / 待辦事項（負責人、期限）。
   逐字稿沒有對應內容的欄位，寫「逐字稿未有提及」。
4) 不要重複貼出整份逐字稿，除非使用者明確要求。
5) 逐字稿來自語音辨識，可能有同音錯字；請合理理解其意思，不要糾正或評論轉寫品質。
"""


def fit_transcript(text: str, char_budget: int = 11000) -> tuple:
    """Trim a transcript to a character budget, keeping the head AND the tail.

    A long meeting transcript does not fit in the local model's context, and
    silently sending only the beginning would produce minutes that quietly
    ignore the second half — so the middle is dropped explicitly and reported
    back to the UI. Returns (text, truncated, omitted_chars).
    """
    text = (text or "").strip()
    if char_budget <= 0 or len(text) <= char_budget:
        return text, False, 0
    head = int(char_budget * 0.6)
    tail = char_budget - head
    omitted = len(text) - head - tail
    marker = f"\n\n……（中間 {omitted:,} 字因長度限制已省略，只保留頭尾）……\n\n"
    return text[:head].rstrip() + marker + text[-tail:].lstrip(), True, omitted


# --------------------------------------------------------------------------
def _ollama_round(convo, base_url, model, num_ctx, num_predict, should_stop=None):
    """Stream a local-LLM reply via llama-server / OpenAI-compatible endpoint."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover — httpx ships with the app
        raise RuntimeError("httpx 未安裝，無法連接本地 LLM 伺服器") from exc

    payload = {
        "messages": convo,
        "stream": True,
        "temperature": 0.3,
        "max_tokens": num_predict,
    }
    if model:
        payload["model"] = model
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
    try:
        endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
        with httpx.stream("POST", endpoint, json=payload, timeout=timeout) as r:
            if r.status_code != 200:
                body = r.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"LLM 伺服器回應 {r.status_code}：{body}")
            for line in r.iter_lines():
                if should_stop and should_stop():
                    return
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except ValueError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or ""
                        piece = content if content else reasoning
                        if piece:
                            yield piece
    except httpx.ConnectError as exc:
        raise RuntimeError(f"無法連接本地 LLM（{base_url}）— 請確認伺服器正在執行") from exc
    except httpx.ReadTimeout as exc:
        raise RuntimeError("LLM 伺服器回應逾時（模型可能仍在生成，請再試一次）") from exc


def chat_stream(messages, transcript, base_url: str = "http://127.0.0.1:8080",
                model: str = "", num_ctx: int = 8192,
                num_predict: int = 2048, should_stop=None):
    """Stream a local-LLM reply about `transcript`, yielding text deltas.

    `messages` is the chat history ([{"role": "user"|"assistant", ...}, ...]).
    The transcript goes into the system message so that it stays byte-identical
    across turns - LLM can then reuse its cached prompt prefix instead of
    re-reading the whole transcript for every follow-up question.
    """
    system = f"{CHAT_SYSTEM}\n\n=== 逐字稿開始 ===\n{transcript}\n=== 逐字稿結束 ==="
    convo = [{"role": "system", "content": system}, *messages]
    if "qwen3" in (model or "").lower():
        convo.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
    for chunk in _ollama_round(convo, base_url, model, num_ctx, num_predict,
                               should_stop=should_stop):
        yield chunk


# --------------------------------------------------------------------------
# Speaker diarization (pyannote.audio 3.1) — label each transcript segment
# with who is speaking. Requires a HuggingFace token for the gated model.
# --------------------------------------------------------------------------

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def bundled_model_dir(model_name: str) -> str | None:
    """Find a pre-downloaded faster-whisper model bundled with the package or on Desktop."""
    base = Path(__file__).resolve().parent
    desktop = Path(os.environ.get("USERPROFILE", "C:/Users/kevin")) / "Desktop"
    candidates = (
        base / "models" / f"faster-whisper-{model_name}",
        base.parent / "models" / f"faster-whisper-{model_name}",
        desktop / f"faster-whisper-{model_name}",
        desktop / model_name,
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "model.bin").is_file():
            return str(candidate)
    return None


def load_ffmpeg_shared_dlls() -> None:
    """Register the bundled FFmpeg shared build (DLLs) so torchcodec
    (pyannote 4.x audio loader) can decode audio on Windows."""
    if os.name != "nt":
        return
    base = Path(__file__).resolve().parent
    shared_dirs = [base / "ffmpeg-shared", base.parent / "ffmpeg-shared"]
    for shared in shared_dirs:
        if not shared.is_dir():
            continue
        bins = sorted(shared.glob("*/bin"))
        if not bins:
            continue
        dll_dir = str(bins[0])
        try:
            os.add_dll_directory(dll_dir)
        except OSError:
            pass
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        break


def _to_wav16k(src: str) -> str:
    """Convert any audio to 16 kHz mono PCM WAV via ffmpeg. pyannote's
    torchcodec loader chokes on some containers (opus/amr/ogg) with sample
    count mismatches — a normalized WAV avoids all format edge cases."""
    dst = os.path.join(tempfile.gettempdir(), f"pyannote_{os.getpid()}_{int(time.time() * 1000)}.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-ar", "16000", "-ac", "1", dst], check=True)
    return dst


def diarize(audio_path: str, token: str, device: str = "cuda") -> tuple[list[dict] | None, str | None]:
    """Run pyannote speaker diarization on an audio file.
    Returns (turns, error): turns is a list of {speaker, start, end} or None;
    error is a human-readable reason when it failed, else None."""
    wav_path = None
    try:
        import torch
        load_ffmpeg_shared_dlls()
        import torchaudio
        # torchaudio >= 2.9 removed list_audio_backends(); pyannote 3.x still
        # calls it at import time — provide a shim (load() works via ffmpeg).
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["ffmpeg"]
        # huggingface_hub 1.x removed the deprecated use_auth_token argument,
        # which pyannote 3.x still passes internally — shim it to token=.
        import huggingface_hub as _hh
        for _name in ("hf_hub_download", "snapshot_download"):
            _fn = getattr(_hh, _name, None)
            if _fn is None or getattr(_fn, "_shimmed", False):
                continue

            def _wrap(fn=_fn):
                def _inner(*args, **kwargs):
                    if "use_auth_token" in kwargs:
                        tok = kwargs.pop("use_auth_token")
                        if tok is not None and "token" not in kwargs:
                            kwargs["token"] = tok
                    return fn(*args, **kwargs)
                _inner._shimmed = True
                return _inner
            setattr(_hh, _name, _wrap(_fn))
        from pyannote.audio import Pipeline

        try:
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
        pipeline.to(torch.device(device))
        wav_path = _to_wav16k(audio_path)
        diarization = pipeline(wav_path)
        # pyannote 4.x wraps the classic Annotation in DiarizeOutput;
        # pyannote 3.x returns the Annotation directly.
        ann = getattr(diarization, "exclusive_speaker_diarization", None) or diarization
        turns = []
        for turn, _, speaker in ann.itertracks(yield_label=True):
            turns.append({"speaker": speaker, "start": float(turn.start), "end": float(turn.end)})
        if not turns:
            return None, "未偵測到可辨識的語音"
        return turns, None
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        if "Torch not compiled with CUDA" in str(exc):
            msg = ("torch 係 CPU 版 — 請用專案 .venv 啟動（start_server.bat），"
                   "唔好用 uv run 或系統 Python")
        return None, msg
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Assign each whisper segment a speaker id (SPEAKER_00 → 1) by
    maximum time overlap with diarization turns. Segments with negligible
    overlap stay unlabeled."""
    if not turns:
        return [{**s, "speaker": None} for s in segments]
    speaker_order = {}
    next_id = 1
    out = []
    for seg in segments:
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
            if ov > best_ov:
                best_ov, best = ov, t["speaker"]
        if best is not None and best_ov > 0.25:
            if best not in speaker_order:
                speaker_order[best] = next_id
                next_id += 1
            out.append({**seg, "speaker": speaker_order[best]})
        else:
            out.append({**seg, "speaker": None})
    return out


def labeled_text(segments: list[dict]) -> str:
    """Render segments with 說話人N： prefixes (used for display + proofread)."""
    lines = []
    for s in segments:
        prefix = f"說話人{s['speaker']}：" if s.get("speaker") else ""
        lines.append(f"{prefix}{s['text']}")
    return "\n".join(lines)


def labeled_srt(segments: list[dict]) -> str:
    blocks = []
    for i, s in enumerate(segments, 1):
        text = f"說話人{s['speaker']}：{s['text']}" if s.get("speaker") else s["text"]
        blocks.append(_make_srt(i, s["start"], s["end"], text))
    return "\n".join(blocks)


def labeled_vtt(segments: list[dict]) -> str:
    blocks = []
    for s in segments:
        text = f"說話人{s['speaker']}：{s['text']}" if s.get("speaker") else s["text"]
        blocks.append(_make_vtt(s["start"], s["end"], text))
    return "WEBVTT\n\n" + "\n".join(blocks)



def load_nvidia_dlls() -> None:
    """Inject pip-installed NVIDIA cuBLAS/cuDNN/cudart DLL dirs into PATH.
    Without this, ctranslate2 fails with 'cublas64_12.dll not found' on Windows."""
    if os.name != "nt":
        return
    import importlib.util
    dll_dirs = []
    for pkg in ("cublas", "cudnn", "cuda_runtime"):
        try:
            spec = importlib.util.find_spec(f"nvidia.{pkg}")
        except ModuleNotFoundError:
            continue
        if not spec or not spec.submodule_search_locations:
            continue
        for base in spec.submodule_search_locations:
            for sub in ("lib", "bin"):
                dll_dir = os.path.join(base, sub)
                if os.path.isdir(dll_dir):
                    dll_dirs.append(dll_dir)
                    try:
                        os.add_dll_directory(dll_dir)
                    except OSError:
                        pass
    if dll_dirs:
        os.environ["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + os.environ.get("PATH", "")


def ensure_deps() -> None:
    """If faster-whisper is missing (wrong interpreter), re-launch this script
    with the project venv so it always works regardless of which python.exe
    is used to invoke it."""
    try:
        import faster_whisper  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    script_dir = Path(__file__).resolve().parent
    venv_py = script_dir / ".venv" / "Scripts" / "python.exe"
    if venv_py.is_file():
        print(f"[transcribe] faster-whisper not found in: {sys.executable}", file=sys.stderr)
        print(f"[transcribe] Re-launching with project venv: {venv_py}", file=sys.stderr)
        import subprocess
        raise SystemExit(subprocess.call([str(venv_py)] + sys.argv))
    print("ERROR: faster-whisper is not installed in this Python.", file=sys.stderr)
    print(f"  You ran: {sys.executable}", file=sys.stderr)
    print(f'  Install deps first:  "{venv_py}" -m pip install -r "{script_dir / "requirements.txt"}"', file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class WhisperEngine:
    """Lazy-loaded, thread-safe wrapper around faster-whisper."""

    def __init__(self) -> None:
        self._model = None
        self._model_name = None
        self._lock = threading.Lock()

    # -- model management ------------------------------------------------

    def cuda_device_count(self) -> int:
        try:
            import ctranslate2
            return int(ctranslate2.get_cuda_device_count())
        except Exception:
            return 0

    def get_model(self, model_name: str):
        model_name = model_name if model_name in SUPPORTED_MODELS else "medium"
        with self._lock:
            if self._model is None or self._model_name != model_name:
                from faster_whisper import WhisperModel
                device = "cuda" if self.cuda_device_count() > 0 else "cpu"
                compute = "float16" if device == "cuda" else "int8"
                bundled = bundled_model_dir(model_name)
                base = Path(__file__).resolve().parent
                models_dir = (base / "models") if (base / "models").is_dir() else (base.parent / "models")
                if bundled:
                    source = bundled
                    where = "bundled model folder"
                    local_only = True
                    target_root = None
                else:
                    source = model_name
                    where = f"auto-downloading to {models_dir}"
                    local_only = False
                    target_root = str(models_dir) if models_dir.is_dir() else None
                print(f"[engine] loading Whisper '{model_name}' from {where} on {device} ({compute})", file=sys.stderr)
                self._model = WhisperModel(source, device=device, compute_type=compute,
                                           download_root=target_root,
                                           local_files_only=local_only)
                self._model_name = model_name
            return self._model

    def unload(self) -> None:
        """Drop the Whisper model and free its VRAM.

        The transcript chat runs the local LLM on the same 8 GB GPU, and
        Whisper's ~1.5 GB would otherwise push the LLM's context cache out of
        memory (Ollama silently spills layers to the CPU and generation crawls).
        The next transcription reloads it from the local cache in a few seconds.
        """
        with self._lock:
            if self._model is None:
                return
            name = self._model_name
            self._model = None
            self._model_name = None
        print(f"[engine] unloading Whisper '{name}' to make room for the LLM", file=sys.stderr)
        try:  # best effort — freeing memory must never break a request
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # -- transcription ----------------------------------------------------

    def transcribe(self, audio_path, model="medium", language=None, should_cancel=None):
        """Generator yielding progress updates.

        Each yielded dict has: phase ('loading'|'transcribing'|'done'),
        progress (0-100), and result fields (language, duration, segments,
        text, srt, vtt, json) filled in as they become available.
        """
        audio_path = str(audio_path)
        m = self.get_model(model)

        kwargs = dict(
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=None if language == "zh" else YUE_PROMPT,
        )
        # NOTE: forcing language="yue" yields English output (faster-whisper
        # bug) — map to "zh" + Cantonese prompt instead. "auto"/None = detect.
        model_language = None if language in (None, "", "auto") else language
        model_language = "zh" if model_language == "yue" else model_language
        if model_language:
            kwargs["language"] = model_language

        yield {"phase": "loading", "progress": 0, "text": "", "segments": [], "error": None}
        segments, info = m.transcribe(audio_path, **kwargs)

        duration = float(info.duration or 0.0)
        segs, text_lines = [], []
        srt_blocks, vtt_blocks = [], []
        for i, seg in enumerate(segments, 1):
            # faster-whisper returns a lazy generator, so checking here makes the
            # 中止 button take effect mid-transcription (one window's granularity)
            if should_cancel and should_cancel():
                raise JobCancelled()
            text = seg.text.strip()
            if not text:
                continue
            segs.append({"id": i, "start": round(float(seg.start), 3),
                         "end": round(float(seg.end), 3), "text": text})
            text_lines.append(text)
            srt_blocks.append(_make_srt(i, seg.start, seg.end, text))
            vtt_blocks.append(_make_vtt(seg.start, seg.end, text))
            progress = min(99.0, (float(seg.end) / duration) * 99.0) if duration else 0.0
            yield {
                "phase": "transcribing",
                "progress": round(progress, 1),
                "language": info.language,
                "language_probability": round(float(info.language_probability), 4),
                "duration": duration,
                "segments": list(segs),
                "segments_count": len(segs),
                "text": "\n".join(text_lines),
                "error": None,
            }

        yield {
            "phase": "done",
            "progress": 100.0,
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "duration": duration,
            "segments": list(segs),
            "segments_count": len(segs),
            "text": "\n".join(text_lines),
            "srt": "\n".join(srt_blocks),
            "vtt": "WEBVTT\n\n" + "\n".join(vtt_blocks),
            "error": None,
        }


# --------------------------------------------------------------------------
# Subtitle helpers
# --------------------------------------------------------------------------

def _fmt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    return f"{h:02d}:{m:02d}:{t % 60:06.3f}"


def _make_srt(i: int, start: float, end: float, text: str) -> str:
    def _f(t):
        h = int(t // 3600)
        m = int(t % 3600 // 60)
        return f"{h:02d}:{m:02d}:{t % 60:06.3f}".replace(".", ",")
    return f"{i}\n{_f(start)} --> {_f(end)}\n{text}\n"


def _make_vtt(start: float, end: float, text: str) -> str:
    return f"{_fmt_ts(start)} --> {_fmt_ts(end)}\n{text}\n"