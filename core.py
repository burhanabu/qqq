"""AccentLive engine: your speech -> text (Whisper) -> clear American voice (Kokoro). Runs fully on this PC, free.

Settings (optional environment variables):
  WHISPER_MODEL  tiny.en | base.en (default) | small.en | medium.en   (smaller = faster, less accurate)
  SILENCE_MS     pause length that ends a phrase (default 260)
  DEVICE         auto (default) | cpu | cuda
"""
import asyncio, collections, json, os, re, threading, time, urllib.request
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).parent
MODELS = HERE / "models"
PORT = 8765
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
SILENCE_MS = int(os.environ.get("SILENCE_MS", "260"))
DEVICE = os.environ.get("DEVICE", "auto")
MAX_SPEED = 1.35  # when speech is piling up, talk faster instead of dropping words
KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
JUNK = {"you", "thank you", "thanks for watching", "bye", "the end"}


def _contexts():
    import ssl
    out = []
    try:
        import truststore  # uses the Windows certificate store (fixes antivirus / proxy certificate errors)
        out.append(truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    except Exception:
        pass
    try:
        import certifi
        out.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    out.append(ssl.create_default_context())
    return out


def fetch(url, dest):
    print("Downloading", dest.name, "(first run only)...")
    tmp = dest.with_name(dest.name + ".part")
    last = None
    for ctx in _contexts():
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = mark = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if done - mark >= 10 << 20:
                        mark = done
                        print("   %d / %d MB" % (done >> 20, total >> 20))
            tmp.replace(dest)
            return
        except Exception as e:
            last = e
    raise last


def ensure_models():
    MODELS.mkdir(exist_ok=True)
    onnx, voices = MODELS / "kokoro.onnx", MODELS / "voices.bin"
    try:
        if not onnx.exists():
            try:
                fetch(KOKORO_BASE + "kokoro-v1.0.int8.onnx", onnx)  # smaller and faster on CPU
            except Exception as e:
                print("Fast model not available, trying the full model:", e)
                fetch(KOKORO_BASE + "kokoro-v1.0.onnx", onnx)
        if not voices.exists():
            fetch(KOKORO_BASE + "voices-v1.0.bin", voices)
    except Exception as e:
        print("\nAUTOMATIC DOWNLOAD FAILED:", e)
        print("Download these 2 files in your browser and put them in:", MODELS)
        print("  1) " + KOKORO_BASE + "kokoro-v1.0.int8.onnx   -> rename to  kokoro.onnx")
        print("  2) " + KOKORO_BASE + "voices-v1.0.bin   -> rename to  voices.bin")
        print("Then run start.bat again.")
        raise SystemExit(1)
    return onnx, voices


def split_pieces(text, first_len=28, next_len=70):
    """Short first piece (heard quickly), then longer pieces; cuts at punctuation first."""
    out = []
    for part in re.split(r"(?<=[.!?,;:])\s+", text.strip()):
        part = part.strip()
        if not part:
            continue
        limit = first_len if not out else next_len
        cur = ""
        for w in part.split():
            if cur and len(cur) + 1 + len(w) > limit:
                out.append(cur)
                cur, limit = w, next_len
            else:
                cur = (cur + " " + w).strip()
        if cur:
            out.append(cur)
    # glue tiny pieces (e.g. "office.") to the piece before it, so nothing is skipped or clipped
    merged = []
    for p in out:
        if merged and len(p) < 10 and len(merged[-1]) + len(p) < 90:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged or [text]


def trim(samples, thr=0.01):
    """Cut the silent start/end that TTS adds, so pieces play back-to-back without gaps."""
    idx = np.where(np.abs(samples) > thr)[0]
    if len(idx) == 0:
        return samples
    start = max(0, idx[0] - 240)
    end = min(len(samples), idx[-1] + 480)
    return samples[start:end]


class Engine:
    def __init__(self):
        from kokoro_onnx import Kokoro
        onnx, voices = ensure_models()
        self.tts = Kokoro(str(onnx), str(voices))
        try:
            self.voices = sorted(self.tts.get_voices())
        except Exception:
            self.voices = ["am_adam", "am_michael", "af_heart", "af_bella"]
        self.cache = collections.OrderedDict()  # repeated phrases come back instantly

        self.tts_lock = threading.Lock()  # Kokoro (espeak) must never run in 2 threads at once
        self.last_used = time.time()
        self.rtf = 0.0  # synth time / audio length; > 0.7 means the PC is struggling to keep up
        self.stt = None  # loaded in the background after the server starts (never blocks startup)
        self.stt_lock = threading.Lock()

    def _load_stt(self):
        import ctranslate2
        from faster_whisper import WhisperModel
        cuda = (ctranslate2.get_cuda_device_count() > 0) if DEVICE == "auto" else DEVICE == "cuda"
        print("Speech recognition:", WHISPER_MODEL, "on", "GPU" if cuda else "CPU")
        stt = None
        if cuda:
            try:
                stt = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
                stt.transcribe(np.zeros(16000, np.float32), language="en")
            except Exception as e:
                print("GPU speech recognition not usable, using CPU instead:", e)
                stt = None
        if stt is None:
            stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                               cpu_threads=max(2, (os.cpu_count() or 4) // 2))
        self.stt = stt

    def ensure_stt(self):
        with self.stt_lock:
            if self.stt is None:
                self._load_stt()

    def preload(self):
        try:
            self.ensure_stt()
            self.transcribe(np.zeros(16000, np.float32))
            print("Speech recognition ready.")
        except Exception as e:
            import traceback
            print("\nSpeech recognition could NOT load (typing/'say' mode still works):", e)
            traceback.print_exc()

    def transcribe(self, audio):
        self.ensure_stt()
        self.last_used = time.time()
        # vad_filter off: our Segmenter already cut the phrase, Whisper's VAD was clipping words
        segs, _ = self.stt.transcribe(
            audio, language="en", beam_size=1, vad_filter=False,
            condition_on_previous_text=False, temperature=0.0, without_timestamps=True)
        text = " ".join(s.text.strip() for s in segs).strip()
        if len(audio) < 16000 * 1.3 and text.lower().strip(" .!?,") in JUNK:
            return ""  # typical Whisper hallucination on noise
        return text

    async def stream(self, text, voice, speed):
        """Cut the text into pieces and send each piece as soon as it is ready. Never skips a piece."""
        if voice not in self.voices:
            voice = "am_adam" if "am_adam" in self.voices else self.voices[0]
        for piece in split_pieces(text):
            if not re.search(r"\w", piece):
                continue
            samples = None
            for attempt in range(2):  # one retry before giving up on a piece
                try:
                    samples, sr = await asyncio.to_thread(self.speak, piece, voice, speed)
                    break
                except Exception as e:
                    print("piece failed (try %d):" % (attempt + 1), repr(piece), "->", e)
            if samples is None:
                continue
            yield samples, sr

    def speak(self, text, voice, speed):
        key = (text.strip().lower(), voice, round(float(speed), 2))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        with self.tts_lock:
            t0 = time.time()
            lang = "en-gb" if voice[:1] == "b" else "en-us"  # bf_ / bm_ = British voices
            samples, sr = self.tts.create(text, voice=voice, speed=float(speed), lang=lang)
        dur = max(0.05, len(samples) / float(sr))
        print("   voice piece: %.2fs to make, %.2fs of audio | %s" % (time.time() - t0, dur, text[:40]))
        self.rtf = 0.7 * self.rtf + 0.3 * ((time.time() - t0) / dur)
        self.last_used = time.time()
        samples = trim(samples)
        self.cache[key] = (samples, sr)
        if len(self.cache) > 1500:
            self.cache.popitem(last=False)
        return samples, sr


class Segmenter:
    """Cuts the live microphone stream into phrases using a simple energy detector."""
    def __init__(self, silence_ms=SILENCE_MS):
        self.frame = 320  # 20 ms at 16 kHz
        self.buf = np.zeros(0, np.float32)
        self.pre = collections.deque(maxlen=12)
        self.cur, self.in_speech = [], False
        self.silence = self.voiced = self.nspeech = 0
        self.noise = 0.005
        self.set_silence(silence_ms)
        self.min_frames, self.max_frames = 6, 600  # at least 120 ms of speech, at most 12 s

    def set_silence(self, ms):
        self.silence_frames = max(5, int(ms / 20))

    def push(self, pcm16):
        out = []
        self.buf = np.concatenate([self.buf, pcm16.astype(np.float32) / 32768.0])
        while len(self.buf) >= self.frame:
            f, self.buf = self.buf[:self.frame], self.buf[self.frame:]
            rms = float(np.sqrt(np.mean(f * f)))
            speech = rms > max(0.010, self.noise * 3.0)
            if not self.in_speech:
                self.pre.append(f)
                if speech:
                    self.voiced += 1
                    if self.voiced >= 3:
                        self.in_speech, self.cur, self.silence, self.nspeech = True, list(self.pre), 0, 3
                else:
                    self.voiced = 0
                    self.noise = 0.95 * self.noise + 0.05 * rms
            else:
                self.cur.append(f)
                self.silence = 0 if speech else self.silence + 1
                self.nspeech += 1 if speech else 0
                if self.silence >= self.silence_frames or len(self.cur) >= self.max_frames:
                    if self.nspeech >= self.min_frames:
                        out.append(np.concatenate(self.cur))
                    self.cur, self.in_speech, self.voiced, self.silence = [], False, 0, 0
                    self.pre.clear()
        return out


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = None


@app.get("/health")
def health():
    return {"ok": True, "stt": WHISPER_MODEL, "voices": engine.voices if engine else []}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    seg = Segmenter()
    cfg = {"voice": "am_adam", "speed": 1.0, "gen": 0, "busy": False}
    audio_q = asyncio.Queue()  # phrases of microphone audio waiting for recognition
    text_q = asyncio.Queue()   # text waiting to be spoken: (text, time, generation)
    prep_q = asyncio.Queue()   # text to prepare in advance (not spoken, only stored for instant playback)
    send_lock = asyncio.Lock()  # keeps "audio header + audio bytes" together, never split by another message

    def compute_speed(text, backlog):
        return min(MAX_SPEED, round(float(cfg["speed"]) + 0.1 * backlog + (0.1 if len(text) > 200 else 0), 2))

    def drain(q):
        while not q.empty():
            q.get_nowait()

    async def stt_worker():
        """Recognizes the next phrase while the previous one is still being spoken."""
        while True:
            audio = await audio_q.get()
            try:
                text = await asyncio.to_thread(engine.transcribe, audio)
                if not text:
                    continue
                async with send_lock:
                    await ws.send_text(json.dumps({"type": "heard", "text": text}))
                text_q.put_nowait((text, time.time(), cfg["gen"]))
            except Exception as e:
                import traceback
                print("stt error:", e)
                traceback.print_exc()

    async def tts_worker():
        while True:
            text, t0, gen = await text_q.get()
            if gen != cfg["gen"]:
                continue  # was cancelled by STOP
            cfg["busy"] = True
            try:
                # phrases waiting? speak them together (fewer calls = no growing delay)
                while len(text) < 400 and not text_q.empty():
                    nt, _, g2 = text_q.get_nowait()
                    if g2 == cfg["gen"]:
                        text += " " + nt
                speed = compute_speed(text, text_q.qsize())
                first = True
                async for samples, sr in engine.stream(text, cfg["voice"], speed):
                    if gen != cfg["gen"]:
                        break  # STOP pressed
                    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
                    async with send_lock:
                        await ws.send_text(json.dumps({"type": "audio", "rate": int(sr)}))
                        await ws.send_bytes(pcm)
                    if first:
                        print("first sound after %.2fs  %s" % (time.time() - t0, text[:60]))
                        first = False
            except Exception as e:
                import traceback
                print("tts error:", e)
                traceback.print_exc()
            finally:
                cfg["busy"] = False

    async def prep_worker():
        """Makes the voice for prepared text in the background (only while nothing is being spoken)."""
        while True:
            text, gen = await prep_q.get()
            try:
                while (cfg["busy"] or not text_q.empty()) and gen == cfg["gen"]:
                    await asyncio.sleep(0.05)
                async for _ in engine.stream(text, cfg["voice"], compute_speed(text, 0)):
                    if gen != cfg["gen"] or cfg["busy"] or not text_q.empty():
                        break  # real speech has priority; what is already made stays stored
                print("prepared:", text[:50])
            except Exception as e:
                print("prepare error:", e)

    tasks = [asyncio.create_task(f()) for f in (stt_worker, tts_worker, prep_worker)]
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            try:  # one bad message must NEVER kill the connection
                if msg.get("bytes"):
                    raw = msg["bytes"]
                    raw = raw[: len(raw) // 2 * 2]  # odd byte count would crash
                    if raw:
                        for sg in seg.push(np.frombuffer(raw, dtype="<i2")):
                            while audio_q.qsize() >= 8:  # only if the PC is badly overloaded
                                audio_q.get_nowait()
                            audio_q.put_nowait(sg)
                elif msg.get("text"):
                    d = json.loads(msg["text"])
                    kind = d.get("type")
                    if kind == "config":
                        cfg["voice"] = d.get("voice", cfg["voice"])
                        cfg["speed"] = d.get("speed", cfg["speed"])
                        if d.get("silence_ms"):
                            seg.set_silence(int(d["silence_ms"]))
                    elif kind == "say" and d.get("text"):
                        txt = str(d["text"])[:20000]
                        print("got text (%d waiting):" % text_q.qsize(), txt[:120])
                        if not any(t == txt for t, _, _ in list(text_q._queue)):
                            text_q.put_nowait((txt, time.time(), cfg["gen"]))
                    elif kind == "prepare":  # {"type":"prepare","text":"..."} or {"type":"prepare","lines":["..",".."]}
                        lines = d.get("lines") or [d.get("text", "")]
                        for ln in lines:
                            if str(ln).strip():
                                prep_q.put_nowait((str(ln)[:20000], cfg["gen"]))
                    elif kind == "stop":  # cancels everything waiting and the sentence being spoken
                        cfg["gen"] += 1
                        drain(text_q)
                        drain(prep_q)
                        async with send_lock:
                            await ws.send_text(json.dumps({"type": "cleared"}))
            except Exception as e:
                print("bad message ignored:", repr(e))
    except Exception as e:
        print("connection closed:", repr(e))
    finally:
        for t in tasks:
            t.cancel()


def prewarm_phrases():
    """Lines in phrases.txt (one per line) are made at start, so they play instantly later."""
    f = HERE / "phrases.txt"
    if not f.exists():
        return
    voice = os.environ.get("PHRASE_VOICE", "am_adam")
    speed = float(os.environ.get("PHRASE_SPEED", "1.0"))
    n = 0
    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            for piece in split_pieces(line):
                if re.search(r"\w", piece):
                    engine.speak(piece, voice, speed)
            n += 1
        except Exception as e:
            print("phrase skipped:", line[:40], e)
    print("Ready-made phrases:", n)


def set_high_priority():
    """Windows: give the engine priority over other programs so the voice never stutters."""
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00008000)  # ABOVE_NORMAL_PRIORITY_CLASS
    except Exception:
        pass


def keep_warm():
    """When idle, quietly run the voice now and then so the first phrase after a pause is not slow.
    Skips itself if real speech is being made (never runs at the same time)."""
    while True:
        time.sleep(15)
        try:
            if engine and time.time() - engine.last_used > 15 and engine.tts_lock.acquire(blocking=False):
                try:
                    engine.tts.create("Okay.", voice=engine.voices[0], speed=1.0, lang="en-us")
                finally:
                    engine.tts_lock.release()
                engine.last_used = time.time()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        set_high_priority()
        print("Loading models. First run downloads about 600 MB, please wait...")
        engine = Engine()
        engine.speak("Ready.", "am_adam", 1.0)
        if os.environ.get("PRELOAD_STT") == "1":
            threading.Thread(target=engine.preload, daemon=True).start()
        threading.Thread(target=keep_warm, daemon=True).start()
        threading.Thread(target=prewarm_phrases, daemon=True).start()
        print("\nAccentLive engine is READY. Keep this window open while you take calls.")
        while True:  # if anything crashes, restart the web server by itself (models stay loaded)
            try:
                uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning",
                            ws_ping_interval=None, ws_ping_timeout=None)  # never drop the connection mid-call
                break
            except KeyboardInterrupt:
                break
            except Exception:
                import traceback
                traceback.print_exc()
                print("Server error, restarting in 1 second...")
                time.sleep(1)
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        input("\nERROR above. Send this text to the developer. Press Enter to close...")
