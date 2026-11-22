"""AccentLive Pro: type a line or press a button, a clear American voice goes into a virtual microphone.
Buttons are voiced once at start-up and kept in memory, so pressing one plays it straight away.
Every call opens with a spoken disclosure (your name, company, AI voice, recording notice)."""
import json, os, queue, subprocess, sys, threading, time, webbrowser
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
BASE = Path(sys.executable).parent if FROZEN else Path(__file__).parent
if FROZEN:  # no console window: keep a log file instead
    sys.stdout = sys.stderr = open(BASE / "log.txt", "w", buffering=1, encoding="utf-8")
if (BASE / "models" / "whisper").exists():  # speech model shipped inside the package (works offline)
    os.environ.setdefault("WHISPER_MODEL", str(BASE / "models" / "whisper"))

import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse

import core  # voice + speech engine (Kokoro / Whisper)

HERE = BASE
core.MODELS = BASE / "models"
CFG_FILE = HERE / "settings.json"
PORT = 8766
DEFAULT = {
    "name": "", "company": "", "voice": "am_adam", "speed": 1.0, "silence": 300,
    "mic": None, "cable": None, "monitor": None, "live": False,
    "buttons": [
        {"label": "Yes", "text": "Yes."},
        {"label": "One moment", "text": "One moment please."},
        {"label": "Repeat", "text": "Could you please repeat that?"},
        {"label": "Thanks", "text": "Thank you, have a great day."},
    ],
}
cfg = dict(DEFAULT)
if CFG_FILE.exists():
    try:
        cfg.update(json.loads(CFG_FILE.read_text("utf-8")))
    except Exception:
        pass

S = {"call": False, "ready": False, "gen": 0, "heard": [], "prepared": 0, "total": 0, "err": ""}
engine = None
text_q, audio_q = queue.Queue(), queue.Queue()
out_streams, mic = {}, {"s": None}
clips, pending = {}, set()  # clips: finished audio per button text; pending: lines queued or playing
seg = core.Segmenter(cfg["silence"])


def disclosure():
    return ("Hi, this is %s from %s. I'm using an AI voice so I'm easy to understand, "
            "and this call may be recorded." % (cfg["name"], cfg["company"]))


# ---------- audio devices ----------
def wasapi():
    for i, h in enumerate(sd.query_hostapis()):
        if "WASAPI" in h["name"]:
            return i
    return None


def devices():
    ha, ins, outs = wasapi(), [], []
    for i, d in enumerate(sd.query_devices()):
        if ha is not None and d["hostapi"] != ha:
            continue
        if d["max_input_channels"] > 0:
            ins.append({"id": i, "name": d["name"]})
        if d["max_output_channels"] > 0:
            outs.append({"id": i, "name": d["name"]})
    return ins, outs


def extra(dev):
    if dev is not None and sd.query_devices(dev)["hostapi"] == wasapi():
        return sd.WasapiSettings(auto_convert=True)
    return None


def out_stream(dev, sr):
    s = out_streams.get((dev, sr))
    if s is None:  # opened once and kept open, so playback starts with no set-up delay
        s = sd.OutputStream(samplerate=sr, device=dev, channels=1, dtype="float32",
                            latency="low", extra_settings=extra(dev))
        s.start()
        out_streams[(dev, sr)] = s
    return s


def play(samples, sr, gen):
    targets = [d for d in (cfg["cable"], cfg["monitor"]) if d is not None]
    if not targets:
        S["err"] = "Choose the virtual cable in settings first."
        return
    try:
        streams = [out_stream(d, sr) for d in targets]
    except Exception as e:
        out_streams.clear()
        S["err"] = "Audio output: %s" % e
        return
    x = samples.astype(np.float32).reshape(-1, 1)
    for i in range(0, len(x), 1200):
        if gen != S["gen"]:
            return  # Stop was pressed
        for s in streams:
            s.write(x[i:i + 1200])


# ---------- speaking ----------
def pieces(text):
    return [p for p in core.split_pieces(text) if any(c.isalnum() for c in p)]


def clip_of(text):
    c = clips.get((text, cfg["voice"], cfg["speed"]))
    if c is None:
        parts = [engine.speak(p, cfg["voice"], cfg["speed"]) for p in pieces(text)]
        if parts:
            c = (np.concatenate([a for a, _ in parts]), parts[0][1])
            clips[(text, cfg["voice"], cfg["speed"])] = c
    return c


def say_text(text, gen):
    c = clips.get((text, cfg["voice"], cfg["speed"]))
    if c is not None:  # prepared button: one finished clip, plays at once
        play(c[0], c[1], gen)
        return
    for p in pieces(text):
        if gen != S["gen"]:
            return
        try:
            samples, sr = engine.speak(p, cfg["voice"], cfg["speed"])  # instant when prepared
        except Exception as e:
            S["err"] = str(e)
            continue
        play(samples, sr, gen)


def tts_worker():
    while True:
        text, gen = text_q.get()
        try:
            if gen == S["gen"] and S["call"]:
                say_text(text, gen)
        finally:
            pending.discard(text)


def prepare():
    """Voice every button now, so nothing has to be made while the customer waits."""
    def run():
        bs = list(cfg["buttons"])
        S["total"], S["prepared"] = len(bs) + 1, 0
        texts = [b["text"] for b in bs]
        if cfg["name"] and cfg["company"]:
            texts.append(disclosure())
        for t in texts:
            try:
                clip_of(t)
            except Exception as e:
                S["err"] = str(e)
            S["prepared"] += 1
    threading.Thread(target=run, daemon=True).start()


# ---------- optional live voice (speak instead of type) ----------
def mic_cb(indata, frames, t, status):
    if S["call"] and cfg["live"]:
        for sg in seg.push(indata[:, 0]):
            audio_q.put(sg)


def sync_mic():
    if mic["s"]:
        try:
            mic["s"].close()
        except Exception:
            pass
        mic["s"] = None
    if not (S["call"] and cfg["live"]):
        return
    try:
        s = sd.InputStream(samplerate=16000, device=cfg["mic"], channels=1, dtype="int16",
                           blocksize=320, callback=mic_cb, extra_settings=extra(cfg["mic"]))
        s.start()
        mic["s"] = s
    except Exception as e:
        S["err"] = "Microphone: %s" % e


def stt_worker():
    while True:
        a = audio_q.get()
        if not (S["call"] and cfg["live"]):
            continue
        try:
            t = engine.transcribe(a)
        except Exception as e:
            S["err"] = "Speech recognition: %s" % e
            continue
        if t:
            S["heard"] = (S["heard"] + [t])[-6:]
            enqueue(t)


def enqueue(text):
    """Queue a line unless the same line is already waiting or playing (stops accidental double speech)."""
    if text in pending:
        return
    pending.add(text)
    text_q.put((text, S["gen"]))


def stop():
    pending.clear()
    S["gen"] += 1
    while not text_q.empty():
        try:
            text_q.get_nowait()
        except queue.Empty:
            break


# ---------- local web interface ----------
app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/state")
def state():
    ins, outs = devices()
    return {"cfg": cfg, "ins": ins, "outs": outs, "voices": engine.voices if engine else [],
            "cable_found": any("CABLE" in d["name"] for d in outs), "call": S["call"], "ready": S["ready"], "heard": S["heard"], "err": S["err"],
            "prepared": S["prepared"], "total": S["total"], "disclosure": disclosure()}


@app.post("/api/config")
def config(body: dict = Body(...)):
    old = (cfg["voice"], cfg["speed"], json.dumps(cfg["buttons"]), cfg["name"], cfg["company"])
    for k in DEFAULT:
        if k in body:
            cfg[k] = body[k]
    cfg["buttons"] = [b for b in cfg["buttons"] if b.get("text")][:30]
    CFG_FILE.write_text(json.dumps(cfg, indent=1), "utf-8")
    seg.set_silence(int(cfg["silence"]))
    S["err"] = ""
    if engine and old != (cfg["voice"], cfg["speed"], json.dumps(cfg["buttons"]), cfg["name"], cfg["company"]):
        prepare()
    sync_mic()
    return {"ok": True}


@app.post("/api/call")
def call(body: dict = Body(...)):
    if body.get("on"):
        if not (cfg["name"].strip() and cfg["company"].strip()):
            return {"error": "Enter your name and company first. They are spoken at the start of every call."}
        if cfg["cable"] is None:
            return {"error": "Choose the virtual cable in settings first."}
        S["call"] = True
        enqueue(disclosure())  # the call always opens with the disclosure
    else:
        stop()
        S["call"] = False
    sync_mic()
    return {"ok": True}


@app.post("/api/say")
def say(body: dict = Body(...)):
    t = str(body.get("text", "")).strip()[:2000]
    if S["call"] and t:
        enqueue(t)
    return {"ok": S["call"]}


@app.post("/api/button")
def button(body: dict = Body(...)):
    i = int(body.get("i", -1))
    if S["call"] and 0 <= i < len(cfg["buttons"]):
        enqueue(cfg["buttons"][i]["text"])
    return {"ok": S["call"]}


@app.post("/api/stop")
def stop_api():
    stop()
    return {"ok": True}


def autoselect():
    _, outs = devices()
    if cfg["cable"] is None:
        for d in outs:
            if "CABLE Input" in d["name"]:
                cfg["cable"] = d["id"]


def boot():
    global engine
    try:
        autoselect()
        engine = core.Engine()
        if cfg["voice"] not in engine.voices:
            cfg["voice"] = "am_adam" if "am_adam" in engine.voices else engine.voices[0]
        engine.speak("Ready.", cfg["voice"], 1.0)
        prepare()
        S["ready"] = True
        if cfg["live"]:
            threading.Thread(target=engine.preload, daemon=True).start()
    except SystemExit:
        S["err"] = "Voice model files are missing. See README.txt."
    except Exception as e:
        S["err"] = "Engine could not start: %s" % e


def open_window():
    url = "http://127.0.0.1:%d" % PORT
    if sys.platform == "win32":
        subprocess.Popen('start "" msedge --app=%s --window-size=1040,780' % url, shell=True)
    else:
        webbrowser.open(url)


if __name__ == "__main__":
    core.set_high_priority()
    for f in (tts_worker, stt_worker, boot):
        threading.Thread(target=f, daemon=True).start()
    threading.Timer(2.0, open_window).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_config=None)
