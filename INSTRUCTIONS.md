# Running HearHere

How to install and run HearHere on each operating system.

> **Platform status.** Recording currently works on **Windows only** — it's the
> priority-1 platform. macOS and Linux capture are not implemented yet (they
> arrive in Batches 3 and 4; see [`TODO.md`](TODO.md)). Everything *except*
> recording — processing an existing recording, the web UI, exports, and the
> remote backend — is cross-platform today.

---

## Windows — step by step from a cloned repo

This assumes you've just done `git clone` on a Windows machine and have nothing
else set up. Do the steps in order. Every command is run in **PowerShell**
(press Start, type "PowerShell", open it).

> **Heads-up:** the Windows record → transcribe path is fully written but has not
> yet been exercised on real audio hardware or with the real models. Expect to
> hit the odd rough edge on first run; the [Troubleshooting](#troubleshooting)
> section covers the likely ones.

### Step 1 — Install Python 3.10+

1. Download the installer from
   [python.org/downloads](https://www.python.org/downloads/) (3.10, 3.11, or
   3.12 — **not** 3.13 yet, some ML deps lag behind).
2. Run it and **tick "Add python.exe to PATH"** on the first screen before
   clicking Install. This is the most common thing people miss.
3. Close and reopen PowerShell, then confirm:

   ```powershell
   python --version
   ```

   You should see `Python 3.1x.x`. If you get "Python was not found" or the
   Microsoft Store opens, PATH wasn't set — re-run the installer and tick the box.

### Step 2 — Go into the repo folder

Replace the path with wherever you cloned it:

```powershell
cd C:\Users\you\hearhere
```

You're in the right place if `dir` shows `pyproject.toml` and a `hearhere`
folder.

### Step 3 — Create and activate a virtual environment

A venv keeps HearHere's dependencies out of your system Python.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**If activation fails** with *"running scripts is disabled on this system"*,
Windows is blocking PowerShell scripts. Allow it for this window only, then
activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

(Or, if you prefer the classic Command Prompt instead of PowerShell, run
`.venv\Scripts\activate.bat` there — no execution-policy issue.)

Once active, your prompt shows `(.venv)` at the start. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

### Step 4 — Install HearHere and the pieces you want

Install from the repo you're standing in. `-e` makes it an *editable* install, so
`git pull` updates take effect without reinstalling.

```powershell
# Minimum to record and get a transcript:
pip install -e ".[capture,asr]"

# Or everything — speaker names, summaries, and the browser UI:
pip install -e ".[capture,asr,diarization,llm,webui]"
```

This takes a while — `asr` pulls in PyTorch and NeMo, which are large. When it
finishes, confirm the command is available:

```powershell
hearhere --version
hearhere --help
```

**GPU note.** By default pip installs the **CPU** build of PyTorch, which works
but transcribes slowly. If you have an NVIDIA GPU and want CUDA acceleration,
install a CUDA build of torch *before* the line above (check your CUDA version at
[pytorch.org](https://pytorch.org/get-started/locally/)):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 5 — (Optional) set up speaker names and summaries

Skip this if you only want a plain transcript — the pipeline **fails soft**:
if diarization or the LLM isn't available it logs a warning and keeps going.

**Speaker separation (diarization)** needs a free Hugging Face token because the
model is license-gated:

1. Create an account at [huggingface.co](https://huggingface.co).
2. Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   and click **Agree / accept** the terms on each.
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   (a "Read" token is enough).
4. Make it available to HearHere for this PowerShell window:

   ```powershell
   $env:HF_TOKEN = "hf_your_token_here"
   ```

   (Or put it in `config.toml` under `[diarization].hf_token` — see Step 6.)

**Summaries** need [Ollama](https://ollama.com) running locally:

1. Download and install Ollama for Windows from [ollama.com](https://ollama.com).
   It runs in the background after install.
2. Pull the model once:

   ```powershell
   ollama pull llama3.1
   ```

### Step 6 — (Optional) create a config file

Defaults work without this. To change where meetings are saved, the language, or
which engines run, copy the example and edit it:

```powershell
copy config.example.toml config.toml
notepad config.toml
```

HearHere looks for `config.toml` in the current folder first, then in
`%USERPROFILE%\.config\hearhere\`. The setting most people change is
`[general].storage_dir` (default `~/HearHere`, i.e.
`C:\Users\you\HearHere`).

### Step 7 — Record your first meeting

Start a real or test call so there's system audio to capture, then:

```powershell
hearhere record --title "Weekly Sync"
```

What happens:

- It records your **microphone** (your voice → `self`) and the **system output**
  — literally what's coming out of your speakers/headphones, via WASAPI loopback,
  so **no virtual cable is needed**.
- The terminal prints `Recording… press Enter to stop.` — leave it running for
  the meeting, then press **Enter**.
- It then transcribes both channels, separates speakers, and writes a summary.
  **The very first run also downloads the Parakeet model (~2 GB)** from Hugging
  Face, so give it time and keep the network on.

When it's done you'll see something like
`Processed 42 segment(s) -> C:\Users\you\HearHere\2026-09-13_weekly-sync`.

The mic is captured with **sounddevice** (PortAudio) and the system output with
**soundcard** (WASAPI loopback). If a device won't open, list the names and pin
them in `config.toml`:

```powershell
hearhere devices    # lists input names ([capture].mic_device) and outputs ([capture].output_device)
```

Handy variants:

```powershell
hearhere record --title "Weekly Sync" --no-process   # just record; transcribe later
hearhere process "C:\Users\you\HearHere\2026-09-13_weekly-sync"   # transcribe a saved recording
hearhere list                                         # list past meetings
```

### Step 8 — Look at the results

Open the meeting folder (`C:\Users\you\HearHere\<date>_<title>\`). Inside:

- `transcript.md` — readable transcript, "Me" vs each speaker.
- `summary.md` — summary, decisions, action items (if the LLM ran).
- `meeting.json` — the full structured record (everything else is exported from
  this).
- `transcript.srt` — subtitles. `audio/` — the two WAVs. `hearhere.log` — the run log.

### Step 9 — Review and tidy up in the browser

```powershell
hearhere ui
```

Open <http://127.0.0.1:8809> in your browser. You can browse every past meeting,
read the transcript and summary, **rename speakers inline** (e.g. "Speaker 1" →
"Anna" — it rewrites `meeting.json` and re-exports automatically), and
**re-export** to other formats. Nothing is uploaded; it only reads what's already
on your disk. Press **Ctrl+C** in PowerShell to stop the server.

### Step 10 — Re-export to other formats anytime (optional)

```powershell
hearhere export "C:\Users\you\HearHere\2026-09-13_weekly-sync" --format md,srt,vtt
```

Exports are regenerated from `meeting.json`, so this never re-runs the models.

### The short version

Once set up, your day-to-day is just:

```powershell
cd C:\Users\you\hearhere
.\.venv\Scripts\Activate.ps1
hearhere record --title "Some Meeting"
hearhere ui
```

---

## macOS (recording not yet supported)

Recording on macOS arrives in **Batch 3** (via BlackHole / a virtual output
device). Until then, `hearhere record` will tell you it's not implemented.

What already works on macOS today:

- **Review existing meetings** — `pip install ".[webui]"` then `hearhere ui`.
- **Process a recording made elsewhere** — drop a meeting folder with
  `audio/self.wav` + `audio/others.wav` under your `storage_dir` and run
  `pip install ".[asr,diarization,llm]"` + `hearhere process <folder>`.
- **Use a remote GPU worker** — see [Remote backend](#remote-backend-any-os).

Installation is the same as Windows minus the `capture` extra:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[asr,diarization,llm,webui]"
```

---

## Linux (recording not yet supported)

Recording on Linux arrives in **Batch 4** (via a PipeWire/PulseAudio `.monitor`
source). Until then, `hearhere record` will tell you it's not implemented.

> **WSL2 note:** WSL2 has no direct audio access — you can't record from it even
> once Linux capture lands. Record on the host OS.

Everything except recording works, exactly as described for macOS above:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[asr,diarization,llm,webui]"
hearhere ui        # review meetings in the browser
```

---

## Remote backend (any OS)

If your laptop is too slow and you have a rented GPU box (e.g. RunPod), you can
offload processing. **Audio leaves your machine in this mode** — HearHere warns
you and asks for confirmation before the first upload.

On the **GPU host**, run the worker:

```bash
pip install ".[remote,asr,diarization,llm]"
hearhere worker --host 0.0.0.0 --port 8808
```

On your **laptop**, point the config at it and switch the backend:

```toml
[compute]
backend = "remote"

[compute.remote]
url   = "https://<your-gpu-host>:8808"
token = ""   # or set HEARHERE_REMOTE_TOKEN
```

Then `hearhere record` / `hearhere process` run the pipeline remotely and pull
the finished meeting back; exports still happen locally.

---

## Troubleshooting

- **`python` not found / the Microsoft Store opens** — Python isn't on PATH.
  Re-run the python.org installer and tick *"Add python.exe to PATH"*.
- **`running scripts is disabled on this system`** (activating the venv) — run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use
  `.venv\Scripts\activate.bat` in Command Prompt.
- **`hearhere` is not recognized** — the venv isn't active. Run
  `.\.venv\Scripts\Activate.ps1` (you should see `(.venv)` in the prompt).
- **`Recording the microphone channel failed…`** — the mic is captured with
  `sounddevice` (PortAudio), which handles most pro/USB interfaces (e.g.
  **Focusrite Scarlett**). If it still can't open, run `hearhere devices` to list
  input names, then set `[capture].mic_device` in `config.toml` to one of them
  (or change the Windows default mic in *Settings → System → Sound*) and
  re-record. HearHere fails immediately with this message instead of silently
  producing an empty recording.
- **`Recording the system-output (loopback) channel failed (AssertionError…)`** —
  the *output* device isn't compatible with `soundcard`'s WASAPI loopback. Run
  `hearhere devices`, set `[capture].output_device` to another playback device
  (or change the Windows default), and re-record.
- **`IndexError: index 0 is out of bounds…` / `PermissionError [WinError 32] …
  manifest.json` during transcription** — this used to happen when a channel was
  captured empty (see the item above); the empty channel is now skipped with a
  warning instead of crashing NeMo. If you still see it, the WAV under `audio/`
  has no audio — re-record after fixing the device.
- **Transcription ran on CPU / `CUDA is not available`** — the default PyTorch is
  CPU-only. On an NVIDIA machine, install a CUDA build of torch (Step 4's GPU
  note) for a large speed-up.
- **`UserWarning: … does not support symlinks` (Hugging Face cache)** — harmless;
  the model still downloads. To silence it, enable Windows *Developer Mode*
  (Settings → Privacy & security → For developers) so the cache can use symlinks.
- **`'record' is not implemented` on macOS/Linux** — expected; recording is
  Windows-only for now (Batches 3/4 add the rest).
- **`The web UI needs the '[webui]' extra`** — run `pip install ".[webui]"`.
- **Diarization is skipped** — set a Hugging Face token (`HF_TOKEN`) and accept
  the pyannote model license; the pipeline fails soft and continues unlabeled.
- **No summary** — make sure Ollama is installed and running and the model is
  pulled (`ollama pull llama3.1`); summaries are skipped if it's unavailable.
- **Slow transcription** — Parakeet runs on CPU but is much faster on a CUDA
  GPU; set `[compute].device = "cuda"` or leave it on `auto`.
