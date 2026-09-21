# Rock-Paper-Scissors vs AI — IET PSBA booth demo

Play rock-paper-scissors against an AI that learns your patterns, using just a webcam.
Built for the IET PSB Academy On Campus booth at Campus Club Fair 2026.

Two versions:

- `rps_booth.py` — the laptop app for the booth. Fullscreen, runs fully offline.
- `index.html` — a browser version people can open on their own phone.

---

## Setup (macOS — tested and working)

**Important: use Python 3.11 or 3.12.** mediapipe does not support 3.13 or 3.14.

```bash
# if you don't have 3.12:
brew install python@3.12

cd rps-booth
$(brew --prefix python@3.12)/bin/python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "mediapipe==0.10.14" "numpy<2" "opencv-python==4.10.0.84"
python rps_booth.py
```

**The versions are pinned on purpose.** mediapipe 1.x removed the `mp.solutions` API this
script uses, and it needs a model file downloaded at runtime — bad news on venue wifi.
0.10.14 runs offline. Don't `pip install mediapipe` unpinned.

On Windows/Linux the same pins work; just create the venv with your own 3.12.

### Every time after that

```bash
cd rps-booth && source venv/bin/activate && python rps_booth.py
```

### Controls

**F** fullscreen · **R** reset scores · **M** mute · **B** show/hide robot · **Q** quit

### The 3D robot opponent

```bash
python rps_booth.py --robot
```

Adds a panel beside the camera with a robot that pumps its fist through the
countdown and throws the AI's actual move. It is a small software 3D renderer in
`robot3d.py`: boxes, 4x4 matrices, a per-pixel z-buffer and flat shading. There
is no 3D engine and no model file to download, which is why it also runs on a
Raspberry Pi.

Plain `python rps_booth.py` runs without it. **B** toggles the panel live, so if
anything misbehaves at the booth you can hide it without restarting.

### Voice, face greeting and the local LLM

The robot greets people it sees (OpenCV face detection), speaks through macOS
`say`, and comments on real patterns in your play. Every line is derived from
the game state - it never invents facts.

Optional: a local LLM rephrases those lines in the robot's voice.

```bash
brew install ollama
brew services start ollama
ollama pull gemma3:4b
python llm.py                          # quick test
python rps_booth.py --robot --llm
```

If Ollama is slow or missing it falls back to the built-in lines. **M** mutes.

### IET facts

Twice per game (after round one, and after the result) the robot shares a fact
about the IET, IET Singapore or IET PSBA. No fact repeats within a game, and the
deck rotates so consecutive players hear different ones. Edit `IET_FACTS` near
the top of `rps_booth.py` to change them. Keep each under 14 words.

### Playing

Hold your hand steady in front of the camera and the countdown starts: Rock,
Paper, Scissors, Shoot, one second per word. Hold your throw until Shoot. Each
player gets three rounds, then the game ends and the next person steps up.

Fist = rock · open palm = paper · two fingers = scissors

---

## Things that will bite you (all hit us during setup)

**macOS camera permission.** First run fails with `not authorized to capture video`. Grant
your terminal app camera access in System Settings → Privacy & Security → Camera, then
**fully quit the terminal (Cmd+Q) and reopen it** — macOS only applies the permission at
app launch, so a new window is not enough.

**Continuity Camera hijacks the camera.** If your iPhone is nearby, macOS may use it
instead of the laptop webcam. Either pass a different index (`python rps_booth.py 1`, then
`2`), or better, turn it off: iPhone → Settings → General → AirPlay & Continuity →
Continuity Camera. Do this before the fair so the phone can't hijack the demo mid-event.

**Harmless noise on startup.** `SymbolDatabase.GetPrototype() is deprecated`,
`Feedback manager requires a model with a single signature inference`, XNNPACK messages —
all normal, ignore them.

---

## Browser version

Open `index.html` over https or localhost (the camera is blocked on plain `file://`):

```bash
python -m http.server 8000     # then visit http://localhost:8000
```

To share it at the fair, host it free on GitHub Pages or Netlify and put the QR code on
the booth poster.

## Running on a Raspberry Pi

Works on a **Raspberry Pi 4 or 5** running **64-bit Raspberry Pi OS**. Use 4GB of
RAM or more, and 8GB if you also want the local LLM. Any USB webcam works, and
so does the Pi Camera Module.

The game detects the Pi and switches to a lighter profile on its own: hands are
tracked on a 640x360 frame, the robot is drawn without supersampling and every
other frame, faces are checked less often, and the LLM uses the 1B model.

**Python version matters.** Current Raspberry Pi OS ships Python 3.13, and
mediapipe 0.10.14 only supports 3.9 to 3.12. `uv` installs a separate 3.12
without touching the system Python:

```bash
sudo apt update && sudo apt install -y espeak-ng alsa-utils
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd ~/rps-booth
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install -r requirements.txt
python rps_booth.py --robot
```

- **Pi Camera Module:** used automatically if no USB camera is found, or force
  it with `--picam`. Frames are read from `rpicam-vid`, so it works with any
  Python version and no picamera2 is needed.
- **Speech:** `espeak-ng` stands in for macOS `say`. Sound effects play through
  `aplay`.
- **Local LLM on the Pi:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull gemma3:1b
  python rps_booth.py --robot --llm
  ```
  Use `--model NAME` to try another model, such as `qwen2.5:0.5b`, which is
  faster still.
- **Testing the Pi profile on a Mac:** `python rps_booth.py --robot --pi`.

Expect about 20 fps on a Pi 5 and 10 to 12 fps on a Pi 4. Hand tracking is the
limit there. The countdown is timed in seconds, not frames, so play feels the
same.

---

## How the AI works (your answer when someone asks)

It is **not** random. It keeps a history of your throws and builds frequency tables over
your last one and two moves — an order-2 Markov model. Once it has seen a pattern enough
times it predicts your next throw and plays the counter. About 12% of the time it throws
randomly on purpose, so it stays beatable.

Humans are bad at being random: they avoid repeating a throw three times, and they tend to
switch after losing. The model just learns those habits.

Measured against a simulated habitual player: **52% win rate for the AI**, versus a 33%
random baseline. Against a genuinely random opponent it drops to 33%, exactly as it
should — which is the proof it is learning patterns rather than cheating.

---

## Booth day checklist

- Charge the laptop and bring the charger — camera plus model drains it fast.
- Test under the actual hall lighting. Backlight from a window is the one thing that
  breaks hand detection; position the laptop so players are lit from the front.
- Keep the scoreboard visible — "AI 47 – Humans 31" is the hook that pulls people over.
- One person on the laptop, one calling out to passers-by. It sells itself once someone
  is playing.
- Press **R** every hour or so if the AI runs away with the score.
