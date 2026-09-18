# Rock-Paper-Scissors vs AI — IET PSBA booth demo

Two versions of the same game:

- `rps_booth.py` — the laptop app you run at the booth. Fullscreen, offline, most reliable.
- `index.html` — a phone/browser version people can open on their own device.

---

## 1. Laptop version (use this at the booth)

Install once:

```
pip install opencv-python mediapipe
```

Run:

```
python rps_booth.py
```

Keys: **F** fullscreen · **R** reset scoreboard · **Q** quit

How to play: hold your hand steady in front of the camera → the ROCK–PAPER–SCISSORS–SHOOT
countdown starts → hold your throw until "SHOOT!". Fist = rock, open palm = paper,
two fingers = scissors.

> If `mediapipe` refuses to install, you are probably on Python 3.13. Use Python 3.11 or 3.12.

## 2. Browser version

Open `index.html` over **https** or from `localhost` — browsers block the camera on plain
`file://` in some cases. Easiest local test:

```
python -m http.server 8000
```

then visit `http://localhost:8000`. To share a link at the fair, drop the file on GitHub
Pages or Netlify (both free) and put the QR code on your booth poster.

---

## How the AI actually works (your answer when people ask)

It is **not** random. It keeps a short history of what you have thrown and builds a
frequency table over the last one and two moves — an order-2 Markov model. When it has
seen a pattern enough times it predicts your next throw and plays the move that beats it.
About 12% of the time it throws randomly on purpose, so it stays beatable and fun.

Humans are bad at being random — they avoid repeating a throw three times, and they tend
to switch after losing. The model just learns those habits.

Measured on a simulated habitual player: **52% win rate for the AI**, versus a 33% random
baseline. Against a genuinely random opponent it drops to 33%, exactly as it should —
proof it is learning patterns and not cheating.

---

## Booth day checklist

- Charge the laptop, and bring the charger. The camera plus model drains fast.
- Test under the actual hall lighting beforehand. Backlight from a window is the one thing
  that breaks hand detection — position the laptop so players are lit from the front.
- Keep the scoreboard visible. "AI 47 – Humans 31" on screen is the hook that makes people
  walk over.
- Have one person on the laptop and one person calling out to passers-by. The demo sells
  itself once someone is playing.
- Press **R** every hour or so if the AI is running away with the score — a crushed
  scoreboard discourages people.
