"""
Rock Paper Scissors vs AI
IET PSB Academy On Campus, Campus Club Fair 2026

Booth demo. A player shows a hand gesture to the webcam, the AI predicts what
they will throw next and plays the counter. Three rounds per player, then the
next person steps up. Runs fully offline.

Run:   python rps_booth.py                  built-in webcam
       python rps_booth.py 1                second camera
       python rps_booth.py --robot          adds the 3D robot opponent
       python rps_booth.py --robot --llm    local Ollama rephrases its lines
       python rps_booth.py --pi             force the Raspberry Pi profile
       python rps_booth.py --picam          use the Pi Camera Module (rpicam-vid)
       python rps_booth.py --model NAME     choose the Ollama model
       python rps_booth.py --no-hof         turn off the Hall of Fame photos
       python rps_booth.py --keep-hof       keep last session's winners on screen
       python rps_booth.py --no-qr          no QR code for winners' photos
Works on macOS and on a 64-bit Raspberry Pi 4 or 5. The Pi profile switches on
automatically: smaller processing frames, lighter robot rendering, and the
1B language model.
Keys:  Q or ESC quit    R reset scores    F fullscreen    M mute
       B show or hide the robot panel
       X delete the newest Hall of Fame photo (if a winner asks)

Hall of Fame: anyone who beats the AI gets a 3 2 1 countdown and a photo.
Photos are saved only on this computer, in the hall_of_fame folder next to
this file, and the latest winners are shown on the start screen. Every time
the booth starts, the Hall of Fame starts empty again.

Stats: every finished game is added to stats/games.csv next to this file
(open it in Excel or Numbers). The start screen shows today's Humans vs AI
score, and it carries on from the file if the booth is restarted.

Keeps the Mac awake while running (macOS caffeinate), so the screen never
sleeps in the middle of the fair.

QR code: next to the winner's photo. Scanning it opens the photo on the phone,
served straight from this laptop, so the phone must be on the same Wi-Fi.
Only photos from the current session can be opened, each with a random link.
"""

import collections
import csv
import http.server
import secrets
import socket
import threading
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave

import cv2
import numpy as np
import mediapipe as mp

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:               # falls back to OpenCV's built-in font
    Image = ImageDraw = ImageFont = None

try:
    import robot3d
except Exception:
    robot3d = None

try:
    import llm as llm_mod
except Exception:
    llm_mod = None

ROBOT_PANEL_W = 460

# ----------------------------------------------------------------------------
# Branding and palette (BGR). Dark blue base, purple accent, green highlight.
# ----------------------------------------------------------------------------
CLUB_NAME = "IET PSB Academy On Campus"

BLUE_DARK = (54, 23, 10)      # #0A1736  bars
BLUE_LINE = (110, 56, 30)     # #1E386E  subtle dividers
PURPLE = (247, 92, 139)       # #8B5CF7  primary accent
GREEN = (153, 211, 52)        # #34D399  highlight, robot speech
WHITE = (245, 245, 245)
GREY = (190, 176, 160)        # #A0B0BE  secondary text
EMPTY = (90, 70, 55)          # unfilled round marker

WIN_C, LOSE_C, TIE_C = GREEN, PURPLE, GREY
ACCENT = PURPLE

# ----------------------------------------------------------------------------
# Game rules and timing. Deliberately unhurried so people can keep up.
# ----------------------------------------------------------------------------
ROUNDS = 3              # every player gets exactly three rounds
STEP = 1.0              # seconds per countdown word
CAPTURE_AT = 3 * STEP + 0.6   # read the hand a moment after "Shoot"
REVEAL_HOLD = 4.0       # how long the result stays up
READY_GAP = 1.8         # breather before the next round can start
GAMEOVER_HOLD = 3.0     # final screen, then back to the start page
HOF_RESULT = 1.8        # winners: result shows this long, then the photo countdown
HOF_STEP = 1.0          # seconds per 3 2 1 count before the photo
HOF_HOLD = 3.5          # how long the new Hall of Fame photo stays up
HOF_QR_HOLD = 14.0      # longer when there is a QR code to scan
TALK_WAIT_MAX = 10.0    # never hold a round longer than this for speech
STABLE_FRAMES = 10      # hand must be held steady this many frames to start
IDLE_SECS = 15          # no hand for this long mid-game, back to attract

MOVES = ("rock", "paper", "scissors")
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
COUNTER = {"rock": "paper", "paper": "scissors", "scissors": "rock"}


# ----------------------------------------------------------------------------
# Text rendering. A clean system sans serif via Pillow, drawn only into the
# small region each line covers so it stays fast.
# ----------------------------------------------------------------------------
_FONT_FILES = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
# Single-weight font files, tried first (Raspberry Pi OS and other Linux).
_WEIGHT_FILES = {
    "light": ["/usr/share/fonts/truetype/piboto/Piboto-Light.ttf",
              "/usr/share/fonts/truetype/noto/NotoSans-Light.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf"],
    "regular": ["/usr/share/fonts/truetype/piboto/Piboto-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "medium": ["/usr/share/fonts/truetype/piboto/Piboto-Bold.ttf",
               "/usr/share/fonts/truetype/noto/NotoSans-Medium.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
}
_WEIGHTS = {
    "light":   ["light", "book", "regular", "roman"],
    "regular": ["regular", "roman", "book", "medium"],
    "medium":  ["medium", "demi bold", "demibold", "semibold", "bold", "regular"],
}
_faces_cache, _font_cache = {}, {}


def _faces(path):
    if path not in _faces_cache:
        found = {}
        for i in range(24):
            try:
                f = ImageFont.truetype(path, 20, index=i)
            except Exception:
                break
            found.setdefault((f.getname()[1] or "").lower(), i)
        _faces_cache[path] = found
    return _faces_cache[path]


def get_font(px, weight="regular"):
    key = (px, weight)
    if key in _font_cache:
        return _font_cache[key]
    font = None
    if ImageFont is not None and not sys.platform.startswith("darwin"):
        for path in _WEIGHT_FILES[weight]:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, px)
                    break
                except Exception:
                    continue
    if font is None and ImageFont is not None:
        for path in _FONT_FILES:
            if not os.path.exists(path):
                continue
            faces = _faces(path)
            if not faces:
                continue
            idx = next((faces[w] for w in _WEIGHTS[weight] if w in faces),
                       next(iter(faces.values())))
            try:
                font = ImageFont.truetype(path, px, index=idx)
                break
            except Exception:
                continue
    _font_cache[key] = font
    return font


def draw_text(img, text, x, y, px, color, weight="regular", align="left",
              shadow=True):
    """Draw text with its baseline at y. Returns the rendered width."""
    if not text:
        return 0
    H, W = img.shape[:2]
    font = get_font(px, weight)
    if font is None:
        scale, th = px / 32.0, max(1, int(round(px / 20)))
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        x = x - tw // 2 if align == "center" else (x - tw if align == "right" else x)
        if shadow:
            cv2.putText(img, text, (x + 1, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), th + 1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    th, cv2.LINE_AA)
        return tw

    spr, one_minus_a, ox, oy, l, tw = _sprite(text, px, weight, color, shadow)
    if align == "center":
        x = x - tw // 2 - l
    elif align == "right":
        x = x - (l + tw)
    x0, y0 = x - ox, y - oy                      # sprite's top-left on screen
    sh, sw = spr.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + sw), min(H, y0 + sh)
    if cx1 <= cx0 or cy1 <= cy0:
        return tw
    sx0, sy0 = cx0 - x0, cy0 - y0
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
    roi = img[cy0:cy1, cx0:cx1]
    blended = roi * one_minus_a[sy0:sy1, sx0:sx1] + spr[sy0:sy1, sx0:sx1]
    roi[:] = blended.astype(np.uint8)
    return tw


# Rendered text is cached as a small premultiplied sprite, so a label that
# stays on screen is drawn by Pillow once, not every frame. This matters a lot
# on a Raspberry Pi.
_sprites = collections.OrderedDict()


def _sprite(text, px, weight, color, shadow):
    key = (text, px, weight, color, shadow)
    hit = _sprites.get(key)
    if hit is not None:
        _sprites.move_to_end(key)
        return hit
    font = get_font(px, weight)
    l, t, r, b = font.getbbox(text, anchor="ls")
    pad = 3
    w, h = (r - l) + 2 * pad + 2, (b - t) + 2 * pad + 3
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ox, oy = pad - l, pad - t                     # baseline origin in sprite
    if shadow:
        d.text((ox + 1, oy + 2), text, font=font, fill=(0, 0, 0, 255), anchor="ls")
    d.text((ox, oy), text, font=font, fill=(color[2], color[1], color[0], 255),
           anchor="ls")
    a = np.asarray(im).astype(np.float32)
    alpha = a[..., 3:4] / 255.0
    spr = a[..., 2::-1] * alpha                   # RGB to BGR, premultiplied
    hit = (spr, 1.0 - alpha, ox, oy, l, r - l)
    _sprites[key] = hit
    if len(_sprites) > 400:
        _sprites.popitem(last=False)
    return hit


def centre(img, text, y, px, color, weight="regular"):
    return draw_text(img, text, img.shape[1] // 2, y, px, color, weight, "center")


_widths = {}


def text_width(text, px, weight="regular"):
    key = (text, px, weight)
    if key not in _widths:
        font = get_font(px, weight)
        if font is None:
            _widths[key] = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 32.0, 1)[0][0]
        else:
            _widths[key] = int(font.getlength(text))
        if len(_widths) > 2000:
            _widths.clear()
    return _widths[key]


def centre_wrapped(img, text, y, px, color, weight="regular", margin=40, gap=1.35,
                   bottom=False):
    """Centre text, wrapping onto extra lines so it never runs off the edge.
    With bottom=True, y is where the last line sits and the text grows upward."""
    max_w = img.shape[1] - 2 * margin
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if cur and text_width(trial, px, weight) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    lines = lines[:3]
    if bottom:
        y -= (len(lines) - 1) * px * gap
    for i, line in enumerate(lines):
        centre(img, line, int(y + i * px * gap), px, color, weight)
    return len(lines)


def dim(img, alpha=0.55):
    cv2.addWeighted(img, 1 - alpha, img * 0, alpha, 0, img)


def blend(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ----------------------------------------------------------------------------
# Platform: Raspberry Pi detection and a camera that also reads the Pi Camera
# Module through rpicam-vid (works with any Python version, no picamera2).
# ----------------------------------------------------------------------------
def is_raspberry_pi():
    try:
        with open("/proc/device-tree/model") as f:
            return "raspberry pi" in f.read().lower()
    except Exception:
        return False


def first_tool(*names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


class Camera:
    """USB or built-in webcam through OpenCV, or the Pi Camera Module
    streamed as raw YUV frames from rpicam-vid."""

    def __init__(self, index=0, size=(1280, 720), force_picam=False):
        self.w, self.h = size
        self.cap = None
        self.proc = None
        if not force_picam:
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            ok = cap.isOpened() and cap.read()[0]
            if ok:
                self.cap = cap
            else:
                cap.release()
        if self.cap is None:
            exe = first_tool("rpicam-vid", "libcamera-vid")
            if exe:
                self.frame_bytes = self.w * self.h * 3 // 2
                self.proc = subprocess.Popen(
                    [exe, "-t", "0", "-n", "--codec", "yuv420",
                     "--width", str(self.w), "--height", str(self.h),
                     "--framerate", "30", "-o", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    bufsize=self.frame_bytes * 2)
                self.source = "Pi Camera Module (" + os.path.basename(exe) + ")"
        else:
            self.source = f"camera {index}"

    def isOpened(self):
        return self.cap is not None or self.proc is not None

    def read(self):
        if self.cap is not None:
            return self.cap.read()
        buf = self.proc.stdout.read(self.frame_bytes)
        if buf is None or len(buf) < self.frame_bytes:
            return False, None
        yuv = np.frombuffer(buf, np.uint8).reshape(self.h * 3 // 2, self.w)
        return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.proc is not None:
            self.proc.terminate()


def crop_16x9(img):
    """Centre-crop any frame to 16:9 so every camera fits the same layout."""
    h, w = img.shape[:2]
    th = w * 9 // 16
    if h > th:
        y = (h - th) // 2
        return img[y:y + th]
    tw = h * 16 // 9
    if w > tw:
        x = (w - tw) // 2
        return img[:, x:x + tw]
    return img


# ----------------------------------------------------------------------------
# Sound effects: tiny WAV files generated at startup. Played with afplay on
# macOS, or aplay / paplay / pw-play on Linux and Raspberry Pi.
# ----------------------------------------------------------------------------
class Sound:
    RATE = 22050

    def __init__(self):
        self.player = first_tool("afplay", "aplay", "paplay", "pw-play")
        self.enabled = self.player is not None
        self.muted = False
        self.dir = os.path.join(tempfile.gettempdir(), "rps_booth_sounds")
        self.paths = {}
        if not self.enabled:
            return
        os.makedirs(self.dir, exist_ok=True)
        self._make("tick",       [(600, 0.07)])
        self._make("shoot",      [(900, 0.16)])
        self._make("win",        [(660, 0.09), (880, 0.09), (1175, 0.16)])
        self._make("lose",       [(440, 0.12), (330, 0.18)])
        self._make("draw",       [(560, 0.08), (0, 0.05), (560, 0.08)])
        self._make("match_win",  [(660, 0.10), (880, 0.10), (1047, 0.10), (1319, 0.26)])
        self._make("match_lose", [(500, 0.14), (400, 0.14), (300, 0.26)])

    def _make(self, name, segments):
        path = os.path.join(self.dir, name + ".wav")
        frames = bytearray()
        for freq, dur in segments:
            n = int(self.RATE * dur)
            for i in range(n):
                if freq <= 0:
                    sample = 0
                else:
                    env = min(1.0, i / 120.0, (n - i) / 120.0)
                    sample = int(12000 * env * math.sin(2 * math.pi * freq * i / self.RATE))
                frames += struct.pack("<h", sample)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.RATE)
            w.writeframes(bytes(frames))
        self.paths[name] = path

    def play(self, name):
        if not self.enabled or self.muted or name not in self.paths:
            return
        try:
            cmd = [self.player]
            if self.player.endswith("aplay"):
                cmd.append("-q")
            subprocess.Popen(cmd + [self.paths[name]],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# What the robot says. Every line is either true of this player or a true fact
# about the IET. Keep lines to 14 words or fewer.
# ----------------------------------------------------------------------------
GREET_LINES = [
    "Hey there. Want to play rock paper scissors?",
    "Hello. Care to take me on at rock paper scissors?",
    "You look like a worthy opponent. Rock paper scissors?",
    "Step up. I bet I can read your mind.",
]
ROUND_LINES = {"human": ["Nice one.", "You got that round."],
               "ai": ["I saw that coming.", "Told you I was watching."],
               "tie": ["Same move. Great minds think alike.", "A draw. We picked the same."]}
MATCH_LINES = {"human": ["You beat me. Well played.", "You win. Impressive."],
               "ai": ["I win this one. Thanks for playing.", "That game is mine."],
               "tie": ["A draw. Nicely matched."]}

# Sources: theiet.org key facts, engx.theiet.org Singapore network pages,
# localevents.theiet.org, and IET PSBA's own event announcements.
IET_FACTS = [
    "The IET has over 157,000 engineering and technology members in 148 countries.",
    "The IET was formed in 2006 when the IEE and the IIE merged.",
    "Worldwide, the IET runs over 1,500 engineering events every year.",
    "IET Singapore runs lectures and technical visits, and most are free to attend.",
    "In 2024, IET Singapore organised 29 events, about two and a half every month.",
    "IET Singapore visits real engineering sites, like rail depots and manufacturing centres.",
    "IET Singapore has a Young Professionals Section for early career engineers.",
    "IET Singapore has a talk on digital rail maintenance on 27 October.",
    "IET PSBA has already held two events this year, with a third coming soon.",
    "On 1 October, IET PSBA visits the Hyundai Motor Group Innovation Centre Singapore.",
    "The 1 October industry visit has only 15 places, so sign up early.",
    "IET PSBA's annual general meeting featured a sharing session on the marine industry.",
    "At our AGM, members learned how AI is used in the marine industry.",
    "Our AGM also included a sharing on prompt complexity scores.",
    "IET PSBA invited four speakers from different engineering fields to meet students.",
    "Our four speakers shared advances in their industries and their work experience.",
]


class FactDeck:
    """Deals IET facts so each one is said at most once per game, and
    consecutive players hear different ones."""

    def __init__(self, facts):
        self.facts = list(facts)
        self.deck = []

    def draw(self, used):
        for _ in range(2):
            if not self.deck:
                self.deck = self.facts[:]
                random.shuffle(self.deck)
            while self.deck:
                f = self.deck.pop()
                if f not in used:
                    used.add(f)
                    return f
        return None


class Brain:
    """Reactive commentary about the current player. Every line must be
    provably true of them. Reset for each new player."""

    def __init__(self, cooldown=3):
        self.recent = collections.deque(maxlen=cooldown)
        self.hist = []
        self.results = []

    def note(self, human_move, result):
        self.hist.append(human_move)
        self.results.append(result)

    def _fresh(self, key, line):
        if key in self.recent:
            return None
        self.recent.append(key)
        return line

    def _run(self):
        n = 0
        for m in reversed(self.hist):
            if m == self.hist[-1]:
                n += 1
            else:
                break
        return n

    def _streak(self, who):
        n = 0
        for r in reversed(self.results):
            if r == who:
                n += 1
            else:
                break
        return n

    def taunt(self, ai, human_wins, ai_wins, round_no):
        """Said as soon as a round begins, so it is finished before the countdown."""
        if round_no == ROUNDS:
            if human_wins >= 2:
                return self._fresh("won_h", "You already have two wins. This last one is for pride.")
            if ai_wins >= 2:
                return self._fresh("won_a", "I already have two wins. Last round for pride.")
            if ai_wins > human_wins:
                return self._fresh("last_a", "Last round, and I am ahead.")
            if human_wins > ai_wins:
                return self._fresh("last_h", "Last round. You are ahead of me.")
            return self._fresh("last_t", "All square. The last round decides it.")
        run = self._run()
        if len(self.hist) >= 2 and run >= 2:
            plural = {"rock": "rocks", "paper": "papers", "scissors": "scissors"}[self.hist[-1]]
            return self._fresh("run", f"That is {run} {plural} in a row. I noticed.")
        move, p, k = ai.read()
        if move is not None and p >= 0.6 and k >= 2:
            return self._fresh("conf", "I have a feeling about this one.")
        if self._streak("tie") >= 2:
            return self._fresh("ties", "We keep landing on the same move.")
        return None

    def round_remark(self, result, human_move, ai_move, expected):
        """Said at the reveal. expected is what the model predicted, if any."""
        if result == "ai" and expected is not None and expected == human_move:
            return f"I expected {human_move}. That is exactly what you played."
        if self._streak("ai") >= 2:
            return f"That is {self._streak('ai')} rounds in a row to me."
        if self._streak("human") >= 2:
            return f"{self._streak('human')} in a row to you. I need to adjust."
        return None


class Chatter:
    """Unprompted small talk, driven by what the camera can actually see."""

    def __init__(self):
        self.last = -1e9
        self.said = collections.deque(maxlen=4)
        self.watching_since = None
        self.played_with_me = False

    def _pick(self, key, line, now, gap=9.0):
        if now - self.last < gap or key in self.said:
            return None
        self.said.append(key)
        self.last = now
        return line

    def idle_line(self, now, face, games, ai_g, human_g, playing):
        if face is None:
            if self.watching_since is not None and self.played_with_me:
                self.watching_since = None
                self.played_with_me = False
                return self._pick("bye", "Thanks for playing. Come back any time.", now)
            self.watching_since = None
            return None
        if self.watching_since is None:
            self.watching_since = now
        if playing:
            self.played_with_me = True
            return None
        watched = now - self.watching_since
        near = face[1] > 0.035
        if watched > 14 and games >= 3:
            pct = int(round(100.0 * human_g / games))
            return self._pick("stats", f"Today I have played {games} games. "
                                       f"Humans won {human_g}, about {pct} percent.",
                              now, gap=14)
        if watched > 9:
            return self._pick("waiting", "You have been watching a while. "
                                         "Are you going to try one?", now, gap=12)
        if near and watched > 4:
            return self._pick("close", "You are close enough. Show me a fist to start.", now)
        return None


class Voice:
    """Text to speech with a short queue so lines never talk over each other.
    macOS 'say', or espeak-ng / espeak / spd-say on Linux and Raspberry Pi."""

    def __init__(self):
        self.tool = first_tool("say", "espeak-ng", "espeak", "spd-say")
        self.enabled = self.tool is not None
        self.muted = False
        self.proc = None
        self.queue = collections.deque()
        self.until = 0.0

    def say(self, text, now=None):
        if not text:
            return
        now = time.time() if now is None else now
        if len(self.queue) >= 2:
            self.queue.popleft()            # never let a backlog build up
        self.queue.append((now, text))

    def busy(self, now):
        talking = self.proc is not None and self.proc.poll() is None
        return talking or now < self.until

    def update(self, now):
        """Start the next queued line if free. Returns the line started."""
        if self.busy(now):
            return None
        while self.queue:
            t, text = self.queue.popleft()
            if now - t > 8.0:
                continue                     # too stale to still be relevant
            self.until = now + 0.6 + 0.36 * len(text.split())
            if self.enabled and not self.muted:
                try:
                    name = os.path.basename(self.tool)
                    if name == "say":
                        cmd = [self.tool, "-r", "180", text]
                    elif name.startswith("espeak"):
                        cmd = [self.tool, "-s", "160", "-v", "en", text]
                    else:
                        cmd = [self.tool, "-w", text]
                    self.proc = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    self.proc = None
            return text
        return None


class FaceWatcher:
    """OpenCV's built-in Haar cascade: tells us when someone walks up."""

    def __init__(self, every=6, scale=0.35):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cc = cv2.CascadeClassifier(path)
        self.ok = not self.cc.empty()
        self.every, self.scale, self.i = every, scale, 0
        self.face = None
        self.box = None             # (x, y, w, h) as fractions of the frame
        self.present = False
        self.arrived = None
        self.last_seen = -1e9
        self.last_greet = -1e9

    def update(self, frame, now):
        self.i += 1
        if not self.ok or (self.i % self.every):
            return self.face
        small = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        found = self.cc.detectMultiScale(gray, 1.2, 5, minSize=(38, 38))
        if len(found):
            x, y, w, h = max(found, key=lambda f: f[2] * f[3])
            H, W = gray.shape[:2]
            self.face = (((x + w / 2) / W) * 2 - 1, (w * h) / float(W * H))
            self.box = (x / W, y / H, w / W, h / H)
            self.last_seen = now
        elif now - self.last_seen > 1.2:
            self.face = None
            self.box = None
        was, self.present = self.present, self.face is not None
        if self.present and not was:
            self.arrived = now
        return self.face

    def should_greet(self, now, cooldown=16.0):
        if not self.present or self.arrived is None:
            return False
        if now - self.arrived > 1.6 or now - self.last_greet < cooldown:
            return False
        self.last_greet = now
        return True


# ----------------------------------------------------------------------------
# Hall of Fame: a photo of everyone who beats the AI.
# ----------------------------------------------------------------------------
HOF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hall_of_fame")
HOF_SIZE = (480, 600)        # saved photo, width x height (4:5 portrait)
HOF_THUMB = (96, 120)
HOF_SHOW = 5                 # winners shown on the start screen


class HallOfFame:
    def __init__(self, folder=HOF_DIR, keep=False):
        self.folder = folder
        self.thumbs = []                 # newest first: (path, image)
        try:
            os.makedirs(folder, exist_ok=True)
            names = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))
        except OSError:
            names = []
        if not keep:                     # a fresh Hall of Fame every session
            for name in names:
                try:
                    os.remove(os.path.join(folder, name))
                except OSError:
                    pass
            names = []
        for name in names[-HOF_SHOW:]:
            img = cv2.imread(os.path.join(folder, name))
            if img is not None:
                self.thumbs.insert(0, (os.path.join(folder, name), self._thumb(img)))
        self.total = len(names)

    @staticmethod
    def _thumb(img):
        return cv2.resize(img, HOF_THUMB, interpolation=cv2.INTER_AREA)

    @staticmethod
    def crop_box(shape, face_box, limit_w):
        """Portrait box (x0, y0, x1, y1) around the face, or the middle of
        the camera area if no face is visible. Stays left of limit_w."""
        H = shape[0]
        W = min(shape[1], limit_w)
        aspect = HOF_SIZE[0] / HOF_SIZE[1]
        if face_box is not None:
            fx, fy, fw, fh = face_box
            cx, cy = (fx + fw / 2) * shape[1], (fy + fh / 2) * H
            bh = min(H, max(fh * H * 3.0, H * 0.55))
            cy += bh * 0.12              # leave room for shoulders
        else:
            cx, cy, bh = W / 2, (H + 60) / 2, H * 0.78
        bw = bh * aspect
        x0 = int(min(max(cx - bw / 2, 0), W - bw))
        y0 = int(min(max(cy - bh / 2, 60), H - bh))
        return x0, y0, int(x0 + bw), int(y0 + bh)

    def add(self, frame, box):
        x0, y0, x1, y1 = box
        photo = cv2.resize(frame[y0:y1, x0:x1], HOF_SIZE, interpolation=cv2.INTER_AREA)
        name = time.strftime("winner_%Y%m%d_%H%M%S.jpg")
        path = os.path.join(self.folder, name)
        try:
            cv2.imwrite(path, photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
        except Exception:
            path = None
        self.thumbs.insert(0, (path, self._thumb(photo)))
        del self.thumbs[HOF_SHOW:]
        self.total += 1
        return photo, path

    def delete_last(self):
        if not self.thumbs:
            return False
        path, _ = self.thumbs.pop(0)
        try:
            if path:
                os.remove(path)
        except OSError:
            pass
        self.total = max(0, self.total - 1)
        return True


# ----------------------------------------------------------------------------
# Photo sharing: a tiny web server on the laptop plus a QR code per photo.
# ----------------------------------------------------------------------------
SHARE_PORT = 8765

SHARE_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hall of Fame</title><style>
body{{margin:0;background:#0A1736;color:#fff;font-family:-apple-system,Helvetica,Arial,sans-serif;
text-align:center;padding:24px 16px}}
h1{{font-weight:500;color:#34D399;font-size:26px;margin:8px 0 4px}}
p{{color:#A0B0BE;margin:6px 0 18px}}
img{{width:100%;max-width:420px;border:3px solid #34D399;border-radius:8px}}
a.b{{display:inline-block;margin-top:18px;padding:12px 22px;border-radius:24px;
background:#8B5CF7;color:#fff;text-decoration:none}}
small{{display:block;color:#A0B0BE;margin-top:14px}}
</style></head><body>
<h1>You beat the AI</h1><p>IET PSB Academy On Campus</p>
<img src="/p/{t}.jpg" alt="Your Hall of Fame photo">
<br><a class="b" href="/p/{t}.jpg" download="hall-of-fame.jpg">Save photo</a>
<small>On iPhone, press and hold the photo, then tap Save to Photos.</small>
</body></html>"""


def lan_ip():
    """This laptop's address on the local Wi-Fi (no data is sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return None if ip.startswith("127.") else ip
    except OSError:
        return None


class PhotoShare:
    def __init__(self, port=SHARE_PORT):
        self.files = {}                   # token -> photo path (this session only)
        self.ip = lan_ip()
        self.port = port
        self.ok = False
        if self.ip is None:
            print("QR codes off: this laptop is not on a network.")
            return
        share = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                m = re.fullmatch(r"/p/([A-Za-z0-9_-]{6,32})(\.jpg)?", self.path)
                path = share.files.get(m.group(1)) if m else None
                if not path or not os.path.exists(path):
                    self._send(404, b"Photo not found", "text/plain")
                elif m.group(2):
                    with open(path, "rb") as f:
                        self._send(200, f.read(), "image/jpeg")
                else:
                    page = SHARE_PAGE.format(t=m.group(1)).encode()
                    self._send(200, page, "text/html; charset=utf-8")

        for p in (port, port + 1, port + 2):
            try:
                self.server = http.server.ThreadingHTTPServer(("0.0.0.0", p), Handler)
                self.port = p
                break
            except OSError:
                self.server = None
        if self.server is None:
            print("QR codes off: could not open a port for the photo server.")
            return
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.ok = True
        print(f"Photo sharing on http://{self.ip}:{self.port} "
              "(phones must be on the same Wi-Fi as this laptop)")

    def add(self, path):
        """Register a photo; returns (url, qr image) or (None, None)."""
        if not self.ok or not path:
            return None, None
        token = secrets.token_urlsafe(8)
        self.files[token] = path
        url = f"http://{self.ip}:{self.port}/p/{token}"
        return url, make_qr(url)

    def close(self):
        if self.ok:
            self.server.shutdown()


def make_qr(text, size=250):
    """Black on white QR code with a quiet zone, as a BGR image."""
    try:
        m = cv2.QRCodeEncoder.create().encode(text)
    except Exception:
        return None
    m = cv2.copyMakeBorder(m, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def paste(img, pic, x, y, border=PURPLE, thick=2):
    """Paste a picture with a thin border, clipped to the image."""
    h, w = pic.shape[:2]
    H, W = img.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return
    img[y:y + h, x:x + w] = pic
    cv2.rectangle(img, (x - 1, y - 1), (x + w, y + h), border, thick, cv2.LINE_AA)


def draw_hof_strip(img, hof):
    """The latest winners along the bottom of the start screen."""
    if hof is None or not hof.thumbs:
        return
    h, w = img.shape[:2]
    tw, th = HOF_THUMB
    gap = 16
    n = len(hof.thumbs)
    x = w // 2 - (n * tw + (n - 1) * gap) // 2
    y = h - 90 - th
    centre(img, "Hall of Fame", y - 14, 18, GREY, "light")
    for _, pic in hof.thumbs:
        paste(img, pic, x, y)
        x += tw + gap


# ----------------------------------------------------------------------------
# The AI: an order 2 Markov predictor over the player's move history.
# ----------------------------------------------------------------------------
class PatternAI:
    def __init__(self, epsilon=0.12):
        self.history = []
        self.tables = [collections.defaultdict(collections.Counter) for _ in range(3)]
        self.epsilon = epsilon
        self.last_reason = "warming up"

    def predict(self):
        if random.random() < self.epsilon:
            self.last_reason = "mixing it up"
            return random.choice(MOVES)
        for order in (2, 1, 0):
            if len(self.history) < order:
                continue
            key = tuple(self.history[-order:]) if order else ()
            counts = self.tables[order].get(key)
            if counts and sum(counts.values()) >= (2 if order else 3):
                guess = counts.most_common(1)[0][0]
                self.last_reason = f"you often play {guess} here"
                return COUNTER[guess]
        self.last_reason = "warming up"
        return random.choice(MOVES)

    def read(self):
        """What the model expects next, and how sure it is: (move, p, samples)."""
        for order in (2, 1, 0):
            if len(self.history) < order:
                continue
            key = tuple(self.history[-order:]) if order else ()
            counts = self.tables[order].get(key)
            if counts:
                tot = sum(counts.values())
                if tot >= (2 if order else 3):
                    move, c = counts.most_common(1)[0]
                    return move, c / tot, tot
        return None, 0.0, 0

    def observe(self, human_move):
        for order in (0, 1, 2):
            if len(self.history) >= order:
                key = tuple(self.history[-order:]) if order else ()
                self.tables[order][key][human_move] += 1
        self.history.append(human_move)

    def reset(self):
        self.__init__(self.epsilon)


# ----------------------------------------------------------------------------
# Gesture recognition
# ----------------------------------------------------------------------------
TIPS = (8, 12, 16, 20)
PIPS = (6, 10, 14, 18)


def classify_hand(lm, aspect=16 / 9):
    """Works at any hand angle (up, sideways, tilted): a finger counts as
    straight when its tip is clearly further from the wrist than its middle
    knuckle. The thumb is ignored because people hold it every which way."""
    def pt(i):
        return (lm[i].x * aspect, lm[i].y)
    wrist = pt(0)
    ratios = []
    for t, p in zip(TIPS, PIPS):
        d_pip = math.dist(wrist, pt(p)) or 1e-6
        ratios.append(math.dist(wrist, pt(t)) / d_pip)
    extended = [r > 1.12 for r in ratios]
    curled = [r < 1.02 for r in ratios]
    index, middle, ring, pinky = extended
    n = sum(extended)
    if index and middle and curled[3] and not (ring and pinky):
        return "scissors"
    if n >= 3:
        return "paper"
    if n <= 1 and sum(curled) >= 3:
        return "rock"
    return None


def hand_size(lm):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def majority(moves, need):
    """Most common real move if it appears at least 'need' times."""
    real = [m for m in moves if m]
    if not real:
        return None
    best = max(set(real), key=real.count)
    return best if real.count(best) >= need else None


def outcome(human, ai):
    if human == ai:
        return "tie"
    return "human" if BEATS[human] == ai else "ai"


# ----------------------------------------------------------------------------
# Screen furniture
# ----------------------------------------------------------------------------
def draw_rounds(img, round_results, round_no, show_current=True, y=96):
    """Three small dots, one per round, filled with who won it."""
    cx = img.shape[1] // 2
    for i in range(ROUNDS):
        x = cx + (i - 1) * 30
        if i < len(round_results):
            col = {"human": WIN_C, "ai": LOSE_C, "tie": TIE_C}[round_results[i]]
            cv2.circle(img, (x, y), 7, col, -1, cv2.LINE_AA)
        else:
            ring = WHITE if show_current and i == len(round_results) else EMPTY
            cv2.circle(img, (x, y), 7, ring, 2, cv2.LINE_AA)


def draw_chrome(img, ai_g, human_g, muted):
    """One slim top bar: club name on the left, games won on the right."""
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 56), BLUE_DARK, -1)
    cv2.line(img, (0, 56), (w, 56), PURPLE, 1, cv2.LINE_AA)
    draw_text(img, CLUB_NAME, 28, 36, 20, WHITE, "light", shadow=False)
    score = f"AI {ai_g}     You {human_g}" + ("     muted" if muted else "")
    draw_text(img, score, w - 28, 36, 20, GREY, "light", "right", shadow=False)


STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats")
STATS_FILE = os.path.join(STATS_DIR, "games.csv")
STATS_COLS = ["date", "time", "result", "player_rounds", "ai_rounds", "draws",
              "player_moves", "ai_moves", "hall_of_fame"]


class Stats:
    """One row per finished game in stats/games.csv."""

    def __init__(self, path=STATS_FILE):
        self.path = path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(STATS_COLS)
        except OSError as e:
            print("Stats file not available:", e)

    def today(self):
        """(games, human wins, ai wins) already logged today."""
        games = human = ai = 0
        day = time.strftime("%Y-%m-%d")
        try:
            with open(self.path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("date") != day:
                        continue
                    games += 1
                    human += row.get("result") == "player"
                    ai += row.get("result") == "ai"
        except OSError:
            pass
        return games, human, ai

    def log(self, result, rounds, player_moves, ai_moves, photo):
        row = [time.strftime("%Y-%m-%d"), time.strftime("%H:%M:%S"),
               {"human": "player", "ai": "ai", "tie": "draw"}[result],
               rounds.count("human"), rounds.count("ai"), rounds.count("tie"),
               " ".join(player_moves), " ".join(ai_moves), "yes" if photo else "no"]
        try:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except OSError:
            pass


def keep_awake():
    """Stop the computer sleeping or dimming while the booth runs."""
    try:
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            # -d display, -i idle, -s system; -w ends it when this program ends
            return subprocess.Popen(["caffeinate", "-dis", "-w", str(os.getpid())],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.environ.get("DISPLAY") and shutil.which("xset"):
            subprocess.run(["xset", "s", "off", "-dpms"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=3)
    except Exception:
        pass
    return None


def draw_attract(img, t, human_g=0, ai_g=0):
    h = img.shape[0]
    dim(img, 0.68)
    col = blend(PURPLE, GREEN, 0.5 + 0.5 * math.sin(t * 1.2))
    if human_g + ai_g > 0:
        # today's score, Humans in green and the AI in purple
        a, b, gap = f"Humans {human_g}", f"AI {ai_g}", 70
        wa, wb = text_width(a, 44, "medium"), text_width(b, 44, "medium")
        x = img.shape[1] // 2 - (wa + gap + wb) // 2
        y = h // 2 - 136
        draw_text(img, a, x, y, 44, GREEN, "medium")
        draw_text(img, b, x + wa + gap, y, 44, PURPLE, "medium")
        centre(img, "today", y + 32, 18, GREY, "light")
    centre(img, "Rock Paper Scissors", h // 2, 64, col, "medium")
    centre(img, "Show your hand to play", h // 2 + 60, 26, WHITE, "light")


# ----------------------------------------------------------------------------
def main():
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    plain = [a for a in sys.argv[1:] if not a.startswith("-")]
    cam = int(plain[0]) if plain else 0
    want_robot = "--robot" in flags
    want_llm = "--llm" in flags
    want_hof = "--no-hof" not in flags
    want_qr = "--no-qr" not in flags
    on_pi = "--pi" in flags or is_raspberry_pi()
    model = None
    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        if i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            plain = [a for a in plain if a != model]
            cam = int(plain[0]) if plain else 0

    # Performance profile. The Pi processes a smaller frame, renders the robot
    # at 1x every other frame, checks for faces less often, and uses the 1B
    # model with a longer wait before falling back.
    if on_pi:
        P = dict(cam_size=(640, 360), proc_w=640, robot_ss=1, robot_every=2,
                 face_every=10, model=model or "gemma3:1b", llm_deadline=4.0,
                 hand_model=0)
        print("Raspberry Pi profile on.")
    else:
        P = dict(cam_size=(1280, 720), proc_w=640, robot_ss=2, robot_every=1,
                 face_every=6, model=model or "gemma3:4b", llm_deadline=2.5,
                 hand_model=1)
    if want_robot and robot3d is None:
        print("robot3d.py not found, running without the robot panel.")
        want_robot = False

    cap = Camera(cam, P["cam_size"], force_picam="--picam" in flags)
    if not cap.isOpened():
        raise SystemExit("Could not open a camera. Is another app using it? "
                         "On a Pi with a Camera Module, try --picam.")
    print("Using", cap.source)

    # Two hands are tracked so a bystander's hand in the background can be
    # ignored: the player's hand is the biggest (closest) one.
    hands = mp.solutions.hands.Hands(
        max_num_hands=2, model_complexity=P["hand_model"],
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawer = mp.solutions.drawing_utils
    dot_style = drawer.DrawingSpec(color=GREEN, thickness=-1, circle_radius=3)
    line_style = drawer.DrawingSpec(color=PURPLE, thickness=2)

    llmv = None
    if want_llm:
        if llm_mod is None:
            print("llm.py not found, running with the built-in lines.")
        else:
            llmv = llm_mod.LLMVoice(model=P["model"], timeout=P["llm_deadline"] + 1.0)
            ok, info = llmv.check()
            print("LLM:", f"connected to Ollama ({P['model']})" if ok else
                  f"not available ({info}), using the built-in lines")
            if ok:
                llmv.warm_up()          # load the model now, not on the first line
            else:
                llmv = None

    if get_font(20) is None:
        print("Note: no system font found, using OpenCV's built-in font.")

    snd = Sound()
    voice = Voice()
    brain = Brain()
    chat = Chatter()
    faces = FaceWatcher(every=P["face_every"],
                        scale=448.0 / (P["proc_w"] or P["cam_size"][0]))
    deck = FactDeck(IET_FACTS)
    hof = HallOfFame(keep="--keep-hof" in flags) if want_hof else None
    share = PhotoShare() if (hof is not None and want_qr) else None
    hof_qr = None
    hof_photo = None           # the winner's photo, shown after the snap
    hof_box = None
    snap_at = None             # when the Hall of Fame photo is taken
    facts_used = set()
    ai = PatternAI()

    says, says_until = "", 0.0
    fact_due = None            # (time, fact): an IET fact waiting its turn
    greet_until = 0.0
    expected = None
    look = 0.0

    ai_games = human_games = games_played = 0
    stats = Stats()
    games_played, human_games, ai_games = stats.today()
    if games_played:
        print(f"Today so far: {games_played} games, Humans {human_games}, AI {ai_games}")
    awake = keep_awake()
    game_ai_moves = []
    human_wins = ai_wins = 0
    round_no = 1
    round_results = []

    state = "attract"
    t0 = 0.0
    wait_from = 0.0
    missed_until = 0.0
    last_tick = -1
    last_hand_seen = time.time()
    stable = collections.deque(maxlen=STABLE_FRAMES)
    recent = collections.deque(maxlen=7)      # raw guesses, for smoothing
    votes = collections.deque(maxlen=40)      # (time, move) around the throw
    gamma_lut, light_check = None, 0
    locked_human = locked_ai = result = None
    reason = ""
    match_result = None
    fullscreen = False
    panel = None
    panel_img = None
    panel_state = None
    frame_i = 0
    show_robot = want_robot

    win = "RPS vs AI"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def speak(line, rephrase=True):
        """Hand a true line to the LLM to rephrase, or straight to the voice.
        IET facts are spoken word for word (rephrase=False) so no detail is
        lost or garbled."""
        if not line:
            return
        if rephrase and llmv is not None and not llmv.pending:
            llmv.request(line, line, time.time(), deadline=P["llm_deadline"])
        else:
            voice.say(line)

    def new_game():
        nonlocal human_wins, ai_wins, round_no, round_results, brain, facts_used
        nonlocal game_ai_moves
        game_ai_moves = []
        human_wins = ai_wins = 0
        round_no = 1
        round_results = []
        ai.reset()
        brain = Brain()            # new player, new memory
        facts_used = set()

    def hand_held_steady(move):
        """True once the same gesture fills most of the last few frames."""
        stable.append(move)
        return len(stable) == stable.maxlen and \
            majority(stable, int(stable.maxlen * 0.8)) is not None

    def talking(now):
        return voice.busy(now) or bool(voice.queue) or bool(llmv and llmv.pending)

    def begin_round(now):
        """Enter the gap before a round and say this round's line straight
        away, so it is finished before Rock Paper Scissors starts."""
        nonlocal state, wait_from
        state, wait_from = "waiting", now
        stable.clear()
        if round_no == 2:
            speak(deck.draw(facts_used), rephrase=False)     # IET fact, mid game
        else:
            speak(brain.taunt(ai, human_wins, ai_wins, round_no))

    def start_countdown(now):
        nonlocal expected, locked_ai, reason, state, t0, last_tick
        voice.queue.clear()               # nothing new starts mid countdown
        if llmv:
            llmv.cancel()
        expected = ai.read()[0]
        locked_ai = ai.predict()          # the AI commits BEFORE you throw
        reason = ai.last_reason
        state, t0, last_tick = "counting", now, -1
        stable.clear()

    # Auto speed: if the frame rate drops, draw the robot more cheaply.
    # Each level is (robot supersampling, redraw the robot every N frames).
    SPEED_LEVELS = [(2, 1), (2, 2), (1, 2), (1, 3)]
    speed = SPEED_LEVELS.index((P["robot_ss"], P["robot_every"])) \
        if (P["robot_ss"], P["robot_every"]) in SPEED_LEVELS else 0
    fps_t, fps_n, fps_report = time.perf_counter(), 0, time.perf_counter()

    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame_i += 1
        fps_n += 1
        if fps_n == 45:
            tnow = time.perf_counter()
            fps = fps_n / max(1e-6, tnow - fps_t)
            fps_t, fps_n = tnow, 0
            if fps < 18 and speed < len(SPEED_LEVELS) - 1:
                speed += 1
                P["robot_ss"], P["robot_every"] = SPEED_LEVELS[speed]
                if panel is not None:
                    panel.ss = P["robot_ss"]
                print(f"Running at {fps:.0f} fps, drawing the robot more cheaply "
                      f"(level {speed}).")
            if tnow - fps_report > 15:
                fps_report = tnow
                print(f"{fps:.0f} fps")
        work = crop_16x9(cv2.flip(raw, 1))
        # hand and face detection run on a small frame; the UI is drawn at 720p
        pw = P["proc_w"]
        if pw and work.shape[1] != pw:
            proc = cv2.resize(work, (pw, pw * 9 // 16), interpolation=cv2.INTER_AREA)
        else:
            proc = work
        frame = work if work.shape[:2] == (720, 1280) else \
            cv2.resize(work, (1280, 720), interpolation=cv2.INTER_LINEAR)
        # an untouched copy for the Hall of Fame photo (only when it is needed)
        clean = frame.copy() if (state == "gameover" and snap_at is not None) else None
        # dim room: brighten the copy the hand tracker sees
        light_check += 1
        if light_check % 15 == 1:
            level = float(cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY).mean())
            if level < 95:
                g = 0.55 if level < 60 else 0.72
                gamma_lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)],
                                     np.uint8)
            else:
                gamma_lut = None
        seen_img = cv2.LUT(proc, gamma_lut) if gamma_lut is not None else proc
        res = hands.process(cv2.cvtColor(seen_img, cv2.COLOR_BGR2RGB))

        raw_move, hand_at = None, None
        if res.multi_hand_landmarks:
            hl = max(res.multi_hand_landmarks, key=lambda h: hand_size(h.landmark))
            drawer.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS,
                                  dot_style, line_style)
            raw_move = classify_hand(hl.landmark)
            hand_at = (int(hl.landmark[0].x * frame.shape[1]),
                       int(hl.landmark[0].y * frame.shape[0]))
            last_hand_seen = time.time()
        # smooth out one-frame flickers: a gesture needs 3 of the last 7 frames
        recent.append(raw_move)
        move_now = majority(recent, 3) if raw_move is not None or hand_at else None
        votes.append((time.time(), raw_move))

        now = time.time()
        show_rounds = False

        # speech plumbing
        if llmv:
            ready = llmv.poll(now)
            if ready:
                voice.say(ready, now)
        if fact_due and now >= fact_due[0] and not (llmv and llmv.pending) \
                and not voice.queue:
            speak(fact_due[1], rephrase=False)
            fact_due = None
        started = voice.update(now)
        if started:
            says, says_until = started, voice.until + 0.8

        seen = faces.update(proc, now)
        look = seen[0] if seen else 0.0
        if state in ("attract", "waiting") and not voice.busy(now) and not voice.queue:
            small = chat.idle_line(now, seen, games_played, ai_games,
                                   human_games, state != "attract")
            if small:
                speak(small)
        if state == "attract" and faces.should_greet(now):
            speak(random.choice(GREET_LINES))
            greet_until = now + 3.4

        panel_on = show_robot and robot3d is not None
        stage = frame[:, :frame.shape[1] - ROBOT_PANEL_W] if panel_on else frame
        mid = stage.shape[0] // 2

        # ------------------------------------------------------------------
        if state == "attract":
            draw_attract(stage, now, human_games, ai_games)
            draw_hof_strip(stage, hof)
            if hand_held_steady(move_now):
                new_game()
                begin_round(now)

        # ------------------------------------------------------------------
        elif state == "waiting":
            show_rounds = True
            # the countdown only starts once the robot has finished talking
            ready = now - wait_from >= READY_GAP and \
                (not talking(now) or now - wait_from > TALK_WAIT_MAX)
            if now < missed_until:
                line = "Try again"
            elif not ready:
                line = f"Round {round_no}"
            else:
                line = "Show your hand"
            centre(stage, line, mid + 20, 40, WHITE, "light")
            if not ready:
                stable.clear()
            elif hand_held_steady(move_now):
                start_countdown(now)
            if now - last_hand_seen > IDLE_SECS:
                state = "attract"
                stable.clear()

        # ------------------------------------------------------------------
        elif state == "counting":
            show_rounds = True
            elapsed = now - t0
            words = ["Rock", "Paper", "Scissors", "Shoot"]
            idx = min(int(elapsed / STEP), 3)
            if idx != last_tick:
                snd.play("shoot" if idx == 3 else "tick")
                last_tick = idx
            word_col = GREEN if idx == 3 else PURPLE
            centre(stage, words[idx], mid + 30, 112, word_col, "medium")
            # read the throw from the frames just after Shoot, and give a
            # short grace period if the hand was not seen yet (the AI's move
            # is already locked, so waiting a moment cannot help it cheat)
            thrown = None
            if elapsed >= CAPTURE_AT:
                window = [m for t, m in votes if now - t <= 0.35]
                thrown = majority(window, 2)
            if elapsed >= CAPTURE_AT and thrown is None:
                if elapsed >= CAPTURE_AT + 0.8:
                    state, wait_from, missed_until = "waiting", now, now + 2.5
                    stable.clear()
            elif thrown is not None:
                locked_human = thrown
                result = outcome(locked_human, locked_ai)
                ai.observe(locked_human)
                round_results.append(result)
                game_ai_moves.append(locked_ai)
                if result == "ai":
                    ai_wins += 1
                    snd.play("lose")
                elif result == "human":
                    human_wins += 1
                    snd.play("win")
                else:
                    snd.play("draw")
                brain.note(locked_human, result)
                speak(brain.round_remark(result, locked_human, locked_ai, expected)
                      or random.choice(ROUND_LINES[result]))
                if panel is not None:
                    panel.set_pose(locked_ai, instant=True)
                state, t0 = "reveal", now

        # ------------------------------------------------------------------
        elif state == "reveal":
            show_rounds = True
            texts = {"human": ("You win", WIN_C),
                     "ai": ("AI wins", LOSE_C),
                     "tie": ("Draw", TIE_C)}
            big, color = texts[result]
            centre(stage, big, mid + 10, 72, color, "medium")
            centre(stage, f"AI played {locked_ai}", mid + 62, 26, GREY, "light")
            if now - t0 > REVEAL_HOLD:
                if round_no >= ROUNDS:
                    if human_wins > ai_wins:
                        match_result = "human"
                        human_games += 1
                        snd.play("match_win")
                    elif ai_wins > human_wins:
                        match_result = "ai"
                        ai_games += 1
                        snd.play("match_lose")
                    else:
                        match_result = "tie"
                        snd.play("draw")
                    games_played += 1
                    stats.log(match_result, round_results, brain.hist, game_ai_moves,
                              match_result == "human" and hof is not None)
                    fact = deck.draw(facts_used)
                    if match_result == "human" and hof is not None:
                        speak("You beat me! Smile for the Hall of Fame.", rephrase=False)
                        snap_at, hof_photo = now + HOF_RESULT + 3 * HOF_STEP, None
                        hold = HOF_QR_HOLD if (share and share.ok) else HOF_HOLD
                        fact_due = (snap_at + hold + 0.3, fact)     # after the photo
                    else:
                        speak(random.choice(MATCH_LINES[match_result]))
                        snap_at = None
                        fact_due = (now + 0.5, fact)                # IET fact, after the result
                    state, t0 = "gameover", now
                    stable.clear()
                else:
                    round_no += 1
                    begin_round(now)

        # ------------------------------------------------------------------
        elif state == "gameover":
            texts = {"human": ("You beat the AI", WIN_C),
                     "ai": ("The AI wins", LOSE_C),
                     "tie": ("It is a draw", TIE_C)}
            big, color = texts[match_result]
            if snap_at is None or now - t0 < HOF_RESULT:
                dim(stage, 0.5)
                centre(stage, big, mid, 72, color, "medium")
                centre(stage, f"{human_wins}  to  {ai_wins}", mid + 56, 30, WHITE, "light")
                draw_rounds(stage, round_results, ROUNDS, show_current=False, y=mid - 100)
                if snap_at is None and now - t0 > GAMEOVER_HOLD:
                    state = "attract"
                    stable.clear()
            elif now < snap_at:
                # 3 2 1 countdown with a frame showing what will be in the photo
                hof_box = HallOfFame.crop_box(frame.shape, faces.box, stage.shape[1])
                x0, y0, x1, y1 = hof_box
                dim(stage, 0.35)
                stage[y0:y1, x0:x1] = clean[y0:y1, x0:x1]
                cv2.rectangle(stage, (x0, y0), (x1, y1), GREEN, 2, cv2.LINE_AA)
                n = int(math.ceil(snap_at - now))
                cx = (x0 + x1) // 2
                draw_text(stage, "Smile", cx, y0 + 44, 30, WHITE, "light", "center")
                centre(stage, str(max(1, n)), mid + 40, 140, GREEN, "medium")
            else:
                if hof_photo is None:
                    box = hof_box or HallOfFame.crop_box(frame.shape, faces.box, stage.shape[1])
                    hof_photo, photo_path = hof.add(clean, box)
                    hof_qr = share.add(photo_path)[1] if share else None
                    snd.play("shoot")
                    if hof_qr is not None:
                        speak("Scan the code to get your photo.", rephrase=False)
                if now - snap_at < 0.18:
                    stage[:] = 255                      # camera flash
                else:
                    dim(stage, 0.7)
                    SW, SH = stage.shape[1], stage.shape[0]
                    if hof_qr is None:
                        ph = SH - 250
                        pw = ph * HOF_SIZE[0] // HOF_SIZE[1]
                        pic = cv2.resize(hof_photo, (pw, ph), interpolation=cv2.INTER_AREA)
                        paste(stage, pic, SW // 2 - pw // 2, 110, GREEN, 3)
                    else:
                        ph = SH - 280
                        pw = ph * HOF_SIZE[0] // HOF_SIZE[1]
                        q = hof_qr.shape[0]
                        gap = 50
                        x = SW // 2 - (pw + gap + q) // 2
                        pic = cv2.resize(hof_photo, (pw, ph), interpolation=cv2.INTER_AREA)
                        paste(stage, pic, x, 100, GREEN, 3)
                        qx, qy = x + pw + gap, 100 + (ph - q) // 2 - 16
                        paste(stage, hof_qr, qx, qy, WHITE, 1)
                        draw_text(stage, "Scan for your photo", qx + q // 2, qy + q + 34,
                                  22, WHITE, "light", "center")
                    centre(stage, "Welcome to the Hall of Fame", SH - 80,
                           32, GREEN, "medium")
                hold = HOF_HOLD if hof_qr is None else HOF_QR_HOLD
                if now - snap_at > hold:
                    state, snap_at = "attract", None
                    stable.clear()

        if hand_at and move_now and state in ("waiting", "counting", "attract"):
            draw_text(frame, move_now, hand_at[0], min(hand_at[1] + 34, frame.shape[0] - 10),
                      22, GREEN, "medium", "center")

        # ---- compose the canvas ------------------------------------------
        if panel_on:
            if panel is None or panel.h != frame.shape[0]:
                panel = robot3d.RobotPanel(ROBOT_PANEL_W, frame.shape[0])
                panel.ss = P["robot_ss"]
                panel.set_pose("rock", instant=True)
            if now < greet_until and state in ("attract", "waiting"):
                r_state, r_el = "greet", 0.0
            elif state == "counting":
                r_state, r_el = "count", now - t0
            elif state in ("reveal", "gameover"):
                r_state, r_el = "reveal", 0.0
            else:
                r_state, r_el = "idle", 0.0
            # on the Pi the robot redraws every other frame unless it changes pose
            if (panel_img is None or r_state != panel_state
                    or frame_i % P["robot_every"] == 0):
                panel_img = panel.draw(now, r_state, r_el, "", look=look)
                panel_state = r_state
            canvas = frame.copy()
            canvas[:, -ROBOT_PANEL_W:] = panel_img
            cv2.line(canvas, (frame.shape[1] - ROBOT_PANEL_W, 0),
                     (frame.shape[1] - ROBOT_PANEL_W, frame.shape[0]), BLUE_LINE, 2)
        else:
            canvas = frame

        cam_area = canvas[:, :canvas.shape[1] - ROBOT_PANEL_W] if panel_on else canvas
        if show_rounds:
            draw_rounds(cam_area, round_results, round_no)
        photo_up = state == "gameover" and snap_at is not None and now >= snap_at
        if now < says_until and says and state != "counting" and not photo_up:
            centre_wrapped(cam_area, says, cam_area.shape[0] - 36, 22, GREEN, "light",
                           margin=80, bottom=True)
        draw_chrome(canvas, ai_games, human_games, snd.muted)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            fact_due = None
            ai_games = human_games = games_played = 0
            new_game()
            state = "attract"
            stable.clear()
        if key == ord('m'):
            snd.muted = not snd.muted
            voice.muted = snd.muted
        if key == ord('x') and hof is not None and hof.delete_last():
            print("Deleted the newest Hall of Fame photo.")
        if key == ord('b') and robot3d is not None:
            show_robot = not show_robot
        if key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)

    if llmv:
        print(f"LLM lines used: {llmv.used_llm}, fallbacks: {llmv.used_fallback}")
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal tally: AI {ai_games}, Humans {human_games}, over {games_played} games.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nStopped. The Hall of Fame will start fresh next time.")
