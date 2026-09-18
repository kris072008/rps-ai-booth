"""
Rock-Paper-Scissors vs AI  -  IET PSB Academy On Campus  -  Campus Club Fair 2026
Run:  python rps_booth.py
Keys: Q or ESC = quit,  R = reset scoreboard,  F = toggle fullscreen
"""

import collections
import math
import random
import time

import cv2
import mediapipe as mp

CLUB_NAME = "IET PSB Academy On Campus"
EVENT_NAME = "Campus Club Fair 2026"
NAVY = (78, 42, 26)
ACCENT = (60, 190, 255)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
WIN_C = (120, 220, 130)
LOSE_C = (110, 110, 240)
TIE_C = (200, 200, 200)

MOVES = ("rock", "paper", "scissors")
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
COUNTER = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
EMOJI = {"rock": "ROCK", "paper": "PAPER", "scissors": "SCISSORS"}


class PatternAI:
    """Order-2 Markov predictor over the player's move history."""

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
                self.last_reason = "you often play " + guess + " here"
                return COUNTER[guess]
        self.last_reason = "warming up"
        return random.choice(MOVES)

    def observe(self, human_move):
        for order in (0, 1, 2):
            if len(self.history) >= order:
                key = tuple(self.history[-order:]) if order else ()
                self.tables[order][key][human_move] += 1
        self.history.append(human_move)

    def reset(self):
        self.__init__(self.epsilon)


TIPS = (8, 12, 16, 20)
PIPS = (6, 10, 14, 18)


def classify_hand(lm):
    span = math.dist((lm[0].x, lm[0].y), (lm[9].x, lm[9].y)) or 1e-6
    margin = span * 0.15
    extended = []
    for tip, pip in zip(TIPS, PIPS):
        extended.append((lm[pip].y - lm[tip].y) > margin)
    n = sum(extended)
    index, middle, ring, pinky = extended
    if n == 0:
        return "rock"
    if index and middle and not ring and not pinky:
        return "scissors"
    if n >= 3:
        return "paper"
    return None


def outcome(human, ai):
    if human == ai:
        return "tie"
    return "human" if BEATS[human] == ai else "ai"


def banner(img, text, y, scale, color, thick=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    x = (img.shape[1] - w) // 2
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_chrome(img, ai_score, human_score, rounds, status, status_color, hint):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 78), NAVY, -1)
    cv2.rectangle(img, (0, 74), (w, 78), ACCENT, -1)
    cv2.putText(img, CLUB_NAME, (24, 34), cv2.FONT_HERSHEY_DUPLEX, 0.78, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, EVENT_NAME + "  |  Beat the AI", (24, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, ACCENT, 1, cv2.LINE_AA)
    score = "AI " + str(ai_score) + "   -   HUMANS " + str(human_score)
    (sw, _), _ = cv2.getTextSize(score, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    cv2.putText(img, score, (w - sw - 24, 48), cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, str(rounds) + " rounds played", (w - sw - 24, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREY, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, h - 92), (w, h), NAVY, -1)
    cv2.rectangle(img, (0, h - 92), (w, h - 88), ACCENT, -1)
    banner(img, status, h - 46, 1.15, status_color, 2)
    if hint:
        banner(img, hint, h - 16, 0.55, GREY, 1)


def main():
    import sys; cap = cv2.VideoCapture(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Is another app using it?")

    hands = mp.solutions.hands.Hands(
        max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.6, min_tracking_confidence=0.5)
    drawer = mp.solutions.drawing_utils

    ai = PatternAI()
    ai_score = human_score = rounds = 0
    state = "waiting"
    t0 = 0.0
    stable = collections.deque(maxlen=6)
    locked_human = locked_ai = result = None
    reason = ""
    fullscreen = False

    win = "RPS vs AI"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        move_now = None
        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            drawer.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)
            move_now = classify_hand(hl.landmark)

        now = time.time()
        status, color, hint = "", WHITE, ""

        if state == "waiting":
            stable.append(move_now)
            if len(stable) == stable.maxlen and stable.count(stable[0]) == len(stable) \
                    and stable[0] is not None:
                state, t0 = "counting", now
                stable.clear()
            status = "SHOW YOUR HAND TO START"
            hint = "Rock = fist   Paper = open palm   Scissors = two fingers"

        elif state == "counting":
            elapsed = now - t0
            words = ["ROCK...", "PAPER...", "SCISSORS...", "SHOOT!"]
            idx = min(int(elapsed / 0.6), 3)
            status, color = words[idx], ACCENT
            hint = "Lock in your throw"
            banner(frame, words[idx], frame.shape[0] // 2, 3.2, ACCENT, 5)
            if elapsed >= 2.6:
                if move_now is None:
                    state = "waiting"
                else:
                    locked_human = move_now
                    locked_ai = ai.predict()
                    reason = ai.last_reason
                    result = outcome(locked_human, locked_ai)
                    ai.observe(locked_human)
                    rounds += 1
                    if result == "ai":
                        ai_score += 1
                    elif result == "human":
                        human_score += 1
                    state, t0 = "reveal", now

        elif state == "reveal":
            texts = {"human": ("YOU WIN!", WIN_C),
                     "ai": ("AI WINS", LOSE_C),
                     "tie": ("DRAW", TIE_C)}
            status, color = texts[result]
            hint = ("You: " + EMOJI[locked_human] + "   |   AI: " + EMOJI[locked_ai]
                    + "   |   AI thought: " + reason)
            banner(frame, "AI PLAYED " + EMOJI[locked_ai], frame.shape[0] // 2 - 40, 1.8, WHITE, 3)
            banner(frame, status, frame.shape[0] // 2 + 50, 2.6, color, 4)
            if now - t0 > 2.8:
                state = "waiting"
                stable.clear()

        draw_chrome(frame, ai_score, human_score, rounds, status, color, hint)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            ai.reset()
            ai_score = human_score = rounds = 0
            state = "waiting"
        if key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                win, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
    print("Final tally - AI", ai_score, "Humans", human_score, "over", rounds, "rounds.")


if __name__ == "__main__":
    main()
