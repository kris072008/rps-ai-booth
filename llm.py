"""
Optional local-LLM voice for the booth robot, via Ollama.

Design rule: the LLM NEVER invents facts. rps_booth.py works out what is
actually true about the player (three rocks in a row, match point, its own win
rate) and this module only rephrases that fact in the robot's voice. If Ollama
is slow, missing, or returns something odd, the caller falls back to the plain
template line - so the worst case is exactly the behaviour without an LLM.

Runs entirely offline once the model is pulled. Uses only the standard library.

Setup on a Mac:
    brew install ollama && brew services start ollama
    ollama pull gemma3:4b

Setup on a Raspberry Pi (64-bit OS, Pi 4 or 5 with 4GB+):
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull gemma3:1b

Then:  python rps_booth.py --robot --llm      (the Pi picks the 1b model)

Why Gemma 3: Google's small Gemma models write noticeably cleaner, more
natural English than Llama 3.2 at the same size, which matters when the line
is read aloud by a speech synthesiser. Try any other model with --model NAME.

Test the lines on their own:  python llm.py
"""

import json
import os
import re
import threading
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")   # gemma3:1b on a Pi

PERSONA = (
    "You are a robot that runs a rock paper scissors booth at a university "
    "campus club fair. The organisation you are helping is the IET, the "
    "Institution of Engineering and Technology. Your task is to be playful with "
    "the participant and brief with your explanations. "
    "You will be given a FACT that is true. Rephrase it as ONE spoken line. "
    "The spoken line can be at most 14 words. Do not invent any details that "
    "are not in the FACT. No emoji, no quotation marks, no stage directions. "
    "Just the line. "
    "Write in simple, correct English that is easy to understand when heard "
    "out loud: complete sentences, everyday words, a capital letter at the "
    "start and a full stop, question mark or exclamation mark at the end. "
    "Talk directly to the player as you, and about yourself as I."
)

# A few worked examples. Small models copy the style of examples far more
# reliably than they follow rules, so these do most of the work on grammar.
EXAMPLES = [
    ("That is 3 rocks in a row. I noticed.",
     "Three rocks in a row? I noticed that."),
    ("I expected paper. That is exactly what you played.",
     "I guessed paper, and that is exactly what you played!"),
    ("2 in a row to you. I need to adjust.",
     "Two in a row to you. I need to change my plan."),
    ("Last round, and I am ahead.",
     "This is the last round, and I am in the lead."),
    ("Hello there. Want to play rock paper scissors?",
     "Hi there! Do you want to play rock paper scissors with me?"),
]

# Anything the model generates is checked before it reaches a speaker in front
# of students and lecturers. A hit on any of these means the line is dropped
# and the original, pre-written fact is spoken instead.
_BANNED = re.compile(
    r"\b(fuck\w*|shit\w*|bitch\w*|bastard\w*|damn\w*|crap\w*|piss\w*|"
    r"dick\w*|cock\w*|pussy|ass|asses|asshole\w*|arse\w*|bollock\w*|bloody|"
    r"wank\w*|twat\w*|cunt\w*|slut\w*|whore\w*|sex\w*|porn\w*|nude\w*|"
    r"idiot\w*|stupid|dumb\w*|moron\w*|loser\w*|suck\w*|pathetic|ugly|fat|"
    r"hell|kill\w*|die|dies|dead|death|murder\w*|hate\w*|gun\w*|weapon\w*|"
    r"drunk|drug\w*|beer|alcohol)\b",
    re.I)

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text):
    return {n.replace(",", "") for n in _NUM.findall(text or "")}


def _clean(text, fact="", truncated=False):
    """Return a safe one-liner, or None if the reply is unusable."""
    if not text:
        return None
    line = text.strip().split("\n")[0].strip()
    line = line.replace("**", "").replace("__", "")
    line = re.sub(r"^(your line|line|robot|answer|spoken line)\s*:\s*", "", line, flags=re.I)
    # normalise punctuation, drop quotes, dashes, pipes and anything non-ASCII
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", ""), ("\u201d", ""),
                 ("\u2014", ", "), ("\u2013", ", "), (" - ", ", "), ("|", " ")):
        line = line.replace(a, b)
    line = line.replace('"', "")
    line = "".join(c for c in line if 32 <= ord(c) < 127)
    line = re.sub(r"\s+", " ", line).strip(" ,")
    if not line:
        return None
    words = line.split()
    if len(words) < 2 or len(words) > 14 or len(line) > 110:
        return None
    # grammar and readability checks, so the speaker never reads out a fragment
    if line[-1] not in ".!?":
        if truncated:
            return None                  # cut off mid-sentence
        line += "."
    line = line[0].upper() + line[1:]
    if len(re.findall(r"[.!?](\s|$)", line)) > 2:     # at most two short sentences
        return None
    low = [re.sub(r"\W", "", w.lower()) for w in words]
    if any(a == b and a for a, b in zip(low, low[1:])):  # "the the"
        return None
    for w in ("fact", "player"):                        # echoing the prompt
        if re.search(rf"\b{w}\b", line, re.I) and not re.search(rf"\b{w}\b", fact, re.I):
            return None
    if re.search(r"[*\[\]()<>#@]", line):          # stage directions, markup
        return None
    if _BANNED.search(line):
        return None
    if line.lower().startswith(("as an ai", "i cannot", "i can't", "sure", "here",
                                "fact:", "line:", "okay", "ok,")):
        return None
    if not _numbers(line) <= _numbers(fact):         # no invented numbers
        return None
    return line


class LLMVoice:
    """Asks Ollama for a line in the background; never blocks the game loop.

    Usage per event:
        llm.request(fact, fallback, now, deadline=1.2)
        ...each frame...
        line = llm.poll(now)      # the LLM line, or the fallback once time is up
        if line: voice.say(line)
    """

    def __init__(self, model=MODEL, host=HOST, timeout=3.0):
        self.model, self.host, self.timeout = model, host, timeout
        self.lock = threading.Lock()
        self.result = None          # text from the model
        self.pending = False
        self.fallback = None
        self.due = 0.0
        self.available = None       # None = not checked yet
        self.used_llm = 0
        self.used_fallback = 0

    # -- connectivity -------------------------------------------------------
    def check(self):
        """One-off probe so the booth can print whether the LLM is live."""
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=2.0) as r:
                names = [m.get("name", "") for m in json.load(r).get("models", [])]
            want = self.model if ":" in self.model else self.model + ":latest"
            self.available = want in names
            if not self.available:
                return False, (f"model {self.model} not pulled yet, run: "
                               f"ollama pull {self.model}")
            return self.available, names
        except Exception as e:
            self.available = False
            return False, str(e)

    # -- generation ---------------------------------------------------------
    def _ask(self, fact, timeout):
        """One chat call. Returns (text, truncated) or (None, False)."""
        messages = [{"role": "system", "content": PERSONA}]
        for f, line in EXAMPLES:
            messages.append({"role": "user", "content": f"FACT: {f}"})
            messages.append({"role": "assistant", "content": line})
        messages.append({"role": "user", "content": f"FACT: {fact}"})
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "60m",            # stay loaded all day at the booth
            "options": {"temperature": 0.5, "top_p": 0.9, "num_predict": 40,
                        "num_ctx": 1024},   # small context: less memory, faster
        }).encode()
        req = urllib.request.Request(
            self.host + "/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        text = data.get("message", {}).get("content", "")
        return text, data.get("done_reason") == "length"

    def warm_up(self):
        """Load the model into memory in the background, so the first real
        line is not slow. Called once at start-up."""
        def run():
            try:
                self._ask("Hello there. Want to play rock paper scissors?", 60)
            except Exception:
                pass
        threading.Thread(target=run, daemon=True).start()

    def cancel(self):
        """Forget the current request (for example, the countdown started)."""
        with self.lock:
            self.fallback = None
            self.result = None

    def _worker(self, fact):
        try:
            text, truncated = self._ask(fact, self.timeout)
        except Exception:
            text, truncated = None, False
        with self.lock:
            self.result = _clean(text, fact, truncated)
            self.pending = False

    def request(self, fact, fallback, now, deadline=1.3):
        """Kick off a rewrite. 'fallback' is spoken if it isn't ready in time."""
        with self.lock:
            if self.pending:            # one at a time; keep it simple
                return
            self.result = None
            self.pending = True
            self.fallback = fallback
            self.due = now + deadline
        threading.Thread(target=self._worker, args=(fact,), daemon=True).start()

    def poll(self, now):
        """Returns a line to speak exactly once per request, or None."""
        with self.lock:
            if self.fallback is None:
                return None
            if self.result is not None:
                line, self.result, self.fallback = self.result, None, None
                self.used_llm += 1
                return line
            if not self.pending or now >= self.due:
                line, self.fallback = self.fallback, None
                self.used_fallback += 1
                return line
            return None


if __name__ == "__main__":
    import sys, time
    llm = LLMVoice()
    ok, info = llm.check()
    print("ollama reachable and model present:", ok)
    if not ok:
        print("  ->", info)
        print("  start it with:  ollama serve   (and: ollama pull " + MODEL + ")")
        sys.exit(1)
    llm.timeout = 60.0
    print("loading the model (first time can take a few seconds)...")
    try:
        llm._ask("Hello.", 60)
    except Exception:
        pass
    for fact in ["That is 3 scissors in a row. I noticed.",
                 "I already have two wins. Last round for pride.",
                 "All square. The last round decides it.",
                 "You are close enough. Show me a fist to start.",
                 "Today I have played 12 games. Humans won 5, about 42 percent.",
                 "I win this one. Thanks for playing."]:
        t = time.time()
        llm.request(fact, "(fallback: the model's line failed the checks)",
                    time.time(), deadline=30.0)
        while True:
            line = llm.poll(time.time())
            if line:
                print(f"\nFACT: {fact}\nSAID: {line}   ({time.time() - t:.1f} s)")
                break
            time.sleep(0.05)
