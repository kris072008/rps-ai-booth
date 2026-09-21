"""
A tiny software 3D renderer and a robot arm that throws rock, paper or scissors.

No 3D engine, no model files, no extra dependencies - just numpy maths and
cv2.fillConvexPoly. Every part is a box in its own local space, positioned by a
chain of 4x4 matrices, then projected, z-buffered per pixel and flat-shaded.

Used by rps_booth.py when you pass --robot.
"""

import math

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Palette (BGR), matched to the booth chrome
# ---------------------------------------------------------------------------
BODY = (168, 150, 138)      # cool grey-blue plating
BODY_DARK = (120, 104, 94)  # joints and struts
ACCENT = (247, 92, 139)     # purple, same as the UI
EYE = (153, 211, 52)        # green visor
BG_TOP = (70, 32, 14)       # dark blue backdrop gradient
BG_BOT = (34, 14, 6)

LIGHT = np.array([-0.45, 0.78, -0.44])
LIGHT /= np.linalg.norm(LIGHT)


# ---------------------------------------------------------------------------
# Matrix helpers - column-vector convention, points are (x, y, z, 1)
# ---------------------------------------------------------------------------
def translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _cross(a, b):
    """Plain cross product. np.cross is ~20x slower for single 3-vectors."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


# A unit box: x and z centred on the origin, y running 0 -> 1.
# Local origin sits at the bottom of the part, which is the joint it pivots on.
_BOX_V = np.array([
    [-.5, 0, -.5], [.5, 0, -.5], [.5, 1, -.5], [-.5, 1, -.5],
    [-.5, 0, .5], [.5, 0, .5], [.5, 1, .5], [-.5, 1, .5],
], dtype=float)

_BOX_F = [
    (0, 1, 2, 3),  # back
    (5, 4, 7, 6),  # front
    (4, 0, 3, 7),  # left
    (1, 5, 6, 2),  # right
    (3, 2, 6, 7),  # top
    (4, 5, 1, 0),  # bottom
]


class Scene:
    """Collects transformed boxes, then rasterises them with a z-buffer."""

    def __init__(self):
        self.faces = []  # (depth, world_pts[4], colour)

    def box(self, mat, w, h, d, colour):
        v = _BOX_V.copy()
        v[:, 0] *= w
        v[:, 1] *= h
        v[:, 2] *= d
        pts = np.hstack([v, np.ones((8, 1))]) @ mat.T
        pts = pts[:, :3]
        for f in _BOX_F:
            quad = pts[list(f)]
            n = _cross(quad[1] - quad[0], quad[2] - quad[0])
            ln = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
            if ln < 1e-9:
                continue
            n /= ln
            shade = 0.38 + 0.62 * max(0.0, float(n @ LIGHT))
            col = tuple(min(255, int(c * shade)) for c in colour)
            self.faces.append((quad, col))

    def render(self, img, f=950.0, cam=(-0.35, 1.30, -6.6), ss=2):
        """Project and rasterise with a per-pixel z-buffer, so near surfaces
        always hide far ones - no see-through artefacts. Rendered at ss x
        resolution and scaled down for smooth anti-aliased edges. Edges are
        outlined wherever two different faces meet on screen."""
        h, w = img.shape[:2]
        H, W, F = h * ss, w * ss, f * ss
        cx, cy = W * 0.5, H * 0.55
        cam = np.array(cam, dtype=float)

        # each pixel's viewing ray, precomputed once per size
        key = (H, W, F)
        if getattr(self, "_rays_key", None) != key:
            self._rx = ((np.arange(W) + 0.5 - cx) / F).astype(np.float32)
            self._ry = (-(np.arange(H) + 0.5 - cy) / F).astype(np.float32)
            self._rays_key = key
        RX, RY = self._rx, self._ry

        # project and cull first, so we know which part of the screen matters
        faces = []
        for quad, col in self.faces:
            view = quad - cam
            if view[:, 2].min() < 0.35:              # behind the camera
                continue
            n = _cross(view[1] - view[0], view[2] - view[0])
            d = float(n @ view[0])
            if d > 0:                                # facing away: culled
                continue
            zs = view[:, 2]
            sx = F * view[:, 0] / zs + cx
            sy = -F * view[:, 1] / zs + cy
            faces.append((sx, sy, n.astype(np.float32), np.float32(d), col))
        if not faces:
            return
        allx = np.concatenate([f[0] for f in faces])
        ally = np.concatenate([f[1] for f in faces])
        bx0, bx1 = max(int(allx.min()) - 2, 0), min(int(allx.max()) + 3, W)
        by0, by1 = max(int(ally.min()) - 2, 0), min(int(ally.max()) + 3, H)
        if bx1 <= bx0 or by1 <= by0:
            return

        faces = faces[:254]                  # ids must fit in one byte
        big = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        colour = big[by0:by1, bx0:bx1]
        bh, bw = colour.shape[:2]
        zbuf = np.full((bh, bw), np.inf, dtype=np.float32)
        ids = np.zeros((bh, bw), dtype=np.uint8)          # 0 = background
        RXb, RYb = RX[bx0:bx1], RY[by0:by1]

        for fid, (sx, sy, n, d, col) in enumerate(faces, start=1):
            sx = sx - bx0
            sy = sy - by0
            x0, x1 = max(int(sx.min()), 0), min(int(sx.max()) + 2, bw)
            y0, y1 = max(int(sy.min()), 0), min(int(sy.max()) + 2, bh)
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            poly = np.stack([sx - x0, sy - y0], axis=1).round().astype(np.int32)
            cv2.fillConvexPoly(mask, poly, 1)
            # exact depth of this face at every pixel of its bounding box:
            # where each pixel's viewing ray meets the face's plane
            denom = n[0] * RXb[None, x0:x1] + n[1] * RYb[y0:y1, None] + n[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                z = d / denom
            zb = zbuf[y0:y1, x0:x1]
            win = (mask != 0) & (z > 0) & (z < zb)
            zb[win] = z[win]
            ids[y0:y1, x0:x1][win] = fid

        # colour every pixel in one pass, via 8-bit lookup tables in OpenCV
        lut = np.zeros((1, 256, 3), np.uint8)
        lut[0, 1:len(faces) + 1] = [f[4] for f in faces]
        ids3 = cv2.merge([ids, ids, ids])
        shaded = cv2.LUT(ids3, lut)
        # outline: wherever the face id changes between neighbouring pixels
        edge = cv2.dilate(ids, None) != cv2.erode(ids, None)
        edge = (edge & (ids > 0)).astype(np.uint8)
        cv2.copyTo(cv2.LUT(ids3, lut >> 1), edge, shaded)
        cv2.copyTo(shaded, (ids > 0).astype(np.uint8), colour)
        colour = big

        img[:] = cv2.resize(colour, (w, h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Finger curl targets, per pose. Two joints per finger, radians.
# order: index, middle, ring, pinky
# ---------------------------------------------------------------------------
_CURL = (1.45, 1.55)      # folded into the palm
_STRAIGHT = (0.06, 0.04)  # extended

POSES = {
    "rock":     dict(fingers=[_CURL] * 4,
                     thumb=(0.95, 0.85)),
    "paper":    dict(fingers=[_STRAIGHT] * 4,
                     thumb=(0.15, 0.10)),
    "scissors": dict(fingers=[_STRAIGHT, _STRAIGHT, _CURL, _CURL],
                     thumb=(1.00, 0.85)),
}

_SPREAD = (-0.20, -0.06, 0.07, 0.20)   # sideways fan so paper reads as a hand


def _lerp(a, b, t):
    return a + (b - a) * t


def _pose_blend(p_from, p_to, t):
    """Ease between two named poses so the hand snaps rather than teleports."""
    t = max(0.0, min(1.0, t))
    t = t * t * (3 - 2 * t)                      # smoothstep
    a, b = POSES[p_from], POSES[p_to]
    fingers = [(_lerp(a["fingers"][i][0], b["fingers"][i][0], t),
                _lerp(a["fingers"][i][1], b["fingers"][i][1], t)) for i in range(4)]
    thumb = (_lerp(a["thumb"][0], b["thumb"][0], t),
             _lerp(a["thumb"][1], b["thumb"][1], t))
    return fingers, thumb


def build_robot(scene, yaw, shoulder, elbow, wrist, fingers, thumb):
    """Assemble the robot from boxes, parented joint to joint."""
    root = translate(-0.45, 0, 0.30) @ rot_y(yaw)

    # Plinth and torso
    scene.box(root @ translate(0, 0, 0), 2.30, 0.28, 1.70, BODY_DARK)
    scene.box(root @ translate(0, 0.28, 0), 1.90, 0.16, 1.40, BODY)
    torso = root @ translate(0, 0.44, 0)
    scene.box(torso, 1.15, 1.45, 0.86, BODY)
    scene.box(torso @ translate(0, 0.45, -0.44), 0.70, 0.36, 0.06, ACCENT)  # chest light

    # Neck and head
    neck = torso @ translate(0, 1.45, 0)
    scene.box(neck, 0.34, 0.22, 0.34, BODY_DARK)
    head = neck @ translate(0, 0.22, 0) @ rot_y(math.sin(yaw * 1.7) * 0.10)
    scene.box(head, 1.00, 0.76, 0.80, BODY)
    scene.box(head @ translate(0, 0.22, -0.41), 0.74, 0.26, 0.05, EYE)      # visor
    scene.box(head @ translate(-0.30, 0.76, 0), 0.10, 0.26, 0.10, BODY_DARK)  # antenna
    scene.box(head @ translate(-0.30, 1.02, 0), 0.16, 0.10, 0.16, ACCENT)

    # Right arm. Every segment is drawn hanging along -y from its joint, so a
    # rotation at the joint swings the whole rest of the limb with it.
    sh = torso @ translate(1.02, 1.02, -0.62) @ rot_z(shoulder) @ rot_x(0.10)
    scene.box(sh @ translate(0, -0.20, 0), 0.42, 0.38, 0.42, BODY_DARK)     # shoulder ball
    scene.box(sh @ translate(0, -1.15, 0), 0.34, 1.15, 0.34, BODY)          # upper arm

    # elbow ~2.5 rad folds the forearm up and forward, toward the camera
    el = sh @ translate(0, -1.15, 0) @ rot_x(elbow)
    scene.box(el @ translate(0, -0.16, 0), 0.34, 0.32, 0.34, BODY_DARK)     # elbow joint
    scene.box(el @ translate(0, -1.00, 0), 0.28, 1.00, 0.28, BODY)          # forearm

    wr = el @ translate(0, -1.00, 0) @ rot_x(wrist)
    scene.box(wr @ translate(0, -0.14, 0), 0.34, 0.18, 0.30, ACCENT)        # wrist band

    # Palm: broad face points at the camera, fingertips continue along -y.
    scene.box(wr @ translate(0, -0.86, 0), 0.76, 0.72, 0.27, BODY)

    # Four fingers, hinged at the knuckle line
    for (c1, c2), sx in zip(fingers, _SPREAD):
        knuckle = wr @ translate(sx * 1.20, -0.86, 0) @ rot_z(sx * 0.5) @ rot_x(c1)
        scene.box(knuckle @ translate(0, -0.36, 0), 0.17, 0.36, 0.19, BODY)
        mid = knuckle @ translate(0, -0.36, 0) @ rot_x(c2)
        scene.box(mid @ translate(0, -0.30, 0), 0.15, 0.30, 0.17, BODY_DARK)

    # Thumb, swinging off the side of the palm
    t1, t2 = thumb
    tb = wr @ translate(-0.46, -0.56, 0.02) @ rot_z(-0.95) @ rot_x(t1)
    scene.box(tb @ translate(0, -0.30, 0), 0.17, 0.30, 0.17, BODY)
    tb2 = tb @ translate(0, -0.30, 0) @ rot_x(t2)
    scene.box(tb2 @ translate(0, -0.24, 0), 0.15, 0.24, 0.15, BODY_DARK)


class RobotPanel:
    """Owns the robot's animation state and renders it into a side panel."""

    def __init__(self, width, height):
        self.w, self.h = width, height
        self._bg = self._gradient()
        self.pose_from = "rock"
        self.pose_to = "rock"
        self.pose_t = 1.0
        self.caption = ""
        self.ss = 2          # supersampling; the Raspberry Pi profile sets 1

    def _gradient(self):
        g = np.zeros((self.h, self.w, 3), np.uint8)
        for y in range(self.h):
            f = y / max(1, self.h - 1)
            g[y, :] = [int(_lerp(BG_TOP[c], BG_BOT[c], f)) for c in range(3)]
        return g

    def set_pose(self, pose, instant=False):
        if pose == self.pose_to and not instant:
            return
        self.pose_from = self.pose_to if not instant else pose
        self.pose_to = pose
        self.pose_t = 1.0 if instant else 0.0

    def draw(self, t, state, count_elapsed, caption="", look=0.0):
        """t: wall clock. state: 'idle' | 'count' | 'reveal' | 'greet'.
        look: -1..1, turns the robot toward where the person is standing."""
        img = self._bg.copy()
        yaw = 0.34 + 0.16 * math.sin(t * 0.55) + float(look) * 0.30

        if state == "greet":
            # waves hello with an open hand
            wave = math.sin(t * 7.5)
            shoulder = 0.12 + 0.26 * wave
            elbow = 2.45 + 0.08 * wave
            wrist = 0.22 * wave
            self.set_pose("paper", instant=True)
        elif state == "count":
            # Three pumps, one per countdown word.
            phase = count_elapsed / 1.0      # one pump per countdown word
            pump = math.sin(min(phase, 3.0) * math.pi)
            shoulder = -0.14 - 0.14 * abs(pump)
            elbow = 2.08 - 0.52 * abs(pump)
            wrist = 0.22 + 0.30 * abs(pump)
            self.set_pose("rock", instant=True)
        elif state == "reveal":
            shoulder = -0.16
            elbow = 2.05
            wrist = 0.26
            self.pose_t = min(1.0, self.pose_t + 0.16)
        else:  # idle
            breathe = math.sin(t * 1.1) * 0.05
            shoulder = -0.12 + breathe
            elbow = 1.98 + breathe * 1.2
            wrist = 0.24 + 0.06 * math.sin(t * 0.9)
            self.set_pose("rock", instant=True)

        fingers, thumb = _pose_blend(self.pose_from, self.pose_to, self.pose_t)

        scene = Scene()
        build_robot(scene, yaw, shoulder, elbow, wrist, fingers, thumb)
        scene.render(img, ss=self.ss)

        # Panel framing
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), (110, 56, 30), 2)
        label = caption or self.caption
        if label:
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, _), _ = cv2.getTextSize(label, font, 0.72, 2)
            x = (self.w - tw) // 2
            y = self.h - 26
            cv2.putText(img, label, (x + 2, y + 2), font, 0.72, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (x, y), font, 0.72, ACCENT, 2, cv2.LINE_AA)
        return img


# ---------------------------------------------------------------------------
# Standalone preview:  python robot3d.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    panel = RobotPanel(460, 720)
    poses = ["rock", "paper", "scissors"]
    i = 0
    t0 = None
    import time as _t
    start = _t.time()
    cv2.namedWindow("robot", cv2.WINDOW_NORMAL)
    while True:
        now = _t.time()
        cycle = (now - start) % 5.0
        if cycle < 2.6:
            state, el = "count", cycle
            t0 = None
        else:
            if t0 is None:
                i = (i + 1) % 3
                panel.set_pose(poses[i])
                t0 = now
            state, el = "reveal", 0.0
        cv2.imshow("robot", panel.draw(now, state, el, poses[i].upper()))
        if cv2.waitKey(16) & 0xFF in (ord('q'), 27):
            break
    cv2.destroyAllWindows()
