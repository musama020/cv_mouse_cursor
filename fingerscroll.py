"""
FingerScroll — Hand-Gesture Vertical Scrolling (Headless / Tray Mode)
---------------------------------------------------------------------
Tracks one hand with MediaPipe Hands. Raise your index finger and move it
up or down to scroll the active window. Runs silently in the system tray —
no terminal window, no preview window (unless launched with --debug).

This is a STANDALONE companion to gazeswitch.py. It uses the webcam
exclusively, so run one at a time (both expose a camera-release hotkey).

Tray icon menu:
  Enable / Disable   — same as F10 hotkey
  Release Camera     — frees the webcam so Meet / Zoom can use it
  Reclaim Camera     — grabs the webcam back
  Quit

Hotkeys (global):
  F10 → toggle finger-scroll on/off
  F11 → release / reclaim camera

Gestures (index-finger curl — works wherever you hold your hand):
  Index finger STRAIGHT            → scroll up   (fixed speed)
  Index finger BENT / half-curled  → scroll down (fixed speed)
  Hand in frame + index            → always decides a direction and scrolls
  Tilt hand SIDEWAYS               → pause (bring it upright to resume)
  Remove hand from frame           → stops
  (gesture_mode extras, opt-in via config):
    2 fingers up               → fast scroll
    3 fingers up               → jump to top (Ctrl+Home)

Setup:
  pip install opencv-python mediapipe pyautogui keyboard numpy pystray pillow

On first run the hand landmark model (~4 MB) is downloaded automatically.

Run silently (no console):
  pythonw fingerscroll.py

Run with live preview window for tuning:
  python fingerscroll.py --debug
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keyboard
import json
import math
import time
import os
import sys
import threading
import urllib.request
import importlib

# pystray + Pillow for tray icon
try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    print("Missing dependencies. Run:  pip install pystray pillow")
    sys.exit(1)

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
CONFIG = {
    # ── Finger-curl model ────────────────────────────────────
    # Direction comes from how STRAIGHT the index finger is, not where the
    # hand is in frame — so it works wherever you hold your hand.
    # We measure "straightness" as the average bend angle across the index
    # joints (MCP→PIP→DIP→TIP). A straight finger ≈ 180°; a curled one less.
    #   straightness >= straight_angle           → scroll UP
    #   anything more bent (incl. half-curl)      → scroll DOWN
    # As long as a hand + extended index is seen, it ALWAYS decides & scrolls.
    "straight_angle":           160.0,  # avg joint angle (deg) at/above which = straight = UP
    "hysteresis":               12.0,   # deg of stickiness so it doesn't flicker at the line
    "scroll_up_speed":          75,      # fixed clicks/tick when scrolling UP
    "scroll_down_speed":        75,      # fixed clicks/tick when scrolling DOWN
    "smoothing":                0.6,    # EMA factor for the straightness value (0=raw, →1=smooth)
    # ── Pause-on-tilt ─────────────────────────────────────────
    # Hand orientation = angle of the wrist→index-base vector from vertical
    # (0° = pointing straight up, 90° = pointing sideways). Tilt the hand past
    # `pause_tilt` toward horizontal and scrolling pauses; bring it upright
    # again to resume. `pause_tilt_release` gives hysteresis so it does not
    # flicker right at the line.
    "pause_on_tilt":            True,   # enable wrist-tilt pause
    "pause_tilt":               55.0,   # deg from vertical → pause when tilted past this
    "pause_tilt_release":       45.0,   # deg from vertical → resume when uprighter than this
    "min_detection_confidence": 0.6,
    "min_tracking_confidence":  0.7,
    "webcam_index":             0,
    "hotkey_toggle":            "F10",  # F9 is taken by GazeSwitch
    "hotkey_camera":            "F11",  # F8 is taken by GazeSwitch
    "config_file":              "fingerscroll_config.json",
    "model_file":               "hand_landmarker.task",
    "gesture_mode":             False,  # enable 2/3-finger extras
    "gesture_mode_speed_mult":  2,      # speed multiplier when 2 fingers raised
}

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ─────────────────────────────────────────────────────────
# AUTO-DOWNLOAD MODEL
# ─────────────────────────────────────────────────────────
def ensure_hand_model():
    path = CONFIG["model_file"]
    if os.path.exists(path):
        return True
    print(f"Downloading hand landmark model (~4 MB)…")
    try:
        def _progress(count, block, total):
            pct = min(int(count * block * 100 / total), 100)
            print(f"\r  {pct}%", end="", flush=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, path, reporthook=_progress)
        print("\nModel downloaded.")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False

# Persisted config keys (a subset of CONFIG worth saving/tuning)
_SAVED_KEYS = [
    "straight_angle", "hysteresis", "scroll_up_speed", "scroll_down_speed",
    "smoothing", "gesture_mode",
    "pause_on_tilt", "pause_tilt", "pause_tilt_release",
]

# MediaPipe Hands landmark indices
WRIST            = 0
THUMB_TIP        = 4
INDEX_TIP        = 8
FINGER_TIPS      = [8, 12, 16, 20]   # index, middle, ring, pinky tips
FINGER_PIPS      = [6, 10, 14, 18]   # the PIP joint two below each tip
# The four index-finger keypoints, base → tip:
INDEX_CHAIN      = [5, 6, 7, 8]      # MCP, PIP, DIP, TIP

cam_w, cam_h = 640, 480

# ─────────────────────────────────────────────────────────
# GESTURE HELPERS
# ─────────────────────────────────────────────────────────
def count_fingers(landmarks):
    """Count raised fingers (excluding thumb). A finger is 'up' when its tip
    is higher on screen (smaller y) than its PIP joint."""
    count = 0
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1
    return count


def index_is_up(landmarks):
    """True when the index finger is extended (tip above its PIP joint)."""
    return landmarks[INDEX_TIP].y < landmarks[FINGER_PIPS[0]].y


def _angle_at(a, b, c):
    """Angle in degrees at vertex b for points a-b-c (uses x,y; z ignored
    because it is noisier on a webcam)."""
    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    dot   = bax * bcx + bay * bcy
    na    = (bax * bax + bay * bay) ** 0.5
    nc    = (bcx * bcx + bcy * bcy) ** 0.5
    if na < 1e-6 or nc < 1e-6:
        return 180.0
    cos = max(-1.0, min(1.0, dot / (na * nc)))
    return math.degrees(math.acos(cos))


def index_straightness(landmarks):
    """Average joint angle (degrees) along the index finger. ~180 = perfectly
    straight; lower = more bent/curled. Robust to where the hand is in frame."""
    p = [landmarks[i] for i in INDEX_CHAIN]          # MCP, PIP, DIP, TIP
    pip_angle = _angle_at(p[0], p[1], p[2])          # bend at PIP
    dip_angle = _angle_at(p[1], p[2], p[3])          # bend at DIP
    return (pip_angle + dip_angle) / 2.0


def hand_tilt(landmarks):
    """Hand orientation as the angle (deg) of the wrist→index-base vector away
    from vertical. 0° = hand pointing straight up, 90° = pointing sideways.
    Used to pause scrolling when the hand is tilted horizontal."""
    wrist = landmarks[WRIST]          # 0
    base  = landmarks[INDEX_CHAIN[0]]  # 5 (index MCP)
    dx = base.x - wrist.x
    dy = base.y - wrist.y             # image y grows downward
    # Angle from the straight-up direction (0, -1).
    return math.degrees(math.atan2(abs(dx), abs(dy)))


# ─────────────────────────────────────────────────────────
# HAND DETECTOR  (mediapipe Tasks API — mp.solutions removed in 0.10.x)
# ─────────────────────────────────────────────────────────
# Drawing utils live at an internal path in mediapipe 0.10.x
_du = importlib.import_module("mediapipe.tasks.python.vision.drawing_utils")
_DrawingSpec   = _du.DrawingSpec
_draw_landmarks = _du.draw_landmarks
_HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

_LANDMARK_SPEC   = _DrawingSpec(color=(0, 255, 0),   thickness=2, circle_radius=3)
_CONNECTION_SPEC = _DrawingSpec(color=(255, 255, 0),  thickness=2)


class HandScrollDetector:
    """Wraps mp.tasks.vision.HandLandmarker (VIDEO mode).
    process() returns a list of NormalizedLandmark for one hand, or None."""

    def __init__(self, model_path):
        VisionRunningMode     = mp.tasks.vision.RunningMode
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        BaseOptions           = mp.tasks.BaseOptions
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=CONFIG["min_detection_confidence"],
            min_hand_presence_confidence=CONFIG["min_detection_confidence"],
            min_tracking_confidence=CONFIG["min_tracking_confidence"],
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._frame_ts  = 0

    def process(self, frame_bgr):
        """Returns list of NormalizedLandmark (21 points) or None."""
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]   # list of NormalizedLandmark

    def draw(self, frame_bgr, landmarks):
        _draw_landmarks(
            frame_bgr, landmarks, _HAND_CONNECTIONS,
            landmark_drawing_spec=_LANDMARK_SPEC,
            connection_drawing_spec=_CONNECTION_SPEC,
        )

    def close(self):
        self.landmarker.close()


# ─────────────────────────────────────────────────────────
# SCROLL CONTROLLER  (finger-curl direction — straight=UP, bent=DOWN)
# ─────────────────────────────────────────────────────────
class ScrollController:
    """Direction from index-finger curl, not hand position.

    Straightness = average bend angle along the index joints (~180° straight,
    lower = curled). As long as a hand + extended index is seen we ALWAYS
    decide a direction and scroll at a fixed speed:

        straightness >= straight_angle        → straight → scroll UP
        anything more bent (incl. half-curl)  → bent     → scroll DOWN

    A hysteresis band around the threshold keeps the decision 'sticky' so a
    finger hovering near the cutoff doesn't flicker UP/DOWN — but it always
    commits to one direction (never rest), per the always-scroll rule.
    """

    def __init__(self):
        self.smooth_a   = None    # EMA-smoothed straightness angle
        self.direction  = +1      # current committed direction (+1 up, -1 down)
        self.last_action = ""     # for debug HUD
        self.last_angle  = 0.0    # last smoothed angle (for debug HUD)

    def reset(self):
        """Call when the hand leaves frame or the index finger drops."""
        self.smooth_a   = None
        self.last_action = ""

    def _smooth(self, angle):
        a = CONFIG["smoothing"]
        if self.smooth_a is None:
            self.smooth_a = angle
        else:
            self.smooth_a = a * self.smooth_a + (1.0 - a) * angle
        return self.smooth_a

    def update(self, straightness, finger_count):
        """straightness: avg index joint angle in degrees (~180 = straight).
        Returns a short action label for the debug HUD."""
        ang = self._smooth(straightness)
        self.last_angle = ang

        thr  = CONFIG["straight_angle"]
        hyst = CONFIG["hysteresis"]

        # Sticky threshold: only flip once the angle crosses well past the line.
        if self.direction > 0:                    # currently UP → need to curl below (thr - hyst)
            if ang < thr - hyst:
                self.direction = -1
        else:                                     # currently DOWN → need to straighten above (thr + hyst)
            if ang > thr + hyst:
                self.direction = +1

        if self.direction > 0:
            amount = CONFIG["scroll_up_speed"]
            label  = "UP"
        else:
            amount = CONFIG["scroll_down_speed"]
            label  = "DOWN"

        if CONFIG["gesture_mode"] and finger_count >= 2:
            amount *= CONFIG["gesture_mode_speed_mult"]

        pyautogui.scroll(self.direction * amount)
        self.last_action = f"{label} x{amount}"
        return self.last_action


# ─────────────────────────────────────────────────────────
# TRAY ICON BUILDER
# ─────────────────────────────────────────────────────────
def _make_icon_image(color=(150, 80, 220)):
    """16×16 filled circle on transparent background (purple by default to
    distinguish from GazeSwitch's green icon)."""
    img  = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((1, 1, 14, 14), fill=color + (255,))
    return img


# ─────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────
class FingerScrollApp:
    def __init__(self, debug=False):
        self.debug        = debug
        self.enabled      = True
        self.cam_released = False
        self.running      = True
        self.lock         = threading.Lock()
        self.cap          = None
        self.detector     = None
        self.controller   = None
        self.tray         = None
        self._top_fired   = False   # debounce for the 3-finger jump-to-top gesture
        self._tilt_paused = False   # sticky pause state from wrist tilt
        self._tilt        = 0.0     # last hand-tilt angle (for debug HUD)

    # ── camera helpers ──────────────────────────────────
    def _open_camera(self):
        cap = cv2.VideoCapture(CONFIG["webcam_index"])
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
            cap.set(cv2.CAP_PROP_FPS,          30)
        return cap

    def release_camera(self):
        with self.lock:
            if not self.cam_released and self.cap is not None:
                self.cap.release()
                self.cam_released = True
                if self.controller:
                    self.controller.reset()
                self._update_tray()

    def reclaim_camera(self):
        with self.lock:
            if self.cam_released:
                self.cap = self._open_camera()
                self.cam_released = False
                self._update_tray()

    def toggle_camera(self):
        if self.cam_released:
            self.reclaim_camera()
        else:
            self.release_camera()

    # ── enable/disable ───────────────────────────────────
    def toggle_enabled(self):
        self.enabled = not self.enabled
        if not self.enabled and self.controller:
            self.controller.reset()
        self._update_tray()

    # ── tray icon ────────────────────────────────────────
    def _tray_title(self):
        if self.cam_released:
            return "FingerScroll — Camera released"
        return "FingerScroll — " + ("Active" if self.enabled else "Disabled")

    def _update_tray(self):
        if self.tray is None:
            return
        if self.cam_released:
            color = (180, 100, 0)     # orange = camera released
        elif self.enabled:
            color = (150, 80, 220)    # purple = active
        else:
            color = (120, 120, 120)   # grey   = disabled
        self.tray.icon  = _make_icon_image(color)
        self.tray.title = self._tray_title()
        self.tray.update_menu()

    def _build_menu(self):
        def on_toggle(icon, item):
            self.toggle_enabled()
        def on_camera(icon, item):
            self.toggle_camera()
        def on_quit(icon, item):
            self.running = False
            icon.stop()
            os._exit(0)

        return pystray.Menu(
            pystray.MenuItem(
                lambda item: "Disable Finger Scroll" if self.enabled else "Enable Finger Scroll",
                on_toggle,
            ),
            pystray.MenuItem(
                lambda item: "Reclaim Camera (F11)" if self.cam_released else "Release Camera for Meet/Zoom (F11)",
                on_camera,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", on_quit),
        )

    def _run_tray(self):
        self.tray = pystray.Icon(
            "FingerScroll",
            _make_icon_image((150, 80, 220)),
            self._tray_title(),
            menu=self._build_menu(),
        )
        self.tray.run()

    # ── debug HUD ─────────────────────────────────────────
    def _draw_hud(self, frame, finger_count, action):
        cv2.putText(frame, "F10=Toggle  F11=Camera  Q=Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        state = "ACTIVE" if self.enabled else "DISABLED"
        cv2.putText(frame,
                    f"{state}  ang={self.controller.last_angle:5.0f}  tilt={self._tilt:4.0f}/{CONFIG['pause_tilt']:.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if action == "PAUSED":
            cv2.putText(frame, "PAUSED (tilt)", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        elif action.startswith("UP"):
            cv2.putText(frame, action, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        elif action.startswith("DOWN"):
            cv2.putText(frame, action, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif action == "TOP":
            cv2.putText(frame, "JUMP TO TOP", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

    # ── scroll loop ───────────────────────────────────────
    def _scroll_loop(self):
        while self.running:
            with self.lock:
                released = self.cam_released
                cap      = self.cap

            if released or cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            hand  = self.detector.process(frame)

            action = ""
            finger_count = 0

            if hand is not None:
                landmarks    = hand          # Tasks API returns list[NormalizedLandmark] directly
                finger_count = count_fingers(landmarks)

                # Wrist-tilt pause (sticky/hysteretic): tilt the hand sideways
                # past `pause_tilt` to pause, bring it upright past
                # `pause_tilt_release` to resume.
                # Also pauses when all 4 fingers are extended (open palm / wrist position).
                self._tilt = hand_tilt(landmarks)
                if CONFIG["pause_on_tilt"]:
                    open_palm = finger_count >= 4
                    if self._tilt_paused:
                        if self._tilt < CONFIG["pause_tilt_release"] and not open_palm:
                            self._tilt_paused = False
                    elif self._tilt > CONFIG["pause_tilt"] or open_palm:
                        self._tilt_paused = True

                # Always-scroll rule: a hand is in front of the camera on purpose,
                # so while one is detected (and not tilt-paused) we decide a
                # direction from index curl and scroll. To stop: tilt the hand
                # sideways, or take it out of frame.
                if self.enabled and not self._tilt_paused:
                    # gesture_mode: 3 fingers up → jump to top, suppress scroll this frame
                    if CONFIG["gesture_mode"] and finger_count >= 3:
                        if not self._top_fired:
                            pyautogui.hotkey("ctrl", "home")
                            self._top_fired = True
                        action = "TOP"
                    else:
                        self._top_fired = False
                        straightness = index_straightness(landmarks)
                        action = self.controller.update(straightness, finger_count)
                elif self._tilt_paused:
                    # Hold direction steady while paused so resume doesn't lurch.
                    self.controller.reset()
                    action = "PAUSED"

                if self.debug:
                    self.detector.draw(frame, hand)
                    tip_x = int(landmarks[INDEX_TIP].x * cam_w)
                    tip_y = int(landmarks[INDEX_TIP].y * cam_h)
                    cv2.circle(frame, (tip_x, tip_y), 8, (255, 0, 255), -1)
            else:
                self.controller.reset()
                self._top_fired = False
                self._tilt_paused = False

            if self.debug:
                self._draw_hud(frame, finger_count, action)
                cv2.imshow("FingerScroll (debug)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    if self.tray:
                        self.tray.stop()
                    break

    # ── entry point ──────────────────────────────────────
    def run(self):
        # Load saved config
        cfg_file = CONFIG["config_file"]
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file) as f:
                    saved = json.load(f)
                for k in _SAVED_KEYS:
                    if k in saved:
                        CONFIG[k] = saved[k]
            except Exception as e:
                print(f"Could not read {cfg_file}: {e}")

        if not ensure_hand_model():
            sys.exit(1)

        pyautogui.FAILSAFE = False

        self.detector   = HandScrollDetector(CONFIG["model_file"])
        self.controller = ScrollController()
        self.cap        = self._open_camera()
        if not self.cap.isOpened():
            print("Could not open webcam.")
            sys.exit(1)

        # Global hotkeys
        keyboard.add_hotkey(CONFIG["hotkey_toggle"], self.toggle_enabled)
        keyboard.add_hotkey(CONFIG["hotkey_camera"], self.toggle_camera)

        # Tray icon in background thread
        t_tray = threading.Thread(target=self._run_tray, daemon=True)
        t_tray.start()

        # Scroll loop on main thread
        self._scroll_loop()

        # Cleanup
        self.running = False
        keyboard.unhook_all()
        if self.cap is not None:
            self.cap.release()
        self.detector.close()
        if self.debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import signal
    debug = "--debug" in sys.argv
    app = FingerScrollApp(debug=debug)
    signal.signal(signal.SIGINT,
                  lambda s, f: (setattr(app, 'running', False),
                                app.tray and app.tray.stop()))
    app.run()
