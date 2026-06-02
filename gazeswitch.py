"""
GazeSwitch — Face Direction Multi-Monitor Cursor Control (Headless / Tray Mode)
--------------------------------------------------------------------------------
Runs silently in the system tray — no terminal window, no preview window.
The webcam is used only by GazeSwitch; release it when you need Google Meet.

Tray icon menu:
  Enable / Disable   — same as F9 hotkey
  Release Camera     — frees the webcam so Meet / Zoom can use it
  Reclaim Camera     — grabs the webcam back
  Quit

Hotkeys (global, work even when tray is focused):
  F9  → toggle gaze control on/off
  F8  → release / reclaim camera

Setup:
  pip install opencv-python mediapipe pyautogui screeninfo keyboard numpy pystray pillow

Run silently (no console):
  pythonw gazeswitch.py
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import screeninfo
import keyboard
import json
import time
import os
import sys
import threading
import urllib.request
import ctypes
import ctypes.wintypes
from collections import deque

# pystray + Pillow for tray icon
try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    print("Missing dependencies. Run:  pip install pystray pillow")
    sys.exit(1)

# DPI awareness so SetCursorPos coordinates match EnumDisplayMonitors
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
CONFIG = {
    "yaw_threshold_left":  10.0,
    "yaw_threshold_right": 0.5,
    "yaw_release":         1,
    "invert_yaw":          True,
    "jump_duration":       0.0,
    "webcam_index":        0,
    "hotkey_toggle":       "F9",
    "hotkey_camera":       "F8",
    "config_file":         "gazeswitch_config.json",
    "model_file":          "face_landmarker.task",
}

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

# ─────────────────────────────────────────────────────────
# AUTO-DOWNLOAD MODEL
# ─────────────────────────────────────────────────────────
def ensure_model():
    path = CONFIG["model_file"]
    if os.path.exists(path):
        return True
    print(f"Downloading face landmark model (~28 MB)…")
    try:
        def _progress(count, block, total):
            pct = min(int(count * block * 100 / total), 100)
            print(f"\r  {pct}%", end="", flush=True)
        urllib.request.urlretrieve(MODEL_URL, path, reporthook=_progress)
        print("\nModel downloaded.")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False

# ─────────────────────────────────────────────────────────
# MONITOR LAYOUT
# ─────────────────────────────────────────────────────────
class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize",    ctypes.c_ulong),
                ("rcMonitor", ctypes.wintypes.RECT),
                ("rcWork",    ctypes.wintypes.RECT),
                ("dwFlags",   ctypes.c_ulong)]

def _enum_logical_monitors():
    rects = []
    def _cb(hMon, hdc, lprc, data):
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(info)
        ctypes.windll.user32.GetMonitorInfoW(hMon, ctypes.byref(info))
        r = info.rcMonitor
        rects.append((r.left, r.top, r.right, r.bottom))
        return True
    _PROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong, ctypes.c_ulong,
                                ctypes.POINTER(ctypes.wintypes.RECT), ctypes.c_long)
    set_ctx = ctypes.windll.user32.SetThreadDpiAwarenessContext
    set_ctx.restype = ctypes.c_void_p
    prev = set_ctx(ctypes.c_void_p(-2))
    try:
        ctypes.windll.user32.EnumDisplayMonitors(None, None, _PROC(_cb), 0)
    finally:
        set_ctx(ctypes.c_void_p(prev))
    return rects

def get_sorted_monitors():
    monitors     = sorted(screeninfo.get_monitors(), key=lambda m: m.x)
    logical_rects = sorted(_enum_logical_monitors(), key=lambda r: r[0])
    centers = []
    for lr in logical_rects:
        centers.append(((lr[0] + lr[2]) // 2, (lr[1] + lr[3]) // 2))
    return monitors, centers

def move_cursor(x, y, duration=0.0):
    _set = ctypes.windll.user32.SetCursorPos
    if duration > 0:
        pt = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        sx, sy = pt.x, pt.y
        steps  = max(2, int(duration * 60))
        delay  = duration / steps
        for i in range(1, steps + 1):
            t = i / steps
            _set(int(sx + (x - sx) * t), int(sy + (y - sy) * t))
            time.sleep(delay)
    else:
        _set(int(x), int(y))

# ─────────────────────────────────────────────────────────
# FACE YAW DETECTOR
# ─────────────────────────────────────────────────────────
class FaceYawDetector:
    LANDMARKS = [1, 33, 263, 61, 291, 199]

    def __init__(self, model_path):
        VisionRunningMode      = mp.tasks.vision.RunningMode
        FaceLandmarkerOptions  = mp.tasks.vision.FaceLandmarkerOptions
        BaseOptions            = mp.tasks.BaseOptions
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.frame_ts   = 0

    def get_yaw(self, frame):
        h, w     = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.frame_ts += 33
        result = self.landmarker.detect_for_video(mp_image, self.frame_ts)
        if not result.face_landmarks:
            return None
        if result.facial_transformation_matrixes:
            matrix  = np.array(result.facial_transformation_matrixes[0].data).reshape(4, 4)
            yaw_rad = np.arctan2(-matrix[2, 0], matrix[0, 0])
            return float(np.degrees(yaw_rad))
        # solvePnP fallback
        lm = result.face_landmarks[0]
        pts_3d, pts_2d = [], []
        for idx in self.LANDMARKS:
            x = lm[idx].x * w; y = lm[idx].y * h; z = lm[idx].z * w
            pts_3d.append([x, y, z]); pts_2d.append([x, y])
        pts_3d = np.array(pts_3d, dtype=np.float64)
        pts_2d = np.array(pts_2d, dtype=np.float64)
        cam    = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
        _, rvec, _ = cv2.solvePnP(pts_3d, pts_2d, cam,
                                   np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _    = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        return float(angles[1])

    def close(self):
        self.landmarker.close()

# ─────────────────────────────────────────────────────────
# CURSOR CONTROLLER
# ─────────────────────────────────────────────────────────
class CursorController:
    def __init__(self, monitors, centers):
        self.monitors   = monitors
        self.centers    = centers
        self.active_idx = None
        pyautogui.FAILSAFE = False

    def jump_to(self, idx):
        if idx == self.active_idx:
            return
        cx, cy = self.centers[idx]
        move_cursor(cx, cy, duration=CONFIG["jump_duration"])
        self.active_idx = idx

# ─────────────────────────────────────────────────────────
# TRAY ICON BUILDER
# ─────────────────────────────────────────────────────────
def _make_icon_image(color=(0, 200, 100)):
    """16×16 filled circle on transparent background."""
    img  = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((1, 1, 14, 14), fill=color + (255,))
    return img

# ─────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────
class GazeSwitchApp:
    def __init__(self):
        self.enabled      = True
        self.cam_released = False
        self.running      = True
        self.lock         = threading.Lock()
        self.cap          = None
        self.detector     = None
        self.tray         = None

    # ── camera helpers ──────────────────────────────────
    def _open_camera(self):
        cap = cv2.VideoCapture(CONFIG["webcam_index"])
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS,          30)
        return cap

    def release_camera(self):
        with self.lock:
            if not self.cam_released and self.cap is not None:
                self.cap.release()
                self.cam_released = True
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
        self._update_tray()

    # ── tray icon ────────────────────────────────────────
    def _tray_title(self):
        if self.cam_released:
            return "GazeSwitch — Camera released"
        return "GazeSwitch — " + ("Active" if self.enabled else "Disabled")

    def _update_tray(self):
        if self.tray is None:
            return
        if self.cam_released:
            color = (180, 100, 0)   # orange = camera released
        elif self.enabled:
            color = (0, 200, 100)   # green  = active
        else:
            color = (120, 120, 120) # grey   = disabled
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
                lambda item: "Disable Gaze Control" if self.enabled else "Enable Gaze Control",
                on_toggle,
            ),
            pystray.MenuItem(
                lambda item: "Reclaim Camera (F8)" if self.cam_released else "Release Camera for Meet/Zoom (F8)",
                on_camera,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", on_quit),
        )

    def _run_tray(self):
        self.tray = pystray.Icon(
            "GazeSwitch",
            _make_icon_image((0, 200, 100)),
            self._tray_title(),
            menu=self._build_menu(),
        )
        self.tray.run()

    # ── gaze loop ────────────────────────────────────────
    def _gaze_loop(self, monitors, centers):
        controller = CursorController(monitors, centers)
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
            yaw   = self.detector.get_yaw(frame)

            if yaw is not None:
                eff_yaw = -yaw if CONFIG["invert_yaw"] else yaw
                if not hasattr(self, '_last_yaw') or abs(yaw - self._last_yaw) > 0.5:
                    print(f"raw={yaw:+.1f}  eff={eff_yaw:+.1f}  active={controller.active_idx}")
                    self._last_yaw = yaw

            if self.enabled and yaw is not None:
                thr_left  = CONFIG["yaw_threshold_left"]
                thr_right = CONFIG["yaw_threshold_right"]
                release   = CONFIG["yaw_release"]
                eff_yaw   = -yaw if CONFIG["invert_yaw"] else yaw
                active    = controller.active_idx

                screen_idx = None
                if eff_yaw < -abs(thr_left):
                    screen_idx = 0
                elif eff_yaw > -8.0:
                    screen_idx = 1
                elif active == 0 and eff_yaw > -release:
                    screen_idx = None
                elif active == 1 and eff_yaw < release:
                    screen_idx = None

                if screen_idx is not None:
                    controller.jump_to(screen_idx)

    # ── entry point ──────────────────────────────────────
    def run(self):
        if not ensure_model():
            sys.exit(1)

        # Load saved calibration
        cfg_file = CONFIG["config_file"]
        if os.path.exists(cfg_file):
            with open(cfg_file) as f:
                saved = json.load(f)
                CONFIG["yaw_threshold_left"]  = saved.get("yaw_threshold_left",  CONFIG["yaw_threshold_left"])
                CONFIG["yaw_threshold_right"] = saved.get("yaw_threshold_right", CONFIG["yaw_threshold_right"])

        monitors, centers = get_sorted_monitors()
        if len(monitors) < 2:
            sys.exit(1)

        self.detector = FaceYawDetector(CONFIG["model_file"])
        self.cap      = self._open_camera()
        if not self.cap.isOpened():
            sys.exit(1)

        # Global hotkeys
        keyboard.add_hotkey(CONFIG["hotkey_toggle"], self.toggle_enabled)
        keyboard.add_hotkey(CONFIG["hotkey_camera"], self.toggle_camera)

        # Tray icon in background thread
        t_tray = threading.Thread(target=self._run_tray, daemon=True)
        t_tray.start()

        # Gaze loop on main thread so print() works in terminal
        self._gaze_loop(monitors, centers)

        # Cleanup
        self.running = False
        keyboard.unhook_all()
        if self.cap is not None:
            self.cap.release()
        self.detector.close()


if __name__ == "__main__":
    import signal
    app = GazeSwitchApp()
    signal.signal(signal.SIGINT, lambda s, f: (setattr(app, 'running', False), app.tray and app.tray.stop()))
    app.run()
