import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import screeninfo
import keyboard
import time
import threading
import urllib.request
import ctypes
import ctypes.wintypes

import config

try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    print("Missing dependencies. Run:  pip install -r requirement.txt")
    sys.exit(1)

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

DEBUG = "--debug" in sys.argv


def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def _download(url, path, label):
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log(f"Downloading {label} model…")
    try:
        def _progress(count, block, total):
            pct = min(int(count * block * 100 / total), 100)
            log(f"\r  {pct}%", end="")
        urllib.request.urlretrieve(url, path, reporthook=_progress)
        log(f"\n{label} model downloaded.")
        return True
    except Exception as e:
        print(f"\nDownload of {label} model failed: {e}")
        return False


class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong),
                ("rcMonitor", ctypes.wintypes.RECT),
                ("rcWork", ctypes.wintypes.RECT),
                ("dwFlags", ctypes.c_ulong)]


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
    monitors = sorted(screeninfo.get_monitors(), key=lambda m: m.x)
    logical_rects = sorted(_enum_logical_monitors(), key=lambda r: r[0])
    centers = [((lr[0] + lr[2]) // 2, (lr[1] + lr[3]) // 2) for lr in logical_rects]
    return monitors, centers


def move_cursor(x, y, duration=0.0):
    _set = ctypes.windll.user32.SetCursorPos
    if duration > 0:
        pt = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        sx, sy = pt.x, pt.y
        steps = max(2, int(duration * 60))
        delay = duration / steps
        for i in range(1, steps + 1):
            t = i / steps
            _set(int(sx + (x - sx) * t), int(sy + (y - sy) * t))
            time.sleep(delay)
    else:
        _set(int(x), int(y))


class FaceYawDetector:
    LANDMARKS = [1, 33, 263, 61, 291, 199]

    def __init__(self, model_path):
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        BaseOptions = mp.tasks.BaseOptions
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=config.SHARED["min_detection_confidence"],
            min_face_presence_confidence=config.SHARED["min_detection_confidence"],
            min_tracking_confidence=config.SHARED["min_tracking_confidence"],
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.frame_ts = 0

    def get_yaw(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.frame_ts += 33
        result = self.landmarker.detect_for_video(mp_image, self.frame_ts)
        if not result.face_landmarks:
            return None
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0].data).reshape(4, 4)
            yaw_rad = np.arctan2(-matrix[2, 0], matrix[0, 0])
            return float(np.degrees(yaw_rad))
        lm = result.face_landmarks[0]
        pts_3d, pts_2d = [], []
        for idx in self.LANDMARKS:
            x = lm[idx].x * w
            y = lm[idx].y * h
            z = lm[idx].z * w
            pts_3d.append([x, y, z])
            pts_2d.append([x, y])
        pts_3d = np.array(pts_3d, dtype=np.float64)
        pts_2d = np.array(pts_2d, dtype=np.float64)
        cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        _, rvec, _ = cv2.solvePnP(pts_3d, pts_2d, cam,
                                  np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        return float(angles[1])

    def close(self):
        self.landmarker.close()


class CursorController:
    def __init__(self, centers):
        self.centers = centers
        self.active_idx = None

    def jump_to(self, idx):
        if idx == self.active_idx or idx >= len(self.centers):
            return
        cx, cy = self.centers[idx]
        move_cursor(cx, cy, duration=config.GAZE["jump_duration"])
        self.active_idx = idx


def _make_icon_image(color):
    img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((1, 1, 14, 14), fill=color + (255,))
    return img


class GazeApp:
    def __init__(self):
        self.enabled = config.GAZE["enabled_on_start"]
        self.cam_released = False
        self.running = True
        self.lock = threading.Lock()
        self.cap = None
        self.detector = None
        self.cursor = None
        self.tray = None
        self._last_yaw = None

    def _open_camera(self):
        cap = cv2.VideoCapture(config.SHARED["webcam_index"])
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.SHARED["cam_width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.SHARED["cam_height"])
            cap.set(cv2.CAP_PROP_FPS, config.SHARED["cam_fps"])
        return cap

    def release_camera(self):
        with self.lock:
            if not self.cam_released and self.cap is not None:
                self.cap.release()
                self.cam_released = True
                log("Camera released.")
                self._update_tray()

    def reclaim_camera(self):
        with self.lock:
            if self.cam_released:
                self.cap = self._open_camera()
                self.cam_released = False
                log("Camera reclaimed.")
                self._update_tray()

    def toggle_camera(self):
        if self.cam_released:
            self.reclaim_camera()
        else:
            self.release_camera()

    def toggle_enabled(self):
        self.enabled = not self.enabled
        log(f"Gaze switch {'ENABLED' if self.enabled else 'DISABLED'}")
        self._update_tray()

    def _tray_title(self):
        if self.cam_released:
            return "GazeSwitch — Camera released"
        return "GazeSwitch — " + ("Active" if self.enabled else "Disabled")

    def _update_tray(self):
        if self.tray is None:
            return
        if self.cam_released:
            color = (180, 100, 0)
        elif self.enabled:
            color = (0, 200, 100)
        else:
            color = (120, 120, 120)
        self.tray.icon = _make_icon_image(color)
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

        hk = config.SHARED
        return pystray.Menu(
            pystray.MenuItem(
                lambda item: f"{'Disable' if self.enabled else 'Enable'} Gaze Switch ({hk['hotkey_gaze']})",
                on_toggle,
            ),
            pystray.MenuItem(
                lambda item: f"{'Reclaim' if self.cam_released else 'Release'} Camera ({hk['hotkey_camera']})",
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

    def _handle_gaze(self, yaw):
        eff_yaw = -yaw if config.GAZE["invert_yaw"] else yaw
        if self._last_yaw is None or abs(yaw - self._last_yaw) > 0.5:
            log(f"yaw raw={yaw:+.1f} eff={eff_yaw:+.1f} active={self.cursor.active_idx}")
            self._last_yaw = yaw
        thr_left = config.GAZE["yaw_threshold_left"]
        release = config.GAZE["yaw_release"]
        active = self.cursor.active_idx
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
            self.cursor.jump_to(screen_idx)

    def _loop(self):
        while self.running:
            with self.lock:
                released = self.cam_released
                cap = self.cap

            if released or cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            yaw = self.detector.get_yaw(frame)

            if self.enabled and yaw is not None:
                self._handle_gaze(yaw)

            if DEBUG:
                state = "ACTIVE" if self.enabled else "DISABLED"
                cv2.putText(frame, f"GAZE {state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"{config.SHARED['hotkey_gaze']}=Toggle  "
                                   f"{config.SHARED['hotkey_camera']}=Camera  Q=Quit",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("GazeSwitch (debug)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    if self.tray:
                        self.tray.stop()
                    break

    def run(self):
        if not _download(config.FACE_MODEL_URL, config.SHARED["face_model_file"], "face"):
            sys.exit(1)

        pyautogui.FAILSAFE = False

        monitors, centers = get_sorted_monitors()
        if len(monitors) < 2:
            print("Gaze switch needs 2+ monitors.")
            sys.exit(1)

        self.detector = FaceYawDetector(config.SHARED["face_model_file"])
        self.cursor = CursorController(centers)
        self.cap = self._open_camera()
        if not self.cap.isOpened():
            print("Could not open webcam.")
            sys.exit(1)

        keyboard.add_hotkey(config.SHARED["hotkey_gaze"], self.toggle_enabled)
        keyboard.add_hotkey(config.SHARED["hotkey_camera"], self.toggle_camera)

        t_tray = threading.Thread(target=self._run_tray, daemon=True)
        t_tray.start()

        log(f"GazeSwitch running. {config.SHARED['hotkey_gaze']}=Toggle  "
            f"{config.SHARED['hotkey_camera']}=Camera")

        self._loop()

        self.running = False
        keyboard.unhook_all()
        if self.cap is not None:
            self.cap.release()
        self.detector.close()
        if DEBUG:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import signal
    app = GazeApp()
    signal.signal(signal.SIGINT,
                  lambda s, f: (setattr(app, "running", False),
                                app.tray and app.tray.stop()))
    app.run()
