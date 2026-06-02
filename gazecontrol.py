import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import screeninfo
import keyboard
import math
import time
import os
import sys
import threading
import urllib.request
import importlib
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


def log(*args):
    if DEBUG:
        print(*args)


WRIST = 0
INDEX_TIP = 8
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
INDEX_CHAIN = [5, 6, 7, 8]


def _download(url, path, label):
    if os.path.exists(path):
        return True
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


def ensure_models():
    ok_face = _download(config.FACE_MODEL_URL,
                        config.SHARED["face_model_file"], "face")
    ok_hand = _download(config.HAND_MODEL_URL,
                        config.SHARED["hand_model_file"], "hand")
    return ok_face, ok_hand


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


def count_fingers(landmarks):
    count = 0
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1
    return count


def _angle_at(a, b, c):
    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    dot = bax * bcx + bay * bcy
    na = (bax * bax + bay * bay) ** 0.5
    nc = (bcx * bcx + bcy * bcy) ** 0.5
    if na < 1e-6 or nc < 1e-6:
        return 180.0
    cos = max(-1.0, min(1.0, dot / (na * nc)))
    return math.degrees(math.acos(cos))


def index_straightness(landmarks):
    p = [landmarks[i] for i in INDEX_CHAIN]
    pip_angle = _angle_at(p[0], p[1], p[2])
    dip_angle = _angle_at(p[1], p[2], p[3])
    return (pip_angle + dip_angle) / 2.0


def hand_tilt(landmarks):
    wrist = landmarks[WRIST]
    base = landmarks[INDEX_CHAIN[0]]
    dx = base.x - wrist.x
    dy = base.y - wrist.y
    return math.degrees(math.atan2(abs(dx), abs(dy)))


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

    def get_yaw_from_image(self, mp_image, w, h):
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


_du = importlib.import_module("mediapipe.tasks.python.vision.drawing_utils")
_DrawingSpec = _du.DrawingSpec
_draw_landmarks = _du.draw_landmarks
_HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
_LANDMARK_SPEC = _DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
_CONNECTION_SPEC = _DrawingSpec(color=(255, 255, 0), thickness=2)


class HandScrollDetector:
    def __init__(self, model_path):
        VisionRunningMode = mp.tasks.vision.RunningMode
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        BaseOptions = mp.tasks.BaseOptions
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.SHARED["min_detection_confidence"],
            min_hand_presence_confidence=config.SHARED["min_detection_confidence"],
            min_tracking_confidence=config.SHARED["min_tracking_confidence"],
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.frame_ts = 0

    def process_image(self, mp_image):
        self.frame_ts += 33
        result = self.landmarker.detect_for_video(mp_image, self.frame_ts)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def draw(self, frame_bgr, landmarks):
        _draw_landmarks(
            frame_bgr, landmarks, _HAND_CONNECTIONS,
            landmark_drawing_spec=_LANDMARK_SPEC,
            connection_drawing_spec=_CONNECTION_SPEC,
        )

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


class ScrollController:
    def __init__(self):
        self.smooth_a = None
        self.direction = +1
        self.last_action = ""
        self.last_angle = 0.0

    def reset(self):
        self.smooth_a = None
        self.last_action = ""

    def _smooth(self, angle):
        a = config.SCROLL["smoothing"]
        if self.smooth_a is None:
            self.smooth_a = angle
        else:
            self.smooth_a = a * self.smooth_a + (1.0 - a) * angle
        return self.smooth_a

    def update(self, straightness, finger_count):
        ang = self._smooth(straightness)
        self.last_angle = ang
        thr = config.SCROLL["straight_angle"]
        hyst = config.SCROLL["hysteresis"]
        if self.direction > 0:
            if ang < thr - hyst:
                self.direction = -1
        else:
            if ang > thr + hyst:
                self.direction = +1
        if self.direction > 0:
            amount = config.SCROLL["scroll_up_speed"]
            label = "UP"
        else:
            amount = config.SCROLL["scroll_down_speed"]
            label = "DOWN"
        if config.SCROLL["gesture_mode"] and finger_count >= 2:
            amount *= config.SCROLL["gesture_mode_speed_mult"]
        pyautogui.scroll(self.direction * amount)
        self.last_action = f"{label} x{amount}"
        return self.last_action


def _make_icon_image(color):
    img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((1, 1, 14, 14), fill=color + (255,))
    return img


class GazeControlApp:
    def __init__(self):
        self.gaze_enabled = config.GAZE["enabled_on_start"]
        self.scroll_enabled = config.SCROLL["enabled_on_start"]
        self.cam_released = False
        self.running = True
        self.lock = threading.Lock()
        self.cap = None
        self.tray = None

        self.face_detector = None
        self.hand_detector = None
        self.cursor = None
        self.scroller = None
        self.has_two_monitors = False

        self._top_fired = False
        self._tilt_paused = False
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
                if self.scroller:
                    self.scroller.reset()
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

    def toggle_gaze(self):
        self.gaze_enabled = not self.gaze_enabled
        log(f"Gaze switch {'ENABLED' if self.gaze_enabled else 'DISABLED'}")
        self._update_tray()

    def toggle_scroll(self):
        self.scroll_enabled = not self.scroll_enabled
        if not self.scroll_enabled and self.scroller:
            self.scroller.reset()
        log(f"Finger scroll {'ENABLED' if self.scroll_enabled else 'DISABLED'}")
        self._update_tray()

    def _tray_title(self):
        if self.cam_released:
            return "GazeControl — Camera released"
        parts = []
        parts.append("Gaze:" + ("on" if self.gaze_enabled else "off"))
        parts.append("Scroll:" + ("on" if self.scroll_enabled else "off"))
        return "GazeControl — " + "  ".join(parts)

    def _update_tray(self):
        if self.tray is None:
            return
        if self.cam_released:
            color = (180, 100, 0)
        elif self.gaze_enabled or self.scroll_enabled:
            color = (0, 200, 100)
        else:
            color = (120, 120, 120)
        self.tray.icon = _make_icon_image(color)
        self.tray.title = self._tray_title()
        self.tray.update_menu()

    def _build_menu(self):
        def on_gaze(icon, item):
            self.toggle_gaze()

        def on_scroll(icon, item):
            self.toggle_scroll()

        def on_camera(icon, item):
            self.toggle_camera()

        def on_quit(icon, item):
            self.running = False
            icon.stop()
            os._exit(0)

        hk = config.SHARED
        return pystray.Menu(
            pystray.MenuItem(
                lambda item: f"{'Disable' if self.gaze_enabled else 'Enable'} Gaze Switch ({hk['hotkey_gaze']})",
                on_gaze,
            ),
            pystray.MenuItem(
                lambda item: f"{'Disable' if self.scroll_enabled else 'Enable'} Finger Scroll ({hk['hotkey_scroll']})",
                on_scroll,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: f"{'Reclaim' if self.cam_released else 'Release'} Camera ({hk['hotkey_camera']})",
                on_camera,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", on_quit),
        )

    def _run_tray(self):
        self.tray = pystray.Icon(
            "GazeControl",
            _make_icon_image((0, 200, 100)),
            self._tray_title(),
            menu=self._build_menu(),
        )
        self.tray.run()

    def _handle_gaze(self, yaw):
        if yaw is None:
            return
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

    def _handle_scroll(self, hand, frame):
        if hand is None:
            self.scroller.reset()
            self._top_fired = False
            self._tilt_paused = False
            return ""
        landmarks = hand
        finger_count = count_fingers(landmarks)
        tilt = hand_tilt(landmarks)
        if config.SCROLL["pause_on_tilt"]:
            open_palm = finger_count >= 4
            if self._tilt_paused:
                if tilt < config.SCROLL["pause_tilt_release"] and not open_palm:
                    self._tilt_paused = False
            elif tilt > config.SCROLL["pause_tilt"] or open_palm:
                self._tilt_paused = True

        action = ""
        if not self._tilt_paused:
            if config.SCROLL["gesture_mode"] and finger_count >= 3:
                if not self._top_fired:
                    pyautogui.hotkey("ctrl", "home")
                    self._top_fired = True
                action = "TOP"
            else:
                self._top_fired = False
                straightness = index_straightness(landmarks)
                action = self.scroller.update(straightness, finger_count)
        else:
            self.scroller.reset()
            action = "PAUSED"

        if DEBUG and frame is not None:
            self.hand_detector.draw(frame, hand)
            cv2.putText(frame, f"scroll: {action} tilt={tilt:.0f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return action

    def _loop(self):
        cam_w = config.SHARED["cam_width"]
        cam_h = config.SHARED["cam_height"]
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
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            if self.gaze_enabled and self.has_two_monitors and self.face_detector:
                yaw = self.face_detector.get_yaw_from_image(mp_image, w, h)
                self._handle_gaze(yaw)

            if self.scroll_enabled and self.hand_detector:
                hand = self.hand_detector.process_image(mp_image)
                self._handle_scroll(hand, frame if DEBUG else None)
            elif DEBUG:
                pass

            if DEBUG:
                state = (f"GAZE:{'on' if self.gaze_enabled else 'off'}  "
                         f"SCROLL:{'on' if self.scroll_enabled else 'off'}")
                cv2.putText(frame, state, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"{config.SHARED['hotkey_gaze']}=Gaze  "
                                   f"{config.SHARED['hotkey_scroll']}=Scroll  "
                                   f"{config.SHARED['hotkey_camera']}=Camera  Q=Quit",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("GazeControl (debug)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    if self.tray:
                        self.tray.stop()
                    break

    def run(self):
        ok_face, ok_hand = ensure_models()
        if not (ok_face or ok_hand):
            print("No models available — cannot start.")
            sys.exit(1)

        pyautogui.FAILSAFE = False

        monitors, centers = get_sorted_monitors()
        self.has_two_monitors = len(monitors) >= 2
        if not self.has_two_monitors:
            log("Fewer than 2 monitors detected — gaze switch disabled.")
            self.gaze_enabled = False

        if ok_face and self.has_two_monitors:
            self.face_detector = FaceYawDetector(config.SHARED["face_model_file"])
            self.cursor = CursorController(centers)
        else:
            self.gaze_enabled = False

        if ok_hand:
            self.hand_detector = HandScrollDetector(config.SHARED["hand_model_file"])
            self.scroller = ScrollController()
        else:
            self.scroll_enabled = False

        self.cap = self._open_camera()
        if not self.cap.isOpened():
            print("Could not open webcam.")
            sys.exit(1)

        keyboard.add_hotkey(config.SHARED["hotkey_gaze"], self.toggle_gaze)
        keyboard.add_hotkey(config.SHARED["hotkey_scroll"], self.toggle_scroll)
        keyboard.add_hotkey(config.SHARED["hotkey_camera"], self.toggle_camera)

        t_tray = threading.Thread(target=self._run_tray, daemon=True)
        t_tray.start()

        log("GazeControl running. Hotkeys: "
            f"{config.SHARED['hotkey_gaze']}=Gaze  "
            f"{config.SHARED['hotkey_scroll']}=Scroll  "
            f"{config.SHARED['hotkey_camera']}=Camera")

        self._loop()

        self.running = False
        keyboard.unhook_all()
        if self.cap is not None:
            self.cap.release()
        if self.face_detector:
            self.face_detector.close()
        if self.hand_detector:
            self.hand_detector.close()
        if DEBUG:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import signal
    app = GazeControlApp()
    signal.signal(signal.SIGINT,
                  lambda s, f: (setattr(app, "running", False),
                                app.tray and app.tray.stop()))
    app.run()
