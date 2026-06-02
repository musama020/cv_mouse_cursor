import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import pyautogui
import keyboard
import math
import time
import threading
import urllib.request
import importlib

import config

try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    print("Missing dependencies. Run:  pip install -r requirement.txt")
    sys.exit(1)

DEBUG = "--debug" in sys.argv


def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


WRIST = 0
INDEX_TIP = 8
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
INDEX_CHAIN = [5, 6, 7, 8]


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

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
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


class ScrollApp:
    def __init__(self):
        self.enabled = config.SCROLL["enabled_on_start"]
        self.cam_released = False
        self.running = True
        self.lock = threading.Lock()
        self.cap = None
        self.detector = None
        self.controller = None
        self.tray = None
        self._top_fired = False
        self._tilt_paused = False
        self._tilt = 0.0

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
                if self.controller:
                    self.controller.reset()
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
        if not self.enabled and self.controller:
            self.controller.reset()
        log(f"Finger scroll {'ENABLED' if self.enabled else 'DISABLED'}")
        self._update_tray()

    def _tray_title(self):
        if self.cam_released:
            return "FingerScroll — Camera released"
        return "FingerScroll — " + ("Active" if self.enabled else "Disabled")

    def _update_tray(self):
        if self.tray is None:
            return
        if self.cam_released:
            color = (180, 100, 0)
        elif self.enabled:
            color = (150, 80, 220)
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
                lambda item: f"{'Disable' if self.enabled else 'Enable'} Finger Scroll ({hk['hotkey_scroll']})",
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
            "FingerScroll",
            _make_icon_image((150, 80, 220)),
            self._tray_title(),
            menu=self._build_menu(),
        )
        self.tray.run()

    def _handle_scroll(self, hand, frame):
        landmarks = hand
        finger_count = count_fingers(landmarks)
        self._tilt = hand_tilt(landmarks)
        if config.SCROLL["pause_on_tilt"]:
            open_palm = finger_count >= 4
            if self._tilt_paused:
                if self._tilt < config.SCROLL["pause_tilt_release"] and not open_palm:
                    self._tilt_paused = False
            elif self._tilt > config.SCROLL["pause_tilt"] or open_palm:
                self._tilt_paused = True

        action = ""
        if self.enabled and not self._tilt_paused:
            if config.SCROLL["gesture_mode"] and finger_count >= 3:
                if not self._top_fired:
                    pyautogui.hotkey("ctrl", "home")
                    self._top_fired = True
                action = "TOP"
            else:
                self._top_fired = False
                straightness = index_straightness(landmarks)
                action = self.controller.update(straightness, finger_count)
        elif self._tilt_paused:
            self.controller.reset()
            action = "PAUSED"

        if DEBUG and frame is not None:
            self.detector.draw(frame, hand)
            cv2.putText(frame, f"scroll: {action} tilt={self._tilt:.0f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return action

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
            hand = self.detector.process(frame)

            if hand is not None:
                self._handle_scroll(hand, frame if DEBUG else None)
            else:
                self.controller.reset()
                self._top_fired = False
                self._tilt_paused = False

            if DEBUG:
                state = "ACTIVE" if self.enabled else "DISABLED"
                cv2.putText(frame, f"SCROLL {state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"{config.SHARED['hotkey_scroll']}=Toggle  "
                                   f"{config.SHARED['hotkey_camera']}=Camera  Q=Quit",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("FingerScroll (debug)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    if self.tray:
                        self.tray.stop()
                    break

    def run(self):
        if not _download(config.HAND_MODEL_URL, config.SHARED["hand_model_file"], "hand"):
            sys.exit(1)

        pyautogui.FAILSAFE = False

        self.detector = HandScrollDetector(config.SHARED["hand_model_file"])
        self.controller = ScrollController()
        self.cap = self._open_camera()
        if not self.cap.isOpened():
            print("Could not open webcam.")
            sys.exit(1)

        keyboard.add_hotkey(config.SHARED["hotkey_scroll"], self.toggle_enabled)
        keyboard.add_hotkey(config.SHARED["hotkey_camera"], self.toggle_camera)

        t_tray = threading.Thread(target=self._run_tray, daemon=True)
        t_tray.start()

        log(f"FingerScroll running. {config.SHARED['hotkey_scroll']}=Toggle  "
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
    app = ScrollApp()
    signal.signal(signal.SIGINT,
                  lambda s, f: (setattr(app, "running", False),
                                app.tray and app.tray.stop()))
    app.run()
