"""
GazeSwitch — Face Direction Multi-Monitor Cursor Control
---------------------------------------------------------
Setup:   [Dell P2419H LCD]  |  [Laptop Screen]
          LEFT (PRIMARY)         RIGHT (x=2880)

Look LEFT  → cursor jumps to Dell LCD
Look RIGHT → cursor returns to Laptop screen

Controls:
  F9  → toggle on/off
  Q   → quit (in preview window)
  C   → run calibration (in preview window)

Requirements:
  pip install opencv-python mediapipe pyautogui screeninfo keyboard numpy

Model file (download once, place in same folder):
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  Save as: face_landmarker.task
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
import urllib.request
import ctypes
import ctypes.wintypes
from collections import deque

# Tell Windows this process is DPI-aware per monitor so that
# EnumDisplayMonitors / GetMonitorInfoW return logical (SetCursorPos-compatible)
# coordinates rather than raw physical pixels.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()       # fallback for older Windows

# ─────────────────────────────────────────────────────────
# CONFIG  — tweak these if behavior feels off
# ─────────────────────────────────────────────────────────
CONFIG = {
    "yaw_threshold_left":  10.0,  # degrees to jump to left monitor (Dell LCD)
    "yaw_threshold_right": 5.0,   # degrees to jump to right monitor (Laptop)
    "yaw_release":         3,     # degrees: yaw must return within this before next jump fires
    "invert_yaw":         True,  # set True if directions are swapped (webcam mirroring)
    "manual_pause_sec":   0.0,   # 0 = disabled; >0 pauses after manual mouse move
    "jump_duration":      0.0,   # 0 = instant snap (ms response); >0 animates
    "webcam_index":       0,     # 0 = built-in webcam
    "hotkey_toggle":      "F9",
    "config_file":        "gazeswitch_config.json",
    "model_file":         "face_landmarker.task",
    "preview_scale":      0.7,
}

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# ─────────────────────────────────────────────────────────
# AUTO-DOWNLOAD MODEL IF MISSING
# ─────────────────────────────────────────────────────────
def ensure_model():
    model_path = CONFIG["model_file"]
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
        return True

    print(f"📥 Downloading face landmark model (~28MB)...")
    print(f"   From: {MODEL_URL}")
    try:
        def progress(count, block_size, total_size):
            pct = int(count * block_size * 100 / total_size)
            pct = min(pct, 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r   [{bar}] {pct}%", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=progress)
        print(f"\n✅ Model downloaded: {model_path}\n")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(f"\n   Please manually download the model file:")
        print(f"   URL: {MODEL_URL}")
        print(f"   Save as: face_landmarker.task")
        print(f"   Place in: {os.getcwd()}\n")
        return False

# ─────────────────────────────────────────────────────────
# MONITOR LAYOUT DETECTION
# ─────────────────────────────────────────────────────────
class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize",     ctypes.c_ulong),
                ("rcMonitor",  ctypes.wintypes.RECT),
                ("rcWork",     ctypes.wintypes.RECT),
                ("dwFlags",    ctypes.c_ulong)]

def _enum_logical_monitors():
    """Return logical monitor rects compatible with SetCursorPos.

    SetThreadDpiAwarenessContext(-4) = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
    makes GetMonitorInfoW return coordinates in the same space SetCursorPos uses,
    regardless of what process-level DPI awareness mediapipe/OpenCV set at import.
    """
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
    # Temporarily switch this thread to system-DPI-aware so EnumDisplayMonitors
    # returns the same logical coordinate space that SetCursorPos operates in.
    set_ctx = ctypes.windll.user32.SetThreadDpiAwarenessContext
    set_ctx.restype = ctypes.c_void_p
    prev = set_ctx(ctypes.c_void_p(-2))   # -2 = DPI_AWARENESS_CONTEXT_SYSTEM_AWARE
    try:
        ctypes.windll.user32.EnumDisplayMonitors(None, None, _PROC(_cb), 0)
    finally:
        set_ctx(ctypes.c_void_p(prev))
    return rects

def get_sorted_monitors():
    monitors = sorted(screeninfo.get_monitors(), key=lambda m: m.x)
    logical_rects = sorted(_enum_logical_monitors(), key=lambda r: r[0])  # sort by left edge

    centers = []
    print(f"\n📺 Detected {len(monitors)} monitors:")
    for i, (m, lr) in enumerate(zip(monitors, logical_rects)):
        cx = (lr[0] + lr[2]) // 2   # logical center x
        cy = (lr[1] + lr[3]) // 2   # logical center y
        centers.append((cx, cy))
        tag = "(PRIMARY)" if m.is_primary else ""
        print(f"   Monitor {i+1}: {m.width}x{m.height} phys @ ({m.x},{m.y}) {tag}")
        print(f"             logical rect ({lr[0]},{lr[1]})-({lr[2]},{lr[3]})  cursor_center=({cx},{cy})")
    return monitors, centers

def move_cursor(x, y, duration=0.0):
    """Move cursor using Win32 SetCursorPos — works across all monitors regardless of DPI."""
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
# FACE YAW DETECTOR  (new mediapipe tasks API)
# Uses facial_transformation_matrix for accurate yaw angle
# ─────────────────────────────────────────────────────────
class FaceYawDetector:
    LANDMARKS = [1, 33, 263, 61, 291, 199]   # for solvePnP fallback

    def __init__(self, model_path):
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        BaseOptions = mp.tasks.BaseOptions

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_facial_transformation_matrixes=True,  # gives us head pose directly
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.frame_ts   = 0   # timestamp counter for VIDEO mode

    def get_yaw(self, frame):
        """
        Returns (yaw_degrees, annotated_frame)
        yaw < 0 → looking LEFT
        yaw > 0 → looking RIGHT
        """
        h, w = frame.shape[:2]
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.frame_ts += 33   # ~30fps in ms

        result = self.landmarker.detect_for_video(mp_image, self.frame_ts)

        if not result.face_landmarks:
            return None, frame

        # ── Method 1: Use transformation matrix (most accurate) ──
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0].data).reshape(4, 4)
            # MediaPipe facial_transformation_matrix is a 4x4 model-to-world matrix.
            # Yaw (left/right rotation around Y-axis): atan2(-R[2,0], R[0,0])
            yaw_rad = np.arctan2(-matrix[2, 0], matrix[0, 0])
            yaw     = np.degrees(yaw_rad)
        else:
            # ── Method 2: solvePnP fallback ──
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
            cam    = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
            _, rvec, _ = cv2.solvePnP(pts_3d, pts_2d, cam,
                                       np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
            rmat, _    = cv2.Rodrigues(rvec)
            angles, *_ = cv2.RQDecomp3x3(rmat)
            yaw        = angles[1]

        # Draw nose arrow indicator
        lm      = result.face_landmarks[0]
        nose_x  = int(lm[1].x * w)
        nose_y  = int(lm[1].y * h)
        arrow_x = int(nose_x + yaw * 2.5)
        cv2.arrowedLine(frame, (nose_x, nose_y), (arrow_x, nose_y),
                        (0, 255, 255), 2, tipLength=0.3)
        cv2.circle(frame, (nose_x, nose_y), 4, (0, 255, 255), -1)

        return yaw, frame

    def close(self):
        self.landmarker.close()

# ─────────────────────────────────────────────────────────
# DIRECTION STABILIZER
# ─────────────────────────────────────────────────────────
class DirectionStabilizer:
    def __init__(self):
        self.buffer = deque(maxlen=CONFIG["stability_frames"])

    def update(self, screen_idx):
        self.buffer.append(screen_idx)
        if len(self.buffer) < self.buffer.maxlen:
            return None

        counts  = {}
        for d in self.buffer:
            counts[d] = counts.get(d, 0) + 1
        dominant = max(counts, key=counts.get)
        ratio    = counts[dominant] / len(self.buffer)
        return dominant if ratio >= CONFIG["stability_ratio"] else None

    def reset(self):
        self.buffer.clear()

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
        labels = ["Laptop (RIGHT)", "Dell LCD (LEFT)"]
        label  = labels[idx] if idx < len(labels) else f"Monitor {idx+1}"
        print(f"   → Jumped to {label}")

# ─────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────
def run_calibration(cap, detector):
    print("\n🎯 CALIBRATION — follow on-screen instructions")
    samples  = {"left": [], "right": []}
    phase    = "left"
    start_t  = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame      = cv2.flip(frame, 1)
        yaw, frame = detector.get_yaw(frame)
        elapsed    = time.time() - start_t

        if phase == "left":
            msg1 = "Look at LEFT screen (Dell LCD)"
            msg2 = f"Get ready... {max(0, int(3-elapsed))}s" if elapsed < 3 \
                   else f"COLLECTING... ({len(samples['left'])} samples)"
            if elapsed >= 3 and yaw is not None:
                samples["left"].append(yaw)
            if elapsed >= 7:
                phase   = "right"
                start_t = time.time()
        else:
            msg1 = "Look at RIGHT screen (Laptop)"
            msg2 = f"Get ready... {max(0, int(3-elapsed))}s" if elapsed < 3 \
                   else f"COLLECTING... ({len(samples['right'])} samples)"
            if elapsed >= 3 and yaw is not None:
                samples["right"].append(yaw)
            if elapsed >= 7:
                break

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 120), (0,0,0), -1)
        cv2.putText(frame, "CALIBRATION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.putText(frame, msg1, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, msg2, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)
        if yaw is not None:
            cv2.putText(frame, f"Yaw: {yaw:+.1f}", (w-150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        small = cv2.resize(frame, (int(w * CONFIG["preview_scale"]),
                                   int(h * CONFIG["preview_scale"])))
        cv2.imshow("GazeSwitch", small)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    if samples["left"] and samples["right"]:
        avg_left  = float(np.mean(samples["left"]))
        avg_right = float(np.mean(samples["right"]))
        threshold = abs((avg_left + avg_right) / 2)

        print(f"\n   Left  yaw avg : {avg_left:+.1f}°")
        print(f"   Right yaw avg : {avg_right:+.1f}°")
        print(f"   Threshold set : ±{threshold:.1f}°")

        with open(CONFIG["config_file"], "w") as f:
            json.dump({"yaw_threshold_left": threshold, "yaw_threshold_right": threshold}, f, indent=2)

        CONFIG["yaw_threshold_left"]  = threshold
        CONFIG["yaw_threshold_right"] = threshold
        print("   ✅ Calibration saved!\n")
    else:
        print("   ❌ Not enough samples — try again\n")

# ─────────────────────────────────────────────────────────
# HUD OVERLAY
# ─────────────────────────────────────────────────────────
def draw_hud(frame, yaw, screen_idx, stable_idx, enabled, paused):
    h, w       = frame.shape[:2]
    threshold  = CONFIG["yaw_threshold_left"]

    # Top bar
    cv2.rectangle(frame, (0,0), (w, 115), (15,15,15), -1)
    cv2.line(frame, (0,115), (w,115), (60,60,60), 1)

    # Status
    if not enabled:
        st, sc = "DISABLED  (F9 to enable)", (60,60,220)
    elif paused:
        st, sc = "PAUSED  (manual mouse)", (60,160,220)
    else:
        st, sc = "ACTIVE", (60,220,60)

    cv2.putText(frame, f"GazeSwitch  |  {st}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)

    # Yaw reading
    if yaw is not None:
        eff = -yaw if CONFIG["invert_yaw"] else yaw
        if   eff < -threshold: direction, dc = "◄ LEFT  (Dell LCD)",   (80,80,255)
        elif eff >  threshold: direction, dc = "RIGHT (Laptop) ►",     (255,80,80)
        else:                  direction, dc = "CENTER",                (180,180,180)
        cv2.putText(frame, f"Yaw: {yaw:+.1f}°   {direction}",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, dc, 2)

    # Active screen
    if stable_idx is not None:
        labels = ["Laptop (RIGHT)", "Dell LCD (LEFT)"]
        label  = labels[stable_idx] if stable_idx < len(labels) else f"Monitor {stable_idx+1}"
        cv2.putText(frame, f"Cursor on: {label}",
                    (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,220,180), 2)

    # Gauge bar
    by   = h - 22
    bx1  = 50
    bx2  = w - 50
    bmid = (bx1 + bx2) // 2
    cv2.rectangle(frame, (bx1, by-8), (bx2, by+8), (40,40,40), -1)
    cv2.rectangle(frame, (bx1, by-8), (bx2, by+8), (90,90,90), 1)

    # Threshold lines
    lmark = bmid + int((-threshold/50)*(bmid-bx1))
    rmark = bmid + int(( threshold/50)*(bx2-bmid))
    cv2.line(frame, (lmark, by-10), (lmark, by+10), (0,200,255), 2)
    cv2.line(frame, (rmark, by-10), (rmark, by+10), (0,200,255), 2)
    cv2.putText(frame, "L", (bx1-18, by+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
    cv2.putText(frame, "R", (bx2+5,  by+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    # Moving dot
    if yaw is not None:
        yc    = max(-50, min(50, yaw))
        dot_x = bmid + int((yc/50)*(bx2-bmid))
        dot_x = max(bx1+8, min(bx2-8, dot_x))
        col   = (80,80,255) if yaw < -threshold else \
                (255,80,80) if yaw >  threshold else \
                (80,255,80)
        cv2.circle(frame, (dot_x, by), 9, col, -1)

    # Controls hint
    cv2.putText(frame, "F9: toggle   C: calibrate   Q: quit",
                (10, h-38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,120,120), 1)
    return frame

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 52)
    print("  GazeSwitch — Face Direction Cursor Control")
    print("=" * 52)

    # Ensure model file exists (auto-download)
    if not ensure_model():
        sys.exit(1)

    # Load saved calibration
    if os.path.exists(CONFIG["config_file"]):
        with open(CONFIG["config_file"]) as f:
            saved = json.load(f)
            CONFIG["yaw_threshold_left"]  = saved.get("yaw_threshold_left",  CONFIG["yaw_threshold_left"])
            CONFIG["yaw_threshold_right"] = saved.get("yaw_threshold_right", CONFIG["yaw_threshold_right"])
        print(f"📂 Calibration loaded: left=±{CONFIG['yaw_threshold_left']:.1f}°  right=±{CONFIG['yaw_threshold_right']:.1f}°")

    # Monitors
    monitors, centers = get_sorted_monitors()
    if len(monitors) < 2:
        print("\n❌ Only 1 monitor detected.")
        print("   Make sure Dell LCD is connected & set to Extend.")
        sys.exit(1)

    print(f"\n🖥️  Your layout:")
    print(f"   [Dell LCD]  x={monitors[0].x}  ← Look LEFT")
    print(f"   [Laptop  ]  x={monitors[1].x}  ← Look RIGHT/CENTER")

    # Init
    print("\n⏳ Loading face landmark model...")
    detector   = FaceYawDetector(CONFIG["model_file"])
    controller = CursorController(monitors, centers)

    cap = cv2.VideoCapture(CONFIG["webcam_index"])
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    enabled = True

    def toggle():
        nonlocal enabled
        enabled = not enabled
        print(f"   GazeSwitch {'ENABLED ✅' if enabled else 'DISABLED ⏸️'}")

    keyboard.add_hotkey(CONFIG["hotkey_toggle"], toggle)

    print(f"\n✅ Running!  Left threshold={CONFIG['yaw_threshold_left']:.1f}°  Right threshold={CONFIG['yaw_threshold_right']:.1f}°")
    print(f"   F9=toggle  |  C=calibrate  |  Q=quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        yaw        = None
        screen_idx = None

        yaw, frame = detector.get_yaw(frame)
        if enabled and yaw is not None:
            thr_left  = CONFIG["yaw_threshold_left"]
            thr_right = CONFIG["yaw_threshold_right"]
            release   = CONFIG["yaw_release"]
            eff_yaw   = -yaw if CONFIG["invert_yaw"] else yaw
            active    = controller.active_idx

            if eff_yaw < -thr_left:
                screen_idx = 0   # left monitor
            elif eff_yaw > thr_right:
                screen_idx = 1   # right monitor
            # hysteresis: once on a screen, only jump away after yaw clears release zone
            elif active == 0 and eff_yaw > -release:
                screen_idx = None
            elif active == 1 and eff_yaw < release:
                screen_idx = None

            if screen_idx is not None:
                controller.jump_to(screen_idx)

        yaw_str = f"{yaw:+.1f}" if yaw is not None else "  --"
        print(f"\r  yaw={yaw_str}°  active={controller.active_idx}    ",
              end="", flush=True)

        # Draw HUD
        scale = CONFIG["preview_scale"]
        small = cv2.resize(frame,
                           (int(frame.shape[1]*scale),
                            int(frame.shape[0]*scale)))
        small = draw_hud(small, yaw, screen_idx, screen_idx,
                         enabled, False)
        cv2.imshow("GazeSwitch", small)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            run_calibration(cap, detector)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 GazeSwitch closed.")

if __name__ == "__main__":
    main()
