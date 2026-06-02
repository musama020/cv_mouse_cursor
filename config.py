"""
GazeControl — user-tunable settings.

Edit the values below to adjust behaviour, then restart the app.
Nothing else in the project needs editing for normal tuning.
"""

# ─────────────────────────────────────────────────────────
# GAZE SWITCH  (face-direction multi-monitor cursor jump)
# ─────────────────────────────────────────────────────────
GAZE = {
    "enabled_on_start":    True,    # start with gaze control active
    "yaw_threshold_left":  10.0,    # look-left angle (deg) that jumps to the left monitor
    "yaw_threshold_right": 0.5,     # look-right angle (deg) that jumps to the right monitor
    "yaw_release":         1.0,     # dead-zone (deg) before the cursor will jump again
    "invert_yaw":          True,    # flip left/right if your camera mirrors you
    "jump_duration":       0.0,     # seconds to glide the cursor (0 = instant teleport)
}

# ─────────────────────────────────────────────────────────
# FINGER SCROLL  (index-finger curl = scroll direction)
# ─────────────────────────────────────────────────────────
SCROLL = {
    "enabled_on_start":    True,    # start with finger scroll active
    "straight_angle":      160.0,   # avg index-joint angle (deg) at/above which = straight = UP
    "hysteresis":          12.0,    # deg of stickiness so direction doesn't flicker at the line
    "scroll_up_speed":     75,      # fixed clicks per tick when scrolling UP
    "scroll_down_speed":   75,      # fixed clicks per tick when scrolling DOWN
    "smoothing":           0.6,     # EMA factor for the straightness value (0 = raw, →1 = smooth)

    # Open-palm / wrist pause: show an open hand (4 fingers up) or tilt the
    # hand sideways past `pause_tilt` to pause scrolling; close back to a
    # pointing index (and bring it upright past `pause_tilt_release`) to resume.
    "pause_on_tilt":       True,
    "pause_tilt":          55.0,    # deg from vertical → pause when tilted past this
    "pause_tilt_release":  45.0,    # deg from vertical → resume when uprighter than this

    # Optional extras (off by default):
    "gesture_mode":          False, # enable 2/3-finger extras
    "gesture_mode_speed_mult": 2,   # speed multiplier when 2 fingers raised
    "_gesture_mode_note":    "3 fingers up = jump to top (Ctrl+Home)",
}

# ─────────────────────────────────────────────────────────
# SHARED  (camera, hotkeys, detection tuning)
# ─────────────────────────────────────────────────────────
SHARED = {
    "webcam_index":             0,       # which camera (0 = default)
    "cam_width":                640,
    "cam_height":               480,
    "cam_fps":                  30,

    "min_detection_confidence": 0.6,
    "min_tracking_confidence":  0.6,

    # Global hotkeys (work anywhere in Windows):
    "hotkey_gaze":   "F9",    # toggle gaze switch on/off
    "hotkey_scroll": "F10",   # toggle finger scroll on/off
    "hotkey_camera": "F8",    # release / reclaim the webcam (for Meet / Zoom)

    # Model files (auto-downloaded on first run if missing):
    "face_model_file": "face_landmarker.task",
    "hand_model_file": "hand_landmarker.task",
}

# ─────────────────────────────────────────────────────────
# MODEL DOWNLOAD URLs  (do not normally need to change)
# ─────────────────────────────────────────────────────────
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
