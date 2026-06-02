# Plan — Finger Gesture Scrolling (`fingerscroll.py`)

A new, **separate** Python module that adds hand/finger-gesture vertical scrolling,
built to match the conventions already used in [gazeswitch.py](gazeswitch.py)
(headless tray app, global hotkeys, camera release for Meet/Zoom, JSON config).

---

## 1. Goal

Track one hand with MediaPipe Hands. Raise the index finger and move it up/down to
scroll the active window. The feature runs as its own tray app so it can be used
independently of (or alongside) GazeSwitch.

```
Camera detects hand → track index-finger tip → moving UP = scroll up, DOWN = scroll down
```

---

## 2. Why a separate file (not editing gazeswitch.py)

- GazeSwitch owns the webcam exclusively; running both at once would fight over the
  camera. A standalone file lets the user pick **one** at a time (or we document the
  conflict and add camera-release parity so they cooperate).
- Keeps the head-yaw monitor-switching logic untouched and low-risk.
- The sample snippet's Haar-cascade + Kalman head cursor is **redundant** — GazeSwitch
  already does head control better via MediaPipe FaceLandmarker. We drop that half and
  keep **only the finger-scroll** portion.

---

## 3. What we keep / change vs. the provided snippet

| Snippet does | Our plan |
|---|---|
| Haar cascade head → cursor | **Drop** (GazeSwitch already handles head control) |
| Kalman cursor smoothing | Drop (not needed for scroll-only) |
| MediaPipe Hands, 1 hand | **Keep** |
| `count_fingers()` helper | **Keep** (drives gesture modes) |
| Index-tip delta → `pyautogui.scroll()` | **Keep**, but smooth + add cooldown |
| `cv2.imshow` preview window | **Drop** for normal run; keep behind a `--debug` flag |
| Hardcoded constants | Move into a `CONFIG` dict + JSON file, like GazeSwitch |
| Bare `while True` loop | Wrap in a tray app class with enable/disable + camera release |

---

## 4. Module structure (`fingerscroll.py`)

Mirror GazeSwitch's layout for consistency:

1. **Header docstring** — usage, deps, `pythonw fingerscroll.py` to run silently.
2. **CONFIG dict** — defaults:
   - `scroll_threshold` (px of finger travel to trigger) — default `15`
   - `scroll_speed` (clicks per trigger) — default `3`
   - `cooldown_frames` — default `5`
   - `min_detection_confidence` / `min_tracking_confidence` — `0.7`
   - `webcam_index`, `hotkey_toggle` (F10 — avoid clash with GazeSwitch F9),
     `hotkey_camera` (F11), `config_file` = `fingerscroll_config.json`
   - `gesture_mode` toggle for the multi-finger extras (off by default)
3. **`count_fingers(hand_landmarks)`** — from snippet (tips 8/12/16/20 vs PIP joints).
4. **`HandScrollDetector`** — wraps `mp.solutions.hands`; `.process(frame)` returns
   `(finger_count, index_tip_y_px)` or `None` when no hand.
5. **`ScrollController`** — owns `prev_finger_y`, `cooldown`; `.update(finger_y)` emits
   `pyautogui.scroll()` calls. Uses Win32 `SetCursorPos`-style discipline? No — scroll
   uses `pyautogui.scroll`, which is fine and matches the snippet.
6. **`FingerScrollApp`** — tray app class copied/adapted from `GazeSwitchApp`:
   - tray icon (purple to distinguish from GazeSwitch green)
   - Enable/Disable, Release/Reclaim Camera, Quit
   - global hotkeys F10 / F11
   - main loop reads frames, runs detector + controller
   - optional `--debug` opens `cv2.imshow` with landmarks + HUD
7. **`__main__`** — SIGINT handler + `app.run()`, same pattern as GazeSwitch.

---

## 5. Scroll logic (refined from snippet)

- Only scroll when index finger is the dominant raised finger (avoid noise from a flat
  palm). Gate on `count_fingers() >= 1` and index extended.
- Compute `delta = finger_y - prev_finger_y`.
- If `abs(delta) > scroll_threshold` and `cooldown == 0`:
  - `delta > 0` (moved down) → `pyautogui.scroll(-scroll_speed)`
  - `delta < 0` (moved up)   → `pyautogui.scroll(+scroll_speed)`
  - set `cooldown = cooldown_frames`
- Decrement cooldown each frame; reset `prev_finger_y = None` when the hand leaves frame.
- **Optional improvement:** scale `scroll_speed` by `abs(delta)` so a bigger sweep
  scrolls faster (proportional rather than fixed).

### Optional gesture extras (behind `gesture_mode`, default off)
- 2 fingers up → fast scroll (×3 speed)
- 3 fingers up → jump to top (`Home` / `Ctrl+Home`)
- Pinch (thumb-tip ↔ index-tip distance) → zoom (`Ctrl + scroll`)

These are stubbed/flagged so the first version stays simple and testable.

---

## 6. Dependencies

Already present for GazeSwitch; only confirm `mediapipe` Hands + `pyautogui`:

```
pip install opencv-python mediapipe pyautogui keyboard numpy pystray pillow
```

(`myenv/` already has these — verify before adding anything.)

---

## 7. Camera coexistence with GazeSwitch

Both apps grab webcam index 0. Options, in order of preference:
1. **Document** that you run one at a time; both expose F-key camera release.
2. (Later) a tiny shared lock-file so whichever starts second backs off.

v1 ships option 1.

---

## 8. Build & verification steps

1. Write `fingerscroll.py` per the structure above.
2. Smoke test imports: `python -c "import fingerscroll"` (no camera grab on import).
3. Run `python fingerscroll.py --debug`, confirm:
   - hand landmarks draw, finger count shows
   - index up-sweep scrolls a long page up, down-sweep scrolls down
   - F10 toggles enable, F11 releases/reclaims camera, tray menu works
4. Tune `scroll_threshold` / `scroll_speed` for comfort; save to JSON config.
5. Run headless via `pythonw fingerscroll.py` and confirm tray-only operation.

---

## 9. Out of scope (v1)

- Horizontal scrolling
- Mouse-button clicks via gestures
- Running head-cursor + finger-scroll in the *same* process (camera contention)
- Multi-hand support
