# GazeControl

Hands-free Windows control with your webcam. Two features in one app:

- **Gaze Switch** — turn your head left/right to jump the mouse cursor between monitors.
- **Finger Scroll** — point your index finger; straighten it to scroll **up**, curl it to scroll **down**. Show an open palm to **pause**.

Both run off a **single shared webcam** and live quietly in the **system tray**. Each feature can be toggled on/off independently.

---

## 1. Install

Requires **Python 3.10+** on Windows.

```powershell
pip install -r requirement.txt
```

On first run the two MediaPipe models (~4 MB hand, ~28 MB face) download automatically into the project folder.

---

## 2. Run

### Debug mode (console logs + live camera preview)

Use this for tuning and to see what the camera sees:

```powershell
python gazecontrol.py --debug
```

- A preview window opens with on-screen state and landmarks.
- Console prints yaw angles and scroll actions.
- Press **Q** in the preview window to quit.

### Background mode (silent, no console, no camera window)

Double-click **`start.vbs`**, or run:

```powershell
wscript start.vbs
```

The app runs entirely in the system tray — no console, no preview window. This is the shippable way to run it day-to-day.

> `start.vbs` launches `pythonw gazecontrol.py`, so Python must be installed and on PATH on the target machine.

---

## 3. Controls

### Global hotkeys (work anywhere in Windows)

| Key   | Action                                              |
|-------|-----------------------------------------------------|
| **F9**  | Toggle **Gaze Switch** on/off                     |
| **F10** | Toggle **Finger Scroll** on/off                   |
| **F8**  | Release / reclaim the camera (free it for Meet/Zoom) |

### Tray menu (right-click the tray icon)

- **Enable / Disable Gaze Switch** (same as F9)
- **Enable / Disable Finger Scroll** (same as F10)
- **Release / Reclaim Camera** (same as F8)
- **Quit**

Tray icon color: **green** = at least one feature active · **grey** = both disabled · **orange** = camera released.

### Stopping one feature

Press its hotkey (**F9** for gaze, **F10** for scroll), or use the tray menu. The other feature keeps running. To stop everything, choose **Quit** in the tray menu.

---

## 4. Gestures (Finger Scroll)

| Gesture                          | Result                          |
|----------------------------------|---------------------------------|
| Index finger **straight**        | Scroll **up**                   |
| Index finger **bent / curled**   | Scroll **down**                 |
| **Open palm** (4 fingers up)     | **Pause** (resume by pointing)  |
| Tilt hand **sideways**           | **Pause** (resume upright)      |
| **Remove hand** from frame       | Stop                            |

Optional extras (enable `gesture_mode` in `config.py`):

| Gesture        | Result                |
|----------------|-----------------------|
| 2 fingers up   | Fast scroll           |
| 3 fingers up   | Jump to top (Ctrl+Home) |

---

## 5. Configuration

All tunable values live in **`config.py`** — edit it and restart the app. No other file needs touching.

- **`GAZE`** — yaw thresholds, invert direction, cursor glide duration.
- **`SCROLL`** — straight-angle threshold, scroll speeds, smoothing, pause behavior, gesture mode.
- **`SHARED`** — camera index/resolution, detection confidence, the F8/F9/F10 hotkeys, model file paths.

Each value has an inline comment explaining what it does.

---

## 6. Files

| File                  | Purpose                                          |
|-----------------------|--------------------------------------------------|
| `gazecontrol.py`      | The merged application (run this)                |
| `config.py`           | All user-tunable settings                        |
| `start.vbs`           | Silent background launcher                       |
| `requirement.txt`     | Python dependencies                              |
| `face_landmarker.task`| Face model (auto-downloaded)                     |
| `hand_landmarker.task`| Hand model (auto-downloaded)                     |

> Note: **Gaze Switch requires 2+ monitors.** With a single monitor it auto-disables and only Finger Scroll runs.
