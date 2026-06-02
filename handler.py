"""
GazeControl launcher / dispatcher.

Usage:
    python handler.py both    [--debug]   # gaze + scroll together (default)
    python handler.py gaze    [--debug]   # gaze switch only
    python handler.py scroll  [--debug]   # finger scroll only

    --debug  shows a live camera preview window and prints logs to console.
    Without it, the app runs silently in the system tray.

Background (silent) launch is done via start.vbs, which calls:
    pythonw handler.py both
"""

import os
import sys
import runpy

APPS = {
    "both":   "combined",
    "gaze":   "gaze",
    "scroll": "scroll",
}

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]
    mode = "both"
    for a in args:
        if a in APPS:
            mode = a
            break
    if mode not in APPS:
        print(__doc__)
        sys.exit(1)

    app_module = os.path.join(ROOT, "apps", APPS[mode] + ".py")

    # Hand the remaining flags (e.g. --debug) to the app via sys.argv.
    forwarded = [a for a in args if a not in APPS]
    sys.argv = [app_module] + forwarded

    runpy.run_path(app_module, run_name="__main__")


if __name__ == "__main__":
    main()
