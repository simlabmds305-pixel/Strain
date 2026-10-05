#!/usr/bin/env python3
# ============================================================================
#  CROP PICKER
#
#  Drag a box round the bead on the first image of a set; it prints the --roi
#  flag to paste into run_local.py.  The same crop is then used for every image
#  in that set.
#
#      python pick_roi.py /path/to/set
#
#  run_local.py --pick-roi calls this and goes straight on with the answer, so
#  you normally do not need to run it yourself.
#
#  Uses matplotlib rather than cv2.selectROI, because matplotlib's Tk backend
#  ships with Python on Windows while a cv2 GUI needs opencv-python built with
#  highgui -- pip's opencv-python-headless has no window at all, and that is a
#  common thing to have installed without knowing it.
# ============================================================================
from __future__ import annotations

import os
import sys

import cv2
import numpy as np


def _clamp_roi(x0, x1, y0, y1, W, H):
    """Order, round and clip a dragged box to the image.  Returns (x0,x1,y0,y1)."""
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(int(W), x1), min(int(H), y1)
    return x0, x1, y0, y1


def roi_is_sane(roi, W, H, min_px=20):
    x0, x1, y0, y1 = roi
    return (x1 - x0) >= min_px and (y1 - y0) >= min_px


def pick(path, start_roi=None):
    """Show the image, let the user drag a box, return (x0, x1, y0, y1) or None."""
    import matplotlib
    # An interactive backend is required here; run_local.py forces Agg for its
    # own plotting, so ask for a real one before pyplot is touched.
    if matplotlib.get_backend().lower() == "agg":
        for backend in ("TkAgg", "QtAgg", "MacOSX"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
    if matplotlib.get_backend().lower() == "agg":
        raise RuntimeError(
            "matplotlib has no window backend here (still on Agg), so the crop "
            "cannot be drawn. Either install one (pip install pyqt5) or pass the "
            "crop directly with --roi x0,x1,y0,y1")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    chosen = {}

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(rgb)
    ax.set_title(
        f"{os.path.basename(path)}   ({W} x {H})\n"
        "Drag a box round the BEAD: background above the apex and on both sides,\n"
        "the bottom edge on or just below the mat.  Redrag to redo.  Close the window when done.",
        fontsize=10)
    ax.set_axis_off()

    def on_select(epress, erelease):
        roi = _clamp_roi(epress.xdata, erelease.xdata, epress.ydata, erelease.ydata, W, H)
        if roi_is_sane(roi, W, H):
            chosen["roi"] = roi
            x0, x1, y0, y1 = roi
            ax.set_xlabel(f"--roi {x0},{x1},{y0},{y1}      "
                          f"({x1-x0} x {y1-y0} px)", fontsize=12, color="#0b0b0b")
            ax.set_frame_on(False)
            fig.canvas.draw_idle()

    sel = RectangleSelector(ax, on_select, useblit=True, button=[1],
                            minspanx=20, minspany=20, spancoords="pixels",
                            interactive=True,
                            props=dict(facecolor="none", edgecolor="#eb6834", lw=2))
    if start_roi:
        x0, x1, y0, y1 = start_roi
        sel.extents = (x0, x1, y0, y1)
        chosen["roi"] = tuple(start_roi)

    plt.tight_layout()
    plt.show()          # blocks until the window is closed
    return chosen.get("roi")


def pick_for_folder(folder, pattern="*", start_roi=None):
    """Pick a crop on the first image of a folder."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bead_pipeline as bp
    paths = bp.list_frames(folder, pattern)
    if not paths:
        raise SystemExit(f"no images found in {folder}")
    return pick(paths[0], start_roi)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: python pick_roi.py <folder-or-image>")
    target = argv[0]
    roi = pick(target) if os.path.isfile(target) else pick_for_folder(target)
    if not roi:
        print("no box drawn - nothing to use (the whole frame will be used "
              "if you run without --roi)")
        return
    x0, x1, y0, y1 = roi
    print()
    print(f"  --roi {x0},{x1},{y0},{y1}")
    print()
    print("  paste that onto the run_local.py command for this set, e.g.")
    print(f"    python run_local.py \"{target}\" --roi {x0},{x1},{y0},{y1} --interval 30")


if __name__ == "__main__":
    main()
