#!/usr/bin/env python3
# ============================================================================
#  MANUAL TRACE  |  separate helper, not part of the main app
#
#  For sets whose first, wet photos are too dark at the base for the automatic
#  outline: trace the bead by hand on those frames and get their volume the
#  same way the app does (stack of discs, cut at the mat row).
#
#  Reads only what a normal run already wrote (analysis/settings.json for the
#  crop, analysis/per_image.csv for the mat row and the app's own volumes) and
#  writes its own folder, analysis/manual_trace/.  Nothing else is changed.
#
#      pythonw manual_trace.py               (or double-click Trace Bead.bat)
#      python  manual_trace.py <set folder>
#
#  In the window:  left click = add a point on the bead's edge
#                  hold the left button and drag = draw freehand along it
#                  right click / Backspace = undo the last click or stroke
#                  c = clear     g = brighter / normal picture
#                  Enter = save this frame and go to the next
#                  n = skip this frame      q = stop and write the results
#  Use the toolbar's zoom/pan freely; clicks only count when zoom/pan is off.
# ============================================================================
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bead_pipeline as bp   # noqa: E402  (only list_frames is used; nothing is changed)


def volume_from_outline(points, shape_hw, roi, floor_y, scale):
    """Volume (px^3, mm^3) of the closed outline `points` (full-image x, y),
    as a stack of discs one pixel thick: each row's width is its diameter --
    the same model as the app's V_disk.  Rows below the mat row are dropped."""
    x0, x1, y0, y1 = roi
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    pts = np.round(np.asarray(points, float) - [x0, y0]).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    if np.isfinite(floor_y):
        cut = int(round(floor_y - y0)) + 1
        if 0 <= cut < h:
            m[cut:] = 0
    rows = np.flatnonzero(m.any(axis=1))
    if not rows.size:
        return 0.0, 0.0, m
    sub = m[rows] > 0
    left = sub.argmax(axis=1)
    right = sub.shape[1] - 1 - sub[:, ::-1].argmax(axis=1)
    widths = (right - left + 1).astype(float)
    v_px = float(np.pi * np.sum((widths / 2.0) ** 2))
    return v_px, v_px / scale ** 3 / 1e9, m


def outline_of(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea)[:, 0, :] if cs else None


class Tracer:
    def __init__(self, folder, frames, df, roi, scale):
        import matplotlib
        if matplotlib.get_backend().lower() == "agg":
            matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        # free the keys used here from matplotlib's own shortcuts (g = grid,
        # c / Backspace = back, q = quit ...), so they do only what is listed
        mine = {"g", "c", "n", "q", "enter", "backspace"}
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[k] = [v for v in plt.rcParams[k] if v not in mine]
        self.plt = plt
        self.folder, self.frames, self.df, self.roi, self.scale = folder, frames, df, roi, scale
        self.results = {}
        self.k = 0
        self.bright = True
        self.stop = False
        self.fig, self.ax = plt.subplots(figsize=(13, 7.5))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.dragging = False
        self.show()

    # ---------------------------------------------------------------- frame
    def row(self, name):
        r = self.df[self.df["image"] == name]
        return r.iloc[0] if len(r) else None

    def show(self):
        if self.k >= len(self.frames):
            self.plt.close(self.fig)
            return
        name = self.frames[self.k]
        x0, x1, y0, y1 = self.roi
        g = cv2.imread(os.path.join(self.folder, name), cv2.IMREAD_GRAYSCALE)
        self.crop = g[y0:y1, x0:x1]
        self.strokes = []          # each click is a stroke of one point; each drag one stroke
        r = self.row(name)
        self.floor = float(r["baseline_y"]) if r is not None else np.nan
        self.app_outline = None    # the app's own outline, worked out once per frame
        try:
            cfg = json.load(open(os.path.join(self.folder, "analysis", "settings.json")))
            cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()}
            m = bp.measure_one((os.path.join(self.folder, name), cfg, True))
            if m.get("ok"):
                c = outline_of(m["mask"] > 0)
                if c is not None:
                    self.app_outline = c + [x0, y0]
        except Exception:
            pass
        self.draw()

    @property
    def pts(self):
        return [p for st in self.strokes for p in st]

    def picture(self):
        if not self.bright:
            return self.crop
        # stretch so the nearly black wet base becomes visible
        f = self.crop.astype(np.float32)
        lo, hi = np.percentile(f, 1), np.percentile(f, 99.5)
        f = np.clip((f - lo) / max(1.0, hi - lo), 0, 1) ** 0.45
        return (f * 255).astype(np.uint8)

    def draw(self):
        ax = self.ax
        lims = (ax.get_xlim(), ax.get_ylim()) if getattr(self, "_drawn", False) else None
        ax.clear()
        x0, x1, y0, y1 = self.roi
        ax.imshow(self.picture(), cmap="gray", extent=(x0, x1, y1, y0), vmin=0, vmax=255)
        name = self.frames[self.k]
        r = self.row(name)
        if self.app_outline is not None:
            c = self.app_outline
            ax.plot(c[:, 0], c[:, 1], color="#ef4444", lw=1, alpha=0.8, label="app's outline")
        if np.isfinite(self.floor):
            ax.axhline(self.floor, color="#22d3ee", ls="--", lw=1.2, label="mat row")
        # the trace is one line whose data is updated in place while dragging
        self.trace_line, = ax.plot([], [], "-", color="#facc15", lw=1.6, label="your trace")
        self.dots, = ax.plot([], [], "o", color="#facc15", ms=3)
        self.update_trace(redraw=False)
        info = ""
        if r is not None:
            s = self.scale ** 3 * 1e9
            info = (f"   app: V_disk {r['V_disk'] / s:.2f}  V_extrap {r['V_extrap'] / s:.2f} mm3")
        ax.set_title(f"[{self.k + 1}/{len(self.frames)}]  {name}{info}\n"
                     "click points or hold and drag along the edge, from one foot over the top "
                     "to the other (the mat row closes it).  right-click: undo last click/stroke"
                     "   g: brightness   Enter: save   n: skip   q: finish", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        self._drawn = True
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- input
    FREEHAND_STEP = 4.0     # px between recorded points while dragging

    def update_trace(self, redraw=True):
        p = np.array(self.pts) if self.pts else np.empty((0, 2))
        self.trace_line.set_data(p[:, 0], p[:, 1]) if len(p) else self.trace_line.set_data([], [])
        # dots only where single clicks were made, so a drawn stroke stays a clean line
        clicks = np.array([st[0] for st in self.strokes if len(st) == 1]) \
            if any(len(st) == 1 for st in self.strokes) else np.empty((0, 2))
        self.dots.set_data(clicks[:, 0], clicks[:, 1]) if len(clicks) else self.dots.set_data([], [])
        if redraw:
            self.fig.canvas.draw_idle()

    def _toolbar_busy(self):
        tb = getattr(self.fig.canvas.manager, "toolbar", None)
        return bool(tb is not None and tb.mode)

    def on_click(self, ev):
        if ev.inaxes is not self.ax or self._toolbar_busy() or ev.xdata is None:
            return
        if ev.button == 1:
            self.strokes.append([(ev.xdata, ev.ydata)])
            self.dragging = True
        elif ev.button == 3 and self.strokes:
            self.strokes.pop()
        self.update_trace()

    def on_move(self, ev):
        if not self.dragging or ev.inaxes is not self.ax or ev.xdata is None:
            return
        st = self.strokes[-1]
        lx, ly = st[-1]
        if np.hypot(ev.xdata - lx, ev.ydata - ly) >= self.FREEHAND_STEP:
            st.append((ev.xdata, ev.ydata))
            self.update_trace()

    def on_release(self, ev):
        if self.dragging and ev.button == 1:
            self.dragging = False
            self.update_trace()

    def on_key(self, ev):
        if ev.key == "backspace" and self.strokes:
            self.strokes.pop()
            self.update_trace()
        elif ev.key == "c":
            self.strokes = []
            self.update_trace()
        elif ev.key == "g":
            self.bright = not self.bright
            self.draw()
        elif ev.key == "enter":
            if len(self.pts) < 5:
                self.ax.set_title("trace at least 5 points before saving (or press n to skip)",
                                  color="red")
                self.fig.canvas.draw_idle()
                return
            self.save_frame()
            self.k += 1
            self._drawn = False
            self.show()
        elif ev.key == "n":
            self.k += 1
            self._drawn = False
            self.show()
        elif ev.key == "q":
            self.plt.close(self.fig)

    def save_frame(self):
        name = self.frames[self.k]
        pts = list(self.pts)
        # close the outline along the mat row: drop both feet onto it
        if np.isfinite(self.floor):
            pts = [(pts[0][0], self.floor)] + pts + [(pts[-1][0], self.floor)]
        v_px, v_mm3, mask = volume_from_outline(pts, self.crop.shape, self.roi, self.floor,
                                                self.scale)
        self.results[name] = dict(points=pts, v_px=v_px, v_mm3=v_mm3, mask=mask)


def write_results(folder, frames, tr, df, roi, scale):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    out = os.path.join(folder, "analysis", "manual_trace")
    os.makedirs(out, exist_ok=True)
    s = scale ** 3 * 1e9
    rows = []
    x0, x1, y0, y1 = roi
    for name, res in tr.results.items():
        r = df[df["image"] == name]
        r = r.iloc[0] if len(r) else None
        rows.append(dict(image=name, V_manual_mm3=res["v_mm3"],
                         V_disk_app_mm3=(r["V_disk"] / s) if r is not None else np.nan,
                         V_extrap_app_mm3=(r["V_extrap"] / s) if r is not None else np.nan,
                         V_used_app_mm3=(r["V_consensus"] / s) if r is not None and
                         np.isfinite(r["V_consensus"]) else np.nan,
                         t_min=(r["t_s"] / 60.0) if r is not None else np.nan,
                         n_points=len(res["points"])))
        g = cv2.imread(os.path.join(folder, name), cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        f = g.astype(np.float32)
        lo, hi = np.percentile(f, 1), np.percentile(f, 99.5)
        ax.imshow(np.clip((f - lo) / max(1.0, hi - lo), 0, 1) ** 0.45, cmap="gray",
                  extent=(x0, x1, y1, y0))
        c = outline_of(res["mask"])
        if c is not None:
            ax.plot(c[:, 0] + x0, c[:, 1] + y0, color="#facc15", lw=1.4, label="your trace")
        ax.set_title(f"{name}: traced volume {res['v_mm3']:.3f} mm3 (brightened picture)")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"trace_{os.path.splitext(name)[0]}.png"), dpi=90)
        plt.close(fig)
        np.save(os.path.join(out, f"trace_{os.path.splitext(name)[0]}_points.npy"),
                np.asarray(res["points"]))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "manual_trace.csv"), index=False)

    # ---- the strain with a traced start
    used = df[df["use"]] if df["use"].any() else df
    last_app = used.iloc[-1]
    setname = os.path.basename(os.path.normpath(folder))
    L = ["=" * 72, f" MANUAL TRACE  {setname}", "=" * 72, ""]
    L.append(f"  crop {roi},  scale {scale} px/um;  volume = stack of discs cut at each frame's "
             f"mat row (as the app's V_disk)")
    L.append("")
    L.append(f"  {'image':22s} {'t (min)':>7s} {'traced':>9s} {'app V_disk':>11s} "
             f"{'app V_extrap':>13s}   (mm^3)")
    for r in rows:
        L.append(f"  {r['image']:22s} {r['t_min']:7.0f} {r['V_manual_mm3']:9.3f} "
                 f"{r['V_disk_app_mm3']:11.3f} {r['V_extrap_app_mm3']:13.3f}")
    L.append("")
    first = [r for r in rows if r["image"] in frames[:-1]] or rows[:1]
    if first:
        f0 = first[0]
        v_end = last_app["V_consensus"] / s
        L.append(f"  traced start ({f0['image']}) -> app's last frame ({last_app['image']}, "
                 f"{v_end:.3f} mm3):")
        L.append(f"     volumetric strain  {100 * (v_end / f0['V_manual_mm3'] - 1):+.2f} %")
        last_tr = [r for r in rows if r["image"] == frames[-1] and r is not f0]
        if last_tr:
            lt = last_tr[0]
            bias = 100 * (lt["V_manual_mm3"] / lt["V_used_app_mm3"] - 1) if \
                np.isfinite(lt["V_used_app_mm3"]) else np.nan
            L.append(f"  traced start -> traced last frame ({lt['image']}):")
            L.append(f"     volumetric strain  "
                     f"{100 * (lt['V_manual_mm3'] / f0['V_manual_mm3'] - 1):+.2f} %   "
                     f"(same hand at both ends, so a habit of tracing a little wide or "
                     f"narrow cancels)")
            L.append(f"  your trace of the last frame vs the app's: {bias:+.2f} %  "
                     f"(how closely your hand matches the app where the app is reliable)")
        L.append(f"  app's own result for comparison: "
                 f"{100 * (v_end / (used.iloc[0]['V_consensus'] / s) - 1):+.2f} %")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(out, "manual_summary.txt"), "w") as fh:
        fh.write(txt)
    print(txt)
    return out


def main():
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    root.withdraw()
    if not folder:
        folder = filedialog.askdirectory(title="Choose the set folder to trace (already run in "
                                         "the app)", mustexist=True)
    if not folder:
        return
    an = os.path.join(folder, "analysis")
    if not (os.path.isfile(os.path.join(an, "per_image.csv")) and
            os.path.isfile(os.path.join(an, "settings.json"))):
        messagebox.showerror("Not analysed yet", "Run this set in the Bead Shrinkage window "
                             "first - the tracer uses its crop and mat row.")
        return
    cfg = json.load(open(os.path.join(an, "settings.json")))
    roi = tuple(int(v) for v in cfg["ROI"]) if cfg.get("ROI") else None
    if roi is None:
        messagebox.showerror("No crop", "This set was run without a crop; rerun it with one.")
        return
    scale = float(cfg.get("SCALE_PX_PER_UM") or bp.DEFAULTS["SCALE_PX_PER_UM"])
    df = pd.read_csv(os.path.join(an, "per_image.csv"))
    n = simpledialog.askinteger("How many?", "Trace how many of the first frames?\n"
                                "(the last frame is added at the end as a check on your "
                                "tracing)", initialvalue=5, minvalue=1, maxvalue=len(df))
    if not n:
        return
    images = list(df["image"])
    frames = images[:n] + ([images[-1]] if images[-1] not in images[:n] else [])
    root.destroy()
    tr = Tracer(folder, frames, df, roi, scale)
    tr.plt.show()
    if tr.results:
        out = write_results(folder, frames, tr, df, roi, scale)
        try:
            os.startfile(os.path.join(out, "manual_summary.txt"))
        except Exception:
            pass


if __name__ == "__main__":
    main()
