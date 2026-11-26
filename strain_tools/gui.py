#!/usr/bin/env python3
# ============================================================================
#  BEAD SHRINKAGE  |  window
#
#  One window over run_local.analyse_set: choose a set, drag the crop on the
#  first image (the same box is shown on the last one), press Run.  Shows the
#  three results -- time-lapse strain, mat rise, first + last strain -- and
#  saves everything beside the images exactly as the command line does.
#
#      pythonw gui.py            (or double-click "Bead Shrinkage" on the desktop)
#
#  Tk only: nothing to install beyond requirements.txt.  Images are handed to
#  Tk as PNG bytes encoded by OpenCV, so Pillow is not needed either.
# ============================================================================
from __future__ import annotations

import base64
import os
import queue
import sys
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                  # noqa: E402
import numpy as np          # noqa: E402
import bead_pipeline as bp  # noqa: E402
import run_local as rl      # noqa: E402
from zoomview import ZoomView  # noqa: E402

VIEW_W = 560            # on-screen width of each image preview
VIEW_H = 377
STRIP_H, STRIP_ZOOM = 96, 2   # the magnified ends of a drawn baseline


def _open(path):
    try:
        os.startfile(path)                     # Windows
    except AttributeError:
        import subprocess
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        root.title("Bead Shrinkage")
        root.minsize(1180, 860)
        self.saved = rl.load_settings()
        self.q = queue.Queue()
        self.busy = False
        self.paths, self.final = [], None
        self.full = (0, 0)            # full image size (W, H)
        self.scale = 1.0              # preview px per image px
        self.roi = None
        self.drag0 = None
        self.roi2 = None              # step 3's own crop for the last image
        self.base = None              # baseline drawn by hand (x0, y0, x1, y1), every frame
        self.first_full = None        # first image at full size, for the magnified ends
        self.res = None

        pad = dict(padx=8, pady=4)
        # ---- folder
        top = ttk.Frame(root)
        top.pack(fill="x", **pad)
        ttk.Label(top, text="Set folder:").pack(side="left")
        self.folder = tk.StringVar(value=self.saved.get("last_dir", ""))
        e = ttk.Entry(top, textvariable=self.folder)
        e.pack(side="left", fill="x", expand=True, padx=6)
        e.bind("<Return>", lambda _e: self.load_folder())
        ttk.Button(top, text="Browse...", command=self.browse).pack(side="left")
        self.info = tk.StringVar(value="Choose a folder of images.")
        ttk.Label(root, textvariable=self.info, foreground="#444").pack(anchor="w", **pad)

        # ---- the two images
        pics = ttk.Frame(root)
        pics.pack(**pad)
        self.lf1 = lf1 = ttk.LabelFrame(pics, text=self.LF1_SERIES)
        lf1.grid(row=0, column=0, padx=4)
        self.lf2 = lf2 = ttk.LabelFrame(pics, text=self.LF2_SAME)
        lf2.grid(row=0, column=1, padx=4)
        self.c1 = tk.Canvas(lf1, width=VIEW_W, height=VIEW_H, bg="#222", highlightthickness=0,
                            cursor="crosshair")
        self.c2 = tk.Canvas(lf2, width=VIEW_W, height=VIEW_H, bg="#222", highlightthickness=0)
        self.c1.pack(padx=4, pady=4)
        self.c2.pack(padx=4, pady=4)
        for c, which in ((self.c1, 1), (self.c2, 2)):
            c.bind("<ButtonPress-1>", lambda ev, w=which: self.drag_start(ev, w))
            c.bind("<B1-Motion>", lambda ev, w=which: self.drag_move(ev, w))
            c.bind("<ButtonRelease-1>", lambda ev, w=which: self.drag_end(ev, w))
        # mouse wheel zooms, right-drag moves, double right-click = whole picture
        self.v1 = ZoomView(tk, self.c1, VIEW_W, VIEW_H, on_change=self.redraw_overlays)
        self.v2 = ZoomView(tk, self.c2, VIEW_W, VIEW_H, on_change=self.redraw_overlays)
        # ---- what the mouse does on the first image, and the baseline
        bl0 = ttk.Frame(root)
        bl0.pack(fill="x", padx=8, pady=(2, 0))
        ttk.Label(bl0, text="On the first image, the left mouse button:").pack(side="left")
        self.mode = tk.StringVar(value="box")
        ttk.Radiobutton(bl0, text="drags the crop box", variable=self.mode, value="box").pack(
            side="left", padx=4)
        ttk.Radiobutton(bl0, text="drags a baseline along the base",
                        variable=self.mode, value="base").pack(side="left")
        ttk.Radiobutton(bl0, text="clicks ONE point on the mat (the code finds the baseline)",
                        variable=self.mode, value="click").pack(side="left", padx=4)
        ttk.Label(bl0, text="   Zoom: mouse wheel.  Move: right-drag.  Whole picture: double "
                  "right-click.", foreground="#555").pack(side="left")
        bl = ttk.Frame(root)
        bl.pack(fill="x", padx=8, pady=(0, 2))
        ttk.Label(bl, text="Baseline (optional, for a base blurred into the mat):").pack(
            side="left")
        for txt, dl, dr in (("   line ▲", -1, -1), ("▼", 1, 1),
                            ("   left ▲", -1, 0), ("▼", 1, 0),
                            ("   right ▲", 0, -1), ("▼", 0, 1)):
            ttk.Button(bl, text=txt.strip(), width=max(3, len(txt.strip()) + 1),
                       command=lambda a=dl, b=dr: self.nudge(a, b)).pack(side="left", padx=1)
        ttk.Button(bl, text="Clear baseline", command=lambda: self.set_base(None)).pack(
            side="left", padx=(8, 0))
        self.base_txt = tk.StringVar(value="none - the code finds the mat edge itself")
        ttk.Label(bl, textvariable=self.base_txt, foreground="#555").pack(side="left", padx=8)
        self.strip = tk.Canvas(root, width=2 * VIEW_W + 24, height=STRIP_H, bg="#111",
                               highlightthickness=0)
        self.strip_pad = pad         # shown only while a baseline is drawn
        self.strip_img = None
        self.strip_after = bl

        # ---- crop + options
        opt = ttk.Frame(root)
        opt.pack(fill="x", **pad)
        ttk.Label(opt, text="Crop  x0,x1,y0,y1:").grid(row=0, column=0, sticky="w")
        self.roi_txt = tk.StringVar()
        re_ = ttk.Entry(opt, textvariable=self.roi_txt, width=26)
        re_.grid(row=0, column=1, sticky="w", padx=6)
        re_.bind("<Return>", lambda _e: self.typed_roi())
        re_.bind("<FocusOut>", lambda _e: self.typed_roi())
        ttk.Button(opt, text="Whole frame", command=lambda: self.set_roi(None)).grid(
            row=0, column=2, sticky="w")

        # before + after only (humidity sets): two photos, a box on each
        self.ba = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Before + after photos only (no time-lapse) - drag a "
                        "separate box on EACH photo", variable=self.ba,
                        command=self.toggle_ba).grid(row=0, column=3, columnspan=3,
                                                     sticky="w", padx=(20, 0))
        ttk.Label(opt, text="Seconds between images:").grid(row=1, column=0, sticky="w")
        self.interval = tk.StringVar(value=str(self.saved.get("interval") or 60))
        ttk.Entry(opt, textvariable=self.interval, width=8).grid(row=1, column=1, sticky="w",
                                                                 padx=6)
        self.by_time = tk.BooleanVar(value=bool(self.saved.get("order_by_time", True)))
        ttk.Checkbutton(opt, text="Put the images in the order they were TAKEN, not by file "
                        "name (for renamed files)", variable=self.by_time,
                        command=lambda: self.load_folder(keep_roi=True)).grid(
            row=1, column=3, columnspan=3, sticky="w", padx=(20, 0))
        self.last_final = tk.BooleanVar(value=bool(self.saved.get("last_is_final", True)))
        self.cb_final = ttk.Checkbutton(opt, text="Last image is the final photo (leave it out "
                                        "of the time-lapse)", variable=self.last_final,
                                        command=self.refresh_info)
        self.cb_final.grid(row=2, column=0, columnspan=3, sticky="w")
        self.first_last = tk.BooleanVar(value=True)
        self.cb_fl = ttk.Checkbutton(opt, text="Also measure the first + last image only",
                                     variable=self.first_last)
        self.cb_fl.grid(row=3, column=0, columnspan=3, sticky="w")
        self.own_crop = tk.BooleanVar(value=False)
        self.cb_own = ttk.Checkbutton(opt, text="Own crop for the last image in first + last "
                                      "(drag it on the last image - for a final photo taken "
                                      "after the stage moved)",
                                      variable=self.own_crop, command=self.toggle_own_crop)
        self.cb_own.grid(row=3, column=3, columnspan=3, sticky="w", padx=(20, 0))
        self.l_roi2 = ttk.Label(opt, text="Last-image crop:")
        self.l_roi2.grid(row=4, column=3, sticky="e", padx=(20, 0))
        self.roi2_txt = tk.StringVar()
        r2 = ttk.Entry(opt, textvariable=self.roi2_txt, width=26)
        r2.grid(row=4, column=4, sticky="w", padx=6)
        r2.bind("<Return>", lambda _e: self.typed_roi2())
        r2.bind("<FocusOut>", lambda _e: self.typed_roi2())
        ttk.Label(opt, text="Leave out (file names, comma separated):").grid(
            row=4, column=0, sticky="w")
        self.skip = tk.StringVar()
        se = ttk.Entry(opt, textvariable=self.skip, width=40)
        se.grid(row=4, column=1, columnspan=2, sticky="w", padx=6)
        se.bind("<FocusOut>", lambda _e: self.load_folder(keep_roi=True))
        se.bind("<Return>", lambda _e: self.load_folder(keep_roi=True))
        ttk.Label(opt, text="Threshold offset:").grid(row=5, column=0, sticky="w")
        self.toff = tk.StringVar(value="0")
        ttk.Entry(opt, textvariable=self.toff, width=8).grid(row=5, column=1, sticky="w",
                                                             padx=6)
        ttk.Label(opt, text="0 = automatic.  Negative (e.g. -40) includes darker parts of the "
                  "bead - use it when the outline stops above the mat on a wet, dark-bottomed "
                  "bead.", foreground="#555").grid(row=5, column=2, columnspan=4, sticky="w")

        # ---- run
        run = ttk.Frame(root)
        run.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run, text="RUN", command=self.start, width=14)
        self.run_btn.pack(side="left")
        self.bar = ttk.Progressbar(run, length=420, mode="determinate")
        self.bar.pack(side="left", padx=10)
        self.status = tk.StringVar()
        ttk.Label(run, textvariable=self.status).pack(side="left")

        # ---- results
        rf = ttk.LabelFrame(root, text="Results")
        rf.pack(fill="both", expand=True, **pad)
        btns = ttk.Frame(rf)
        btns.pack(side="bottom", fill="x", padx=4, pady=(0, 4))
        sb = ttk.Scrollbar(rf, orient="vertical")
        sb.pack(side="right", fill="y", pady=4)
        self.out = tk.Text(rf, height=7, font=("Consolas", 10), wrap="none",
                           yscrollcommand=sb.set)
        self.out.pack(fill="both", expand=True, padx=4, pady=4)
        sb.config(command=self.out.yview)
        self.b_sum = ttk.Button(btns, text="Open summary", state="disabled",
                                command=lambda: _open(os.path.join(self.res["outdir"],
                                                                   "summary.txt")))
        self.b_plot = ttk.Button(btns, text="Open strain plot", state="disabled",
                                 command=lambda: _open(os.path.join(
                                     self.res["outdir"], "volumetric_strain.png")))
        self.b_mat = ttk.Button(btns, text="Open mat plot", state="disabled",
                                command=lambda: _open(os.path.join(self.res["outdir"],
                                                                   "mat_height.png")))
        self.b_pair = ttk.Button(btns, text="Open first + last picture", state="disabled",
                                 command=lambda: _open(os.path.join(self.res["pairdir"],
                                                                    "pair.png")))
        self.b_dir = ttk.Button(btns, text="Open results folder", state="disabled",
                                command=lambda: _open(self.res["outdir"]))
        for b in (self.b_sum, self.b_plot, self.b_mat, self.b_pair, self.b_dir):
            b.pack(side="left", padx=3)
        # master analysis: every set side by side (master_analysis.py)
        self.b_master = ttk.Button(btns, text="Open master analysis", state="disabled",
                                   command=lambda: _open(self.master_out()))
        self.b_master.pack(side="right", padx=3)
        ttk.Button(btns, text="Master folders...", command=self.master_folders).pack(
            side="right", padx=3)
        self.master_busy = False
        if os.path.isdir(self.master_out() or ""):
            self.b_master.state(["!disabled"])

        if self.folder.get() and os.path.isdir(self.folder.get()):
            self.load_folder()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self.poll)

    def on_close(self):
        """Closing while a run or a master refresh is going cuts it off halfway
        (a refresh closed this way once left findings.txt from the run before)."""
        if self.busy or getattr(self, "master_busy", False):
            from tkinter import messagebox
            what = "a run" if self.busy else "the master analysis refresh"
            if not messagebox.askyesno("Still working",
                                       f"{what[0].upper() + what[1:]} is still going. Closing "
                                       f"now stops it halfway and leaves its files incomplete."
                                       "\n\nClose anyway?", parent=self.root):
                return
        self.root.destroy()

    # ------------------------------------------------------------ folder
    def browse(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Choose the folder of bead images",
                                    initialdir=self.folder.get() or os.path.expanduser("~"),
                                    mustexist=True, parent=self.root)
        if d:
            self.folder.set(d)
            self.skip.set("")
            self.load_folder()

    def skips(self):
        return [s.strip() for s in self.skip.get().split(",") if s.strip()]

    def load_folder(self, keep_roi=False):
        d = self.folder.get().strip().strip('"')
        if not os.path.isdir(d):
            self.info.set(f"Not a folder: {d}")
            return
        allp = [p for p in bp.list_frames(d) if os.path.basename(p) not in set(self.skips())]
        if self.by_time.get():                   # renamed files: the order they were taken
            allp = bp.sort_by_capture_time(allp)
        if len(allp) < 2:
            self.info.set(f"Only {len(allp)} image(s) in this folder - need at least 2.")
            self.paths = []
            return
        self.all_paths = allp
        first, last = allp[0], allp[-1]
        b1, b2 = cv2.imread(first), cv2.imread(last)
        if b1 is None or b2 is None:
            self.info.set("Could not read the first or last image.")
            return
        H, W = b1.shape[:2]
        self.full = (W, H)
        self.first_full = b1
        self.v1.set_image(b1, keep_view=keep_roi)
        self.v2.set_image(b2, keep_view=keep_roi)
        self.names = (os.path.basename(first), os.path.basename(last))
        if not keep_roi:
            # the crop used last time on THIS folder, if any (the bead is not in
            # the same place in every set, so another set's crop is not offered)
            per = self.saved.get("rois", {})
            per2 = self.saved.get("rois_last", {})
            self.toff.set(str(self.saved.get("thresh_offsets", {}).get(d, 0)))
            self.roi2 = tuple(per2[d]) if d in per2 else None
            # a folder of just two photos is a before + after set (humidity)
            self.ba.set(len(allp) == 2)
            self.own_crop.set(self.roi2 is not None and not self.ba.get())
            self.set_roi(tuple(per[d]) if d in per else None)
            bases = self.saved.get("bases", {})
            self.set_base(tuple(bases[d]) if d in bases else None)
        else:
            self.set_roi(self.roi)
            self.set_base(self.base)
        self.toggle_ba()

    # ---------------------------------------------------------- baseline
    def set_base(self, base):
        """The bead's base, drawn by hand on the first image and used for EVERY
        frame (a continuous run: the camera and mat do not move)."""
        self.base = tuple(int(round(v)) for v in base) if base else None
        self._draw_base_lines()
        if self.base:
            x0, y0, x1, y1 = self.base
            self.base_txt.set(f"drawn: Y {y0} -> {y1}, every frame")
        else:
            self.base_txt.set("none - the code finds the mat edge itself")
        self.draw_strip()

    def nudge(self, dl, dr):
        if not self.base:
            self.status.set("Draw a baseline first: choose 'the baseline' and drag along the "
                            "bead's base on the first image.")
            return
        x0, y0, x1, y1 = self.base
        self.set_base((x0, y0 + dl, x1, y1 + dr))

    def _measure_first(self):
        """The first image measured the way the run will measure it (box +
        drawn base), or None."""
        if not getattr(self, "all_paths", None):
            return None
        cfg = dict(bp.DEFAULTS, ROI=self.roi, BASE_LINE=tuple(self.base), MAT_CHROMA_MIN=255,
                   THRESH_OFFSET=int(float(self.toff.get() or 0)) if self.toff.get().strip()
                   .lstrip("-").isdigit() else 0)
        m = bp.measure_one((self.all_paths[0], cfg, True))
        return m if m.get("ok") else None

    def draw_strip(self):
        """Where the bead meets the drawn base, magnified: its left and right
        corners on the first image, with the outline the run will measure."""
        s = self.strip
        s.delete("all")
        if self.base and not s.winfo_ismapped():
            s.pack(after=self.strip_after, **self.strip_pad)
            self.out.config(height=4)            # keep the window on the screen
        elif not self.base and s.winfo_ismapped():
            s.pack_forget()
            self.out.config(height=7)
        if not self.base or self.first_full is None:
            return
        bgr = self.first_full
        H, W = bgr.shape[:2]
        W2 = int(s.cget("width"))
        lx0, ly0, lx1, ly1 = self.base
        slope = (ly1 - ly0) / (lx1 - lx0) if lx1 != lx0 else 0.0
        m = self._measure_first()
        if m is not None:              # the bead's own corners: outermost in the bottom band
            low = m["Y"] >= m["Y"][-1] - bp.BASE_LINE_BAND_PX
            corners = [(float(m["xl"][low].min()), "LEFT corner"),
                       (float(m["xr"][low].max()), "RIGHT corner")]
        else:
            corners = [(lx0, "LEFT end of line"), (lx1, "RIGHT end of line")]
        half = W2 // 2
        ww, wh = int(half / STRIP_ZOOM), int(STRIP_H / STRIP_ZOOM)
        tiles, marks = [], []
        for k, (ex, name) in enumerate(corners):
            ey = ly0 + slope * (ex - lx0)
            cx0 = int(max(0, min(W - ww, ex - ww // 2)))
            cy0 = int(max(0, min(H - wh, ey - wh * 0.6)))
            win = cv2.normalize(bgr[cy0:cy0 + wh, cx0:cx0 + ww], None, 0, 255, cv2.NORM_MINMAX)
            tile = cv2.resize(win, (half, STRIP_H), interpolation=cv2.INTER_NEAREST)
            if m is not None:          # the measured outline, in yellow
                mx0, mx1, my0, my1 = m["roi"]
                sub = np.zeros((wh, ww), np.uint8)
                a0, a1 = max(cx0, mx0), min(cx0 + ww, mx1)
                b0, b1 = max(cy0, my0), min(cy0 + wh, my1)
                if a1 > a0 and b1 > b0:
                    sub[b0 - cy0:b1 - cy0, a0 - cx0:a1 - cx0] = \
                        m["mask"][b0 - my0:b1 - my0, a0 - mx0:a1 - mx0]
                big = cv2.resize(sub, (half, STRIP_H), interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours((big > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(tile, cs, -1, (0, 220, 255), 1)
            tiles.append(tile)
            marks.append((k * half, (ly0 + slope * (cx0 - lx0) - cy0) * STRIP_ZOOM,
                          (ly0 + slope * (cx0 + ww - lx0) - cy0) * STRIP_ZOOM, name))
        ok, png = cv2.imencode(".png", cv2.hconcat(tiles))
        self.strip_img = self.tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))
        s.create_image(0, 0, anchor="nw", image=self.strip_img)
        for off, ya, yb, name in marks:
            s.create_line(off, ya, off + half, yb, fill="#ff2d95", width=2)
            s.create_text(off + 6, 4, anchor="nw", fill="#ffcc00",
                          text=f"{name} of the bead, {STRIP_ZOOM}x   pink = the line, "
                               "yellow = what will be measured")
        s.create_line(half, 0, half, STRIP_H, fill="#ffffff")

    def toggle_ba(self):
        ba = self.ba.get()
        if ba and getattr(self, "all_paths", None) and len(self.all_paths) != 2:
            self.ba.set(False)
            self.status.set(f"Before + after needs a folder of exactly 2 photos - this one has "
                            f"{len(self.all_paths)}.")
            ba = False
        self.lf1.config(text=self.LF1_BA if ba else self.LF1_SERIES)
        for w in (self.cb_final, self.cb_fl, self.cb_own):
            w.state(["disabled"] if ba else ["!disabled"])
        self.l_roi2.config(text="After-photo crop:" if ba else "Last-image crop:")
        if ba and not self.roi2 and self.roi:
            self.roi2 = self.roi
        self.toggle_own_crop()
        self.refresh_info()

    def two_boxes(self):
        """True when the second picture takes its own box."""
        return self.ba.get() or self.own_crop.get()

    def refresh_info(self):
        if not getattr(self, "all_paths", None):
            return
        n = len(self.all_paths)
        if self.ba.get():
            self.info.set(f"BEFORE + AFTER   before: {self.names[0]}   after: {self.names[1]}"
                          "   -> drag a box round the bead on EACH photo, then RUN")
            return
        series = n - 1 if self.last_final.get() else n
        fin = (f"   final photo: {self.names[1]}" if self.last_final.get() else "")
        self.info.set(f"{n} images   first: {self.names[0]}   last: {self.names[1]}   "
                      f"-> time-lapse uses {series}{fin}")

    def _view(self, c):
        return self.v1 if c is self.c1 else self.v2

    def _draw_base_lines(self):
        for c in (self.c1, self.c2):
            c.delete("base")
            if self.base:
                v = self._view(c)
                x0, y0, x1, y1 = self.base
                c.create_line(*v.to_canvas(x0, y0), *v.to_canvas(x1, y1), fill="#ff2d95",
                              width=2, tags="base")

    def redraw_overlays(self):
        """After a zoom or a move: the box(es) and the baseline, in the new view."""
        self._box(self.c1, self.roi, "#ff3b30")
        self.redraw_last()
        self._draw_base_lines()

    # -------------------------------------------------------------- crop
    LF1_SERIES = "First image - drag a box round the bead"
    LF1_BA = "BEFORE photo - drag a box round the bead"
    LF2_SAME = "Last image - the same box, check the bead fits"
    LF2_OWN = "Last image - drag its OWN box (used only in first + last)"
    LF2_BA = "AFTER photo - drag its OWN box round the bead"

    def _box(self, c, roi, colour):
        c.delete("box")
        if roi:
            v = self._view(c)
            (x0, y0), (x1, y1) = v.to_canvas(roi[0], roi[2]), v.to_canvas(roi[1], roi[3])
            c.create_rectangle(x0, y0, x1, y1, outline=colour, width=2, tags="box")

    def set_roi(self, roi):
        self.roi = tuple(int(v) for v in roi) if roi else None
        self.roi_txt.set(",".join(str(v) for v in self.roi) if self.roi else "")
        self._box(self.c1, self.roi, "#ff3b30")
        self.redraw_last()
        if getattr(self, "base", None):
            self.draw_strip()                    # the corners depend on the box too

    def set_roi2(self, roi):
        self.roi2 = tuple(int(v) for v in roi) if roi else None
        self.redraw_last()

    def redraw_last(self):
        own = self.two_boxes()
        r = (self.roi2 or self.roi) if own else self.roi
        self._box(self.c2, r, "#34c759" if own and self.roi2 else "#ff3b30")
        self.roi2_txt.set(",".join(str(v) for v in self.roi2) if own and self.roi2 else "")
        self.c2.config(cursor="crosshair" if own else "")

    def toggle_own_crop(self):
        own = self.two_boxes()
        self.lf2.config(text=self.LF2_BA if self.ba.get() else
                        self.LF2_OWN if own else self.LF2_SAME)
        if own and not self.roi2 and self.roi:
            self.roi2 = self.roi             # start from the time-lapse's box
        self.redraw_last()

    def typed_roi2(self):
        t = self.roi2_txt.get().strip()
        if not self.two_boxes():
            return
        try:
            self.set_roi2(rl.parse_roi(t) if t else None)
        except Exception:
            self.status.set("Last-image crop must be four numbers: x0,x1,y0,y1")

    def typed_roi(self):
        t = self.roi_txt.get().strip()
        if not t:
            self.set_roi(None)
            return
        try:
            self.set_roi(rl.parse_roi(t))
        except Exception:
            self.status.set("Crop must be four numbers: x0,x1,y0,y1")

    def _drag_ok(self, which):
        return which == 1 or self.two_boxes()

    def drag_start(self, ev, which=1):
        self.drag0 = (ev.x, ev.y) if self._drag_ok(which) else None

    def drag_move(self, ev, which=1):
        if not self.drag0:
            return
        c = self.c1 if which == 1 else self.c2
        c.delete("drag")
        if which == 1 and self.mode.get() == "click":
            return
        if which == 1 and self.mode.get() == "base":
            c.create_line(*self.drag0, ev.x, ev.y, fill="#ff2d95", width=2, dash=(4, 2),
                          tags="drag")
            return
        c.create_rectangle(*self.drag0, ev.x, ev.y, outline="#ffcc00", width=2,
                           dash=(4, 2), tags="drag")

    def drag_end(self, ev, which=1):
        if not self.drag0 or not self.full[0]:
            return
        (self.c1 if which == 1 else self.c2).delete("drag")
        (ax, ay), self.drag0 = self.drag0, None
        W, H = self.full
        v = self.v1 if which == 1 else self.v2
        (ax, ay), (bx, by) = v.to_full(ax, ay), v.to_full(ev.x, ev.y)   # image px
        if which == 1 and self.mode.get() == "click":
            self.status.set("finding the mat edge near your click...")
            self.root.update_idletasks()
            try:
                line, how = bp.suggest_base_line(self.all_paths[0], self.roi, (bx, by))
            except Exception as e:
                self.status.set(f"could not read the first image: {e}")
                return
            self.set_base(line)
            self.status.set("baseline found from the mat's colour - check the corners below"
                            if how == "found" else
                            "no mat edge found near your click - drew a level line through it; "
                            "move it with the arrows")
            return
        if which == 1 and self.mode.get() == "base":
            (px0, py0), (px1, py1) = sorted([(ax, ay), (bx, by)])
            if px1 - px0 >= 20:
                self.set_base((px0, py0, px1, py1))
            return
        x0, x1 = sorted((ax, bx))
        y0, y1 = sorted((ay, by))
        x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
        x1, y1 = min(W, int(round(x1))), min(H, int(round(y1)))
        if x1 - x0 >= 20 and y1 - y0 >= 20:
            (self.set_roi if which == 1 else self.set_roi2)((x0, x1, y0, y1))

    # --------------------------------------------------------------- run
    def start(self):
        if self.busy:
            return
        self.typed_roi()
        self.typed_roi2()
        d = self.folder.get().strip().strip('"')
        if not getattr(self, "all_paths", None) or not os.path.isdir(d):
            self.status.set("Choose a folder first.")
            return
        if self.base and (self.ba.get() or self.own_crop.get() or self.last_final.get()):
            # One drawn baseline is used for EVERY image.  Separate photos (before /
            # after, or a final photo) are taken after the stage has moved: on
            # f95%_8hr_3 the base sat 25 px lower in the second photo, and one line
            # for both read -27.8 % where the right line on each gives -21.4 %.
            from tkinter import messagebox
            messagebox.showwarning(
                "Baseline and separate photos",
                "The drawn baseline is used for every image, so it only suits a "
                "continuous time-lapse.\n\nThis run includes a separate photo (before / "
                "after, or a final photo), where the base usually sits somewhere else.\n\n"
                "Either clear the baseline here, or measure the photos in Humidity Cycles, "
                "where each photo gets its own baseline.", parent=self.root)
            return
        if not self.roi:
            from tkinter import messagebox
            if not messagebox.askyesno("No crop", "No box is drawn, so the whole frame will "
                                       "be measured. Go ahead?", parent=self.root):
                return
        try:
            iv = float(self.interval.get()) if self.interval.get().strip() else None
        except ValueError:
            self.status.set("Seconds between images must be a number.")
            return
        try:
            toff = int(float(self.toff.get() or 0))
        except ValueError:
            self.status.set("Threshold offset must be a whole number, e.g. 0 or -40.")
            return
        cfg = dict(SCALE_PX_PER_UM=bp.DEFAULTS["SCALE_PX_PER_UM"], ROI=self.roi,
                   INVERT=False, MANUAL_THRESH=None, THRESH_OFFSET=toff, SEGMENT_METHOD="otsu",
                   MAT_CHROMA_MIN=bp.DEFAULTS["MAT_CHROMA_MIN"], BASELINE="blue",
                   BASELINE_TUNE=0.0, MAT_MODE="auto", CLIP_AT_BASELINE=True,
                   INTERVAL_S=iv, TIME_REGEX=None, ORDER_BY_TIME=self.by_time.get())
        bases = dict(self.saved.get("bases", {}))
        if self.base:
            # the floor every frame is measured to; above it the blue smear of a
            # blurred mat counts as bead (no colour filter)
            cfg.update(BASE_LINE=tuple(self.base), MAT_CHROMA_MIN=255)
            bases[d] = list(self.base)
        else:
            bases.pop(d, None)
        rois = dict(self.saved.get("rois", {}))
        if self.roi:
            rois[d] = list(self.roi)
        ba = self.ba.get()
        if ba and not self.roi2:
            self.roi2 = self.roi
        pair_roi = self.roi2 if self.two_boxes() and self.roi2 else None
        toffs = dict(self.saved.get("thresh_offsets", {}))
        toffs[d] = toff
        rois_last = dict(self.saved.get("rois_last", {}))
        if pair_roi:
            rois_last[d] = list(pair_roi)
        else:
            rois_last.pop(d, None)
        self.saved = dict(self.saved, last_dir=d, interval=iv, rois=rois, rois_last=rois_last,
                          thresh_offsets=toffs, bases=bases,
                          last_is_final=self.last_final.get(),
                          order_by_time=self.by_time.get(),
                          roi=list(self.roi) if self.roi else None)
        rl.save_settings(self.saved)

        self.busy = True
        self.run_btn.state(["disabled"])
        for b in (self.b_sum, self.b_plot, self.b_mat, self.b_pair, self.b_dir):
            b.state(["disabled"])
        self.out.delete("1.0", "end")
        self.bar["value"] = 0
        self.status.set("measuring images...")
        args = (d, cfg, self.skips(), self.last_final.get(), self.first_last.get(), pair_roi,
                ba)
        threading.Thread(target=self.work, args=args, daemon=True).start()

    def work(self, d, cfg, skip, last_final, first_last, pair_roi=None, ba=False):
        try:
            prog = lambda i, n: self.q.put(("prog", i, n))  # noqa: E731
            if ba:
                res = rl.analyse_before_after(d, cfg, cfg.get("ROI"), pair_roi, "analysis",
                                              skip, quiet=True, progress=prog)
            else:
                res = rl.analyse_set(d, cfg, "analysis", "*", skip, last_final, first_last,
                                     quiet=True, pair_roi=pair_roi, progress=prog)
            self.q.put(("done", res))
        except BaseException as e:
            self.q.put(("fail", f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"))

    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "prog":
                    _, i, n = msg
                    self.bar["maximum"] = n
                    self.bar["value"] = i
                    self.status.set(f"measuring images... {i} / {n}"
                                    if i < n else "working out strains, plots, first + last...")
                elif msg[0] == "done":
                    self.finish(msg[1])
                elif msg[0] == "master_done":
                    _, out, n = msg
                    self.master_busy = False
                    self.status.set(f"master analysis refreshed - {n} sets, in {out}")
                    self.b_master.state(["!disabled"])
                elif msg[0] == "master_fail":
                    self.master_busy = False
                    self.status.set("master analysis FAILED - see below")
                    self.out.insert("end", "\n\nMASTER ANALYSIS FAILED\n" + msg[1])
                elif msg[0] == "fail":
                    self.busy = False
                    self.run_btn.state(["!disabled"])
                    self.status.set("FAILED - see below")
                    self.out.insert("end", msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def finish(self, res):
        self.res = res
        self.busy = False
        self.run_btn.state(["!disabled"])
        self.bar["value"] = self.bar["maximum"]
        self.status.set(f"done - saved in {res['outdir']}")
        self.out.insert("end", rl.results_text(res))
        if res.get("mode") == "before_after":
            # two photos: summary, the outlines (pair.png) and the folder -- no
            # time plots, and these sets are not part of the time-lapse master
            for b in (self.b_sum, self.b_pair, self.b_dir):
                b.state(["!disabled"])
            return
        for b in (self.b_sum, self.b_plot, self.b_mat, self.b_dir):
            b.state(["!disabled"])
        if res.get("pairdir"):
            self.b_pair.state(["!disabled"])
        self.root.after(300, lambda: self.ask_master(res["folder"]))

    # ------------------------------------------------------- master analysis
    def master_roots(self):
        return [r for r in self.saved.get("master_roots", []) if os.path.isdir(r)]

    def master_out(self):
        out = self.saved.get("master_out")
        if out:
            return out
        roots = self.master_roots()
        return os.path.join(roots[0], "Master Analysis") if roots else None

    def remember_master(self, roots, out=None):
        self.saved = dict(self.saved, master_roots=roots,
                          master_out=out or self.saved.get("master_out")
                          or os.path.join(roots[0], "Master Analysis"))
        rl.save_settings(self.saved)

    def master_folders(self):
        """Choose which data folders the master analysis reads."""
        from tkinter import filedialog, messagebox
        roots = self.master_roots()
        txt = "\n".join(roots) or "(none yet)"
        ans = messagebox.askyesnocancel(
            "Master analysis folders",
            f"The master analysis reads every set in:\n\n{txt}\n\n"
            "Yes = add another data folder\nNo = start the list again\nCancel = keep it",
            parent=self.root)
        if ans is None:
            return
        if ans is False:
            roots = []
        d = filedialog.askdirectory(title="Data folder holding set folders", mustexist=True,
                                    parent=self.root)
        if d:
            d = os.path.normpath(d)
            if d not in [os.path.normpath(r) for r in roots]:
                roots.append(d)
        if roots:
            out = self.saved.get("master_out") if ans else os.path.join(roots[0],
                                                                       "Master Analysis")
            self.remember_master(roots, out)
            self.status.set("master analysis reads: " + " ; ".join(roots))

    def ask_master(self, set_folder):
        """After a run: refresh the master analysis?"""
        from tkinter import messagebox
        if self.master_busy:
            return
        parent = os.path.normpath(os.path.dirname(os.path.normpath(set_folder)))
        roots = self.master_roots()
        # only the remembered data folders: a run from another folder (older
        # sets) must not slip into the master analysis.  The run's own folder
        # is used only when nothing has been chosen yet.
        if not roots:
            roots = [parent]
        elif parent not in [os.path.normpath(r) for r in roots]:
            self.status.set(f"note: this set's folder ({parent}) is not in the master "
                            f"analysis - add it with 'Master folders...' if you want it")
        out = self.master_out() or os.path.join(roots[0], "Master Analysis")
        if not messagebox.askyesno(
                "Refresh master analysis?",
                "Refresh the master analysis (all sets side by side) with this result?\n\n"
                "Reads every set in:\n" + "\n".join(roots) + f"\n\nWrites to:\n{out}\n\n"
                "(Only these folders. Use 'Master folders...' to change them.)",
                parent=self.root):
            return
        self.remember_master(roots, out)
        self.master_busy = True
        self.status.set("refreshing the master analysis...")
        threading.Thread(target=self.master_work, args=(roots, out), daemon=True).start()

    def master_work(self, roots, out):
        try:
            import master_analysis as ma
            m, _ = ma.run(roots, out)
            self.q.put(("master_done", out, len(m)))
        except BaseException as e:
            self.q.put(("master_fail", f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"))


def main():
    import tkinter as tk
    try:
        from ctypes import windll                       # sharp text on a scaled display
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
