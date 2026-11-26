#!/usr/bin/env python3
# ============================================================================
#  HUMIDITY CYCLES  |  window
#
#  For the humidity sets: a bead photographed fresh, then after 72 % RH, after
#  97 % RH, ...  Before / after PHOTOS only; time-lapses go in Bead Shrinkage.
#
#  Choose the folder, check the list (it is in the order the photos were TAKEN,
#  from the camera's clock), give every step a label, drag a box round the bead
#  on each one, RUN.  Results go to <folder>/humidity_analysis/.
#
#      pythonw humidity_gui.py        (or double-click "Humidity Cycles")
#
#  Separate from the Bead Shrinkage window: nothing there is changed.
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
import humidity as hu       # noqa: E402
from zoomview import ZoomView  # noqa: E402
import run_local as rl      # noqa: E402

VIEW_W, VIEW_H = 680, 380
STRIP_H, STRIP_ZOOM = 150, 2    # the two magnified ends of a drawn baseline
KEY = "humidity"            # this window's part of app_settings.json


def _tkey(t):
    """A photo's identity for remembering its label and box: its capture time."""
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _open(path):
    if not os.path.exists(path):
        return
    try:
        os.startfile(path)
    except AttributeError:
        import subprocess
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        root.title("Humidity Cycles")
        root.minsize(1200, 820)
        self.q = queue.Queue()
        self.busy = False
        self.items = []             # dict(path, name, time, label, roi, base)
        self.sel = None             # index of the selected item
        self.scale = 1.0
        self.full = (0, 0)
        self.drag0 = None
        self.photo = None
        self.res = None
        self.cache = {}
        allset = rl.load_settings()
        self.mine = allset.get(KEY, {})
        pad = dict(padx=8, pady=3)

        # ---- folder
        top = ttk.Frame(root)
        top.pack(fill="x", **pad)
        ttk.Label(top, text="Set folder:").pack(side="left")
        self.folder = tk.StringVar(value=self.mine.get("last_dir", ""))
        e = ttk.Entry(top, textvariable=self.folder)
        e.pack(side="left", fill="x", expand=True, padx=6)
        e.bind("<Return>", lambda _e: self.load_folder())
        ttk.Button(top, text="Browse...", command=self.browse).pack(side="left")

        # ---- order, options
        opt = ttk.Frame(root)
        opt.pack(fill="x", **pad)
        ttk.Label(opt, text="Order:").grid(row=1, column=0, sticky="w")
        self.order = tk.StringVar(value="time")
        ttk.Radiobutton(opt, text="when each photo was taken (camera clock)",
                        variable=self.order, value="time",
                        command=self.rebuild).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Radiobutton(opt, text="file name", variable=self.order, value="name",
                        command=self.rebuild).grid(row=1, column=2, sticky="w", padx=4)
        ttk.Label(opt, text="Threshold offset:").grid(row=1, column=3, sticky="e", padx=(24, 4))
        self.toff = tk.StringVar(value="0")
        te = ttk.Entry(opt, textvariable=self.toff, width=6)
        te.grid(row=1, column=4, sticky="w")
        te.bind("<Return>", lambda _e: self.draw_preview())     # strip follows the offset
        te.bind("<FocusOut>", lambda _e: self.draw_preview())
        ttk.Label(opt, text="(0 = automatic; e.g. -40 if the outline stops above the mat)",
                  foreground="#555").grid(row=1, column=5, sticky="w", padx=6)

        self.info = tk.StringVar(value="Choose a folder.")
        ttk.Label(root, textvariable=self.info, foreground="#444").pack(anchor="w", **pad)
        self.warn = tk.StringVar()
        ttk.Label(root, textvariable=self.warn, foreground="#c0392b",
                  justify="left").pack(anchor="w", padx=8)

        # ---- list + picture
        mid = ttk.Frame(root)
        mid.pack(fill="x", **pad)
        left = ttk.Frame(mid)
        left.pack(side="left", fill="y")
        cols = ("n", "label", "files", "taken", "box", "base")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=10,
                                 selectmode="browse")
        for c, w, t in (("n", 34, "#"), ("label", 160, "label"), ("files", 250, "photo"),
                        ("taken", 140, "taken"), ("box", 44, "box"), ("base", 70, "baseline")):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w" if c not in ("box", "base") else "center")
        self.tree.pack(fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.on_select())
        ed = ttk.Frame(left)
        ed.pack(fill="x", pady=(6, 0))
        ttk.Label(ed, text="Label:").pack(side="left")
        self.lab = tk.StringVar()
        le = ttk.Entry(ed, textvariable=self.lab, width=24)
        le.pack(side="left", padx=4)
        le.bind("<Return>", lambda _e: self.set_label())
        ttk.Button(ed, text="Set", command=self.set_label, width=5).pack(side="left")
        ttk.Label(left, text="e.g. fresh / after 72% RH / after 97% RH  (a 97 in the label "
                  "colours it blue, a 72 orange)", foreground="#555",
                  wraplength=500).pack(anchor="w", pady=(2, 6))
        bb = ttk.Frame(left)
        bb.pack(fill="x")
        ttk.Button(bb, text="Copy this box to ALL", command=lambda: self.copy_box("all")).pack(
            side="left")
        ttk.Button(bb, text="... to the ones BELOW", command=lambda: self.copy_box("below")
                   ).pack(side="left", padx=4)
        ttk.Button(bb, text="Clear this box", command=self.clear_box).pack(side="left")

        # ---- baseline drawn by hand (blurred base: blue smear counted as bead)
        bl = ttk.LabelFrame(left, text="Baseline (optional) - where the bead sits on the mat")
        bl.pack(fill="x", pady=(10, 0))
        mr = ttk.Frame(bl)
        mr.pack(fill="x", padx=4, pady=2)
        ttk.Label(mr, text="Dragging on the picture draws:").pack(side="left")
        self.mode = tk.StringVar(value="box")
        ttk.Radiobutton(mr, text="the box", variable=self.mode, value="box",
                        command=self.draw_preview).pack(
            side="left", padx=4)
        ttk.Radiobutton(mr, text="the baseline (drag along the base)", variable=self.mode,
                        value="base", command=self.draw_preview).pack(side="left")
        mr2 = ttk.Frame(bl)
        mr2.pack(fill="x", padx=4)
        ttk.Label(mr2, text="Clicking on the picture:").pack(side="left")
        ttk.Radiobutton(mr2, text="ONE point on the mat - the code finds the line",
                        variable=self.mode, value="click",
                        command=self.draw_preview).pack(side="left", padx=4)
        nr = ttk.Frame(bl)
        nr.pack(fill="x", padx=4, pady=2)
        for txt, dl, dr in (("whole line ▲", -1, -1), ("▼", 1, 1),
                            ("   left end ▲", -1, 0), ("▼", 1, 0),
                            ("   right end ▲", 0, -1), ("▼", 0, 1)):
            ttk.Button(nr, text=txt.strip(), width=max(3, len(txt.strip()) + 1),
                       command=lambda a=dl, b=dr: self.nudge(a, b)).pack(side="left", padx=1)
        ttk.Button(nr, text="Clear baseline", command=self.clear_base).pack(side="left",
                                                                          padx=(10, 0))
        ttk.Label(bl, text="With a baseline the bead is measured down to the line; everything "
                  "below it is dropped and everything above it counts as bead, blue smear from "
                  "a blurred mat included.  The arrows move it 1 pixel; check it in the zoomed "
                  "strip under the picture.", foreground="#555", wraplength=520,
                  justify="left").pack(anchor="w", padx=4, pady=(0, 4))

        right = ttk.Frame(mid)
        right.pack(side="left", padx=(12, 0))
        self.lf = ttk.LabelFrame(right, text="Select a step on the left")
        self.lf.pack()
        self.cv = tk.Canvas(self.lf, width=VIEW_W, height=VIEW_H, bg="#222",
                            highlightthickness=0, cursor="crosshair")
        self.cv.pack(padx=4, pady=4)
        self.cv.bind("<ButtonPress-1>", self.drag_start)
        self.cv.bind("<B1-Motion>", self.drag_move)
        self.cv.bind("<ButtonRelease-1>", self.drag_end)
        # mouse wheel zooms, right-drag moves, double right-click = whole picture
        self.view = ZoomView(tk, self.cv, VIEW_W, VIEW_H, on_change=self.draw_overlays)
        ttk.Label(self.lf, text="Zoom: mouse wheel.  Move: right-click and drag.  Whole "
                  "picture: double right-click.", foreground="#555").pack(anchor="w", padx=4)
        self.strip = tk.Canvas(self.lf, width=VIEW_W, height=STRIP_H, bg="#111",
                               highlightthickness=0)
        self.strip.pack(padx=4, pady=(0, 4))
        self.strip_img = None
        self.full_img = (None, None)          # (path, full-resolution BGR) of the preview
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
        self.out = tk.Text(rf, height=6, font=("Consolas", 9), wrap="none",
                           yscrollcommand=sb.set)
        self.out.pack(fill="both", expand=True, padx=4, pady=4)
        sb.config(command=self.out.yview)
        self.rbtn = []
        ttk.Label(btns, text="vs fresh:").pack(side="left", padx=(2, 0))
        for text, name in (("Strain at every step", "strain_steps.png"),
                           ("Strain vs time", "strain_vs_time.png"),
                           ("|  vs photo before:", None),
                           ("Change at every step", "change_steps.png"),
                           ("Change vs time", "change_vs_time.png"),
                           ("Change bars", "step_changes.png"),
                           ("|", None),
                           ("Volume only", "volume_steps.png"),
                           ("Photo sheet", "photo_sheet.png"),
                           ("Outlines", "outlines.png"),
                           ("Summary", "summary.txt"),
                           ("Results folder", "")):
            if name is None:                         # a separator label, not a button
                ttk.Label(btns, text=text).pack(side="left", padx=(6, 0))
                continue
            b = ttk.Button(btns, text=text, state="disabled",
                           command=lambda n=name: _open(os.path.join(self.res["outdir"], n)
                                                        if n else self.res["outdir"]))
            b.pack(side="left", padx=3)
            self.rbtn.append(b)

        if self.folder.get() and os.path.isdir(self.folder.get()):
            self.load_folder()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self.poll)

    # ------------------------------------------------------------ folder
    def on_close(self):
        if self.busy:
            from tkinter import messagebox
            if not messagebox.askyesno("Still working", "A run is still going. Closing now "
                                       "stops it halfway.\n\nClose anyway?", parent=self.root):
                return
        self.root.destroy()

    def browse(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Choose the humidity set folder",
                                    initialdir=self.folder.get() or os.path.expanduser("~"),
                                    mustexist=True, parent=self.root)
        if d:
            self.folder.set(os.path.normpath(d))
            self.load_folder()

    def dkey(self):
        return os.path.normpath(self.folder.get().strip().strip('"'))

    def load_folder(self):
        d = self.dkey()
        if not os.path.isdir(d):
            self.info.set(f"Not a folder: {d}")
            return
        self.status.set("reading the photos' capture times...")
        self.root.update_idletasks()
        self.photos_all = hu.list_photos(d, "time")
        self.cache.clear()
        self.items, self.sel = [], None      # another folder's labels must not carry over
        n = len(self.photos_all)
        if n < 2 or n > hu.MAX_PHOTOS:
            self.photos_all = []
            self.fill_tree()
            self.cv.delete("all")
            self.status.set("")
            self.info.set(f"{n} photo(s) directly in this folder.")
            self.warn.set(
                f"{n} photos is a time-lapse. This window is for before / after photos only - "
                "run time-lapses in Bead Shrinkage." if n > hu.MAX_PHOTOS else
                "Need at least 2 photos." if n else
                "Sub-folders are not read. Choose the folder that holds the before / after "
                "photos themselves; run time-lapses in Bead Shrinkage.")
            return
        saved = self.mine.get("sets", {}).get(d, {})
        self.order.set(saved.get("order", "time"))
        self.toff.set(str(saved.get("toff", 0)))
        self.status.set("")
        self.rebuild(saved)

    def rebuild(self, saved=None):
        """Make the list of steps, one per photo."""
        if not getattr(self, "photos_all", None):
            return
        d = self.dkey()
        if saved is None:
            saved = self.mine.get("sets", {}).get(d, {})
        # Remembered per photo by its CAPTURE TIME, which renaming a file cannot
        # change (keyed by file name, a label stuck to the wrong photo after two
        # names were swapped).  Only labels typed by hand are remembered; the
        # rest always come from the current file names.
        typed = dict(saved.get("labels_typed", {}),
                     **{_tkey(it["time"]): it["label"] for it in self.items
                        if it["label"] != it.get("default")})
        rois = dict(saved.get("rois_t", {}),
                    **{_tkey(it["time"]): it["roi"] for it in self.items if it.get("roi")})
        rois_by_name = saved.get("rois", {})          # older saves
        bases = dict(saved.get("bases_t", {}),
                     **{_tkey(it["time"]): it["base"] for it in self.items if it.get("base")})
        ph = sorted(self.photos_all, key=(lambda p: p["time"]) if self.order.get() == "time"
                    else (lambda p: hu.bp.natural_key(p["path"])))
        clash = hu.order_clash(self.photos_all)
        defaults = hu.default_photo_labels(ph)

        def kept(m, p):
            # remembered under the camera's clock; a few sets saved on 24-25 Nov
            # 2026 are under the file's own time instead
            return m.get(_tkey(p["time"])) or m.get(_tkey(p["file_time"]))
        self.items = [dict(path=p["path"], name=p["name"], time=p["time"],
                           label=kept(typed, p) or lab, default=lab,
                           roi=kept(rois, p) or rois_by_name.get(p["name"]),
                           base=kept(bases, p))
                      for p, lab in zip(ph, defaults)]
        n = len(self.photos_all)
        self.info.set(f"{n} photos -> {n} steps.  Select each step, check its label, drag a box "
                      "round the bead.  (A box drawn also fills any step that has none yet.)")
        warn = ""
        if clash:
            warn = (clash + "\nThe list follows the camera clock. If a label in a file name is "
                    "on the wrong photo, fix the label here (the file is not renamed).")
        self.warn.set(warn)
        self.fill_tree()
        if self.items:
            self.tree.selection_set(str(0))

    def fill_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, it in enumerate(self.items):
            files, taken = it["name"], it["time"].strftime("%d-%m %H:%M")
            self.tree.insert("", "end", iid=str(i), values=(
                i + 1, it["label"], files, taken, "yes" if it.get("roi") else "-",
                "drawn" if it.get("base") else "-"))

    def refresh_row(self, i):
        it = self.items[i]
        vals = list(self.tree.item(str(i), "values"))
        vals[1], vals[4] = it["label"], "yes" if it.get("roi") else "-"
        vals[5] = "drawn" if it.get("base") else "-"
        self.tree.item(str(i), values=vals)

    # ------------------------------------------------------------ editing
    def on_select(self):
        s = self.tree.selection()
        if not s:
            return
        self.sel = int(s[0])
        self.lab.set(self.items[self.sel]["label"])
        self.draw_preview()

    def set_label(self):
        if self.sel is None:
            return
        self.items[self.sel]["label"] = self.lab.get().strip() or self.items[self.sel]["label"]
        self.refresh_row(self.sel)
        if self.sel + 1 < len(self.items):           # straight on to the next one
            self.tree.selection_set(str(self.sel + 1))
            self.tree.see(str(self.sel + 1))
            self.on_select()

    def copy_box(self, which):
        if self.sel is None or not self.items[self.sel].get("roi"):
            self.status.set("Draw a box on the selected step first.")
            return
        roi = self.items[self.sel]["roi"]
        rng = range(len(self.items)) if which == "all" else range(self.sel + 1, len(self.items))
        for i in rng:
            self.items[i]["roi"] = roi
            self.refresh_row(i)
        self.status.set("box copied - check each step: the bead may sit elsewhere in some photos")

    def clear_box(self):
        if self.sel is not None:
            self.items[self.sel]["roi"] = None
            self.refresh_row(self.sel)
            self.draw_preview()

    def clear_base(self):
        if self.sel is not None:
            self.items[self.sel]["base"] = None
            self.refresh_row(self.sel)
            self.draw_preview()

    def nudge(self, dl, dr):
        """Move the baseline's left / right end by whole pixels (down = +)."""
        if self.sel is None or not self.items[self.sel].get("base"):
            self.status.set("Draw a baseline first: choose 'the baseline' and drag along the "
                            "bead's base.")
            return
        x0, y0, x1, y1 = self.items[self.sel]["base"]
        self.items[self.sel]["base"] = (x0, y0 + dl, x1, y1 + dr)
        self.draw_preview()

    def _full(self, path):
        if self.full_img[0] != path:
            self.full_img = (path, cv2.imread(path))
        return self.full_img[1]

    def draw_strip(self, path, it):
        """The base, zoomed: STRIP_ROWS image rows round the line across the box,
        stretched to STRIP_H screen px, with the line on it."""
        self.strip.delete("all")
        base = it.get("base")
        if not base:
            self.strip.create_text(VIEW_W // 2, STRIP_H // 2, fill="#888",
                                   text="the zoomed base appears here once a baseline is drawn")
            return
        bgr = self._full(path)
        if bgr is None:
            return
        H, W = bgr.shape[:2]
        lx0, ly0, lx1, ly1 = base
        slope = (ly1 - ly0) / (lx1 - lx0) if lx1 != lx0 else 0.0
        # the bead's own corners on the line, measured as the run will measure it
        # exactly the settings the run will use -- the threshold offset included,
        # or the strip shows an outline the run never measures
        try:
            toff = int(float(self.toff.get() or 0))
        except ValueError:
            toff = 0
        cfg = dict(hu.bp.DEFAULTS, **hu.base_cfg(toff))
        cfg.update(ROI=it.get("roi"), PAIR_MODE=True, **hu.base_override(base))
        m = hu.bp.measure_one((path, cfg, True))
        if m.get("ok"):
            low = m["Y"] >= m["Y"][-1] - hu.bp.BASE_LINE_BAND_PX
            ends = [float(m["xl"][low].min()), float(m["xr"][low].max())]
            names = ("LEFT corner (yellow = measured, pink = line)", "RIGHT corner")
        else:
            m, ends, names = None, [lx0, lx1], ("LEFT end of the line", "RIGHT end of the line")
        half = VIEW_W // 2
        ww, wh = int(half / STRIP_ZOOM), int(STRIP_H / STRIP_ZOOM)   # image px per panel
        panels = []
        for k, ex in enumerate(ends):
            ey = ly0 + slope * (ex - lx0)
            cx0 = int(max(0, min(W - ww, ex - ww // 2)))
            cy0 = int(max(0, min(H - wh, ey - wh * 0.6)))
            win = cv2.normalize(bgr[cy0:cy0 + wh, cx0:cx0 + ww], None, 0, 255,
                                cv2.NORM_MINMAX)          # stretch: the blur is low-contrast
            tile = cv2.resize(win, (half, STRIP_H), interpolation=cv2.INTER_NEAREST)
            if m is not None:                             # the measured outline, yellow
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
            panels.append(tile)
            # where the line runs through this panel
            ya = (ly0 + slope * (cx0 - lx0) - cy0) * STRIP_ZOOM
            yb = (ly0 + slope * (cx0 + ww - lx0) - cy0) * STRIP_ZOOM
            panels[-1] = (panels[-1], k * half, ya, yb, (ex - cx0) * STRIP_ZOOM + k * half,
                          (ey - cy0) * STRIP_ZOOM)
        img = cv2.hconcat([p[0] for p in panels])
        ok, png = cv2.imencode(".png", img)
        self.strip_img = self.tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))
        self.strip.create_image(0, 0, anchor="nw", image=self.strip_img)
        for _, off, ya, yb, ex, ey in panels:
            self.strip.create_line(off, ya, off + half, yb, fill="#ff2d95", width=2)
            self.strip.create_oval(ex - 4, ey - 4, ex + 4, ey + 4, outline="#ffcc00", width=2)
        self.strip.create_line(half, 0, half, STRIP_H, fill="#ffffff")
        for k, name in enumerate(names):
            self.strip.create_text(k * half + 6, 4, anchor="nw", fill="#ffcc00",
                                   text=f"{name}, {STRIP_ZOOM}x")

    # ------------------------------------------------------------ picture
    def _image(self, path):
        if path not in self.cache:
            bgr = cv2.imread(path)
            if bgr is None:
                return None
            H, W = bgr.shape[:2]
            sc = min(VIEW_W / W, VIEW_H / H)
            small = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            ok, png = cv2.imencode(".png", small)
            self.cache[path] = (self.tk.PhotoImage(data=base64.b64encode(png.tobytes())
                                                   .decode("ascii")), sc, (W, H))
            if len(self.cache) > 24:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[path]

    def draw_preview(self):
        if self.sel is None or self.sel >= len(self.items):
            self.cv.delete("all")
            return
        it = self.items[self.sel]
        path = it["path"]
        bgr = self._full(path)
        if bgr is None:
            self.cv.delete("all")
            self.lf.config(text=f"could not read {os.path.basename(path)}")
            return
        # the same photo again (a nudge, a new label): keep the zoom where it is
        self.view.set_image(bgr, keep_view=(path == getattr(self, "shown", None)))
        self.shown = path
        self.full = self.view.size()
        self.lf.config(text=f"{self.sel + 1}. {it['label']}  -  {os.path.basename(path)}"
                       + {"base": "  -  drag along the bead's BASE",
                          "click": "  -  click ONE point on the mat"}.get(
                           self.mode.get(), "  -  drag a box round the bead"))
        self.draw_overlays()
        self.draw_strip(path, it)

    def draw_overlays(self):
        """Box and baseline in the current zoom (also after every zoom / move)."""
        self.cv.delete("box")
        self.cv.delete("base")
        if self.sel is None or self.sel >= len(self.items):
            return
        it, v = self.items[self.sel], self.view
        if it.get("roi"):
            r = it["roi"]
            self.cv.create_rectangle(*v.to_canvas(r[0], r[2]), *v.to_canvas(r[1], r[3]),
                                     outline="#34c759", width=2, tags="box")
        if it.get("base"):
            x0, y0, x1, y1 = it["base"]
            self.cv.create_line(*v.to_canvas(x0, y0), *v.to_canvas(x1, y1), fill="#ff2d95",
                                width=2, tags="base")

    def drag_start(self, ev):
        self.drag0 = (ev.x, ev.y) if self.sel is not None else None

    def drag_move(self, ev):
        if self.drag0 and self.mode.get() != "click":
            self.cv.delete("drag")
            if self.mode.get() == "base":
                self.cv.create_line(*self.drag0, ev.x, ev.y, fill="#ff2d95", width=2,
                                    dash=(4, 2), tags="drag")
            else:
                self.cv.create_rectangle(*self.drag0, ev.x, ev.y, outline="#ffcc00", width=2,
                                         dash=(4, 2), tags="drag")

    def drag_end(self, ev):
        if not self.drag0 or not self.full[0]:
            return
        self.cv.delete("drag")
        (ax, ay), self.drag0 = self.drag0, None
        W, H = self.full
        (ax, ay), (bx, by) = self.view.to_full(ax, ay), self.view.to_full(ev.x, ev.y)
        if self.mode.get() == "click":
            it = self.items[self.sel]
            self.status.set("finding the mat edge near your click...")
            self.root.update_idletasks()
            try:
                line, how = hu.bp.suggest_base_line(
                    it["path"], it.get("roi"), (bx, by))
            except Exception as e:
                self.status.set(f"could not read the photo: {e}")
                return
            it["base"] = tuple(round(v) for v in line)
            self.refresh_row(self.sel)
            self.draw_preview()
            self.status.set("baseline found from the mat's colour - check the magnified ends"
                            if how == "found" else
                            "no mat edge found near your click - drew a level line through it; "
                            "move it with the arrows")
            return
        if self.mode.get() == "base":
            (px0, py0), (px1, py1) = sorted([(ax, ay), (bx, by)])
            if px1 - px0 < 20:
                return
            self.items[self.sel]["base"] = (round(px0), round(py0), round(px1), round(py1))
            self.refresh_row(self.sel)
            self.draw_preview()
            return
        x0, x1 = sorted((ax, bx))
        y0, y1 = sorted((ay, by))
        x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
        x1, y1 = min(W, int(round(x1))), min(H, int(round(y1)))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return
        roi = (x0, x1, y0, y1)
        self.items[self.sel]["roi"] = roi
        self.refresh_row(self.sel)
        for i, it in enumerate(self.items):            # fill steps with no box yet
            if not it.get("roi"):
                it["roi"] = roi
                self.refresh_row(i)
        self.draw_preview()

    # --------------------------------------------------------------- run
    def remember(self):
        d = self.dkey()
        sets = dict(self.mine.get("sets", {}))
        old = sets.get(d, {})
        here = {_tkey(it["time"]) for it in self.items}
        typed = {k: v for k, v in old.get("labels_typed", {}).items() if k not in here}
        typed.update({_tkey(it["time"]): it["label"] for it in self.items
                      if it["label"] != it.get("default")})
        sets[d] = dict(order=self.order.get(),
                       toff=self.toff.get(), labels_typed=typed,
                       rois_t=dict(old.get("rois_t", {}),
                                   **{_tkey(it["time"]): list(it["roi"]) for it in self.items
                                      if it.get("roi")}),
                       bases_t={**{k: v for k, v in old.get("bases_t", {}).items()
                                   if k not in here},
                                **{_tkey(it["time"]): list(it["base"]) for it in self.items
                                   if it.get("base")}})
        self.mine = dict(self.mine, last_dir=d, sets=sets)
        allset = rl.load_settings()
        allset[KEY] = self.mine
        rl.save_settings(allset)

    def start(self):
        from tkinter import messagebox
        if self.busy or not self.items:
            return
        missing = [str(i + 1) for i, it in enumerate(self.items) if not it.get("roi")]
        if missing:
            messagebox.showinfo("Boxes missing", "Draw a box round the bead on step(s) "
                                + ", ".join(missing) + " first.", parent=self.root)
            return
        try:
            toff = int(float(self.toff.get() or 0))
        except ValueError:
            self.status.set("Threshold offset must be a whole number, e.g. 0 or -40.")
            return
        if len(self.items) < 2:
            self.status.set("Need at least two steps.")
            return
        self.remember()
        self.busy = True
        self.run_btn.state(["disabled"])
        for b in self.rbtn:
            b.state(["disabled"])
        self.out.delete("1.0", "end")
        self.bar["value"] = 0
        self.status.set("measuring...")
        items = [dict(it) for it in self.items]
        threading.Thread(target=self.work, args=(self.dkey(), items, toff),
                         daemon=True).start()

    def work(self, d, items, toff):
        try:
            prog = lambda i, n: self.q.put(("prog", i, n))  # noqa: E731
            res = hu.analyse_photos(d, [dict(path=it["path"], label=it["label"],
                                             roi=it["roi"], base=it.get("base"))
                                        for it in items], toff, progress=prog)
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
                    self.status.set(f"measuring... {i} / {n}" if i < n else
                                    "strains and plots...")
                elif msg[0] == "done":
                    self.res = msg[1]
                    self.busy = False
                    self.run_btn.state(["!disabled"])
                    self.bar["value"] = self.bar["maximum"]
                    self.status.set(f"done - saved in {self.res['outdir']}")
                    self.out.insert("end", hu.summary_text(self.res))
                    for b in self.rbtn:
                        b.state(["!disabled"])
                elif msg[0] == "fail":
                    self.busy = False
                    self.run_btn.state(["!disabled"])
                    self.status.set("FAILED - see below")
                    self.out.insert("end", msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self.poll)


def main():
    import tkinter as tk
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
