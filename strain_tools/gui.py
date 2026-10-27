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
import bead_pipeline as bp  # noqa: E402
import run_local as rl      # noqa: E402

VIEW_W = 560            # on-screen width of each image preview
VIEW_H = 377


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
        lf1 = ttk.LabelFrame(pics, text="First image - drag a box round the bead")
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
        self.img1 = self.img2 = None

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

        ttk.Label(opt, text="Seconds between images:").grid(row=1, column=0, sticky="w")
        self.interval = tk.StringVar(value=str(self.saved.get("interval") or 60))
        ttk.Entry(opt, textvariable=self.interval, width=8).grid(row=1, column=1, sticky="w",
                                                                 padx=6)
        self.last_final = tk.BooleanVar(value=bool(self.saved.get("last_is_final", True)))
        ttk.Checkbutton(opt, text="Last image is the final photo (leave it out of the "
                        "time-lapse)", variable=self.last_final,
                        command=self.refresh_info).grid(row=2, column=0, columnspan=3,
                                                        sticky="w")
        self.first_last = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Also measure the first + last image only",
                        variable=self.first_last).grid(row=3, column=0, columnspan=3,
                                                       sticky="w")
        self.own_crop = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Own crop for the last image in first + last (drag it on "
                        "the last image - for a final photo taken after the stage moved)",
                        variable=self.own_crop, command=self.toggle_own_crop).grid(
            row=3, column=3, columnspan=3, sticky="w", padx=(20, 0))
        ttk.Label(opt, text="Last-image crop:").grid(row=4, column=3, sticky="e",
                                                     padx=(20, 0))
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
        self.out = tk.Text(rf, height=14, font=("Consolas", 10), wrap="none",
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
        root.after(100, self.poll)

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
        self.scale = min(VIEW_W / W, VIEW_H / H)
        self.img1 = self._photo(b1)
        self.img2 = self._photo(b2)
        for c, im in ((self.c1, self.img1), (self.c2, self.img2)):
            c.delete("all")
            c.create_image(0, 0, anchor="nw", image=im)
        self.names = (os.path.basename(first), os.path.basename(last))
        if not keep_roi:
            # the crop used last time on THIS folder, if any (the bead is not in
            # the same place in every set, so another set's crop is not offered)
            per = self.saved.get("rois", {})
            per2 = self.saved.get("rois_last", {})
            self.toff.set(str(self.saved.get("thresh_offsets", {}).get(d, 0)))
            self.roi2 = tuple(per2[d]) if d in per2 else None
            self.own_crop.set(self.roi2 is not None)
            self.set_roi(tuple(per[d]) if d in per else None)
        else:
            self.set_roi(self.roi)
        self.toggle_own_crop()
        self.refresh_info()

    def refresh_info(self):
        if not getattr(self, "all_paths", None):
            return
        n = len(self.all_paths)
        series = n - 1 if self.last_final.get() else n
        fin = (f"   final photo: {self.names[1]}" if self.last_final.get() else "")
        self.info.set(f"{n} images   first: {self.names[0]}   last: {self.names[1]}   "
                      f"-> time-lapse uses {series}{fin}")

    def _photo(self, bgr):
        small = cv2.resize(bgr, None, fx=self.scale, fy=self.scale,
                           interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", small)
        return self.tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))

    # -------------------------------------------------------------- crop
    LF2_SAME = "Last image - the same box, check the bead fits"
    LF2_OWN = "Last image - drag its OWN box (used only in first + last)"

    def _box(self, c, roi, colour):
        c.delete("box")
        if roi:
            x0, x1, y0, y1 = (v * self.scale for v in roi)
            c.create_rectangle(x0, y0, x1, y1, outline=colour, width=2, tags="box")

    def set_roi(self, roi):
        self.roi = tuple(int(v) for v in roi) if roi else None
        self.roi_txt.set(",".join(str(v) for v in self.roi) if self.roi else "")
        self._box(self.c1, self.roi, "#ff3b30")
        self.redraw_last()

    def set_roi2(self, roi):
        self.roi2 = tuple(int(v) for v in roi) if roi else None
        self.redraw_last()

    def redraw_last(self):
        own = self.own_crop.get()
        r = (self.roi2 or self.roi) if own else self.roi
        self._box(self.c2, r, "#34c759" if own and self.roi2 else "#ff3b30")
        self.roi2_txt.set(",".join(str(v) for v in self.roi2) if own and self.roi2 else "")
        self.c2.config(cursor="crosshair" if own else "")

    def toggle_own_crop(self):
        own = self.own_crop.get()
        self.lf2.config(text=self.LF2_OWN if own else self.LF2_SAME)
        if own and not self.roi2 and self.roi:
            self.roi2 = self.roi             # start from the time-lapse's box
        self.redraw_last()

    def typed_roi2(self):
        t = self.roi2_txt.get().strip()
        if not self.own_crop.get():
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
        return which == 1 or self.own_crop.get()

    def drag_start(self, ev, which=1):
        self.drag0 = (ev.x, ev.y) if self._drag_ok(which) else None

    def drag_move(self, ev, which=1):
        if not self.drag0:
            return
        c = self.c1 if which == 1 else self.c2
        c.delete("drag")
        c.create_rectangle(*self.drag0, ev.x, ev.y, outline="#ffcc00", width=2,
                           dash=(4, 2), tags="drag")

    def drag_end(self, ev, which=1):
        if not self.drag0 or not self.full[0]:
            return
        (self.c1 if which == 1 else self.c2).delete("drag")
        (ax, ay), self.drag0 = self.drag0, None
        W, H = self.full
        x0, x1 = sorted((ax / self.scale, ev.x / self.scale))
        y0, y1 = sorted((ay / self.scale, ev.y / self.scale))
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
                   INTERVAL_S=iv, TIME_REGEX=None)
        rois = dict(self.saved.get("rois", {}))
        if self.roi:
            rois[d] = list(self.roi)
        pair_roi = self.roi2 if self.own_crop.get() and self.roi2 else None
        toffs = dict(self.saved.get("thresh_offsets", {}))
        toffs[d] = toff
        rois_last = dict(self.saved.get("rois_last", {}))
        if pair_roi:
            rois_last[d] = list(pair_roi)
        else:
            rois_last.pop(d, None)
        self.saved = dict(self.saved, last_dir=d, interval=iv, rois=rois, rois_last=rois_last,
                          thresh_offsets=toffs,
                          last_is_final=self.last_final.get(),
                          roi=list(self.roi) if self.roi else None)
        rl.save_settings(self.saved)

        self.busy = True
        self.run_btn.state(["disabled"])
        for b in (self.b_sum, self.b_plot, self.b_mat, self.b_pair, self.b_dir):
            b.state(["disabled"])
        self.out.delete("1.0", "end")
        self.bar["value"] = 0
        self.status.set("measuring images...")
        args = (d, cfg, self.skips(), self.last_final.get(), self.first_last.get(), pair_roi)
        threading.Thread(target=self.work, args=args, daemon=True).start()

    def work(self, d, cfg, skip, last_final, first_last, pair_roi=None):
        try:
            res = rl.analyse_set(d, cfg, "analysis", "*", skip, last_final, first_last,
                                 quiet=True, pair_roi=pair_roi,
                                 progress=lambda i, n: self.q.put(("prog", i, n)))
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
        if parent not in [os.path.normpath(r) for r in roots]:
            roots = roots + [parent]
        out = self.master_out() or os.path.join(roots[0], "Master Analysis")
        if not messagebox.askyesno(
                "Refresh master analysis?",
                "Refresh the master analysis (all sets side by side) with this result?\n\n"
                "Reads every set in:\n" + "\n".join(roots) + f"\n\nWrites to:\n{out}\n\n"
                "(Use 'Master folders...' to add or change the data folders.)",
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
