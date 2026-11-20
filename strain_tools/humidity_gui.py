#!/usr/bin/env python3
# ============================================================================
#  HUMIDITY CYCLES  |  window
#
#  For the humidity sets: a bead photographed fresh, then after 72 % RH, after
#  97 % RH, ... (BEFORE / AFTER PHOTOS), or one folder of time-lapse
#  recordings separated by the exposures (TIME-LAPSE BLOCKS).
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
import humidity as hu       # noqa: E402
import run_local as rl      # noqa: E402

VIEW_W, VIEW_H = 680, 440
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
        self.items = []             # dict(paths, name, time, label, roi)
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

        # ---- kind, order, options
        opt = ttk.Frame(root)
        opt.pack(fill="x", **pad)
        ttk.Label(opt, text="This folder is:").grid(row=0, column=0, sticky="w")
        self.kind = tk.StringVar(value="photos")
        ttk.Radiobutton(opt, text="Before / after photos (one photo per step)",
                        variable=self.kind, value="photos",
                        command=self.rebuild).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Radiobutton(opt, text="Time-lapse blocks (recordings split by the gaps)",
                        variable=self.kind, value="timelapse",
                        command=self.rebuild).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(opt, text="Order:").grid(row=1, column=0, sticky="w")
        self.order = tk.StringVar(value="time")
        ttk.Radiobutton(opt, text="when each photo was taken (camera clock)",
                        variable=self.order, value="time",
                        command=self.rebuild).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Radiobutton(opt, text="file name", variable=self.order, value="name",
                        command=self.rebuild).grid(row=1, column=2, sticky="w", padx=4)
        ttk.Label(opt, text="New block after a pause of (min):").grid(row=0, column=3,
                                                                      sticky="e", padx=(24, 4))
        self.gap = tk.StringVar(value="10")
        ge = ttk.Entry(opt, textvariable=self.gap, width=6)
        ge.grid(row=0, column=4, sticky="w")
        ge.bind("<Return>", lambda _e: self.rebuild())
        ge.bind("<FocusOut>", lambda _e: self.rebuild())
        ttk.Label(opt, text="Threshold offset:").grid(row=1, column=3, sticky="e", padx=(24, 4))
        self.toff = tk.StringVar(value="0")
        ttk.Entry(opt, textvariable=self.toff, width=6).grid(row=1, column=4, sticky="w")
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
        cols = ("n", "label", "files", "taken", "box")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=10,
                                 selectmode="browse")
        for c, w, t in (("n", 34, "#"), ("label", 170, "label"), ("files", 270, "photo(s)"),
                        ("taken", 150, "taken"), ("box", 48, "box")):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w" if c != "box" else "center")
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
        self.show_last = tk.BooleanVar(value=False)
        self.cb_last = ttk.Checkbutton(right, text="show this block's LAST frame (check the bead "
                                       "stays inside the box)", variable=self.show_last,
                                       command=self.draw_preview)
        self.cb_last.pack(anchor="w")

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
        self.out = tk.Text(rf, height=8, font=("Consolas", 9), wrap="none",
                           yscrollcommand=sb.set)
        self.out.pack(fill="both", expand=True, padx=4, pady=4)
        sb.config(command=self.out.yview)
        self.rbtn = []
        for text, name in (("Strain at every step", "strain_steps.png"),
                           ("What each step did", "step_changes.png"),
                           ("Strain vs time", "strain_vs_time.png"),
                           ("Photo sheet", "photo_sheet.png"),
                           ("Outlines", "outlines.png"),
                           ("Summary", "summary.txt"),
                           ("Results folder", "")):
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
        if len(self.photos_all) < 2:
            self.info.set(f"{len(self.photos_all)} image(s) here - need at least 2.")
            self.items = []
            self.fill_tree()
            return
        saved = self.mine.get("sets", {}).get(d, {})
        self.kind.set(saved.get("kind") or ("timelapse" if len(self.photos_all) >= 20
                                            else "photos"))
        self.order.set(saved.get("order", "time"))
        self.gap.set(str(saved.get("gap", 10)))
        self.toff.set(str(saved.get("toff", 0)))
        self.status.set("")
        self.rebuild(saved)

    def rebuild(self, saved=None):
        """Make the list of steps (photos or blocks) from the photos."""
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
        ph = sorted(self.photos_all, key=(lambda p: p["time"]) if self.order.get() == "time"
                    else (lambda p: hu.bp.natural_key(p["path"])))
        clash = hu.order_clash(self.photos_all)
        if self.kind.get() == "photos":
            defaults = hu.default_photo_labels(ph)
            self.items = [dict(paths=[p["path"]], name=p["name"], time=p["time"],
                               label=typed.get(_tkey(p["time"]), lab), default=lab,
                               roi=rois.get(_tkey(p["time"])) or rois_by_name.get(p["name"]))
                          for p, lab in zip(ph, defaults)]
            self.cb_last.state(["disabled"])
            self.show_last.set(False)
        else:
            try:
                gap = float(self.gap.get())
            except ValueError:
                gap = hu.GAP_MIN
            blocks = hu.find_blocks(sorted(self.photos_all, key=lambda p: p["time"]), gap)
            defaults = hu.default_block_labels(blocks)
            self.items = [dict(paths=[p["path"] for p in b], name=b[0]["name"],
                               time=b[0]["time"], end=b[-1]["time"],
                               label=typed.get(_tkey(b[0]["time"]), lab), default=lab,
                               roi=rois.get(_tkey(b[0]["time"]))
                               or rois_by_name.get(b[0]["name"]))
                          for b, lab in zip(blocks, defaults)]
            self.cb_last.state(["!disabled"])
        n = len(self.photos_all)
        if self.kind.get() == "photos":
            self.info.set(f"{n} photos -> {len(self.items)} steps.  Select each step, check its "
                          "label, drag a box round the bead.  (A box drawn also fills any step "
                          "that has none yet.)")
        else:
            self.info.set(f"{n} frames -> {len(self.items)} recordings (a new one after every "
                          f"pause longer than {self.gap.get()} min).  Label each block and draw "
                          "its box; tick 'last frame' to check the bead stays inside.")
        warn = ""
        if clash and self.kind.get() == "photos":
            warn = (clash + "\nThe list follows the camera clock. If a label in a file name is "
                    "on the wrong photo, fix the label here (the file is not renamed).")
        if self.kind.get() == "photos" and n > 20:
            warn += ("\n" if warn else "") + (f"{n} photos as single steps - is this really a "
                                              "time-lapse? Choose 'Time-lapse blocks'.")
        self.warn.set(warn)
        self.fill_tree()
        if self.items:
            self.tree.selection_set(str(0))

    def fill_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, it in enumerate(self.items):
            if len(it["paths"]) == 1:
                files, taken = it["name"], it["time"].strftime("%d-%m %H:%M")
            else:
                files = f"{len(it['paths'])} frames from {it['name']}"
                mins = (it["end"] - it["time"]).total_seconds() / 60
                taken = f"{it['time'].strftime('%d-%m %H:%M')} +{mins:.0f}m"
            self.tree.insert("", "end", iid=str(i), values=(
                i + 1, it["label"], files, taken, "yes" if it.get("roi") else "-"))

    def refresh_row(self, i):
        it = self.items[i]
        vals = list(self.tree.item(str(i), "values"))
        vals[1], vals[4] = it["label"], "yes" if it.get("roi") else "-"
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
        self.cv.delete("all")
        if self.sel is None or self.sel >= len(self.items):
            return
        it = self.items[self.sel]
        last = self.show_last.get() and len(it["paths"]) > 1
        path = it["paths"][-1] if last else it["paths"][0]
        got = self._image(path)
        if got is None:
            self.lf.config(text=f"could not read {os.path.basename(path)}")
            return
        self.photo, self.scale, self.full = got
        self.cv.create_image(0, 0, anchor="nw", image=self.photo)
        self.lf.config(text=f"{self.sel + 1}. {it['label']}  -  {os.path.basename(path)}"
                       + ("  (LAST frame of the block)" if last else "")
                       + "  -  drag a box round the bead")
        if it.get("roi"):
            x0, x1, y0, y1 = (v * self.scale for v in it["roi"])
            self.cv.create_rectangle(x0, y0, x1, y1, outline="#34c759", width=2, tags="box")

    def drag_start(self, ev):
        self.drag0 = (ev.x, ev.y) if self.sel is not None else None

    def drag_move(self, ev):
        if self.drag0:
            self.cv.delete("drag")
            self.cv.create_rectangle(*self.drag0, ev.x, ev.y, outline="#ffcc00", width=2,
                                     dash=(4, 2), tags="drag")

    def drag_end(self, ev):
        if not self.drag0 or not self.full[0]:
            return
        self.cv.delete("drag")
        (ax, ay), self.drag0 = self.drag0, None
        W, H = self.full
        x0, x1 = sorted((ax / self.scale, ev.x / self.scale))
        y0, y1 = sorted((ay / self.scale, ev.y / self.scale))
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
        sets[d] = dict(kind=self.kind.get(), order=self.order.get(), gap=self.gap.get(),
                       toff=self.toff.get(), labels_typed=typed,
                       rois_t=dict(old.get("rois_t", {}),
                                   **{_tkey(it["time"]): list(it["roi"]) for it in self.items
                                      if it.get("roi")}))
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
        kind = self.kind.get()
        items = [dict(it) for it in self.items]
        threading.Thread(target=self.work, args=(self.dkey(), kind, items, toff),
                         daemon=True).start()

    def work(self, d, kind, items, toff):
        try:
            prog = lambda i, n: self.q.put(("prog", i, n))  # noqa: E731
            if kind == "photos":
                res = hu.analyse_photos(d, [dict(path=it["paths"][0], label=it["label"],
                                                 roi=it["roi"]) for it in items],
                                        toff, progress=prog)
            else:
                res = hu.analyse_blocks(d, [dict(paths=it["paths"], label=it["label"],
                                                 roi=it["roi"]) for it in items],
                                        toff, progress=prog)
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
                                    "strains, plots and tracking sheets...")
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
