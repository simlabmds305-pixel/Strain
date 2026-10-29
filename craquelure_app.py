"""
Craquelure app - one window for the whole job, no commands to type.

  1. Open paintings (or a folder of them).
  2. Drag boxes on the painting round cracked paint. Click a box to select it.
     Zoom in with the mouse wheel (or + / -), move around with a right-drag,
     "Fit" shows the whole painting again.
  3. For the selected box you see the enhanced, binary and skeleton pictures
     and Bucklow's classification (scores per tradition + descriptor words).
  4. Fill in the painting's width, the grain / longest-side direction and the
     support - the classification uses them.
  5. "Save all" writes every box's pictures to <painting>_cracks folders and one
     results table (craquelure_results.csv) next to the paintings.

Boxes and settings are remembered (craquelure_session.json next to the
paintings), so you can close the app and carry on later.

Start it by double-clicking Craquelure.bat (or: python craquelure_app.py).
Needs craquelure_extract.py and bucklow_classify.py in the same folder.
"""

import json
import os
import sys
import traceback

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import craquelure_extract as ce          # noqa: E402
import bucklow_classify as bc            # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
SESSION = "craquelure_session.json"
RESULTS = "craquelure_results.csv"
TRAD_COLOURS = {"Italian": "#2a78d6", "Flemish": "#eb6834", "Dutch": "#1baf7a", "French": "#eda100"}
BOX_COLOUR, SEL_COLOUR = "#ffd400", "#ff3030"


def is_output(path):
    """Pictures the program wrote itself - never offered as paintings."""
    d = os.path.basename(os.path.dirname(path))
    return d.endswith("_cracks")


def analyse_box(img, roi, st):
    """Process one box with the current settings -> dict of pictures and results."""
    x, y, w, h = roi
    crop = img[y:y + h, x:x + w]
    p = ce.auto_params(crop, st["colour"])
    p["sensitivity"], p["cleanup"], p["spurs"] = st["sensitivity"], st["cleanup"], st["spurs"]
    enh = ce.extract(crop, p)["enhanced"]
    area_cm = None
    if st.get("width_cm"):
        area_cm = st["width_cm"] * w / img.shape[1]
    r = bc.analyse(None, "area" if area_cm else None, area_cm,
                   None if st["grain"] == "unknown" else st["grain"],
                   None if st["support"] == "unknown" else st["support"],
                   st["sensitivity"], st["cleanup"], st["spurs"], gray=enh)
    r.update(crop=crop, enhanced=enh, params=p, roi=list(roi))
    return r


def run():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Craquelure")
    root.geometry("1500x900")
    try:
        root.state("zoomed")                      # full screen on Windows
    except tk.TclError:
        pass

    paintings = []          # list of paths
    data = {}               # path -> {"boxes": [[x,y,w,h], ...], "width_cm": "", "grain": ..., "support": ...}
    cache = {}              # (path, box index) -> analyse_box result
    S = dict(cur=None, img=None, sel=None, scale=1.0, off=(0, 0), drag=None, photo=None,
             zoom=None, centre=(0, 0), pan=None,
             tabs={}, job=None, session_dir=None)

    # ------------------------------------------------------------------ layout
    top = ttk.Frame(root, padding=(8, 6)); top.pack(fill="x")
    ttk.Button(top, text="Open paintings...", command=lambda: open_files()).pack(side="left")
    ttk.Button(top, text="Open folder...", command=lambda: open_folder()).pack(side="left", padx=4)
    ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
    ttk.Button(top, text="< Prev", width=8, command=lambda: step(-1)).pack(side="left")
    name_lbl = ttk.Label(top, text="no painting open", width=40, anchor="center",
                         font=("Segoe UI", 11, "bold"))
    name_lbl.pack(side="left", padx=6)
    ttk.Button(top, text="Next >", width=8, command=lambda: step(1)).pack(side="left")
    ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
    ttk.Button(top, text="Save all", command=lambda: save_all()).pack(side="left")
    ttk.Button(top, text="Open results folder", command=lambda: open_results()).pack(side="left", padx=4)
    prog = ttk.Progressbar(top, length=180, mode="determinate"); prog.pack(side="left", padx=8)

    status = ttk.Label(root, text="Open paintings or a folder to begin.", anchor="w", padding=(8, 3))
    status.pack(side="bottom", fill="x")

    body = ttk.Panedwindow(root, orient="horizontal"); body.pack(fill="both", expand=True)
    left = ttk.Frame(body, padding=4); centre = ttk.Frame(body, padding=4); right = ttk.Frame(body, padding=4)
    body.add(left, weight=1); body.add(centre, weight=5); body.add(right, weight=4)

    # left: paintings + per-painting settings
    ttk.Label(left, text="Paintings").pack(anchor="w")
    plist = tk.Listbox(left, width=28, exportselection=False); plist.pack(fill="both", expand=True)
    pf = ttk.LabelFrame(left, text="This painting", padding=6); pf.pack(fill="x", pady=(6, 0))
    v_width, v_grain, v_support = tk.StringVar(), tk.StringVar(value="unknown"), tk.StringVar(value="unknown")
    ttk.Label(pf, text="Width of the whole\npainting (cm)").grid(row=0, column=0, sticky="w")
    ttk.Entry(pf, textvariable=v_width, width=8).grid(row=0, column=1, sticky="w", padx=4)
    ttk.Label(pf, text="Grain (panel) /\nlongest side (canvas)").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Combobox(pf, textvariable=v_grain, values=["unknown", "vertical", "horizontal"],
                 state="readonly", width=10).grid(row=1, column=1, sticky="w", padx=4)
    ttk.Label(pf, text="Support").grid(row=2, column=0, sticky="w")
    ttk.Combobox(pf, textvariable=v_support, values=["unknown", "panel", "canvas"],
                 state="readonly", width=10).grid(row=2, column=1, sticky="w", padx=4)

    cf = ttk.LabelFrame(left, text="Crack detection (all paintings)", padding=6); cf.pack(fill="x", pady=(6, 0))
    v_colour = tk.StringVar(value="auto")
    v_sens, v_clean, v_spurs = tk.IntVar(value=10), tk.IntVar(value=20), tk.IntVar(value=8)
    ttk.Label(cf, text="Cracks are").grid(row=0, column=0, sticky="w")
    ttk.Combobox(cf, textvariable=v_colour, values=["auto", "dark", "light"], state="readonly",
                 width=8).grid(row=0, column=1, sticky="w")
    for r, (lbl, var, mx) in enumerate([("Sensitivity", v_sens, 20), ("Clean-up", v_clean, 200),
                                        ("Trim spurs", v_spurs, 50)], start=1):
        ttk.Label(cf, text=lbl).grid(row=r, column=0, sticky="w")
        tk.Scale(cf, variable=var, from_=0, to=mx, orient="horizontal", length=130,
                 command=lambda _v: settings_changed()).grid(row=r, column=1, sticky="w")

    # centre: the painting
    ttk.Label(centre, text="Drag a box round cracked paint. Click a box to select it.",
              anchor="w").pack(fill="x")
    zbar = ttk.Frame(centre); zbar.pack(fill="x", pady=(2, 2))
    ttk.Label(zbar, text="Zoom: mouse wheel   Move: right-drag",
              anchor="w", foreground="#52514e").pack(side="left")
    ttk.Button(zbar, text="Fit", width=5, command=lambda: zoom_fit()).pack(side="right")
    ttk.Button(zbar, text="1:1", width=5, command=lambda: zoom_to(1.0)).pack(side="right")
    ttk.Button(zbar, text="-", width=3, command=lambda: zoom_by(1 / 1.5)).pack(side="right")
    zoom_lbl = ttk.Label(zbar, text="", width=7, anchor="center"); zoom_lbl.pack(side="right")
    ttk.Button(zbar, text="+", width=3, command=lambda: zoom_by(1.5)).pack(side="right")
    canvas = tk.Canvas(centre, bg="#2b2b2b", highlightthickness=0, cursor="crosshair")
    canvas.pack(fill="both", expand=True)
    cb = ttk.Frame(centre); cb.pack(fill="x", pady=(4, 0))
    box_lbl = ttk.Label(cb, text="no box"); box_lbl.pack(side="left")
    ttk.Button(cb, text="Delete box", command=lambda: delete_box()).pack(side="right")

    # right: pictures of the selected box + classification
    nb = ttk.Notebook(right); nb.pack(fill="both", expand=True)
    for name in ("Enhanced", "Binary", "Skeleton", "Overlay"):
        c = tk.Canvas(nb, bg="#2b2b2b", highlightthickness=0)
        nb.add(c, text=name); S["tabs"][name] = c
        c.bind("<Configure>", lambda e: show_box())
    rf = ttk.LabelFrame(right, text="Bucklow classification", padding=6); rf.pack(fill="x", pady=(6, 0))
    bars = tk.Canvas(rf, height=150, bg="white", highlightthickness=0); bars.pack(fill="x")
    words = tk.Text(rf, height=13, wrap="word", font=("Consolas", 10)); words.pack(fill="x")

    # ------------------------------------------------------------- session
    def session_path():
        return os.path.join(S["session_dir"], SESSION) if S["session_dir"] else None

    def load_session():
        p = session_path()
        if p and os.path.exists(p):
            try:
                with open(p) as f:
                    saved = json.load(f)
                for k, v in saved.items():
                    if k in data:
                        data[k].update(v)
            except (OSError, ValueError):
                pass

    def save_session():
        p = session_path()
        if not p:
            return
        try:
            old = {}
            if os.path.exists(p):
                with open(p) as f:
                    old = json.load(f)
            old.update(data)
            with open(p, "w") as f:
                json.dump(old, f, indent=1)
        except (OSError, ValueError):
            pass

    # ----------------------------------------------------------- paintings
    def add_paintings(paths):
        new = []
        for p in paths:
            p = os.path.abspath(p)
            if os.path.isdir(p):
                new += sorted(os.path.join(p, f) for f in os.listdir(p)
                              if f.lower().endswith(IMG_EXTS))
            elif p.lower().endswith(IMG_EXTS):
                new.append(p)
        new = [p for p in new if not is_output(p) and p not in paintings]
        if not new:
            status.config(text="no paintings found there"); return
        if S["session_dir"] is None:
            S["session_dir"] = os.path.dirname(new[0])
        for p in new:
            paintings.append(p)
            data[p] = dict(boxes=[], width_cm="", grain="unknown", support="unknown")
        load_session()
        refresh_list()
        if S["cur"] is None:
            select_painting(paintings.index(new[0]))
        status.config(text=f"{len(paintings)} painting(s). Drag boxes round cracked paint.")

    def open_files():
        ps = filedialog.askopenfilenames(title="Open paintings",
                                         filetypes=[("Images", " ".join("*" + e for e in IMG_EXTS)),
                                                    ("All", "*.*")])
        if ps:
            add_paintings(list(ps))

    def open_folder():
        d = filedialog.askdirectory(title="Folder of paintings")
        if d:
            add_paintings([d])

    def refresh_list():
        cur = S["cur"]
        plist.delete(0, "end")
        for p in paintings:
            n = len(data[p]["boxes"])
            plist.insert("end", os.path.splitext(os.path.basename(p))[0] + (f"   ({n} box)" if n == 1 else
                                                                            f"   ({n} boxes)" if n else ""))
        if cur is not None:
            plist.selection_set(paintings.index(cur))

    def select_painting(i):
        if not paintings:
            return
        i = max(0, min(i, len(paintings) - 1))
        p = paintings[i]
        status.config(text=f"loading {os.path.basename(p)} ..."); root.update_idletasks()
        img = ce.imread(p)
        if img is None:
            status.config(text=f"could not read {p}"); return
        S["cur"], S["img"], S["sel"] = p, img, (0 if data[p]["boxes"] else None)
        S["zoom"], S["centre"] = None, (img.shape[1] / 2, img.shape[0] / 2)     # start fitted
        d = data[p]
        loading[0] = True
        v_width.set(d["width_cm"]); v_grain.set(d["grain"]); v_support.set(d["support"])
        loading[0] = False
        name_lbl.config(text=f"{os.path.basename(p)}   ({i + 1}/{len(paintings)})")
        refresh_list(); draw_painting(); show_box()
        status.config(text=f"{os.path.basename(p)}: {img.shape[1]} x {img.shape[0]} px. "
                           f"Drag a box round cracked paint.")

    def step(k):
        if S["cur"] is not None:
            select_painting(paintings.index(S["cur"]) + k)

    loading = [False]

    def painting_settings_changed(*_a):
        if loading[0] or S["cur"] is None:
            return
        d = data[S["cur"]]
        d["width_cm"], d["grain"], d["support"] = v_width.get(), v_grain.get(), v_support.get()
        for k in [k for k in cache if k[0] == S["cur"]]:
            del cache[k]
        save_session(); schedule_box()

    for v in (v_width, v_grain, v_support):
        v.trace_add("write", painting_settings_changed)
    v_colour.trace_add("write", lambda *a: settings_changed())

    def settings_changed():
        cache.clear(); schedule_box()

    def settings_for(p):
        d = data[p]
        try:
            wcm = float(str(d["width_cm"]).replace(",", ".")) if str(d["width_cm"]).strip() else None
        except ValueError:
            wcm = None
        return dict(width_cm=wcm, grain=d["grain"], support=d["support"], colour=v_colour.get(),
                    sensitivity=v_sens.get(), cleanup=v_clean.get(), spurs=v_spurs.get())

    # ------------------------------------------------------------ drawing
    def fit_scale():
        cw, ch = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        H, W = S["img"].shape[:2]
        return min(cw / W, ch / H)

    def view_scale():
        """Screen pixels per painting pixel (S['zoom'] None = fit the whole painting)."""
        return S["zoom"] if S["zoom"] else fit_scale()

    def clamp_centre():
        H, W = S["img"].shape[:2]
        sc = view_scale()
        cw, ch = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        cx, cy = S["centre"]
        hw, hh = cw / 2 / sc, ch / 2 / sc                # half the view, in painting pixels
        cx = W / 2 if hw * 2 >= W else min(max(cx, hw), W - hw)
        cy = H / 2 if hh * 2 >= H else min(max(cy, hh), H - hh)
        S["centre"] = (cx, cy)

    def draw_painting():
        canvas.delete("all")
        img = S["img"]
        if img is None:
            return
        cw, ch = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        clamp_centre()
        sc = view_scale()
        H, W = img.shape[:2]
        cx, cy = S["centre"]
        # only the visible part of the painting is resized - fast even for huge scans
        ix0, iy0 = max(int(cx - cw / 2 / sc), 0), max(int(cy - ch / 2 / sc), 0)
        ix1, iy1 = min(int(np.ceil(cx + cw / 2 / sc)) + 1, W), min(int(np.ceil(cy + ch / 2 / sc)) + 1, H)
        part = img[iy0:iy1, ix0:ix1]
        size = (max(int(round((ix1 - ix0) * sc)), 1), max(int(round((iy1 - iy0) * sc)), 1))
        interp = cv2.INTER_AREA if sc < 1 else (cv2.INTER_LINEAR if sc < 4 else cv2.INTER_NEAREST)
        disp = cv2.cvtColor(cv2.resize(part, size, interpolation=interp), cv2.COLOR_BGR2RGB)
        S["photo"] = ImageTk.PhotoImage(Image.fromarray(disp))
        ox, oy = cw / 2 - cx * sc, ch / 2 - cy * sc       # screen position of painting pixel (0,0)
        S["scale"], S["off"] = sc, (ox, oy)
        canvas.create_image(ox + ix0 * sc, oy + iy0 * sc, image=S["photo"], anchor="nw")
        for i, (x, y, w, h) in enumerate(data[S["cur"]]["boxes"]):
            col = SEL_COLOUR if i == S["sel"] else BOX_COLOUR
            x0, y0 = ox + x * sc, oy + y * sc
            canvas.create_rectangle(x0, y0, x0 + w * sc, y0 + h * sc, outline=col, width=3 if i == S["sel"] else 2)
            canvas.create_text(x0 + 6, y0 + 4, text=str(i + 1), anchor="nw", fill=col,
                               font=("Segoe UI", 12, "bold"))
        zoom_lbl.config(text=f"{sc * 100:.0f}%")
        n = len(data[S["cur"]]["boxes"])
        box_lbl.config(text=(f"box {S['sel'] + 1} of {n} selected" if S["sel"] is not None else
                             "no box - drag one on the painting"))

    def zoom_to(new, ex=None, ey=None):
        """Zoom to 'new' screen px per painting px, keeping the point under (ex, ey) still."""
        if S["img"] is None:
            return
        new = min(max(new, fit_scale() * 0.5), 16.0)
        cw, ch = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        ex = cw / 2 if ex is None else ex
        ey = ch / 2 if ey is None else ey
        sc = view_scale()
        cx, cy = S["centre"]
        px, py = cx + (ex - cw / 2) / sc, cy + (ey - ch / 2) / sc     # painting point under cursor
        S["zoom"] = new
        S["centre"] = (px - (ex - cw / 2) / new, py - (ey - ch / 2) / new)
        draw_painting()

    def zoom_by(f, ex=None, ey=None):
        if S["img"] is not None:
            zoom_to(view_scale() * f, ex, ey)

    def zoom_fit():
        if S["img"] is not None:
            S["zoom"] = None; draw_painting()

    def on_wheel(e):
        up = (getattr(e, "delta", 0) > 0) or getattr(e, "num", 0) == 4
        zoom_by(1.25 if up else 1 / 1.25, e.x, e.y)

    def pan_start(e):
        S["pan"] = (e.x, e.y, S["centre"]); canvas.config(cursor="fleur")

    def pan_move(e):
        if S.get("pan"):
            x0, y0, (cx, cy) = S["pan"]
            sc = view_scale()
            S["centre"] = (cx - (e.x - x0) / sc, cy - (e.y - y0) / sc)
            draw_painting()

    def pan_end(_e):
        S["pan"] = None; canvas.config(cursor="crosshair")

    def to_img(ex, ey):
        sc, (ox, oy) = S["scale"], S["off"]
        H, W = S["img"].shape[:2]
        return (min(max((ex - ox) / sc, 0), W), min(max((ey - oy) / sc, 0), H))

    def on_press(e):
        if S["img"] is None:
            return
        S["drag"] = (e.x, e.y)
        canvas.delete("rubber")

    def on_move(e):
        if S["drag"]:
            canvas.delete("rubber")
            canvas.create_rectangle(*S["drag"], e.x, e.y, outline=SEL_COLOUR, width=2, dash=(4, 3),
                                    tags="rubber")

    def on_release(e):
        if not S["drag"]:
            return
        (sx, sy), S["drag"] = S["drag"], None
        canvas.delete("rubber")
        x0, y0 = to_img(min(sx, e.x), min(sy, e.y))
        x1, y1 = to_img(max(sx, e.x), max(sy, e.y))
        boxes = data[S["cur"]]["boxes"]
        if abs(e.x - sx) < 6 and abs(e.y - sy) < 6:          # a click: select the box under it
            hit = [i for i, (x, y, w, h) in enumerate(boxes) if x <= x0 <= x + w and y <= y0 <= y + h]
            if hit:
                S["sel"] = hit[-1]; draw_painting(); show_box()
            return
        w, h = int(x1 - x0), int(y1 - y0)
        if w < 20 or h < 20:
            status.config(text="box too small - drag a bigger one"); return
        boxes.append([int(x0), int(y0), w, h])
        S["sel"] = len(boxes) - 1
        save_session(); refresh_list(); draw_painting(); show_box()

    def delete_box():
        if S["cur"] is None or S["sel"] is None:
            return
        boxes = data[S["cur"]]["boxes"]
        del boxes[S["sel"]]
        for k in [k for k in cache if k[0] == S["cur"]]:
            del cache[k]
        S["sel"] = (len(boxes) - 1) if boxes else None
        save_session(); refresh_list(); draw_painting(); show_box()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<Configure>", lambda e: draw_painting())
    canvas.bind("<MouseWheel>", on_wheel)                    # Windows / Mac
    canvas.bind("<Button-4>", on_wheel); canvas.bind("<Button-5>", on_wheel)   # Linux
    for btn in ("3", "2"):                                    # right or middle button drags the view
        canvas.bind(f"<ButtonPress-{btn}>", pan_start)
        canvas.bind(f"<B{btn}-Motion>", pan_move)
        canvas.bind(f"<ButtonRelease-{btn}>", pan_end)
    root.bind("<plus>", lambda e: zoom_by(1.5)); root.bind("<equal>", lambda e: zoom_by(1.5))
    root.bind("<minus>", lambda e: zoom_by(1 / 1.5)); root.bind("<Key-0>", lambda e: zoom_fit())
    root.bind("<Delete>", lambda e: delete_box())
    root.bind("<Right>", lambda e: step(1))
    root.bind("<Left>", lambda e: step(-1))
    plist.bind("<<ListboxSelect>>", lambda e: plist.curselection() and select_painting(plist.curselection()[0]))

    # ----------------------------------------------------------- selected box
    def result_for(p, i, img=None):
        key = (p, i)
        if key not in cache:
            img = S["img"] if img is None else img
            cache[key] = analyse_box(img, data[p]["boxes"][i], settings_for(p))
        return cache[key]

    def schedule_box():
        if S["job"]:
            root.after_cancel(S["job"])
        S["job"] = root.after(250, show_box)

    def fit(c, rgb):
        cw, ch = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        sc = min(cw / rgb.shape[1], ch / rgb.shape[0], 6.0)
        size = (max(int(rgb.shape[1] * sc), 1), max(int(rgb.shape[0] * sc), 1))
        return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_NEAREST), sc

    def show_box():
        S["job"] = None
        for c in S["tabs"].values():
            c.delete("all")
        bars.delete("all"); words.delete("1.0", "end")
        if S["cur"] is None or S["sel"] is None:
            return
        status.config(text="working on the box ..."); root.update_idletasks()
        try:
            r = result_for(S["cur"], S["sel"])
        except Exception as ex:                                  # never let one box kill the app
            status.config(text=f"could not process this box: {ex}")
            traceback.print_exc(); return
        g3 = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        pics = {"Enhanced": g3(r["enhanced"])}
        if "binary" in r:
            pics["Binary"] = g3(np.where(r["binary"], 0, 255).astype(np.uint8))
            pics["Overlay"] = cv2.cvtColor(r["crop"], cv2.COLOR_BGR2RGB)
        S["keep"] = []
        for name, c in S["tabs"].items():
            if name == "Skeleton" and "skeleton" in r:
                sk = r["skeleton"].astype(np.uint8)
                disp, sc = fit(c, g3(np.full(sk.shape, 255, np.uint8)))
                k = max(int(np.ceil(1 / sc)), 1) if sc < 1 else 1
                skd = cv2.resize(cv2.dilate(sk, np.ones((k, k), np.uint8)), disp.shape[1::-1],
                                 interpolation=cv2.INTER_NEAREST) > 0
                disp[skd] = 0
            elif name == "Overlay" and "skeleton" in r:
                disp, sc = fit(c, pics["Overlay"])
                k = max(int(np.ceil(1 / sc)), 1) if sc < 1 else 1
                skd = cv2.resize(cv2.dilate(r["skeleton"].astype(np.uint8), np.ones((k, k), np.uint8)),
                                 disp.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
                disp[skd] = (230, 30, 30)
            elif name in pics:
                disp, _ = fit(c, pics[name])
            else:
                continue
            ph = ImageTk.PhotoImage(Image.fromarray(disp)); S["keep"].append(ph)
            c.create_image(max(c.winfo_width(), 50) // 2, max(c.winfo_height(), 50) // 2, image=ph)
        show_result(r)
        status.config(text=f"box {S['sel'] + 1}: {r['roi'][2]} x {r['roi'][3]} px, "
                           f"{'light' if r['params']['light'] else 'dark'} cracks, "
                           f"crack width ~{r['params']['crack_width']} px")

    def show_result(r):
        if r.get("error"):
            words.insert("end", f"No classification: {r['error']}.\n"
                                "Draw a bigger box, or raise Sensitivity.")
            return
        w = max(bars.winfo_width(), 300)
        for i, (t, s) in enumerate(r["scores"].items()):
            y = 8 + i * 35
            bars.create_text(6, y, text=t, anchor="nw", fill="#0b0b0b",
                             font=("Segoe UI", 9, "bold" if i == 0 else "normal"))
            bars.create_rectangle(6, y + 16, 6 + max((w - 64) * s, 2), y + 28,
                                  fill=TRAD_COLOURS[t.split()[0]], outline="")
            bars.create_text(w - 50, y + 22, text=f"{s:.2f}", anchor="w", fill="#52514e",
                             font=("Segoe UI", 9))
        words.insert("end", f"Best match:  {r['best']}\n\n")
        for d, v in r["values"].items():
            words.insert("end", f"  {r['words'][d]:<38s}{v:.2f}\n")
        for n in r.get("notes", []):
            words.insert("end", "\nNote: " + n)
        words.insert("end", "\n\nA pointer after Bucklow (1997), not a verdict.")

    # ------------------------------------------------------------- saving
    def save_all():
        jobs = [(p, i) for p in paintings for i in range(len(data[p]["boxes"]))]
        if not jobs:
            messagebox.showinfo("Nothing to save", "Drag at least one box on a painting first."); return
        rows, prog["maximum"], prog["value"] = [], len(jobs), 0
        imgs = {}
        for k, (p, i) in enumerate(jobs, 1):
            status.config(text=f"saving {k}/{len(jobs)}: {os.path.basename(p)} box {i + 1} ...")
            root.update()
            try:
                if p not in imgs:
                    imgs.clear(); imgs[p] = S["img"] if p == S["cur"] else ce.imread(p)
                img = imgs[p]
                r = result_for(p, i, img)
                stem = os.path.splitext(os.path.basename(p))[0]
                out_dir = os.path.splitext(p)[0] + "_cracks"
                res = {"enhanced": r["enhanced"],
                       "binary": r.get("binary", np.zeros(r["enhanced"].shape, bool)),
                       "skeleton": r.get("skeleton", np.zeros(r["enhanced"].shape, bool))}
                tag = f"{stem}_sel{i + 1}"
                ce.save(out_dir, tag, r["crop"], res, r["params"], r["roi"], [img.shape[1], img.shape[0]])
                row = {"painting": os.path.basename(p), "box": i + 1}
                row.update(bc.result_row(tag, r))
                row.update(width_cm=data[p]["width_cm"], grain=data[p]["grain"], support=data[p]["support"])
                rows.append(row)
            except Exception as ex:
                rows.append({"painting": os.path.basename(p), "box": i + 1, "error": str(ex)})
                traceback.print_exc()
            prog["value"] = k
        out = os.path.join(S["session_dir"], RESULTS)
        try:
            bc.write_csv(out, rows)
        except PermissionError:
            messagebox.showerror("Could not save", f"{out} is open in another program (Excel?). "
                                                   "Close it and press Save all again."); return
        save_session()
        status.config(text=f"saved {len(rows)} box(es): pictures in each painting's _cracks folder, "
                           f"table in {out}")
        messagebox.showinfo("Saved", f"Saved {len(rows)} box(es).\n\nPictures: <painting>_cracks folders\n"
                                     f"Table: {out}")

    def open_results():
        d = S["session_dir"]
        if not d:
            return
        if sys.platform.startswith("win"):
            os.startfile(d)                              # noqa: S606 - opens Explorer
        else:
            status.config(text=d)

    if len(sys.argv) > 1:
        root.after(300, lambda: add_paintings(sys.argv[1:]))
    root.mainloop()


if __name__ == "__main__":
    try:
        run()
    except Exception:                                   # show the error even without a console
        msg = traceback.format_exc()
        try:
            import tkinter as tk
            from tkinter import messagebox
            r = tk.Tk(); r.withdraw()
            messagebox.showerror("Craquelure - error", msg[-1500:])
        except Exception:
            print(msg)
        raise
