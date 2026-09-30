"""
Craquelure (crack pattern) extraction from painting photos.

Follows the extraction stage of
    El-Youssef, Bucklow & Maev, "The development of a diagnostic method for
    geographical and condition-based analysis of artworks using craquelure
    pattern recognition techniques", Insight 56(3), 2014.

Pipeline (paper section "Crack detection and extraction"):
    1. greyscale
    2. morphological CLOSING with a disk slightly wider than the cracks
       -> the dark, thin cracks are filled in with the surrounding paint colour
    3. closed - original  (= black-hat)  -> an image that contains only the cracks
    4. OFFSET THRESHOLDING: split into square tiles, Otsu-threshold each tile,
       do it again with the tile grid shifted by half a tile, keep only the
       pixels that are cracks in BOTH results (true cracks agree, noise doesn't).
       A minimum threshold stops tiles without cracks from "forcing" cracks.
    5. clean: delete isolated specks / short fragments
    6. thin to 1-pixel-wide crack lines (skeleton)

Usage
    python craquelure_extract.py                 # file dialog to pick an image
    python craquelure_extract.py painting.jpg    # or give the path
    python craquelure_extract.py painting.jpg --light   # cracks brighter than paint

Then:
    1. Drag a rectangle over the area you want, press ENTER/SPACE
       (you can draw several; press ESC when finished with all of them).
    2. Two windows open for each area: a narrow "sliders" window on the left
       and a large "pictures" window beside it. Move the sliders until the
       enhanced picture shows the cracks clearly. Keys (click either window first):
          s  save this area and go to the next one
          n  skip this area
          q  quit
          v  change the picture view: original | enhanced  ->  enhanced only
             ->  all four (adds the paper's binary and skeleton)
       The "light cracks" switch (0/1) is for areas where the cracks look
       lighter than the paint, e.g. pale cracks on dark cloth.
    Results go to  <image name>_cracks/  next to the image. The main one is
    *_enhanced.png: paint flattened to an even grey, cracks in black (use the
    "contrast" and "texture cut" sliders for it). *_binary.png and
    *_skeleton.png are the paper's thresholded and thinned versions.

Requirements:  pip install opencv-python numpy scikit-image
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from skimage.morphology import skeletonize


# ----------------------------------------------------------------------------
# processing steps
# ----------------------------------------------------------------------------

def to_gray(bgr, flatten_bg=0):
    """Greyscale, optionally with slow illumination / paint-colour changes removed.

    flatten_bg: blur radius (px) of the background estimate; 0 = off.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    if flatten_bg > 0:
        k = 2 * flatten_bg + 1
        bg = cv2.GaussianBlur(gray.astype(np.float32), (k, k), 0)
        flat = gray.astype(np.float32) - bg + 128.0
        gray = np.clip(flat, 0, 255).astype(np.uint8)
    return gray


def crack_response(gray, crack_width, light_cracks=False, denoise=0):
    """Steps 2-3: closing, then subtract -> greyscale image of cracks only."""
    if denoise > 0:
        gray = cv2.bilateralFilter(gray, 2 * denoise + 1, 25, denoise)
    d = 2 * int(crack_width) + 1                      # disk wider than a crack
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    if light_cracks:                                  # bright cracks -> opening
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)
        resp = cv2.subtract(gray, opened)
        return resp, opened
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    resp = cv2.subtract(closed, gray)
    return resp, closed


def _tile_otsu(resp, tile, offset, min_thresh):
    """Otsu threshold each tile of a grid shifted by `offset` pixels."""
    h, w = resp.shape
    out = np.zeros_like(resp, dtype=bool)
    ys = list(range(-offset if offset else 0, h, tile))
    xs = list(range(-offset if offset else 0, w, tile))
    for y0 in ys:
        for x0 in xs:
            ya, yb = max(y0, 0), min(y0 + tile, h)
            xa, xb = max(x0, 0), min(x0 + tile, w)
            if yb - ya < 2 or xb - xa < 2:
                continue
            block = resp[ya:yb, xa:xb]
            t, _ = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t = max(t, min_thresh)
            out[ya:yb, xa:xb] = block > t
    return out


def offset_threshold(resp, tile, min_thresh):
    """Step 4: the paper's offset thresholding (AND of two shifted tilings)."""
    a = _tile_otsu(resp, tile, 0, min_thresh)
    b = _tile_otsu(resp, tile, tile // 2, min_thresh)
    return a & b


def clean(binary, min_area, bridge=0):
    """Step 5: optionally bridge 1-2 px gaps, then drop small components."""
    m = binary.astype(np.uint8)
    if bridge > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def thin(binary, min_len=0):
    """Step 6: 1-px skeleton; optionally drop skeleton pieces shorter than min_len."""
    sk = skeletonize(binary)
    if min_len > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_len
        sk = keep[lab]
    return sk


def enhance(bgr, crack_width, light=False, contrast=10, texture=16):
    """Flat, high-contrast crack picture: paint -> even light grey, cracks -> black.

    Each pixel is divided by the local paint level (the morphologically closed
    image, i.e. the painting with its cracks filled in), so every paint colour
    ends up the same grey and only the cracks stay dark.
      contrast: higher = more of the crack pixels pushed all the way to black
      texture : higher = faint brush / canvas texture suppressed more
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    if light:
        g = 257.0 - g
    g = cv2.GaussianBlur(g, (0, 0), 0.6)
    d = 2 * int(crack_width) + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    paint = cv2.morphologyEx(g, cv2.MORPH_CLOSE, se)
    paint = cv2.GaussianBlur(paint, (0, 0), max(crack_width, 1))
    depth = np.clip(1.0 - g / paint, 0, 1)                  # 0 on paint, >0 in a crack
    ref = max(float(np.percentile(depth, 100 - 0.03 * max(contrast, 1))), 1e-3)
    x = np.clip(depth / ref, 0, 1) ** (max(texture, 5) / 10.0)
    return (ENH_BG * (1 - x)).astype(np.uint8)


ENH_BG = 195   # grey level of the paint in the enhanced picture


def extract(bgr, p):
    """Run the whole pipeline with parameter dict p. Returns dict of images."""
    gray = to_gray(bgr, p["flatten_bg"])
    resp, closed = crack_response(gray, p["crack_width"], p["light"], p["denoise"])
    # scale by the 99.5th percentile, not the max, so one dark stain or hole
    # does not squash every real crack into the bottom few grey levels
    hi = max(float(np.percentile(resp, 99.5)), 1.0)
    resp_n = np.clip(resp.astype(np.float32) * (255.0 / hi), 0, 255).astype(np.uint8)
    binary = offset_threshold(resp_n, max(p["tile"], 4), p["min_thresh"])
    binary = clean(binary, p["min_area"], p["bridge"])
    skel = thin(binary, p["min_len"])
    enh = enhance(bgr, p["crack_width"], p["light"], p["contrast"], p["texture"])
    return dict(gray=gray, closed=closed, response=resp_n, enhanced=enh,
                binary=binary, skeleton=skel)


# ----------------------------------------------------------------------------
# simple crack-network statistics (nodes / ends / islands, as in paper Table 2)
# ----------------------------------------------------------------------------

def network_stats(skel):
    sk = skel.astype(np.uint8)
    k = np.ones((3, 3), np.float32); k[1, 1] = 0
    nb = cv2.filter2D(sk, cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT)
    ends = int(((nb == 1) & (sk == 1)).sum())
    # merge adjacent junction pixels into a single node
    junc = ((nb >= 3) & (sk == 1)).astype(np.uint8)
    nodes = cv2.connectedComponents(junc, connectivity=8)[0] - 1
    # islands = regions fully enclosed by cracks (holes of the crack network)
    inv = (1 - cv2.dilate(sk, np.ones((3, 3), np.uint8))).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    islands = len([i for i in range(1, n) if i not in border])
    # direction histogram (up-down, left-right, two diagonals) from pixel steps
    ys, xs = np.nonzero(sk)
    s = set(zip(ys.tolist(), xs.tolist()))
    dirs = {"vertical": (1, 0), "horizontal": (0, 1), "diag_TR_BL": (1, -1), "diag_TL_BR": (1, 1)}
    hist = {k_: sum((y + dy, x + dx) in s for y, x in s) for k_, (dy, dx) in dirs.items()}
    tot = sum(hist.values()) or 1
    hist = {k_: round(v / tot, 3) for k_, v in hist.items()}
    return dict(crack_length_px=int(sk.sum()), crack_density=round(float(sk.mean()), 5),
                end_points=ends, nodes=nodes, islands=islands, direction_hist=hist)


# ----------------------------------------------------------------------------
# interactive UI
# ----------------------------------------------------------------------------

SLIDERS = [  # name, key, max, default
    ("light cracks", "light", 1, 0),          # 0 = dark cracks, 1 = light cracks
    ("crack width px", "crack_width", 25, 2),
    ("contrast", "contrast", 50, 10),         # enhanced picture: how black the cracks go
    ("texture cut", "texture", 40, 16),       # enhanced picture: remove faint texture
    ("tile px", "tile", 200, 40),
    ("min thresh", "min_thresh", 255, 60),
    ("min area px", "min_area", 500, 30),
    ("min skel len", "min_len", 300, 10),
    ("bridge gaps px", "bridge", 5, 0),
    ("denoise", "denoise", 10, 0),
    ("flatten bg px", "flatten_bg", 200, 0),
]

VIEW_WIN = "craquelure - pictures   (v = change view)"
CTRL_WIN = "sliders   (s = save  n = skip  q = quit  v = view)"
CTRL_W = 420          # width of the slider window, px
VIEWS = ["original | enhanced", "enhanced only", "all four"]


def screen_size():
    """Usable screen size in px (falls back to 1600x900)."""
    try:
        import ctypes                                   # Windows
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        pass
    try:
        import tkinter as tk
        r = tk.Tk(); r.withdraw()
        wh = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return wh
    except Exception:
        return 1600, 900


def pick_image():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Pick a painting image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), ("All", "*.*")])
    root.destroy()
    return path


def fit_to_screen(img, max_w=1400, max_h=850, upscale=1.0):
    """Scale img to fit max_w x max_h; small images may grow up to `upscale` x."""
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, upscale)
    if abs(s - 1) < 1e-3:
        return 1.0, img
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST   # nearest keeps cracks crisp
    return s, cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)), interpolation=interp)


def check_gui():
    """Stop with a clear fix if this OpenCV build cannot open windows."""
    try:
        cv2.namedWindow("_gui_check")
        cv2.destroyWindow("_gui_check")
    except cv2.error:
        sys.exit(
            "This OpenCV build has no window support (it is the 'headless' version),\n"
            "so the area picker and sliders cannot open. Fix it with:\n\n"
            "    python -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless opencv-python\n"
            "    python -m pip install opencv-python\n\n"
            "then run this script again.")


def select_rois(img):
    sw, sh = screen_size()
    s, disp = fit_to_screen(img, sw - 80, sh - 140)
    print("Draw a box, ENTER/SPACE to accept it. Draw more if you like. ESC when done.")
    rois = cv2.selectROIs("select crack area(s)  -  ESC when done", disp, showCrosshair=False)
    cv2.destroyWindow("select crack area(s)  -  ESC when done")
    out = []
    for x, y, w, h in (rois if len(rois) else []):
        if w > 5 and h > 5:
            out.append(tuple(int(round(v / s)) for v in (x, y, w, h)))
    return out


def panel(crop, res, view, max_w, max_h):
    """Picture for the view window, already scaled to max_w x max_h, with labels."""
    def gray3(g):
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    enh = gray3(res["enhanced"])
    if view == 1:
        tiles = [[("enhanced", enh)]]
    elif view == 2:
        bin_img = gray3(np.where(res["binary"], 0, 255).astype(np.uint8))
        skel = cv2.dilate(res["skeleton"].astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
        overlay = crop.copy(); overlay[skel] = (0, 0, 255)
        tiles = [[("original", crop), ("enhanced", enh)],
                 [("binary", bin_img), ("skeleton overlay", overlay)]]
    else:
        tiles = [[("original", crop), ("enhanced", enh)]]
    rows, cols = len(tiles), len(tiles[0])
    gap = 6
    h, w = crop.shape[:2]
    s, _ = fit_to_screen(crop, (max_w - gap * (cols - 1)) / cols,
                         (max_h - gap * (rows - 1)) / rows, upscale=6.0)
    tw, th = max(int(w * s), 1), max(int(h * s), 1)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
    canvas = np.full((rows * th + gap * (rows - 1), cols * tw + gap * (cols - 1), 3), 60, np.uint8)
    for r, row in enumerate(tiles):
        for c, (label, img) in enumerate(row):
            y, x = r * (th + gap), c * (tw + gap)
            canvas[y:y + th, x:x + tw] = cv2.resize(img, (tw, th), interpolation=interp)
            cv2.putText(canvas, label, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, label, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 220, 255), 2, cv2.LINE_AA)
    return canvas


def _ctrl_image():
    """Small help strip shown under the sliders."""
    img = np.full((70, CTRL_W, 3), 40, np.uint8)
    for i, t in enumerate(["s = save   n = skip   q = quit", "v = change picture view"]):
        cv2.putText(img, t, (10, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return img


def tune(crop, params, view=0):
    """Sliders in their own narrow window, pictures in a large one.

    Returns (action, params, results, view)."""
    sw, sh = screen_size()
    max_w, max_h = sw - CTRL_W - 60, sh - 120          # room left for the pictures

    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    for name, key, mx, _ in SLIDERS:
        cv2.createTrackbar(name, CTRL_WIN, int(params[key]), mx, lambda v: None)
    cv2.imshow(CTRL_WIN, _ctrl_image())
    cv2.resizeWindow(CTRL_WIN, CTRL_W, 40 * len(SLIDERS) + 70)
    cv2.moveWindow(CTRL_WIN, 0, 0)

    cv2.namedWindow(VIEW_WIN, cv2.WINDOW_NORMAL)
    cv2.moveWindow(VIEW_WIN, CTRL_W + 20, 0)

    last, res, shown_view = None, None, None
    while True:
        try:
            cur = {key: cv2.getTrackbarPos(name, CTRL_WIN) for name, key, _, _ in SLIDERS}
        except cv2.error:                              # slider window was closed
            cv2.destroyAllWindows()
            return "q", last, res, view
        cur["crack_width"] = max(cur["crack_width"], 1)
        if cur != last:
            res = extract(crop, cur)
            last = dict(cur)
            shown_view = None
        if shown_view != view:
            pic = panel(crop, res, view, max_w, max_h)
            cv2.imshow(VIEW_WIN, pic)
            cv2.resizeWindow(VIEW_WIN, pic.shape[1], pic.shape[0])
            shown_view = view
        k = cv2.waitKey(50) & 0xFF
        if k == ord("v"):
            view = (view + 1) % len(VIEWS)
            print(f"  view: {VIEWS[view]}")
        elif k in (ord("s"), ord("n"), ord("q"), 27):
            cv2.destroyWindow(VIEW_WIN); cv2.destroyWindow(CTRL_WIN)
            return (chr(k) if k != 27 else "q"), last, res, view
        if (cv2.getWindowProperty(VIEW_WIN, cv2.WND_PROP_VISIBLE) < 1 or
                cv2.getWindowProperty(CTRL_WIN, cv2.WND_PROP_VISIBLE) < 1):
            cv2.destroyAllWindows()
            return "q", last, res, view


def imwrite(path, img):
    """cv2.imwrite that also works for non-ASCII paths on Windows, and says so if it fails."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(path)


def imread(path):
    """cv2.imread that also works for non-ASCII paths on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def save(out_dir, tag, crop, res, params, roi):
    os.makedirs(out_dir, exist_ok=True)
    b = os.path.join(out_dir, tag)
    imwrite(b + "_original.png", crop)
    imwrite(b + "_enhanced.png", res["enhanced"])                            # main result
    imwrite(b + "_response.png", 255 - res["response"])                       # dark cracks
    imwrite(b + "_binary.png", np.where(res["binary"], 0, 255).astype(np.uint8))
    imwrite(b + "_skeleton.png", np.where(res["skeleton"], 0, 255).astype(np.uint8))
    ov = crop.copy(); ov[res["skeleton"]] = (0, 0, 255)
    imwrite(b + "_overlay.png", ov)
    stats = network_stats(res["skeleton"])
    with open(b + "_info.json", "w") as f:
        json.dump(dict(roi_xywh=roi, params=params, stats=stats), f, indent=2)
    print(f"  SAVED -> {b}_enhanced.png  (and _binary, _skeleton, ... in the same folder)")
    print(f"  stats: {stats}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="painting image (omit for a file dialog)")
    ap.add_argument("--light", action="store_true", help="start with the light-cracks switch on")
    ap.add_argument("--out", help="output folder (default <image>_cracks)")
    args = ap.parse_args()

    path = args.image or pick_image()
    if not path:
        sys.exit("no image chosen")
    img = imread(path)
    if img is None:
        sys.exit(f"could not read {path}")
    out_dir = os.path.abspath(args.out or os.path.splitext(path)[0] + "_cracks")
    print(f"results will be saved in: {out_dir}")
    stem = os.path.splitext(os.path.basename(path))[0]

    check_gui()
    rois = select_rois(img)
    if not rois:
        rois = [(0, 0, img.shape[1], img.shape[0])]
        print("no box drawn - using the whole image")

    params = {key: d for _, key, _, d in SLIDERS}
    params["light"] = int(args.light)
    view = 0
    for i, (x, y, w, h) in enumerate(rois, 1):
        crop = img[y:y + h, x:x + w]
        print(f"area {i}/{len(rois)}: x={x} y={y} w={w} h={h}")
        print("  click on either window, then press  s = save,  n = skip,  q = quit,  v = view")
        action, used, res, view = tune(crop, params, view)
        if action == "q":
            break
        params = used                                  # carry settings to next area
        if action == "s":
            save(out_dir, f"{stem}_roi{i}", crop, res, used, [x, y, w, h])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
