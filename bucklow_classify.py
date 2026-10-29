"""
Classify craquelure with Bucklow's descriptors.

    Bucklow, S. (1997) "The description of craquelure patterns",
    Studies in Conservation 42(3), 129-140.

Bucklow describes a crack pattern with a few dichotomies and shows that their
combinations separate four technical traditions:

    descriptor                          poles
    1 predominant direction             NO DIRECTION  / DIRECTION
      ... and its orientation           PARALLEL / PERPENDICULAR to the grain
                                        (panels) or to the longest side (canvas)
    2 local change of direction (~1mm)  SMOOTH        / JAGGED
      global change of direction        STRAIGHT      / CURVED
    3 relation between directions       SQUARE        / NOT SQUARE islands
    4 distance between cracks           SMALL (<5 mm) / LARGE (>15 mm) islands
    5 thickness of cracks               UNIFORM       / SECONDARY network
    6 junctions / terminations          CONNECTED     / BROKEN network
    7 organization                      ORDERED       / RANDOM

    Italian 14th-15th c. panel   often DIRECTION, usually PERPENDICULAR to grain,
                                 usually SMALL-MEDIUM islands, usually JAGGED,
                                 sometimes distinct SECONDARY network
    Flemish 15th-16th c. panel   nearly always DIRECTION, usually PARALLEL to grain,
                                 usually very SMALL islands, usually SMOOTH,
                                 usually STRAIGHT, usually SQUARE, often UNIFORM,
                                 usually highly ORDERED
    Dutch 17th c. canvas         usually DIRECTION, usually PERPENDICULAR to the
                                 longest side, usually MEDIUM islands, usually
                                 JAGGED, often SQUARE
    French 18th c. canvas        usually NO DIRECTION, LARGE islands, SMOOTH,
                                 CURVED, NOT SQUARE, RANDOM

Bucklow's descriptions were made by eye; here each descriptor is MEASURED from
the ENHANCED picture that craquelure_extract.py saves (*_enhanced.png: dark cracks
on even grey paint) - the crack lines are found in it the same way the extraction
does - turned into a 0..1 value between the two poles, and every tradition is
scored by how well the measured values agree with its profile ("nearly always"
counts more than "usually", "often", "sometimes"). The thresholds are set from
the paper's own figures - it is a rule-based approximation of the method, not
Bucklow's discriminant analysis, and it only knows these four traditions.

Checked on the 17 craquelure figures in Bucklow's paper (known tradition and
scale): 14 of 17 placed in the right tradition when the support is given
(--support), 10 of 17 without it. Two misses were Italian panels taken for
Flemish, one a heavily textured French canvas taken for Dutch. The thresholds
were set on those same figures, so expect lower accuracy
on new paintings, and treat the result as a pointer, not a verdict.

Usage
    python bucklow_classify.py                 opens the window (the easy way)
    python bucklow_classify.py C:\\cracks      window with every *_enhanced.png found
    python bucklow_classify.py C:\\cracks --print --scale-cm 7 --grain vertical --support panel
                                               no window: prints + bucklow_results.csv

  In the window: "Open pictures..." or "Open folder...", pick an area in the list,
  set its scale, grain direction and support, and the result appears on the right
  (scores per tradition and the descriptor words); red lines on the picture show
  the cracks that were measured - if they miss cracks or follow texture, adjust
  sensitivity / clean-up. "Classify all" runs every area, "Save results (CSV)"
  writes a table for Excel. Settings are kept per area; "Use these settings for
  all" copies them.

  Settings (give them when you know them):
    scale        width of the area in cm, or of the whole painting (needs results
                 saved by a recent craquelure_extract.py). Without it, island size
                 (SMALL/LARGE) is left out.
    grain        panels: direction of the wood grain in the picture; canvas:
                 direction of the painting's LONGEST side. Without it, the
                 orientation term is left out.
    support      panel or canvas, if known; the other support's traditions are
                 then marked as unlikely.

Requirements:  pip install opencv-python numpy scikit-image pillow
               (craquelure_extract.py must be in the same folder)
"""

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np
from skimage.morphology import skeletonize

# ---------------------------------------------------------------------------
# descriptor thresholds: (value at which the pattern is half-way between the
# poles, spread). Each descriptor is mapped to 0..1 with a logistic.
# ---------------------------------------------------------------------------
TH = {
    "direction": (0.28, 0.06),   # orientation anisotropy (doubled-angle resultant)
    "jagged":    (0.075, 0.008), # 1 - chord/path over ~1 mm
    "curved":    (12.0, 3.0),    # mean turn (deg) between consecutive ~3 mm stretches
    "square":    (0.35, 0.07),   # four-fold symmetry of crack directions
    "secondary": (0.40, 0.05),   # spread (CV) of crack widths
    "broken":    (0.50, 0.08),   # share of free ends among nodes
    "ordered":   (0.33, 0.06),   # sqrt(anisotropy * four-fold symmetry)
}
SMALL_MM, LARGE_MM = 5.0, 15.0   # Bucklow's island size limits
LARGE_MID_MM = 6.0               # measured spacing at which islands count as large:
                                 # the measured spacing runs below the island size
                                 # (secondary cracks and fragments add length)
LOCAL_MM = 1.0                   # "locally" = over about 1 mm
TURN_MM = 3.0                    # "globally": heading change between ~3 mm stretches
MIN_PIECE_MM = 2.0               # crack pieces shorter than this are ignored


def logistic(x, mid, spread):
    return float(1.0 / (1.0 + np.exp(-(x - mid) / spread)))


# ---------------------------------------------------------------------------
# skeleton graph
# ---------------------------------------------------------------------------

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def neighbour_count(sk):
    k = np.ones((3, 3), np.float32); k[1, 1] = 0
    return cv2.filter2D(sk.astype(np.uint8), cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT)


def trace_segments(sk):
    """Split the skeleton at junctions into ordered pixel paths.

    Returns (segments, junction_mask, end_mask); each segment is an (N,2) array
    of (y, x) in order, with the touching junction pixels added at its ends."""
    sk = sk.astype(bool)
    nb = neighbour_count(sk)
    junc = sk & (nb >= 3)
    ends = sk & (nb == 1)
    body = sk & ~junc
    n, lab = cv2.connectedComponents(body.astype(np.uint8), connectivity=8)
    H, W = sk.shape
    segs = []
    pts_by_label = [[] for _ in range(n)]
    ys, xs = np.nonzero(body)
    for y, x in zip(ys, xs):
        pts_by_label[lab[y, x]].append((y, x))
    for li in range(1, n):
        pts = pts_by_label[li]
        if not pts:
            continue
        pset = set(pts)
        # start at an end of the path (a pixel with <=1 neighbour inside the segment)
        def inner_nb(p):
            return [(p[0] + dy, p[1] + dx) for dy, dx in NB8 if (p[0] + dy, p[1] + dx) in pset]
        start = next((p for p in pts if len(inner_nb(p)) <= 1), pts[0])
        path, seen, cur = [start], {start}, start
        while True:
            nxt = [q for q in inner_nb(cur) if q not in seen]
            if not nxt:
                break
            # prefer 4-neighbours so the path does not skip corners
            nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            cur = nxt[0]; seen.add(cur); path.append(cur)
        # attach junction pixels touching either end
        for end_i in (0, -1):
            p = path[end_i]
            for dy, dx in NB8:
                q = (p[0] + dy, p[1] + dx)
                if 0 <= q[0] < H and 0 <= q[1] < W and junc[q]:
                    if end_i == 0:
                        path.insert(0, q)
                    else:
                        path.append(q)
                    break
        segs.append(np.array(path, dtype=np.float32))
    return segs, junc, ends


def path_length(p):
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(p, axis=0).T)))


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------

def measure(skel, binary=None, px_per_mm=None):
    """Measure Bucklow's descriptors on one crack skeleton (bool array)."""
    sk = skeletonize(skel.astype(bool))
    H, W = sk.shape
    local_px = max(int(round(LOCAL_MM * px_per_mm)), 6) if px_per_mm else 10
    # ignore a thin border (scan/crop edges) and crack pieces shorter than ~2 mm,
    # which are mostly paint texture, not cracks
    b = max(local_px // 2, 3)
    sk[:b] = sk[-b:] = False; sk[:, :b] = sk[:, -b:] = False
    n, lab, st, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_PIECE_MM * local_px
    sk = keep[lab]
    segs, junc, ends = trace_segments(sk)
    total_len = sum(path_length(s) for s in segs)
    m = {"crack_length_px": round(total_len, 1)}
    if total_len < 50:
        m["error"] = "too few cracks to describe"
        return m

    # 1 direction: orientation of short chords along every crack, length weighted
    angs, wts = [], []
    for s in segs:
        step = max(local_px, 4)
        for i in range(0, len(s) - step, step):
            d = s[i + step] - s[i]
            L = float(np.hypot(*d))
            if L > 0:
                angs.append(np.arctan2(-d[0], d[1]))      # image y points down
                wts.append(L)
    angs, wts = np.array(angs), np.array(wts)
    z2 = np.sum(wts * np.exp(2j * angs)) / wts.sum()
    m["anisotropy"] = round(float(abs(z2)), 3)
    theta = (np.degrees(np.angle(z2)) / 2.0) % 180.0          # 0 = horizontal, 90 = vertical
    m["main_angle_deg"] = round(float(theta), 1)
    # 4-fold symmetry of directions (a grid of two perpendicular directions)
    z4 = np.sum(wts * np.exp(4j * angs)) / wts.sum()
    m["fourfold"] = round(float(abs(z4)), 3)

    # 2a local: jaggedness over ~1 mm
    jag, jw = [], []
    for s in segs:
        n = local_px
        for i in range(0, len(s) - n, max(n // 2, 1)):
            piece = s[i:i + n + 1]
            pl = path_length(piece)
            if pl > 0:
                jag.append(1 - np.hypot(*(piece[-1] - piece[0])) / pl); jw.append(pl)
    m["jaggedness"] = round(float(np.average(jag, weights=jw)) if jag else 0.0, 3)

    # 2b global: how much the crack's heading turns between consecutive
    # ~TURN_MM stretches (smoothed over ~1 mm first, so jaggedness doesn't count)
    turns, tw = [], []
    step = max(int(TURN_MM * local_px / LOCAL_MM), 2 * local_px)
    k = max(local_px, 3)
    for s in segs:
        if len(s) < 2 * step + k:
            continue
        sm = np.stack([np.convolve(s[:, j], np.ones(k) / k, mode="valid") for j in (0, 1)], 1)
        heads = []
        for i in range(0, len(sm) - step, step):
            d = sm[i + step] - sm[i]
            heads.append(np.arctan2(d[0], d[1]))
        for h0, h1 in zip(heads, heads[1:]):
            turns.append(abs((np.degrees(h1 - h0) + 180) % 360 - 180)); tw.append(step)
    m["turning_deg"] = round(float(np.average(turns, weights=tw)) if turns else 0.0, 2)

    # 3 square: cracks in two perpendicular families = four-fold symmetry of
    # the crack directions (measured above as "fourfold")
    jl = cv2.connectedComponentsWithStats(junc.astype(np.uint8), connectivity=8)
    centres = jl[3][1:]
    m["junctions"] = int(len(centres))

    # 4 island size: mean spacing between cracks = 2 * area / crack length
    spacing_px = 2.0 * H * W / total_len
    m["spacing_px"] = round(spacing_px, 1)
    if px_per_mm:
        m["spacing_mm"] = round(spacing_px / px_per_mm, 2)

    # 5 thickness: spread of crack widths along the skeleton
    if binary is not None and binary.any():
        dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
        w = 2 * dist[sk & binary.astype(bool)]
        w = w[w > 0]
        m["width_cv"] = round(float(w.std() / w.mean()), 3) if len(w) > 20 else None
        m["width_median_px"] = round(float(np.median(w)), 2) if len(w) else None
    else:
        m["width_cv"] = None

    # 6 connected / broken: free ends among all nodes
    n_end = int(ends.sum())
    m["end_points"] = n_end
    m["end_share"] = round(n_end / max(n_end + len(centres), 1), 3)

    # 7 ordered / random: Bucklow's "synthesis" of a predominant direction and a
    # regular (grid-like) layout - the geometric mean of the two measures
    m["order"] = round(float(np.sqrt(m["anisotropy"] * m["fourfold"])), 3)
    return m


def describe(m, px_per_mm=None, grain=None):
    """Turn measurements into Bucklow's poles: dict descriptor -> 0..1 (1 = second pole)."""
    p = {}
    p["direction"] = logistic(m["anisotropy"], *TH["direction"])
    if grain:
        ref = 90.0 if grain == "vertical" else 0.0
        diff = abs(((m["main_angle_deg"] - ref) + 90) % 180 - 90)      # 0..90
        p["perpendicular"] = logistic(diff, 45, 10)                    # 1 = perpendicular
    p["jagged"] = logistic(m["jaggedness"], *TH["jagged"])
    p["curved"] = logistic(m["turning_deg"], *TH["curved"])
    p["square"] = logistic(m["fourfold"], *TH["square"])
    if px_per_mm and "spacing_mm" in m:
        p["large"] = logistic(np.log(m["spacing_mm"]), np.log(LARGE_MID_MM), 0.25)
    if m.get("width_cv") is not None:
        p["secondary"] = logistic(m["width_cv"], *TH["secondary"])
    p["broken"] = logistic(m["end_share"], *TH["broken"])
    p["random"] = 1.0 - logistic(m["order"], *TH["ordered"])
    return p


WORDS = {
    "direction":     ("NO DIRECTION", "DIRECTION"),
    "perpendicular": ("PARALLEL to grain/longest side", "PERPENDICULAR to grain/longest side"),
    "jagged":        ("SMOOTH", "JAGGED"),
    "curved":        ("STRAIGHT", "CURVED"),
    "square":        ("NOT SQUARE", "SQUARE"),
    "large":         ("SMALL", "LARGE"),
    "secondary":     ("UNIFORM", "SECONDARY"),
    "broken":        ("CONNECTED", "BROKEN"),
    "random":        ("ORDERED", "RANDOM"),
}

# Bucklow's profiles: descriptor -> (expected value 0..1, weight from his wording)
NEARLY_ALWAYS, USUALLY, OFTEN, SOMETIMES = 1.0, 0.8, 0.6, 0.4
PROFILES = {
    "Italian 14th-15th c. panel": {
        "direction": (1, OFTEN), "perpendicular": (1, USUALLY), "large": (0.25, USUALLY),
        "jagged": (1, USUALLY), "secondary": (1, SOMETIMES)},
    "Flemish 15th-16th c. panel": {
        "direction": (1, NEARLY_ALWAYS), "perpendicular": (0, USUALLY), "large": (0, USUALLY),
        "jagged": (0, USUALLY), "curved": (0, USUALLY), "square": (1, USUALLY),
        "secondary": (0, OFTEN), "random": (0, USUALLY)},
    "Dutch 17th c. canvas": {
        "direction": (1, USUALLY), "perpendicular": (1, USUALLY), "large": (0.5, USUALLY),
        "jagged": (1, USUALLY), "square": (1, OFTEN)},
    "French 18th c. canvas": {
        "direction": (0, USUALLY), "large": (1, USUALLY), "jagged": (0, USUALLY),
        "curved": (1, USUALLY), "square": (0, USUALLY), "random": (1, USUALLY)},
}
SUPPORT = {"Italian 14th-15th c. panel": "panel", "Flemish 15th-16th c. panel": "panel",
           "Dutch 17th c. canvas": "canvas", "French 18th c. canvas": "canvas"}


def classify(p, support=None):
    """Score each tradition 0..1 by weighted agreement with its profile."""
    scores = {}
    for name, prof in PROFILES.items():
        num = den = 0.0
        for d, (exp, w) in prof.items():
            if d not in p:
                continue                                       # not measured: no evidence
            # 'perpendicular' only means something when there IS a direction
            if d == "perpendicular":
                w *= max(p["direction"], 0.5)                 # weaker when direction is weak
            num += w * (1 - abs(p[d] - exp)); den += w
        s = num / den if den else 0.0
        if support and SUPPORT[name] != support:
            s *= 0.5                                           # wrong support: unlikely
        scores[name] = round(s, 3)
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# cracks from the enhanced picture
# ---------------------------------------------------------------------------

DEFAULTS = dict(sensitivity=10, cleanup=20, spurs=8)


def cracks_from_enhanced(gray, sensitivity=10, cleanup=20, spurs=8):
    """Binary and skeleton from an enhanced picture (dark cracks on even grey paint).

    Same method as craquelure_extract.py: pixels clearly darker than the paint
    start a crack, fainter ones join only where they connect (hysteresis); then
    small pieces are removed and the cracks are thinned and de-spurred. Works on
    any picture with dark cracks on lighter, fairly even paint."""
    from skimage.filters import apply_hysteresis_threshold
    import craquelure_extract as ce
    g = gray.astype(np.float32)
    paint = max(float(np.median(g)), 1.0)                     # the paint is the commonest tone
    dark = np.clip((paint - g) / paint, 0, 1)                 # 0 = paint, 1 = black
    hi = float(np.clip(0.8 - 0.04 * sensitivity, 0.05, 0.95))
    binary = apply_hysteresis_threshold(dark, 0.375 * hi, hi)
    binary = ce.clean(binary, cleanup)
    skel = ce.thin(binary, cleanup // 2, spurs)
    return binary, skel


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def imread_gray(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def find_enhanced(paths):
    """*_enhanced.png in the given folders (and their *_cracks folders), or the files given."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*_enhanced.png")))
            out += sorted(glob.glob(os.path.join(p, "*_cracks", "*_enhanced.png")))
        elif os.path.isfile(p) and p.lower().endswith(IMG_EXTS):
            out.append(p)
    return list(dict.fromkeys(out))


def area_name(path):
    b = os.path.basename(path)
    return b[:-len("_enhanced.png")] if b.endswith("_enhanced.png") else os.path.splitext(b)[0]


def read_info(path):
    """The _info.json saved next to an _enhanced.png, if any."""
    if path.endswith("_enhanced.png"):
        ipath = path[:-len("_enhanced.png")] + "_info.json"
        if os.path.exists(ipath):
            with open(ipath) as f:
                return json.load(f)
    return {}


def px_per_mm(width_px, info, scale_mode, scale_cm):
    """scale_mode: 'area' (width of this picture in cm), 'painting' (width of the
    whole painting in cm, needs image size in _info.json) or None."""
    if not scale_cm or scale_cm <= 0:
        return None
    if scale_mode == "area":
        return width_px / (scale_cm * 10)
    if scale_mode == "painting" and info.get("image_wh"):
        return info["image_wh"][0] / (scale_cm * 10)
    return None


def analyse(path, scale_mode=None, scale_cm=None, grain=None, support=None,
            sensitivity=10, cleanup=20, spurs=8, gray=None):
    """Everything for one area: cracks, measurements, descriptors and scores."""
    if gray is None:
        gray = imread_gray(path)
    if gray is None:
        return {"error": "could not read the image"}
    info = read_info(path) if path else {}
    binary, skel = cracks_from_enhanced(gray, sensitivity, cleanup, spurs)
    ppm = px_per_mm(gray.shape[1], info, scale_mode, scale_cm)
    m = measure(skel, binary, ppm)
    res = dict(gray=gray, binary=binary, skeleton=skel, ppm=ppm, measures=m)
    if "error" in m:
        res["error"] = m["error"]
        return res
    p = describe(m, ppm, grain)
    res.update(values=p, scores=classify(p, support),
               words={d: WORDS[d][int(v >= 0.5)] for d, v in p.items()})
    res["best"] = next(iter(res["scores"]))
    notes = []
    if ppm is None:
        notes.append("no scale: island size (SMALL/LARGE) left out"
                     + (" - this picture has no saved painting size, give the area width instead"
                        if scale_mode == "painting" and scale_cm and not info.get("image_wh") else ""))
    if not grain:
        notes.append("grain / longest side unknown: orientation left out")
    res["notes"] = notes
    return res


def result_row(name, res):
    row = {"area": name, "best_match": res.get("best", ""), "error": res.get("error", "")}
    row.update({f"score: {t}": s for t, s in res.get("scores", {}).items()})
    row.update({d: res["words"][d] for d in WORDS if d in res.get("words", {})})
    row.update({f"value_{d}": round(v, 3) for d, v in res.get("values", {}).items()})
    row.update(res.get("measures", {}))
    return row


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

TRAD_COLOURS = {"Italian": "#c0504d", "Flemish": "#4f81bd", "Dutch": "#e3a32b", "French": "#8064a2"}


def run_gui(initial=()):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Bucklow craquelure classifier")
    root.geometry("1400x820")
    files, settings, results = [], {}, {}
    last = dict(scale_mode="none", scale_cm="", grain="unknown", support="unknown", **DEFAULTS)
    state = {"photo": None, "job": None}

    # ---- layout -----------------------------------------------------------
    top = ttk.Frame(root, padding=6); top.pack(side="top", fill="x")
    body = ttk.Panedwindow(root, orient="horizontal"); body.pack(fill="both", expand=True)
    left = ttk.Frame(body, padding=6); mid = ttk.Frame(body, padding=6); right = ttk.Frame(body, padding=6)
    body.add(left, weight=1); body.add(mid, weight=4); body.add(right, weight=2)

    ttk.Label(left, text="Areas (enhanced pictures)").pack(anchor="w")
    lb = tk.Listbox(left, width=34, exportselection=False); lb.pack(fill="both", expand=True)
    status = ttk.Label(root, text="Open enhanced pictures (*_enhanced.png) or a folder to begin.",
                       anchor="w", padding=4)
    status.pack(side="bottom", fill="x")

    canvas = tk.Canvas(mid, bg="#3a3a3a", highlightthickness=0); canvas.pack(fill="both", expand=True)
    show_cracks = tk.BooleanVar(value=True)
    ttk.Checkbutton(mid, text="show detected cracks (red)", variable=show_cracks,
                    command=lambda: draw()).pack(anchor="w")

    # settings
    sf = ttk.LabelFrame(right, text="Settings for this area", padding=6); sf.pack(fill="x")
    v_mode = tk.StringVar(); v_cm = tk.StringVar(); v_grain = tk.StringVar(); v_support = tk.StringVar()
    v_sens = tk.IntVar(); v_clean = tk.IntVar(); v_spurs = tk.IntVar()
    ttk.Label(sf, text="Scale").grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(sf, text="unknown", value="none", variable=v_mode).grid(row=0, column=1, sticky="w")
    ttk.Radiobutton(sf, text="width of this area (cm)", value="area", variable=v_mode).grid(row=1, column=1, sticky="w")
    ttk.Radiobutton(sf, text="width of whole painting (cm)", value="painting", variable=v_mode).grid(row=2, column=1, sticky="w")
    ttk.Entry(sf, textvariable=v_cm, width=8).grid(row=1, column=2, rowspan=2, padx=4)
    ttk.Label(sf, text="Grain (panel) / longest\nside (canvas) runs").grid(row=3, column=0, sticky="w", pady=(6, 0))
    ttk.Combobox(sf, textvariable=v_grain, values=["unknown", "vertical", "horizontal"],
                 state="readonly", width=12).grid(row=3, column=1, sticky="w", pady=(6, 0))
    ttk.Label(sf, text="Support").grid(row=4, column=0, sticky="w", pady=(6, 0))
    ttk.Combobox(sf, textvariable=v_support, values=["unknown", "panel", "canvas"],
                 state="readonly", width=12).grid(row=4, column=1, sticky="w", pady=(6, 0))
    for r, (lbl, var, mx) in enumerate([("Sensitivity", v_sens, 20), ("Clean-up", v_clean, 200),
                                        ("Trim spurs", v_spurs, 50)], start=5):
        ttk.Label(sf, text=lbl).grid(row=r, column=0, sticky="w")
        tk.Scale(sf, variable=var, from_=0, to=mx, orient="horizontal", length=170,
                 showvalue=True, command=lambda _v: schedule()).grid(row=r, column=1, columnspan=2, sticky="w")
    bf = ttk.Frame(right, padding=(0, 6)); bf.pack(fill="x")
    ttk.Button(bf, text="Use these settings for all", command=lambda: copy_to_all()).pack(side="left")
    ttk.Button(bf, text="Classify all", command=lambda: classify_all()).pack(side="left", padx=4)

    rf = ttk.LabelFrame(right, text="Result", padding=6); rf.pack(fill="both", expand=True)
    bars = tk.Canvas(rf, height=150, bg="white", highlightthickness=0); bars.pack(fill="x")
    txt = tk.Text(rf, height=14, width=44, wrap="word", font=("Consolas", 10)); txt.pack(fill="both", expand=True)

    # ---- helpers ----------------------------------------------------------
    def current():
        sel = lb.curselection()
        return files[sel[0]] if sel else None

    def get_settings():
        try:
            cm = float(v_cm.get()) if v_cm.get().strip() else None
        except ValueError:
            cm = None
        return dict(scale_mode=v_mode.get(), scale_cm=v_cm.get(), cm=cm, grain=v_grain.get(),
                    support=v_support.get(), sensitivity=v_sens.get(), cleanup=v_clean.get(),
                    spurs=v_spurs.get())

    def put_settings(st):
        v_mode.set(st["scale_mode"]); v_cm.set(st["scale_cm"]); v_grain.set(st["grain"])
        v_support.set(st["support"]); v_sens.set(st["sensitivity"]); v_clean.set(st["cleanup"])
        v_spurs.set(st["spurs"])

    def run(path, st):
        mode = None if st["scale_mode"] == "none" else st["scale_mode"]
        try:
            cm = float(st["scale_cm"]) if str(st["scale_cm"]).strip() else None
        except ValueError:
            cm = None
        return analyse(path, mode, cm,
                       None if st["grain"] == "unknown" else st["grain"],
                       None if st["support"] == "unknown" else st["support"],
                       st["sensitivity"], st["cleanup"], st["spurs"])

    def label_for(path):
        r = results.get(path)
        tag = ""
        if r and r.get("best"):
            tag = "  ->  " + r["best"].split()[0]
        elif r and r.get("error"):
            tag = "  ->  (no result)"
        return area_name(path) + tag

    def refresh_list():
        sel = lb.curselection()
        lb.delete(0, "end")
        for p in files:
            lb.insert("end", label_for(p))
        if sel:
            lb.selection_set(sel[0])

    def add_files(new):
        new = [p for p in find_enhanced(new) if p not in files]
        for p in new:
            files.append(p); settings[p] = dict(last)
        refresh_list()
        if new and not lb.curselection():
            lb.selection_set(len(files) - len(new)); on_select()
        status.config(text=f"{len(files)} area(s) loaded. Set the scale, grain and support, "
                           f"then read the result on the right.")

    def open_files():
        ps = filedialog.askopenfilenames(title="Open enhanced pictures",
                                         filetypes=[("Enhanced pictures", "*_enhanced.png"),
                                                    ("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                                                    ("All", "*.*")])
        if ps:
            add_files(list(ps))

    def open_folder():
        d = filedialog.askdirectory(title="Folder with *_cracks folders or *_enhanced.png files")
        if d:
            add_files([d])
            if not files:
                messagebox.showinfo("Nothing found", "No *_enhanced.png found there.\n"
                                    "Run craquelure_extract.py first, or open pictures directly.")

    def on_select(_e=None):
        p = current()
        if not p:
            return
        put_settings(settings[p])
        classify_current()

    def schedule(*_a):
        if state["job"]:
            root.after_cancel(state["job"])
        state["job"] = root.after(300, classify_current)

    def classify_current():
        state["job"] = None
        p = current()
        if not p:
            return
        st = get_settings(); settings[p] = dict(st); last.update(st)
        status.config(text=f"classifying {area_name(p)} ..."); root.update_idletasks()
        results[p] = run(p, st)
        refresh_list(); draw(); show_result()
        status.config(text=f"{area_name(p)}: done")

    def copy_to_all():
        st = get_settings()
        for p in files:
            settings[p] = dict(st)
        status.config(text="settings copied to all areas - press 'Classify all'")

    def classify_all():
        for i, p in enumerate(files, 1):
            status.config(text=f"classifying {i}/{len(files)}: {area_name(p)} ..."); root.update()
            results[p] = run(p, settings[p])
        refresh_list(); draw(); show_result()
        status.config(text=f"classified {len(files)} area(s) - 'Save results' writes them to a CSV file")

    def save_csv():
        done = [p for p in files if p in results]
        if not done:
            messagebox.showinfo("Nothing yet", "Classify some areas first."); return
        out = filedialog.asksaveasfilename(title="Save results", defaultextension=".csv",
                                           initialfile="bucklow_results.csv",
                                           filetypes=[("CSV (opens in Excel)", "*.csv")])
        if out:
            write_csv(out, [result_row(area_name(p), results[p]) for p in done])
            status.config(text=f"saved {out}")

    def draw():
        canvas.delete("all")
        p = current()
        r = results.get(p) if p else None
        if not r or "gray" not in r:
            return
        img = cv2.cvtColor(r["gray"], cv2.COLOR_GRAY2RGB)
        cw, ch = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        sc = min(cw / img.shape[1], ch / img.shape[0])
        size = (max(int(img.shape[1] * sc), 1), max(int(img.shape[0] * sc), 1))
        disp = cv2.resize(img, size, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_NEAREST)
        if show_cracks.get():
            k = max(int(np.ceil(1 / sc)), 1) if sc < 1 else 1
            sk = cv2.dilate(r["skeleton"].astype(np.uint8), np.ones((k, k), np.uint8))
            sk = cv2.resize(sk, size, interpolation=cv2.INTER_NEAREST) > 0
            disp[sk] = (230, 30, 30)
        state["photo"] = ImageTk.PhotoImage(Image.fromarray(disp))
        canvas.create_image(cw // 2, ch // 2, image=state["photo"])

    def show_result():
        bars.delete("all"); txt.delete("1.0", "end")
        p = current()
        r = results.get(p) if p else None
        if not r:
            return
        if r.get("error"):
            txt.insert("end", f"No result: {r['error']}.\nTry a higher sensitivity or a bigger area.")
            return
        w = max(bars.winfo_width(), 300)
        for i, (t, s) in enumerate(r["scores"].items()):
            y = 10 + i * 34
            col = TRAD_COLOURS[t.split()[0]]
            bars.create_text(6, y + 2, text=t, anchor="nw", font=("Segoe UI", 9, "bold" if i == 0 else "normal"))
            bars.create_rectangle(6, y + 17, 6 + (w - 60) * s, y + 29, fill=col, outline="")
            bars.create_text(w - 48, y + 23, text=f"{s:.2f}", anchor="w", font=("Segoe UI", 9))
        txt.insert("end", "Best match:  " + r["best"] + "\n\n")
        txt.insert("end", "Descriptors (0 = first word, 1 = second word):\n")
        for d, v in r["values"].items():
            txt.insert("end", f"  {r['words'][d]:<38s}{v:.2f}\n")
        mm = r["measures"]
        txt.insert("end", "\n")
        if r["ppm"]:
            txt.insert("end", f"Scale: {r['ppm']:.1f} px/mm,  crack spacing ~{mm.get('spacing_mm', 0):.1f} mm\n")
        for n in r["notes"]:
            txt.insert("end", "Note: " + n + "\n")
        txt.insert("end", "\nA rule-based pointer after Bucklow (1997), not a verdict.")

    ttk.Button(top, text="Open pictures...", command=open_files).pack(side="left")
    ttk.Button(top, text="Open folder...", command=open_folder).pack(side="left", padx=4)
    ttk.Button(top, text="Save results (CSV)...", command=save_csv).pack(side="left")
    lb.bind("<<ListboxSelect>>", on_select)
    for v in (v_mode, v_grain, v_support):
        v.trace_add("write", schedule)
    v_cm.trace_add("write", schedule)
    canvas.bind("<Configure>", lambda e: draw())
    rf.bind("<Configure>", lambda e: show_result())
    put_settings(last)
    if initial:
        root.after(200, lambda: add_files(list(initial)))
    root.mainloop()


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="*_enhanced.png files, _cracks folders, or a folder of them")
    ap.add_argument("--print", dest="no_gui", action="store_true",
                    help="no window: print the results and write the CSV")
    sc = ap.add_mutually_exclusive_group()
    sc.add_argument("--scale-cm", type=float, help="width of the analysed area in cm")
    sc.add_argument("--painting-width-cm", type=float,
                    help="width of the whole painting in cm (needs image size in _info.json)")
    ap.add_argument("--grain", choices=["vertical", "horizontal"],
                    help="panel: wood grain direction in the image; canvas: direction of the LONGEST side")
    ap.add_argument("--support", choices=["panel", "canvas"], help="if known")
    ap.add_argument("--sensitivity", type=int, default=DEFAULTS["sensitivity"])
    ap.add_argument("--cleanup", type=int, default=DEFAULTS["cleanup"])
    ap.add_argument("--spurs", type=int, default=DEFAULTS["spurs"])
    ap.add_argument("--csv", default="bucklow_results.csv", help="output table (default bucklow_results.csv)")
    args = ap.parse_args()

    if not args.no_gui:
        run_gui(args.paths)
        return
    if not args.paths:
        sys.exit("--print needs pictures or a folder")

    files = find_enhanced(args.paths)
    if not files:
        sys.exit("no *_enhanced.png found - run craquelure_extract.py first, or give pictures directly")
    mode, cm = (("area", args.scale_cm) if args.scale_cm else
                ("painting", args.painting_width_cm) if args.painting_width_cm else (None, None))
    rows = []
    for f in files:
        name = area_name(f)
        res = analyse(f, mode, cm, args.grain, args.support,
                      args.sensitivity, args.cleanup, args.spurs)
        rows.append(result_row(name, res))
        if res.get("error"):
            print(f"\n{name}: {res['error']}"); continue
        print(f"\n{name}")
        print("  " + ", ".join(f"{res['words'][d]} ({v:.2f})" for d, v in res["values"].items()))
        for t, s in res["scores"].items():
            print(f"    {s:5.2f}  {t}{'   <- best match' if t == res['best'] else ''}")
        for n in res["notes"]:
            print(f"    ({n})")
    write_csv(args.csv, rows)
    print(f"\nsaved {os.path.abspath(args.csv)}")


if __name__ == "__main__":
    main()
