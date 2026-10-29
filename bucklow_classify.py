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
the crack skeleton (and binary) that craquelure_extract.py saves, turned into a
0..1 value between the two poles, and every tradition is scored by how well the
measured values agree with its profile ("nearly always" counts more than
"usually", "often", "sometimes"). The thresholds are set from the paper's own
figures - it is a rule-based approximation of the method, not Bucklow's
discriminant analysis, and it only knows these four traditions.

Checked on the 17 craquelure figures in Bucklow's paper (known tradition and
scale): 14 of 17 placed in the right tradition when the support is given
(--support), 11 of 17 without it. Two misses were Italian panels taken for
Flemish, one a heavily textured French canvas taken for Dutch. The thresholds
were set on those same figures, so expect lower accuracy
on new paintings, and treat the result as a pointer, not a verdict.

Usage
    python bucklow_classify.py C:\\cracks\\SK-A-2344_cracks
    python bucklow_classify.py C:\\cracks              (all *_cracks folders inside)
    python bucklow_classify.py some_skeleton.png

  Options (both matter - give them when you know them):
    --scale-cm 7          width of the analysed area in cm (or --px-per-mm 20, or
                          --painting-width-cm 41 to work it out from the whole
                          painting; needs results saved by a recent
                          craquelure_extract.py). Without a scale, island size
                          (SMALL/LARGE) is left out.
    --grain vertical      panels: direction of the wood grain in the image;
                          canvas: direction of the painting's LONGEST side.
                          Without it, the orientation term is left out.
    --support panel|canvas   if known; the traditions of the other support are
                          then marked as unlikely.

  Output: a table in the prompt and bucklow_results.csv (one row per area: the
  measured values, the descriptor words, and a score per tradition).

Requirements:  pip install opencv-python numpy scikit-image
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
# files
# ---------------------------------------------------------------------------

def imread_gray(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None


def find_skeletons(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*_skeleton.png")))
            out += sorted(glob.glob(os.path.join(p, "*_cracks", "*_skeleton.png")))
        elif os.path.isfile(p):
            out.append(p)
    return list(dict.fromkeys(out))


def load(path):
    """Skeleton (+ binary, + info) for one area. Cracks are black on white."""
    sk_img = imread_gray(path)
    if sk_img is None:
        return None, None, {}
    sk = sk_img < 128
    if sk.mean() > 0.5:                                        # white cracks on black
        sk = ~sk
    binary, info = None, {}
    if path.endswith("_skeleton.png"):
        bpath = path[:-len("_skeleton.png")] + "_binary.png"
        ipath = path[:-len("_skeleton.png")] + "_info.json"
        b = imread_gray(bpath) if os.path.exists(bpath) else None
        if b is not None:
            binary = b < 128
        if os.path.exists(ipath):
            with open(ipath) as f:
                info = json.load(f)
    return sk, binary, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="*_skeleton.png files, _cracks folders, or a folder of them")
    sc = ap.add_mutually_exclusive_group()
    sc.add_argument("--scale-cm", type=float, help="width of the analysed area in cm")
    sc.add_argument("--px-per-mm", type=float, help="pixels per millimetre")
    sc.add_argument("--painting-width-cm", type=float,
                    help="width of the whole painting in cm (needs image size in _info.json)")
    ap.add_argument("--grain", choices=["vertical", "horizontal"],
                    help="panel: wood grain direction in the image; canvas: direction of the LONGEST side")
    ap.add_argument("--support", choices=["panel", "canvas"], help="if known")
    ap.add_argument("--csv", default="bucklow_results.csv", help="output table (default bucklow_results.csv)")
    args = ap.parse_args()

    files = find_skeletons(args.paths)
    if not files:
        sys.exit("no *_skeleton.png found - run craquelure_extract.py first")

    rows = []
    for f in files:
        sk, binary, info = load(f)
        if sk is None:
            print(f"could not read {f}"); continue
        ppm = None
        if args.px_per_mm:
            ppm = args.px_per_mm
        elif args.scale_cm:
            ppm = sk.shape[1] / (args.scale_cm * 10)
        elif args.painting_width_cm and info.get("image_wh"):
            ppm = info["image_wh"][0] / (args.painting_width_cm * 10)
        m = measure(sk, binary, ppm)
        name = os.path.basename(f).replace("_skeleton.png", "")
        if "error" in m:
            print(f"\n{name}: {m['error']}"); continue
        p = describe(m, ppm, args.grain)
        scores = classify(p, args.support)
        words = {d: WORDS[d][int(v >= 0.5)] for d, v in p.items()}
        best = next(iter(scores))
        print(f"\n{name}")
        print("  " + ", ".join(f"{words[d]} ({v:.2f})" for d, v in p.items()))
        for t, s in scores.items():
            print(f"    {s:5.2f}  {t}{'   <- best match' if t == best else ''}")
        if ppm is None:
            print("    (no scale given: island size left out - use --scale-cm)")
        if not args.grain:
            print("    (no --grain given: orientation to grain / longest side left out)")
        row = {"area": name, "best_match": best}
        row.update({f"score: {t}": s for t, s in scores.items()})
        row.update({d: words[d] for d in WORDS if d in words})
        row.update({f"value_{d}": round(v, 3) for d, v in p.items()})
        row.update({k: v for k, v in m.items()})
        rows.append(row)

    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print(f"\nsaved {os.path.abspath(args.csv)}")


if __name__ == "__main__":
    main()
