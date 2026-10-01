"""
Craquelure (crack pattern) extraction from painting photos.

Follows the extraction stage of
    El-Youssef, Bucklow & Maev, "The development of a diagnostic method for
    geographical and condition-based analysis of artworks using craquelure
    pattern recognition techniques", Insight 56(3), 2014.

Pipeline:
    1. greyscale
    2. morphological CLOSING with a disk slightly wider than the cracks
       -> the dark, thin cracks are filled in with the surrounding paint colour
    3. ENHANCED picture: original / closed -> every paint colour becomes the
       same light grey and only the cracks stay dark
    4. BINARY: the dark parts of the enhanced picture (hysteresis threshold).
       With --paper, the paper's OFFSET THRESHOLDING is used instead: Otsu per
       square tile, repeated with the tiles shifted by half a tile, keeping only
       pixels that are cracks in BOTH (true cracks agree, noise doesn't).
    5. clean: delete isolated specks / short fragments
    6. thin to 1-pixel-wide crack lines (skeleton)

Usage
    python craquelure_extract.py SK-A-2344.jpg            # draw box(es) yourself, tune with sliders
    python craquelure_extract.py SK-A-2344.jpg --auto     # areas found for you, check & tune them
    python craquelure_extract.py SK-A-2344.jpg --batch    # areas found and saved, no windows
    python craquelure_extract.py C:\\cracks --batch       # ... for every image in a folder
    python craquelure_extract.py C:\\cracks --select      # draw boxes on every image in turn,
                                                          #   each box processed automatically
    (no image name opens a file dialog; with --select you can Ctrl-click several)

--auto / --batch: automatic area finder
    The painting is cut into overlapping squares scored on crack evidence (long,
    clean crack lines that stand out from the paint's noise), flatness (no
    objects or folds) and exposure. --auto shows the result (ENTER = use these
    areas, ESC = draw your own) and opens the sliders for each; --batch saves
    straight away. Options: --areas 3 (how many), --size 800 (square size, px).
    Two maps are saved:
       *_areas.jpg     where the good areas are (red = good), numbered
       *_crackmap.jpg  every crack found (red), areas outlined - check the areas
                       really sit on cracks

--select: you choose, the program processes
    Each image opens in turn. Drag a box over cracked paint and press ENTER to
    keep it; draw more if you like; press ESC when done (ESC with no box skips
    that image). Every box gets automatic settings - crack colour (dark or light,
    whichever stands out more; light only on darker paint) and crack width
    measured from the box - and is saved. Ctrl+C in the prompt stops.
    Add --tune to open the slider windows for each box instead.

Results go to  <image name>_cracks/  next to each image; names say how they were made:
    <name>_roi1_...      boxes you drew and tuned (no option)
    <name>_checked1_...  --auto areas you checked and saved with s
    <name>_auto1_...     --batch on one image
    <name>_F_auto1_...   --batch on a folder
    <name>_sel1_...      --select boxes (plus <name>_selection.jpg showing them)
    for each:  _enhanced.png  paint flattened to an even grey, cracks in black
               _binary.png    cracks black on white
               _skeleton.png  cracks thinned to 1-pixel lines
               _overlay.png, _original.png, _info.json (settings and crack counts)

Tuning windows: s = save, n = skip, q = quit, v = change picture view.
Sliders, in order of use:
    light cracks    0 = dark cracks, 1 = cracks paler than the paint
    crack width     slightly wider than the thickest crack (pixels)
    contrast        higher = cracks in the enhanced picture go blacker
    texture cut     higher = more faint brush/canvas texture removed
    sensitivity     binary: higher = fainter cracks kept, lower = only strong ones
    clean-up        binary: delete pieces smaller than this many pixels
    trim spurs      skeleton: cut short side "ticks" shorter than this

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


def prune(sk, max_len):
    """Cut short dead-end side branches ('ticks') off a skeleton.

    Classic pruning: peel free ends off one pixel at a time, max_len times, so
    every branch shorter than max_len disappears; then grow the surviving ends
    back along the original skeleton so long cracks keep their full length.
    """
    if max_len <= 0:
        return sk
    k = np.ones((3, 3), np.float32); k[1, 1] = 0
    orig = sk.astype(np.uint8)
    cur = orig.copy()
    for _ in range(int(max_len)):
        nb = cv2.filter2D(cur, cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT)
        ends = (cur == 1) & (nb <= 1)
        if not ends.any():
            break
        cur[ends] = 0
    nb = cv2.filter2D(cur, cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT)
    grow = ((cur == 1) & (nb == 1)).astype(np.uint8)       # surviving free ends
    for _ in range(int(max_len)):
        grow = cv2.dilate(grow, np.ones((3, 3), np.uint8)) & orig
    return (cur | grow).astype(bool)


def thin(binary, min_len=0, spur_len=0):
    """Step 6: 1-px skeleton; trim spurs shorter than spur_len, then drop
    skeleton pieces shorter than min_len."""
    sk = prune(skeletonize(binary), spur_len)
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


def binary_from_enhanced(enh, sensitivity):
    """Cracks = what is dark in the enhanced picture (hysteresis threshold).

    Clearly dark pixels start a crack; fainter pixels are kept only where they
    connect to one, so cracks stay continuous but loose texture is dropped.
    sensitivity 0..20: higher finds fainter cracks.
    """
    from skimage.filters import apply_hysteresis_threshold
    dark = (ENH_BG - enh.astype(np.float32)) / ENH_BG          # 0 = paint, 1 = black
    hi = float(np.clip(0.8 - 0.04 * sensitivity, 0.05, 0.95))
    return apply_hysteresis_threshold(dark, 0.375 * hi, hi)


def extract(bgr, p, paper=False):
    """Run the whole pipeline with parameter dict p. Returns dict of images.

    paper=False: binary = dark parts of the enhanced picture (default).
    paper=True : binary = the paper's offset (shifted-tile Otsu) threshold of the
                 closing-minus-original image.
    Both use the same two sliders: sensitivity and clean-up.
    """
    enh = enhance(bgr, p["crack_width"], p["light"], p["contrast"], p["texture"])
    sens, cleanup = p["sensitivity"], p["cleanup"]
    if paper:
        gray = to_gray(bgr)
        resp, _ = crack_response(gray, p["crack_width"], p["light"])
        # scale by the 99.5th percentile, not the max, so one dark stain or hole
        # does not squash every real crack into the bottom few grey levels
        hi = max(float(np.percentile(resp, 99.5)), 1.0)
        resp_n = np.clip(resp.astype(np.float32) * (255.0 / hi), 0, 255).astype(np.uint8)
        binary = offset_threshold(resp_n, PAPER_TILE, max(0, 120 - 6 * sens))
    else:
        binary = binary_from_enhanced(enh, sens)
    binary = clean(binary, cleanup)
    skel = thin(binary, cleanup // 2, p["spurs"])
    return dict(enhanced=enh, binary=binary, skeleton=skel)


PAPER_TILE = 40   # tile size (px) for the paper's offset threshold


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
# automatic area finder
# ----------------------------------------------------------------------------

AUTO_MAX_SIDE = 2500   # the painting is scored at this size (px, longest side)
AUTO_LONG = 25         # crack pieces at least this long (scored px) count as "network"
AUTO_COLOUR = 6.0      # colour spread (Lab units) at which flatness drops to 37 %
AUTO_MIN_REL = 0.3     # areas scoring below 30 % of the best one are not offered
AUTO_FLAT_MIN = 0.0    # score multiplier for a busy area (a plain one gets 1.0)
AUTO_DENSE = 0.04      # crack-line density above which "cracks" start to look like noise
AUTO_LIGHT_MAX = 120   # light cracks are only looked for where the paint is darker than this


def _box_mean(integral, x, y, w, h):
    """Mean of a map over a box, from its integral image (cv2.integral)."""
    s = integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x]
    return s / float(w * h)


def crack_depth(bgr, crack_width, light):
    """How much darker (grey levels) each pixel is than the paint around it."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    if light:
        g = 257.0 - g
    g = cv2.GaussianBlur(g, (0, 0), 0.6)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * crack_width + 1,) * 2)
    paint = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_CLOSE, se), (0, 0), crack_width)
    return np.clip(paint - g, 0, None)


def score_maps(bgr, light, crack_width):
    """Per-pixel maps the area score is built from (for one crack polarity).

    Returns maps whose window means give: skeleton density, long-network
    density, crack contrast (depth on the skeleton) and background noise
    (mean and mean-square of depth away from cracks)."""
    p = {key: d for _, key, _, d in SLIDERS}
    p["light"] = int(light)
    p["crack_width"] = int(crack_width)
    res = extract(bgr, p)
    sk = res["skeleton"].astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)
    long_ok = np.zeros(n, dtype=bool)
    long_ok[1:] = stats[1:, cv2.CC_STAT_AREA] >= AUTO_LONG * max(crack_width / 2.0, 1.0)
    long_net = long_ok[lab].astype(np.float32)             # skeleton in long, connected cracks
    D = crack_depth(bgr, p["crack_width"], light)
    bg = (cv2.dilate(res["binary"].astype(np.uint8),
                     np.ones((2 * crack_width + 1,) * 2, np.uint8)) == 0).astype(np.float32)
    skf = sk.astype(np.float32)
    return [skf, long_net, D * skf, bg, D * bg, D * D * bg]


def blockwise_maps(bgr, light, block, crack_width, small_hw, pad=None):
    """score_maps at FULL resolution, block by block, each block's maps then
    averaged down to the small scoring grid (small_hw).

    Full resolution: the finder sees the same cracks as the tuning window.
    Block by block: each block's contrast is scaled on its own, like a
    hand-picked box (scaling over the whole painting lets busy dark areas
    hide faint cracks)."""
    H, W = bgr.shape[:2]
    h, w = small_hw
    sy, sx = h / H, w / W
    pad = pad or 4 * crack_width + 8
    out = [np.zeros((h, w), np.float32) for _ in range(6)]
    for y0 in range(0, H, block):
        for x0 in range(0, W, block):
            y1, x1 = min(y0 + block, H), min(x0 + block, W)
            ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
            yb, xb = min(y1 + pad, H), min(x1 + pad, W)
            ms = score_maps(bgr[ya:yb, xa:xb], light, crack_width)
            ty0, ty1 = int(round(y0 * sy)), int(round(y1 * sy))
            tx0, tx1 = int(round(x0 * sx)), int(round(x1 * sx))
            if ty1 <= ty0 or tx1 <= tx0:
                continue
            for o, m in zip(out, ms):
                m = m[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
                o[ty0:ty1, tx0:tx1] = cv2.resize(m, (tx1 - tx0, ty1 - ty0),
                                                 interpolation=cv2.INTER_AREA)
    return out


def painting_crack_width(bgr, light, n=5):
    """Typical crack width of the whole painting, from a few sample spots."""
    H, W = bgr.shape[:2]
    s = max(min(H, W) // 6, 128)
    spots = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)][:n]
    widths = []
    for fx, fy in spots:
        x, y = int(fx * W - s / 2), int(fy * H - s / 2)
        x, y = max(0, min(x, W - s)), max(0, min(y, H - s))
        widths.append(estimate_crack_width(bgr[y:y + s, x:x + s], light))
    return int(min(widths))            # the cleanest spot: busy ones read too wide


def find_areas(bgr, size=None, n_areas=3, polarity="auto"):
    """Find the areas of a painting that are best for crack analysis.

    Cracks are detected at full resolution, block by block. Each square
    window (overlapping, stride = size/4) is then scored on
      crack evidence: long, connected crack lines (relative to the painting's
                      best), that are clean (not specks) and stand out from the
                      paint's own noise (contrast)            - the main factor
      flatness      : little colour spread, i.e. no objects, outlines or folds
                      (a bonus, not a requirement)
      exposure      : no blown highlights or crushed shadows
    Both dark and light cracks are tried (polarity="auto"), or only one.
    Returns (areas, heat, crackmap): areas = [(x, y, w, h, score, light), ...]
    in full-image pixels, best first and not overlapping; heat is a 0..1 map;
    crackmap is a picture of every crack found, with the areas drawn on it.
    """
    H, W = bgr.shape[:2]
    size = int(size or max(min(H, W) // 5, min(300, min(H, W) // 2)))   # >= 300 px if it fits
    size = max(64, min(size, H, W))
    sc = min(1.0, AUTO_MAX_SIDE / max(H, W))
    small = cv2.resize(bgr, (max(int(W * sc), 1), max(int(H * sc), 1)),
                       interpolation=cv2.INTER_AREA) if sc < 1 else bgr
    h, w = small.shape[:2]
    ws = max(int(size * sc), 32)
    step = max(ws // 4, 4)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # flatness: colour spread of the painting with cracks closed and detail blurred
    # away. Plain paint has almost none; any object, outline or fold adds a lot.
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    I_col = []
    for i in range(3):
        ch = cv2.GaussianBlur(cv2.morphologyEx(lab[..., i], cv2.MORPH_CLOSE, se), (0, 0), 3)
        I_col += [cv2.integral(ch), cv2.integral(ch * ch)]
    clip = ((gray >= 250) | (gray <= 5)).astype(np.float32)
    I_clip = cv2.integral(clip)
    I_gray = cv2.integral(gray.astype(np.float32))

    pols = {"auto": (0, 1), "dark": (0,), "light": (1,)}[polarity]
    block = max(size // 2, 128)
    raw, maps, cws = {}, {}, {}
    for pol in pols:
        cws[pol] = painting_crack_width(bgr, pol)
        print(f"  looking for {'light' if pol else 'dark'} cracks "
              f"(typical width ~{cws[pol]} px) ...")
        raw[pol] = blockwise_maps(bgr, pol, block, cws[pol], (h, w))
        maps[pol] = [cv2.integral(m) for m in raw[pol]]

    ys = list(range(0, h - ws + 1, step)) or [0]
    xs = list(range(0, w - ws + 1, step)) or [0]
    wins = []
    for y in ys:
        for x in xs:
            var = 0.0
            for I1, I2 in zip(I_col[0::2], I_col[1::2]):
                m = _box_mean(I1, x, y, ws, ws)
                var += max(_box_mean(I2, x, y, ws, ws) - m * m, 0.0)
            g = np.sqrt(var)                                            # colour spread
            c = _box_mean(I_clip, x, y, ws, ws)
            bright = _box_mean(I_gray, x, y, ws, ws)
            for pol in pols:
                sk, net, dsk, bg, dbg, d2bg = (_box_mean(I, x, y, ws, ws) for I in maps[pol])
                clean_frac = net / sk if sk > 0 else 0.0
                contrast = dsk / sk if sk > 0 else 0.0
                if bg > 0:
                    mu = dbg / bg
                    noise = np.sqrt(max(d2bg / bg - mu * mu, 0.0))
                else:
                    noise = np.inf
                snr = contrast / (noise + 1e-6)
                wins.append([x, y, net, clean_frac, g, c, snr, sk, bright, pol])
    if not wins:
        return [], np.zeros((H, W), np.float32), small.copy()

    arr = np.array([wv[2:9] for wv in wins], dtype=np.float64)
    net, clean_frac, colour, clip_, snr, dens, bright = arr.T
    is_light = np.array([wv[-1] for wv in wins]) == 1
    # real craquelure covers a few % of the area; a denser "crack" mesh is paint
    # texture or noise (seen in dark, busy areas), however long its lines are
    plausible = np.clip(1 - (dens - AUTO_DENSE) / AUTO_DENSE, 0, 1)
    ok = plausible > 0.5
    net_ref = max(np.percentile(net[ok] if ok.any() else net, 90), 1e-9)   # "a lot of cracks" here
    evidence = (np.clip(net / net_ref, 0, 1)                  # much long crack line
                * clean_frac                                  # lines, not specks
                * np.clip((snr - 2.0) / 4.0, 0, 1) ** 2       # stands out from paint noise
                * plausible)                                  # not an implausibly dense mesh
    if len(pols) == 2:
        # the true crack colour is the one that stands out more from the paint;
        # the other polarity only picks up rims and texture, so it may not compete
        pair = snr.reshape(-1, 2)                             # (dark, light) per window
        loser = np.where(pair[:, 0] >= pair[:, 1], 1, 0)
        evidence.reshape(-1, 2)[np.arange(len(pair)), loser] = 0
    # light cracks (pale ground showing through) only make sense on darker paint;
    # on bright paint a "light crack" is just a highlight rim or texture
    evidence[is_light & (bright > AUTO_LIGHT_MAX)] = 0
    flat = np.exp(-(colour / AUTO_COLOUR) ** 2)               # no objects or folds
    score = evidence * (AUTO_FLAT_MIN + (1 - AUTO_FLAT_MIN) * flat) * np.clip(1 - 5 * clip_, 0, 1)
    if score.max() > 0:
        score = score / score.max()
    find_areas.debug = (wins, arr, score, sc, ws)             # for tuning / inspection

    heat = np.zeros((h, w), np.float32)
    for (x, y, *_), s_ in zip(wins, score):
        heat[y:y + ws, x:x + ws] = np.maximum(heat[y:y + ws, x:x + ws], s_)

    # crack map: every crack found, in the polarity that scores better locally
    best_pol = np.zeros((h, w), np.int8)
    if len(pols) == 2:
        ev = {}
        for pol in pols:
            e = np.zeros((h, w), np.float32)
            for (x, y, *_r), s_ in zip(wins, evidence):
                if _r[-1] == pol:
                    e[y:y + ws, x:x + ws] = np.maximum(e[y:y + ws, x:x + ws], s_)
            ev[pol] = e
        best_pol = (ev[1] > ev[0]).astype(np.int8)
    else:
        best_pol[:] = pols[0]
    cracks = np.zeros((h, w), bool)
    for pol in pols:
        cracks |= (raw[pol][0] > 0) & (best_pol == pol)
    crackmap = (cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                * 0.55).astype(np.uint8)
    crackmap[cracks] = (0, 0, 255)

    heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)

    areas, taken = [], []
    for i in np.argsort(-score):
        if score[i] < AUTO_MIN_REL or len(areas) >= n_areas:
            break                                                 # rest are too poor to offer
        x, y = wins[i][0], wins[i][1]
        if any(abs(x - tx) < ws and abs(y - ty) < ws for tx, ty in taken):
            continue                                              # overlaps a better area
        taken.append((x, y))
        fx, fy = int(round(x / sc)), int(round(y / sc))
        areas.append((fx, fy, min(size, W - fx), min(size, H - fy), float(score[i]), wins[i][-1]))
    for i, (x, y, ww, hh, s_, lt) in enumerate(areas, 1):
        p0 = (int(x * sc), int(y * sc)); p1 = (int((x + ww) * sc), int((y + hh) * sc))
        cv2.rectangle(crackmap, p0, p1, (255, 255, 255), 2)
        cv2.putText(crackmap, f"#{i}", (p0[0] + 5, p0[1] + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return areas, heat, crackmap


def estimate_crack_width(bgr, light):
    """Typical crack width (px) in an area, from the binary's distance transform."""
    p = {key: d for _, key, _, d in SLIDERS}
    p["light"] = int(light)
    res = extract(bgr, p)
    sk = res["skeleton"]
    if sk.sum() < 20:
        return p["crack_width"]
    dist = cv2.distanceTransform(res["binary"].astype(np.uint8), cv2.DIST_L2, 3)
    width = 2 * float(np.median(dist[sk]))                     # full width across the crack
    return int(np.clip(round(width) + 1, 1, 15))                # closing disk a bit wider


def area_map(bgr, areas, heat):
    """Overview picture: painting, heat tint (red = good) and numbered boxes.
    Drawn at most AUTO_MAX_SIDE px on its longest side, so huge scans stay light."""
    sc = min(1.0, AUTO_MAX_SIDE / max(bgr.shape[:2]))
    if sc < 1:
        size = (int(bgr.shape[1] * sc), int(bgr.shape[0] * sc))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        heat = cv2.resize(heat, size, interpolation=cv2.INTER_LINEAR)
        areas = [(int(x * sc), int(y * sc), int(w * sc), int(h * sc), s_, lt)
                 for x, y, w, h, s_, lt in areas]
    over = bgr.copy()
    tint = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(over, 0.65, tint, 0.35, 0)
    t = max(2, int(max(bgr.shape[:2]) / 400))
    for i, (x, y, w, h, s_, light) in enumerate(areas, 1):
        cv2.rectangle(over, (x, y), (x + w, y + h), (255, 255, 255), t * 2)
        cv2.rectangle(over, (x, y), (x + w, y + h), (0, 0, 0), t)
        label = f"#{i} {s_:.2f} {'light' if light else 'dark'}"
        fs = t * 0.6
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)[0][0]
        if tw > w - 6 * t:                                   # keep the label inside the box
            fs *= (w - 6 * t) / tw
        org = (x + t * 3, y + int(t * 3 + 22 * fs))
        cv2.putText(over, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), t * 3, cv2.LINE_AA)
        cv2.putText(over, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t, cv2.LINE_AA)
    return over


# ----------------------------------------------------------------------------
# interactive UI
# ----------------------------------------------------------------------------

SLIDERS = [  # name, key, max, default
    ("light cracks", "light", 1, 0),          # 0 = dark cracks, 1 = light cracks
    ("crack width px", "crack_width", 25, 2),
    ("contrast", "contrast", 50, 10),         # enhanced picture: how black the cracks go
    ("texture cut", "texture", 40, 16),       # enhanced picture: remove faint texture
    ("sensitivity", "sensitivity", 20, 10),   # binary: higher = fainter cracks kept
    ("clean-up", "cleanup", 200, 20),         # binary: delete pieces smaller than this
    ("trim spurs px", "spurs", 50, 8),        # skeleton: cut side ticks shorter than this
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


def pick_images():
    """File dialog: pick one or many paintings (Ctrl/Shift-click for many)."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title="Pick painting image(s)  -  Ctrl/Shift-click to pick many",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"), ("All", "*.*")])
    root.destroy()
    return list(paths)


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


def select_rois(img, title="select crack area(s)"):
    """Let the user draw boxes on the image; returns boxes in full-image pixels."""
    sw, sh = screen_size()
    s, disp = fit_to_screen(img, sw - 80, sh - 140)
    win = f"{title}   -   drag a box, ENTER = keep it, draw more, ESC = done"
    rois = cv2.selectROIs(win, disp, showCrosshair=False)
    cv2.destroyWindow(win)
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
        tiles = [[("original", crop), ("enhanced", enh)],
                 [("binary", bin_img), ("skeleton", None)]]
    else:
        tiles = [[("original", crop), ("enhanced", enh)]]
    rows, cols = len(tiles), len(tiles[0])
    gap = 6
    h, w = crop.shape[:2]
    s, _ = fit_to_screen(crop, (max_w - gap * (cols - 1)) / cols,
                         (max_h - gap * (rows - 1)) / rows, upscale=6.0)
    tw, th = max(int(w * s), 1), max(int(h * s), 1)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
    if view == 2:
        # 1-px lines vanish when shrunk: thicken them just enough for the screen
        k = max(int(np.ceil(1 / s)), 1) if s < 1 else 1
        sk = cv2.dilate(res["skeleton"].astype(np.uint8), np.ones((k, k), np.uint8))
        tiles[1][1] = ("skeleton", gray3(np.where(sk > 0, 0, 255).astype(np.uint8)))
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


def tune(crop, params, view=0, paper=False):
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
            res = extract(crop, cur, paper)
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
    imwrite(b + "_binary.png", np.where(res["binary"], 0, 255).astype(np.uint8))
    imwrite(b + "_skeleton.png", np.where(res["skeleton"], 0, 255).astype(np.uint8))
    ov = crop.copy(); ov[res["skeleton"]] = (0, 0, 255)
    imwrite(b + "_overlay.png", ov)
    stats = network_stats(res["skeleton"])
    with open(b + "_info.json", "w") as f:
        json.dump(dict(roi_xywh=roi, params=params, stats=stats), f, indent=2)
    print(f"  SAVED -> {b}_enhanced.png  (and _binary, _skeleton, ... in the same folder)")
    print(f"  stats: {stats}")


IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def crack_snr(crop, light, crack_width):
    """How clearly cracks of one colour stand out from the paint's own noise."""
    sk, _, dsk, bg, dbg, d2bg = (float(m.mean()) for m in score_maps(crop, light, crack_width))
    if sk <= 0 or bg <= 0:
        return 0.0
    noise = np.sqrt(max(d2bg / bg - (dbg / bg) ** 2, 0.0))
    return (dsk / sk) / (noise + 1e-6)


def auto_params(crop, polarity="auto"):
    """Automatic settings for one area: crack colour and crack width.

    Dark or light cracks: whichever stands out more from the paint (light cracks
    only on darker paint, where pale ground can show through). Crack width:
    measured from the cracks themselves. Other sliders keep their defaults."""
    p = {key: d for _, key, _, d in SLIDERS}
    cands = {"auto": (0, 1), "dark": (0,), "light": (1,)}[polarity]
    if polarity == "auto" and cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean() > AUTO_LIGHT_MAX:
        cands = (0,)
    best = None
    for light in cands:
        cw = estimate_crack_width(crop, light)
        snr = crack_snr(crop, light, cw) if len(cands) > 1 else 0.0
        if best is None or snr > best[0]:
            best = (snr, light, cw)
    p["light"], p["crack_width"] = int(best[1]), int(best[2])
    return p


def selection_map(img, rois):
    """Small picture of the painting with the drawn boxes numbered."""
    sc = min(1.0, AUTO_MAX_SIDE / max(img.shape[:2]))
    out = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else img.copy()
    t = max(2, int(max(out.shape[:2]) / 500))
    for i, (x, y, w, h) in enumerate(rois, 1):
        p0, p1 = (int(x * sc), int(y * sc)), (int((x + w) * sc), int((y + h) * sc))
        cv2.rectangle(out, p0, p1, (255, 255, 255), t * 2)
        cv2.rectangle(out, p0, p1, (0, 0, 0), t)
        org = (p0[0] + 3 * t, p0[1] + 12 * t)
        cv2.putText(out, f"#{i}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (0, 0, 0), 3 * t, cv2.LINE_AA)
        cv2.putText(out, f"#{i}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (255, 255, 255), t, cv2.LINE_AA)
    return out


def run_selected(path, args, k, n):
    """Draw boxes on one painting; each box is processed automatically and saved."""
    img = imread(path)
    if img is None:
        print(f"could not read {path} - skipped")
        return
    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.abspath(args.out or os.path.splitext(path)[0] + "_cracks")
    print(f"\n[{k}/{n}] {path}")
    rois = select_rois(img, f"[{k}/{n}] {stem}")
    if not rois:
        print("  no box drawn - skipped")
        return
    print(f"  results will be saved in: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    imwrite(os.path.join(out_dir, stem + "_selection.jpg"), selection_map(img, rois))
    for i, (x, y, w, h) in enumerate(rois, 1):
        crop = img[y:y + h, x:x + w]
        p = auto_params(crop, args.polarity)
        print(f"  box {i}: x={x} y={y} {w}x{h}  {'light' if p['light'] else 'dark'} cracks, "
              f"crack width ~{p['crack_width']} px")
        if args.tune:
            action, p, res, _ = tune(crop, p, 0, args.paper)
            if action == "q":
                sys.exit("stopped")
            if action != "s":
                continue
        else:
            res = extract(crop, p, args.paper)
        save(out_dir, f"{stem}_sel{i}", crop, res, p, [x, y, w, h])


def auto_areas(img, args):
    """Find the best areas and the settings for each: [(roi, params), ...]."""
    print("finding the best areas for crack analysis ...")
    areas, heat, crackmap = find_areas(img, args.size, args.areas, args.polarity)
    out = []
    for i, (x, y, w, h, score, light) in enumerate(areas, 1):
        crop = img[y:y + h, x:x + w]
        p = {key: d for _, key, _, d in SLIDERS}
        p["light"] = int(light)
        p["crack_width"] = estimate_crack_width(crop, light)
        print(f"  area {i}: x={x} y={y} size={w}x{h}  score={score:.2f}  "
              f"{'light' if light else 'dark'} cracks  crack width~{p['crack_width']} px")
        out.append(((x, y, w, h), p))
    if not areas:
        print("  no area with clear cracks found")
    return out, area_map(img, areas, heat), crackmap


def show_area_map(amap, crackmap):
    """Show the area map and the crack map side by side;
    ENTER/SPACE = use these areas, ESC = draw my own."""
    win = "left: best areas (red = good)   right: cracks found (red)   ENTER = analyse,  ESC = draw my own"
    sw, sh = screen_size()
    cm = cv2.resize(crackmap, (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST)
    both = np.hstack([amap, np.full((amap.shape[0], 8, 3), 60, np.uint8), cm])
    _, disp = fit_to_screen(both, sw - 80, sh - 140)
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, disp)
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (13, 10, 32):
            cv2.destroyWindow(win)
            return True
        if k in (27, ord("q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(win)
            return False


def run_batch(path, args, kind="auto"):
    """No windows: find the best areas and save everything for them.
    kind names the files: "auto" (one image) or "F_auto" (a folder run)."""
    img = imread(path)
    if img is None:
        print(f"could not read {path} - skipped")
        return
    out_dir = os.path.abspath(args.out or os.path.splitext(path)[0] + "_cracks")
    stem = os.path.splitext(os.path.basename(path))[0]
    print(f"\n{path}\nresults will be saved in: {out_dir}")
    found, amap, crackmap = auto_areas(img, args)
    os.makedirs(out_dir, exist_ok=True)
    imwrite(os.path.join(out_dir, stem + "_areas.jpg"), amap)
    imwrite(os.path.join(out_dir, stem + "_crackmap.jpg"), crackmap)
    for i, ((x, y, w, h), p) in enumerate(found, 1):
        crop = img[y:y + h, x:x + w]
        save(out_dir, f"{stem}_{kind}{i}", crop, extract(crop, p, args.paper), p, [x, y, w, h])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="painting image (omit for a file dialog); with "
                                              "--batch or --select also several images or a folder")
    ap.add_argument("--auto", action="store_true",
                    help="find the best areas automatically, then tune them in the windows")
    ap.add_argument("--batch", action="store_true",
                    help="find the best areas and save the results without any windows")
    ap.add_argument("--select", action="store_true",
                    help="draw boxes on every image in turn; each box is processed automatically")
    ap.add_argument("--tune", action="store_true",
                    help="with --select: open the slider windows for each box instead")
    ap.add_argument("--light", action="store_true", help="start with the light-cracks switch on")
    ap.add_argument("--paper", action="store_true",
                    help="make the binary with the paper's offset (shifted-tile Otsu) threshold")
    ap.add_argument("--areas", type=int, default=3, help="how many areas to find (default 3)")
    ap.add_argument("--size", type=int, help="area size in pixels (default: 1/5 of the "
                                             "painting's shorter side, at least 300)")
    ap.add_argument("--polarity", choices=["auto", "dark", "light"], default="auto",
                    help="crack colour (default: chosen automatically)")
    ap.add_argument("--out", help="output folder (default <image>_cracks)")
    args = ap.parse_args()

    paths, from_folder = [], False
    for p in args.images:
        if os.path.isdir(p):
            from_folder = True
            paths += sorted(os.path.join(p, f) for f in os.listdir(p)
                            if f.lower().endswith(IMG_EXTS))
        else:
            paths.append(p)

    if args.select:                                    # draw boxes on every image
        if not paths:
            paths = pick_images()
        if not paths:
            sys.exit("no image chosen")
        check_gui()
        print(f"{len(paths)} image(s). For each: drag a box over cracks, ENTER to keep it, "
              f"draw more if you like, ESC when done (ESC with no box skips the image).")
        try:
            for k, p in enumerate(paths, 1):
                run_selected(p, args, k, len(paths))
        except KeyboardInterrupt:
            print("\nstopped - everything saved so far is kept")
        cv2.destroyAllWindows()
        print("\nall done")
        return

    if args.batch:                                     # no windows
        if not paths:
            paths = pick_images()[:1]
        if not paths:
            sys.exit("no image chosen")
        if len(paths) > 1 and args.out:
            print("note: --out is ignored for several images; each gets its own _cracks folder")
            args.out = None
        kind = "F_auto" if from_folder else "auto"
        if from_folder:
            print(f"{len(paths)} images")
        for f in paths:
            run_batch(f, args, kind)
        return

    if len(paths) > 1:
        sys.exit("one image at a time here - for several use --select or --batch")
    path = paths[0] if paths else (pick_images() or [None])[0]
    if not path:
        sys.exit("no image chosen")
    img = imread(path)
    if img is None:
        sys.exit(f"could not read {path}")
    out_dir = os.path.abspath(args.out or os.path.splitext(path)[0] + "_cracks")
    print(f"results will be saved in: {out_dir}")
    stem = os.path.splitext(os.path.basename(path))[0]

    check_gui()
    base = {key: d for _, key, _, d in SLIDERS}
    base["light"] = int(args.light)
    jobs = []                                          # [(roi, params or None, tag)]
    if args.auto:
        found, amap, crackmap = auto_areas(img, args)
        os.makedirs(out_dir, exist_ok=True)
        imwrite(os.path.join(out_dir, stem + "_areas.jpg"), amap)
        imwrite(os.path.join(out_dir, stem + "_crackmap.jpg"), crackmap)
        if found and show_area_map(amap, crackmap):
            jobs = [(roi, p, f"{stem}_checked{i}") for i, (roi, p) in enumerate(found, 1)]
        elif not found:
            print("no suitable area found - draw your own")
    if not jobs:
        rois = select_rois(img)
        if not rois:
            rois = [(0, 0, img.shape[1], img.shape[0])]
            print("no box drawn - using the whole image")
        jobs = [(roi, None, f"{stem}_roi{i}") for i, roi in enumerate(rois, 1)]

    params, view = base, 0
    for i, ((x, y, w, h), p_auto, tag) in enumerate(jobs, 1):
        crop = img[y:y + h, x:x + w]
        print(f"area {i}/{len(jobs)}: x={x} y={y} w={w} h={h}")
        print("  click on either window, then press  s = save,  n = skip,  q = quit,  v = view")
        start = dict(p_auto) if p_auto else params     # auto areas start from their own settings
        action, used, res, view = tune(crop, start, view, args.paper)
        if action == "q":
            break
        params = used                                  # carry settings to next hand-drawn area
        if action == "s":
            save(out_dir, tag, crop, res, used, [x, y, w, h])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
