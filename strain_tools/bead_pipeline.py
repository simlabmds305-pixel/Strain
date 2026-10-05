#!/usr/bin/env python3
# ============================================================================
#  BEAD VOLUME -> SHRINKAGE STRAIN  |  measurement engine
#
#  Side-view images of a gesso bead drying on a mat.  Each image gives a
#  silhouette; the silhouette is integrated as a stack of discs to give a
#  volume; the volume series gives a strain.
#
#  Four volume estimators are computed for every image.  They differ ONLY in
#  how they treat the rows between the bead's widest row and the mat, which is
#  the part of the silhouette that front lighting loses to the bead's own
#  shadow.  Each estimator carries a confidence scored against evidence that is
#  INDEPENDENT of its own value, the trusted ones are combined into a
#  consensus, and their spread is turned into a per-image verdict.
#
#  Engine only: no plotting, no file writing, no printing.  See run_local.py.
# ============================================================================
from __future__ import annotations

import math
import os
import re
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

# ---------------------------------------------------------------- defaults
# Nothing here is specific to one set of images.  run_local.py overrides any
# of it from the command line, and whatever was actually used is written to
# <folder>/analysis/settings.json beside the results.
DEFAULTS = dict(
    SCALE_PX_PER_UM=0.256,     # 1 um = 0.256 px   (Olympus SZX16, 1x, 3088 px wide)
    ROI=None,                  # (x0, x1, y0, y1) in full-image pixels; None = whole frame
    INVERT=False,
    MANUAL_THRESH=None,
    THRESH_OFFSET=0,
    SEGMENT_METHOD="otsu",     # otsu | adaptive | edges
    BLOB_CHOICE="best",
    MAT_CHROMA_MIN=75,         # drop pixels this colourful: the mat, never the bead
    CLOSE_PX=5,
    FILL_HOLES=True,
    A_METHOD="robust",
    BASELINE="blue",           # "blue" | "auto" | a number (full-image row Y)
    BASELINE_TUNE=0.0,
    CLIP_AT_BASELINE=True,
    CLIP_TOLERANCE_PX=10,
    INTERVAL_S=None,           # seconds between frames; None -> x axis is image number
    TIME_REGEX=None,           # e.g. r"_(\d+)min" to read the time out of the file name
    OUTLIER_Z=2.5,             # brightness-outlier threshold (MAD units)
    REFERENCE_INDEX=0,         # which usable image is V0
)

# --- which estimators are allowed to DECIDE the answer ----------------------
# V_disk is the pure measurement: it integrates exactly the rows the threshold
# found, and assumes nothing.  V_extrap is the physical estimate: it continues
# each side of the bead down to the mat along the slope the edge had while it
# was still crisp.  They bracket the truth, and -- the useful part -- they
# AGREE only when the mask already reached the mat, which is precisely the
# condition under which the measurement can be trusted.  So their spread is a
# real test, not a formality.
#
# V_trunc (stop at the widest row) is a hard lower bound and V_base (fill at
# constant radius) an intermediate; both are reported and plotted, but neither
# can by itself drag a good image into CONFLICT.
TRUST_METHODS = ("V_disk", "V_extrap")

METHODS = ("V_disk", "V_trunc", "V_base", "V_extrap")

# Confidence floor: below this an estimator is not worth listening to AT ALL,
# and does not get to decide the run.  See choose_deciders.
MIN_CONF = 0.35

# When is a single FRAME broken?  Not by comparison with a constant: the whole
# run's confidences move together (they share a baseline, a mat, a lamp), so a
# fixed cut lands wherever that common level happens to sit and then splits
# neighbouring frames on noise.  A frame is broken when it is much worse than
# the run it belongs to -- with an absolute floor for genuine rubbish.
FRAME_REJECT_FRAC = 0.5      # of the run's median best-decider confidence
FRAME_REJECT_ABS = 0.10

# Agreement tolerances, as a FRACTION of the consensus volume.  Volumes run to
# ~1e8 px^3 and shrink by tens of percent over a run, so an absolute tolerance
# in px^3 would mean nothing; relative is the only scale that holds across
# magnifications and bead sizes.
CERT_TOL = 0.02      # trusted estimators within 2%  -> CERTIFIED
LIKELY_TOL = 0.05    # within 5%                     -> LIKELY

# Calibration of the confidence scores (all in units of the bead's own height,
# so they hold for any bead size or magnification).
GAP_FULL = 0.15      # mask ending this far from the mat scores V_disk zero
TAPER_FULL = 0.25    # base narrowed to (1 - this) of the widest row scores zero
TRUNC_FULL = 0.03    # V_trunc discarding this share of the volume scores it zero
FILL_BASE = 0.15     # a constant-radius fill stays fair over a short run
FILL_EXTRAP = 0.30   # a straight continuation stays fair over a longer one
HOLDOUT_FULL = 3.0   # px of held-out edge-prediction error that scores V_extrap zero

# A baseline inferred from the masks themselves ("auto") cannot be used to
# judge how far those same masks fall short of it -- that is circular, and a
# circular score reads high exactly when it is least deserved.  Cap it.
#
# This value must stay CLEARLY ABOVE MIN_CONF.  Set equal to it (both were 0.35)
# every greyscale run landed exactly on the decision gate: measured confidences
# of 0.345, 0.348, 0.350 flipped frames between SINGLE and REJECT on rounding
# noise alone, and one 4-frame set was thrown out entirely with masks that were
# in fact perfect (base_taper 0.997, reaching the mat on every frame).  An
# inferred baseline makes a run weaker, not void; the warning text says so.
AUTO_BASELINE_CONF = 0.65


# ------------------------------------------------------------------- helpers
def natural_key(path):
    s = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_frames(folder, pattern="*"):
    """Image files in a folder, in natural order (img2 before img10)."""
    import glob
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    if pattern and pattern not in ("*", "*.*"):
        files = glob.glob(os.path.join(folder, pattern))
    else:
        files = [p for p in glob.glob(os.path.join(folder, "*"))
                 if p.lower().endswith(exts)]
    files = [p for p in files if os.path.isfile(p)]
    return sorted(files, key=natural_key)


def clip01(v):
    return float(np.clip(v, 0.0, 1.0))


# ---------------------------------------------------------------- SEGMENTING
def threshold_crop(img, invert=False, manual_thresh=None, method="otsu", offset=0):
    """Foreground mask of an already-cropped grayscale image."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    if method == "adaptive":
        bs = max(11, (min(img.shape) // 3) | 1)
        mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, bs, -5)
    elif method == "edges":
        v = float(np.median(blur))
        edges = cv2.Canny(blur, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(edges)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    elif manual_thresh is not None:
        _, mask = cv2.threshold(blur, manual_thresh, 255, cv2.THRESH_BINARY)
    else:
        level, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if offset:
            _, mask = cv2.threshold(blur, max(0, min(255, level + offset)), 255,
                                    cv2.THRESH_BINARY)
    if invert:
        mask = cv2.bitwise_not(mask)
    return mask


def clean_mask(mask, close_px=5, fill_holes=True):
    """Seal thin gaps and fill interior holes (glare) so the silhouette stays solid."""
    if close_px and close_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_px), int(close_px)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if fill_holes:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        mask = filled
    return mask


def profile_from_mask(mask):
    """Per-row left edge, right edge, width and pixel count, top to bottom.

    Vectorised.  The obvious loop -- `xs[ys == y]` for each row -- rescans the
    whole coordinate list once per row, which is quadratic in the silhouette
    and was the single slowest thing in the pipeline: on a 1000-row bead it is
    ~10^9 comparisons, and it is paid again for every candidate blob.  argmax
    on a boolean array finds the first and last set pixel per row in one pass.
    """
    m = mask > 0
    idx = np.flatnonzero(m.any(axis=1))
    if idx.size == 0:
        return None
    sub = m[idx]
    lefts = sub.argmax(axis=1)
    rights = sub.shape[1] - 1 - sub[:, ::-1].argmax(axis=1)
    counts = sub.sum(axis=1)
    widths = (rights - lefts + 1).astype(float)
    return idx, widths, lefts.astype(int), rights.astype(int), counts.astype(int)


def mask_quality(mask, ref_area=None):
    """How much does this blob look like a bead sitting on the bottom of the crop?"""
    h, w = mask.shape
    pr = profile_from_mask(mask)
    if pr is None:
        return {"score": -1.0, "area_frac": 0.0, "solidity": 0.0,
                "touches": "nothing", "dome": 0.0}
    rows, widths, lefts, rights, counts = pr
    area = int(counts.sum())
    area_frac = area / float(h * w)
    t_top = bool(rows[0] == 0)
    t_bot = bool(rows[-1] == h - 1)
    t_l = bool(lefts.min() == 0)
    t_r = bool(rights.max() == w - 1)
    dome = float(np.mean(np.diff(widths) >= 0)) if widths.size > 2 else 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidity = float(cv2.contourArea(c) / hull_area) if hull_area > 0 else 0.0

    size = (area / ref_area) if ref_area else (area_frac / 0.10)
    score = 1.5 * dome
    score += 1.5 * solidity - 0.75
    score += 0.5 * min(1.0, size)
    score += 0.5 if t_bot else -0.5
    score -= 1.0 * (t_top + t_l + t_r)
    if area < max(200, 0.001 * h * w):
        score -= 1.5
    if area_frac > 0.75:
        score -= 1.5
    touches = ",".join([n for n, f in (("top", t_top), ("bottom", t_bot),
                                       ("left", t_l), ("right", t_r)) if f]) or "none"
    return {"score": float(score), "area_frac": float(area_frac), "solidity": solidity,
            "touches": touches, "dome": dome}


def pick_blob(mask, mode="best", max_candidates=6):
    """Choose the bead's blob.  The largest bright thing is not always the bead."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise RuntimeError("no foreground blob found - check threshold / invert")
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas) + 1
    if mode != "best":
        return np.uint8(labels == order[0]) * 255
    floor = max(0.05 * areas.max(), 50)
    cands = [i for i in order[:max_candidates] if stats[i, cv2.CC_STAT_AREA] >= floor]
    ref = float(max(stats[i, cv2.CC_STAT_AREA] for i in cands))
    scored = []
    for i in cands:
        blob = np.uint8(labels == i) * 255
        scored.append((mask_quality(blob, ref)["score"], stats[i, cv2.CC_STAT_AREA], i))
    best = max(sc for sc, _, _ in scored)
    i = max((a, i) for sc, a, i in scored if sc >= best - 0.25)[1]
    return np.uint8(labels == i) * 255


def base_radius(widths_px, method="robust"):
    """Half-width of the bead's base; 'robust' ignores a one-row flare at the contact line."""
    if method != "robust" or widths_px.size < 20:
        return float(widths_px.max()) / 2.0
    thr = np.percentile(widths_px, 95)
    return float(np.median(widths_px[widths_px >= thr])) / 2.0


def mat_line_from_chroma(sat, x_bead, y_band, chroma_step=25):
    """Top edge of a coloured mat, from an already-computed chroma image.

    A coloured mat against a dark background gives a step in chroma.  That step
    is a property of the mat, not of how the bead is lit, so unlike a grey
    level it stays put as the bead dries.  The bead hides the mat underneath
    it, so the edge is read from the columns either side and fitted as a line,
    which also handles a mat that is not level.

    Chroma (max channel - min channel), never HSV saturation: saturation
    divides by brightness, so on a dark background it is noise.  Returns
    (slope, intercept) in full-image coordinates, or None.
    """
    H, W = sat.shape
    y0, y1 = y_band
    xl, xr = x_bead
    cols = list(range(max(2, xl - 120), max(3, xl - 8), 4)) + \
           list(range(min(W - 3, xr + 8), min(W - 2, xr + 120), 4))
    xs, ys = [], []
    for x in cols:
        col = cv2.GaussianBlur(sat[y0:y1, x - 2:x + 3].mean(axis=1), (1, 9), 0).ravel()
        if col.size < 12 or col.max() - col.min() < chroma_step:
            continue
        xs.append(x)
        ys.append(y0 + int(np.argmax(np.gradient(col))))
    if len(xs) < 6:
        return None
    xs, ys = np.array(xs, float), np.array(ys, float)
    ok = np.abs(ys - np.median(ys)) < 40
    if ok.sum() < 6:
        return None
    slope, intercept = np.polyfit(xs[ok], ys[ok], 1)
    return float(slope), float(intercept)


# ----------------------------------------------------------- STAGE 1: per image
def measure_one(args):
    """Segment one image and return everything later stages need.

    Pure function of (path, cfg) so it can run in a worker process.  Reads the
    file ONCE: the colour frame is decoded, the grey image and the chroma image
    are derived from that same array, and the mat-edge fit is taken from it too
    -- the notebook decoded the same file three times per image.
    """
    path, cfg, want_mask = args
    try:
        return _measure_one(path, cfg, want_mask)
    except Exception as e:                       # one bad frame must not kill a run
        return {"path": path, "image": os.path.basename(path), "ok": False,
                "error": f"{type(e).__name__}: {e}"}


def _measure_one(path, cfg, want_mask=False):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    H, W = bgr.shape[:2]

    roi = cfg.get("ROI")
    x0, x1, y0, y1 = roi if roi else (0, W, 0, H)
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    y0, y1 = max(0, int(y0)), min(H, int(y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError(f"the crop {tuple(roi)} does not overlap this {W}x{H} image at all")
    if roi:
        # A crop picked on one set and reused on another of a different size is
        # the usual cause, and "every image failed to segment" says nothing
        # about it.  Name the real problem.
        want = (int(roi[1]) - int(roi[0])) * (int(roi[3]) - int(roi[2]))
        if want > 0 and (x1 - x0) * (y1 - y0) < 0.7 * want:
            raise ValueError(
                f"the crop {tuple(roi)} mostly falls outside this {W}x{H} image - only "
                f"{x1 - x0}x{y1 - y0} px of it land on the frame. These images are a "
                f"different size from the ones the crop was picked on, so pick a crop "
                f"for this set (run it on its own with --pick-roi)")

    crop = bgr[y0:y1, x0:x1]
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(img.mean())

    mask = threshold_crop(img, invert=cfg["INVERT"], manual_thresh=cfg["MANUAL_THRESH"],
                          method=cfg["SEGMENT_METHOD"], offset=cfg["THRESH_OFFSET"])

    # Drop anything too colourful to be the bead.  On a white bead against a
    # coloured mat this is what stops the two merging once the bead's shadowed
    # base reaches the mat's own grey level.  On a grey image it does nothing.
    chroma_crop = None
    if cfg["MAT_CHROMA_MIN"] < 255:
        c16 = crop.astype(np.int16)
        chroma_crop = (c16.max(axis=2) - c16.min(axis=2))
        coloured = chroma_crop >= cfg["MAT_CHROMA_MIN"]
        if float(coloured.mean()) > 0.02:        # a real population, not a stray pixel
            mask[coloured] = 0

    mask = clean_mask(mask, close_px=cfg["CLOSE_PX"], fill_holes=cfg["FILL_HOLES"])
    mask = pick_blob(mask, cfg["BLOB_CHOICE"])

    pr = profile_from_mask(mask)
    if pr is None:
        raise RuntimeError("empty silhouette after segmentation")
    rows, widths, lefts, rights, counts = pr

    mh, mw = mask.shape
    clipped = ",".join([n for n, f in (("top", rows[0] == 0),
                                       ("left", lefts.min() == 0),
                                       ("right", rights.max() == mw - 1)) if f])

    out = {
        "path": path, "image": os.path.basename(path), "ok": True, "error": "",
        "roi": (x0, x1, y0, y1), "frame_w": W, "frame_h": H,
        "brightness": brightness,
        "clipped": clipped,
        "touches_roi_bottom": bool(rows[-1] == mh - 1),
        # profile in FULL-image coordinates, so every later stage is frame-absolute
        "Y": (rows + y0).astype(np.int32),
        "widths": widths.astype(np.float32),
        "xl": (lefts + x0).astype(np.float32),
        "xr": (rights + x0).astype(np.float32),
        "counts": counts.astype(np.int32),
    }

    # The mat's edge, from this image's own colour.  Computed here so the file
    # is not decoded a second time for it.
    out["mat_fit"] = None
    if cfg["BASELINE"] == "blue" and chroma_crop is not None:
        sat_full = np.zeros((H, W), np.float32)
        sat_full[y0:y1, x0:x1] = chroma_crop
        fit = mat_line_from_chroma(sat_full, (int(out["xl"].min()), int(out["xr"].max())),
                                   (y0, y1))
        if fit:
            xc = float((out["xl"] + out["xr"])[len(rows) // 2] / 2.0)
            out["mat_fit"] = fit
            out["mat_y"] = fit[0] * xc + fit[1] + cfg["BASELINE_TUNE"]
            out["mat_tilt_deg"] = float(np.degrees(np.arctan(fit[0])))

    if want_mask:
        out["mask"] = mask
        out["img"] = img
    return out


# ------------------------------------------------- STAGE 2: baseline for the set
def resolve_baseline(frames, cfg):
    """One mat row for the whole set, plus how much that row can be trusted.

    Returns (y_base, source, confidence, notes).  `source` is "blue" (read off
    the mat's own colour), "given" (you supplied it) or "auto" (inferred from
    the deepest silhouette).  "auto" is derived FROM the masks, so it cannot
    honestly be used to judge how far those masks fall short of it -- the
    confidence it carries is capped accordingly.
    """
    notes = []
    want = cfg["BASELINE"]

    if want not in ("blue", "auto", None):
        return float(want), "given", 1.0, notes

    if want == "blue":
        found = [f["mat_y"] for f in frames if f.get("mat_y") is not None]
        if found:
            med = float(np.median(found))
            agree = [v for v in found if abs(v - med) <= 30]
            if len(agree) >= max(3, 0.6 * len(found)):
                tilt = np.median([f.get("mat_tilt_deg", 0.0) for f in frames
                                  if f.get("mat_y") is not None])
                notes.append(f"mat row from the mat's colour: Y = {np.median(agree):.0f} "
                             f"(agreed on {len(agree)} of {len(found)} images to within 30 px, "
                             f"tilt {tilt:+.2f} deg)")
                # Confidence: how many images saw the same edge, and how tightly.
                frac = len(agree) / float(len(frames))
                tight = clip01(1 - float(np.std(agree)) / 15.0)
                # The floor matters more than it looks.  This value multiplies
                # every confidence in the run, so if it lands near MIN_CONF the
                # whole run sits on the decision gate and frames split between
                # CERTIFIED and REJECT on the third decimal place.  With the old
                # floor of 0.4 a real run scored 0.901 x 0.4 = 0.36 against a
                # gate of 0.35, and did exactly that.  Getting here already
                # required agreement within 30 px on 60% of frames, which is
                # decent evidence; score it like evidence.
                conf = clip01(0.5 + 0.5 * frac) * (0.65 + 0.35 * tight)
                return float(np.median(agree)), "blue", conf, notes
            notes.append(f"the colour edge disagrees with itself: {len(found)} images gave rows "
                         f"spanning {np.ptp(found):.0f} px. The mat cannot move, so that is not "
                         f"the mat -- most likely it is not coloured here, or it is outside the "
                         f"crop. Falling back to 'auto'.")
        else:
            notes.append("no colour edge found for the mat (grey mat, or a greyscale image) "
                         "- falling back to 'auto'.")

    bottoms = np.array([f["Y"][-1] for f in frames], float)
    y_base = float(bottoms.max())
    p90 = float(np.percentile(bottoms, 90))
    notes.append(f"mat row inferred from this set: Y = {y_base:.0f} (deepest silhouette; "
                 f"90% of them stop at or above Y = {p90:.0f})")
    notes.append("inferred this way the row can only ever be as deep as the deepest silhouette, "
                 "so if the shadow eats every base it is too high and every volume below is an "
                 "underestimate. The confidence scores are capped to say so.")
    return y_base, "auto", AUTO_BASELINE_CONF, notes


# --------------------------------------------- STAGE 3: volumes + confidences
def _edge_slope_fit(Y, xl, xr, i_w, reach):
    """Fit each side's slope ABOVE the widest row, holding out the last rows.

    Returns (ml, cl, mr, cr, holdout_rms_px).  The fit uses rows
    [i_w-n_fit, i_w-n_hold] and is scored on [i_w-n_hold, i_w], which it never
    saw.  Held-out error is the test that means something: scoring a fit on the
    points it was fitted to measures nothing at all, the way a template matched
    against a copy of itself always wins.

    Both windows are sized from `reach`, the number of rows the extrapolation
    actually has to cover.  A bead's edge is an arc, so a fixed 120-row band
    measures the bead's CURVATURE rather than its noise, and reports metres of
    error for an extrapolation that only has to reach a few pixels.  Predicting
    a window the size of the reach asks the question that is actually being
    asked: can this edge be continued THAT far?
    """
    n_hold = int(np.clip(round(reach), 10, 80))
    n_fit = int(min(i_w, max(40, 3 * n_hold)))
    lo = max(0, i_w - n_fit)
    mid = max(lo + 10, i_w - n_hold)

    if i_w - lo < 10:
        return None
    if mid - lo < 10 or i_w - mid < 5:        # too few rows to hold anything out
        ml, cl = np.polyfit(Y[lo:i_w + 1], xl[lo:i_w + 1], 1)
        mr, cr = np.polyfit(Y[lo:i_w + 1], xr[lo:i_w + 1], 1)
        return ml, cl, mr, cr, np.nan

    ml, cl = np.polyfit(Y[lo:mid], xl[lo:mid], 1)
    mr, cr = np.polyfit(Y[lo:mid], xr[lo:mid], 1)
    yh = Y[mid:i_w + 1]
    el = xl[mid:i_w + 1] - (ml * yh + cl)
    er = xr[mid:i_w + 1] - (mr * yh + cr)
    rms = float(np.sqrt(np.mean(np.concatenate([el, er]) ** 2)))
    # refit on everything for the actual extrapolation, now that it is scored
    ml, cl = np.polyfit(Y[lo:i_w + 1], xl[lo:i_w + 1], 1)
    mr, cr = np.polyfit(Y[lo:i_w + 1], xr[lo:i_w + 1], 1)
    return ml, cl, mr, cr, rms


def volumes_for(frame, y_base, baseline_conf, cfg):
    """The four volume estimators for one image, each with a confidence.

    Every confidence is scored against evidence INDEPENDENT of the number it is
    scoring: how far the mask stopped short of the mat, how much the silhouette
    narrowed before it ended, how far an estimator has to reach to get to the
    mat, and held-out edge-prediction error.  None of them is the estimator
    grading its own homework.
    """
    Y = frame["Y"].astype(float)
    w = frame["widths"].astype(float)
    xl = frame["xl"].astype(float)
    xr = frame["xr"].astype(float)

    # Nothing below the mat is bead.  Clipping there is exactly a truncation of
    # the profile -- the mask was already cleaned and the blob already chosen --
    # so there is no need to re-segment the image, as the notebook did.
    if y_base is not None and cfg["CLIP_AT_BASELINE"]:
        if Y[-1] > y_base + cfg["CLIP_TOLERANCE_PX"]:
            keep = Y <= y_base
            if keep.sum() >= 0.5 * len(Y):
                Y, w, xl, xr = Y[keep], w[keep], xl[keep], xr[keep]
                frame["clipped_at_mat"] = True
            else:
                frame["clip_refused"] = True

    m = {}
    r = w / 2.0
    h_px = float(len(Y))          # provisional: replaced below once the mat is known
    i_w = int(np.argmax(w))
    m["h_px"] = h_px
    m["a_px"] = base_radius(w, cfg["A_METHOD"])
    m["width_max_px"] = float(w.max())
    m["n_rows"] = int(len(Y))
    m["Y_apex_full"] = int(Y[0])
    m["Y_widest_full"] = int(Y[i_w])
    m["Y_bottom_full"] = int(Y[-1])
    m["clipped_at_mat"] = bool(frame.get("clipped_at_mat", False))
    m["clip_refused"] = bool(frame.get("clip_refused", False))

    # --- the four estimators ------------------------------------------------
    m["V_disk"] = float(np.pi * np.sum(r ** 2))                 # every row as found
    m["V_trunc"] = float(np.pi * np.sum(r[:i_w + 1] ** 2))      # stop at the widest row

    if y_base is None:
        y_base_eff = float(Y[-1])
    else:
        y_base_eff = float(y_base)
    fill = max(0.0, y_base_eff - Y[i_w])
    m["baseline_y"] = y_base_eff
    m["rows_short"] = float(y_base_eff - Y[-1])
    m["fill_px"] = fill

    # Height is APEX TO MAT, not the number of rows the mask happens to have.
    # Both ends are then measured independently of where the threshold gave up:
    # the apex from the top of the silhouette, the mat from its own colour.
    # Counting mask rows instead inherits every wobble at the bottom -- a mask
    # allowed to leak up to CLIP_TOLERANCE_PX past the mat reports a height
    # that much too large, which on one real run put a 37 um sawtooth through
    # the height curve and moved the vertical strain from -15.3% to -13.6%.
    # Measured on that run: frame-to-frame jitter 0.45 px this way against
    # 2.14 px counting rows, and the answer matches a separate run of the same
    # bead under a different crop to 0.2 points.
    # n_rows still records what the mask actually found.
    if y_base is not None:
        h_px = float(y_base_eff - Y[0] + 1)
        m["h_px"] = h_px
    m["fill_px"] = fill

    # How wide the silhouette still was where it MET THE MAT.  Two details, both
    # learned from a real run where this read 0.0006 for a silhouette that was
    # full width at the contact line:
    #   - rows below the mat are excluded.  A mask allowed to leak a few pixels
    #     past the mat (anything inside CLIP_TOLERANCE_PX is deliberately not
    #     clipped) ends in a 1-3 px spike of mat, and the last row is then that
    #     spike rather than the bead.
    #   - several rows, not one.  A single row is one threshold decision.
    # Read off the final row alone it collapsed to ~0 on 30 of 81 frames, took
    # V_disk's confidence to zero with it, and quarantined the entire second
    # half of a run whose volumes were in fact smooth to better than 1%.
    above = Y <= y_base_eff + 0.5
    w_t = w[above] if above.any() else w
    k = int(min(5, len(w_t)))
    m["base_taper"] = float(np.median(w_t[-k:]) / w.max()) if w.max() else np.nan

    # constant-radius fill from the widest row down to the mat
    m["V_base"] = m["V_trunc"] + float(np.pi * (w[i_w] / 2.0) ** 2 * fill)

    # each side continued to the mat along the slope it had where it was crisp
    holdout = np.nan
    m["V_extrap"] = m["V_base"]
    m["base_width_extrap_px"] = float(w[i_w])
    fitres = _edge_slope_fit(Y, xl, xr, i_w, fill)
    if fitres is not None and fill > 0:
        ml, cl, mr, cr, holdout = fitres
        ys = np.arange(Y[i_w] + 1, y_base_eff + 1, dtype=float)
        we = np.maximum((mr * ys + cr) - (ml * ys + cl) + 1.0, 0.0)
        m["V_extrap"] = m["V_trunc"] + float(np.pi * np.sum((we / 2.0) ** 2))
        m["base_width_extrap_px"] = float(we[-1]) if we.size else float(w[i_w])
    elif fill <= 0:
        holdout = 0.0
    m["edge_holdout_px"] = holdout

    # --- confidences --------------------------------------------------------
    # Each estimator is graded on how much of its volume it INVENTED rather
    # than measured, and on how defensible that invention is:
    #
    #     conf = (measured share) + (invented share) x (is the invention sound?)
    #
    # so an estimator that invents nothing scores 1 whatever else is true, and
    # one that invents half its volume can score at most 0.5 + 0.5 x soundness.
    # Every input is read off mask-versus-mat geometry or held-out prediction
    # error -- never off the estimator's own output.
    gap = abs(m["rows_short"])            # how far the mask stopped from the mat
    taper_ok = clip01(1 - (1 - m["base_taper"]) / TAPER_FULL)

    def blend(invented, soundness):
        inv = clip01(invented)
        return clip01((1.0 - inv) + inv * clip01(soundness))

    # V_disk invents nothing: it integrates exactly the rows the threshold
    # found.  Its risk is the opposite one -- rows it never found.  So it is
    # graded on how far short of the mat the mask stopped, and on whether the
    # silhouette was still widening when it ended (a bead that narrows before
    # the mat is a bead whose base went into shadow).
    c_disk = clip01(1 - gap / (GAP_FULL * h_px)) * taper_ok

    # V_trunc throws away everything below the widest row, so it is a hard
    # lower bound rather than an attempt at the truth.  Judge it on the volume
    # it discarded, not on the row count, and let it collapse quickly: it is
    # only worth believing when it discarded essentially nothing.
    thrown = (m["V_disk"] - m["V_trunc"]) / m["V_disk"] if m["V_disk"] > 0 else 1.0
    c_trunc = clip01(1 - thrown / TRUNC_FULL)

    # V_base fills from the widest row to the mat at constant radius -- fair
    # over a short run, a cylinder stuck on the bottom over a long one.
    inv_base = (m["V_base"] - m["V_trunc"]) / m["V_base"] if m["V_base"] > 0 else 1.0
    c_base = blend(inv_base, 1 - fill / (FILL_BASE * h_px))

    # V_extrap continues each side along the slope it had where the edge was
    # still crisp.  Its invention is sound in proportion to how straight those
    # edges proved on rows the fit never saw, over a window the size of the
    # reach, and to how far past the data it has to go.
    inv_ex = (m["V_extrap"] - m["V_trunc"]) / m["V_extrap"] if m["V_extrap"] > 0 else 1.0
    if fitres is None and fill > 0:
        sound = 0.0                       # no fit at all: this is just V_base
    elif np.isfinite(holdout):
        sound = clip01(1 - holdout / HOLDOUT_FULL) * clip01(1 - fill / (FILL_EXTRAP * h_px))
    else:
        sound = 0.5 * clip01(1 - fill / (FILL_EXTRAP * h_px))   # unscored: half credit
    c_extrap = blend(inv_ex, sound)

    # A silhouette cut off by the crop is a property of the crop, not the bead.
    penalty = 0.3 if frame.get("clipped") else 1.0

    # Everything that compares the mask to the mat inherits the mat row's own
    # confidence.  V_trunc alone does not -- it never uses the baseline.
    conf = {
        "V_disk": c_disk * baseline_conf * penalty,
        "V_trunc": c_trunc * penalty,
        "V_base": c_base * baseline_conf * penalty,
        "V_extrap": c_extrap * baseline_conf * penalty,
    }
    for k, v in conf.items():
        m[f"conf_{k}"] = float(v)
    return m


def choose_deciders(conf_table, trust=TRUST_METHODS):
    """Decide ONCE for the whole run which estimators are allowed to vote.

    Deciding this per frame looks reasonable and is not: two frames of the same
    bead under the same light differ only in noise, so a hard confidence gate
    applied frame by frame puts one at 0.36 and its neighbour at 0.34 and gives
    them opposite verdicts.  Worse, it inverts the meaning -- a frame whose
    V_disk is slightly BETTER clears the gate, meets V_extrap, disagrees with
    it and is thrown out, while its neighbour with a worse V_disk never faces
    the comparison and survives.  Judging each estimator on its median
    confidence over the set removes the knife edge: either the threshold found
    this bead's base in this run, or it did not.

    Returns (deciders, median_confidences).
    """
    med = {k: float(np.nanmedian(conf_table[k])) for k in METHODS}
    ok = [k for k in METHODS if med[k] >= MIN_CONF]
    trusted = [k for k in ok if k in trust]
    if trusted:
        return trusted, med
    if ok:
        return ok, med
    return [max(med, key=med.get)], med


def verdict_for(m, deciders, weights, reject_below=MIN_CONF):
    """Consensus volume and verdict tier for one frame, from the run's deciders.

    `weights` are the run's MEDIAN confidences, deliberately not this frame's.
    Weighting each frame by its own confidence looks more responsive and
    silently destroys the measurement: the weights then drift over the run (as
    the bead shrinks, a mask that stops a fixed 16 px short of the mat loses a
    growing FRACTION of the bead, so V_disk's confidence decays), the blend
    slides from one biased estimator toward another, and that slide is added
    to the strain as shrinkage that never happened.  Measured on a synthetic
    run with a known answer: per-frame weights gave -10.05% against a true
    -11.199%, worse than ANY of the four estimators alone, while fixed weights
    gave -11.3%.  A bias only divides out of a ratio if it is the same at both
    ends, so the weights must be too.

    A decider votes on every frame where its value is finite.  Re-applying the
    confidence gate frame by frame would put back the knife edge that
    choose_deciders exists to remove; the per-frame confidence still sets each
    estimator's WEIGHT in the consensus, which is where it belongs.  A frame on
    which every decider's confidence has collapsed is a broken frame, and that
    is the one case the gate still decides.
    """
    usable = [k for k in deciders if np.isfinite(m.get(k, np.nan))]
    if usable and max(m.get(f"conf_{k}", 0.0) for k in usable) < reject_below:
        usable = []

    if not usable:
        return dict(V_consensus=np.nan, spread=np.nan, spread_pct=np.nan,
                    tier="REJECT", deciders="", n_usable=0)

    v = np.array([m[k] for k in usable], float)
    cw = np.array([weights[k] for k in usable], float)
    if not cw.sum():
        cw = np.ones_like(cw)
    cons = float(np.sum(v * cw) / cw.sum())
    spread = float(v.max() - v.min())
    rel = spread / cons if cons else np.inf

    if len(usable) >= 2 and rel <= CERT_TOL:
        tier = "CERTIFIED"
    elif len(usable) >= 2 and rel <= LIKELY_TOL:
        tier = "LIKELY"
    elif len(usable) >= 2:
        tier = "CONFLICT"
    else:
        tier = "SINGLE"
    return dict(V_consensus=cons, spread=spread, spread_pct=100 * rel, tier=tier,
                deciders=" + ".join(usable), n_usable=len(usable))


def verdict_text(row):
    t = row["tier"]
    if t == "CERTIFIED":
        return (f"CERTIFIED {row['V_consensus']:.4g} px^3  "
                f"({row['deciders']} agree within {row['spread_pct']:.1f}%)")
    if t == "LIKELY":
        return (f"LIKELY {row['V_consensus']:.4g} px^3  "
                f"({row['deciders']} within {row['spread_pct']:.1f}% - good, recheck)")
    if t == "CONFLICT":
        return (f"CONFLICT - estimators disagree by {row['spread_pct']:.1f}% -> the base is "
                f"lost to shadow in this frame, do not use it")
    if t == "SINGLE":
        return f"SINGLE estimator only: {row['deciders']} (uncorroborated, use with caution)"
    return "REJECT - no estimator confident, recapture this frame"


# ------------------------------------------------------ STAGE 4: the whole set
def mark_brightness_outliers(df, z=2.5):
    """Flag frames whose mean brightness is a sudden outlier and interpolate their
    volumes from the neighbours, so one exposure glitch cannot spike the curve."""
    b = df["brightness"].to_numpy(float)
    med = np.median(b)
    mad = np.median(np.abs(b - med)) + 1e-9
    bad = np.abs(b - med) > z * 1.4826 * mad
    df = df.copy()
    df["outlier"] = bad
    if bad.any() and (~bad).sum() >= 2:
        gi = np.flatnonzero(~bad)
        bi = np.flatnonzero(bad)
        for col in list(METHODS) + ["V_consensus", "h_px", "a_px"]:
            if col in df.columns:
                df.loc[df.index[bi], col] = np.interp(bi, gi, df[col].to_numpy(float)[gi])
    return df


def analyse_folder(paths, cfg=None, workers=None, progress=None):
    """Measure a whole set.  Returns (DataFrame, info dict).

    One row per image: the four volumes in px^3, um^3 and mm^3, their
    confidences, the consensus, the verdict tier, the quarantine flag, and the
    strain referred to the first usable image.
    """
    import pandas as pd

    cfg = dict(DEFAULTS, **(cfg or {}))
    paths = list(paths)
    if not paths:
        raise ValueError("no images to analyse")

    # ---- stage 1: segment every image, in parallel -------------------------
    jobs = [(p, cfg, False) for p in paths]
    if workers is None:
        workers = min(len(jobs), os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            frames = list(ex.map(measure_one, jobs, chunksize=1))
    else:
        frames = [measure_one(j) for j in jobs]
    if progress:
        progress(len(frames), len(frames))

    failures = [(f["path"], f["error"]) for f in frames if not f["ok"]]
    good = [f for f in frames if f["ok"]]
    if not good:
        # Pass the first real reason up: "every image failed" on its own sends
        # you looking at the threshold when the crop is what is wrong.
        why = failures[0][1] if failures else "no reason recorded"
        raise RuntimeError(f"every image failed. First one ({os.path.basename(failures[0][0])}) "
                           f"says: {why}" if failures else "every image failed")

    # ---- stage 2: one mat row for the set ----------------------------------
    y_base, src, base_conf, notes = resolve_baseline(good, cfg)

    # ---- stage 3: volumes, confidences, verdicts ---------------------------
    scale = cfg["SCALE_PX_PER_UM"]
    rows = []
    for i, f in enumerate(good):
        m = volumes_for(f, y_base, base_conf, cfg)
        m["index"] = i
        m["image"] = f["image"]
        m["path"] = f["path"]
        m["label"] = os.path.splitext(f["image"])[0]
        m["brightness"] = f["brightness"]
        m["clipped"] = f["clipped"]
        m["touches_roi_bottom"] = f["touches_roi_bottom"]
        m["t_s"] = _time_for(f["path"], i, cfg)
        rows.append(m)

    # Who decides, for this run as a whole, then the per-frame verdicts.
    conf_table = {k: np.array([r[f"conf_{k}"] for r in rows], float) for k in METHODS}
    deciders, med_conf = choose_deciders(conf_table)
    best = np.array([max((r.get(f"conf_{k}", 0.0) for k in deciders), default=0.0)
                     for r in rows], float)
    reject_below = max(FRAME_REJECT_ABS, FRAME_REJECT_FRAC * float(np.nanmedian(best)))
    for m in rows:
        m.update(verdict_for(m, deciders, med_conf, reject_below))

    df = pd.DataFrame(rows)

    # ---- stage 4: outliers, quarantine, strain -----------------------------
    df = mark_brightness_outliers(df, cfg["OUTLIER_Z"])

    # ---- quarantine --------------------------------------------------------
    # What to throw out depends on what the number is FOR.  The strain is a
    # ratio, V/V0, so a bias that is the same in every frame divides out of it
    # exactly: if V_disk and V_extrap disagree by 12% on all 40 frames, that is
    # a statement about the absolute volume (the base is lost to shadow), not a
    # reason to discard the run -- discarding it would leave a biased subset
    # and a strain measured over a shorter span, which is worse than the
    # disagreement was.  What does corrupt a strain is a frame that disagrees
    # much more than the run normally does, or one whose mask is simply broken.
    # So: quarantine the OUTLIERS, and report the systematic part as a caveat
    # on the absolute volume.
    # Compared against its NEIGHBOURS, not against the run's own median.  The
    # disagreement usually drifts upward over a run -- the base is lost to
    # shadow a little more in every frame -- and a global median then reads
    # that trend as a heap of outliers and quarantines the whole back half.
    # On a real 81-frame run it cut the curve off at frame 50 and reported
    # -10.4% where the full run gives about -12.2%.  A local median flags what
    # we actually want flagged: a frame that disagrees far more than the frames
    # either side of it.
    s = df["spread_pct"].to_numpy(float)
    fin = np.isfinite(s)
    if fin.sum() >= 5:
        local = pd.Series(s).rolling(9, center=True, min_periods=3).median().to_numpy()
        resid = s - local
        r = resid[np.isfinite(resid)]
        mad_r = float(np.nanmedian(np.abs(r - np.nanmedian(r)))) * 1.4826
        df["spread_outlier"] = np.where(np.isfinite(resid),
                                        resid > max(3 * mad_r, 1.0), False)
    else:
        df["spread_outlier"] = np.zeros(len(df), bool)
    df["broken"] = (df["tier"] == "REJECT") | df["clipped"].astype(bool).to_numpy()
    df["use"] = ~(df["broken"].to_numpy() | df["spread_outlier"].to_numpy())

    if scale:
        for k in list(METHODS) + ["V_consensus"]:
            df[f"{k}_um3"] = df[k] / scale ** 3
            df[f"{k}_mm3"] = df[f"{k}_um3"] / 1e9
        df["h_um"] = df["h_px"] / scale
        df["a_um"] = df["a_px"] / scale

    usable = df.index[df["use"].to_numpy()]
    if len(usable):
        ref = usable[min(cfg["REFERENCE_INDEX"], len(usable) - 1)]
        V0 = float(df.loc[ref, "V_consensus"])
    else:
        ref, V0 = df.index[0], float(df.loc[df.index[0], "V_consensus"])

    df["V_over_V0"] = df["V_consensus"] / V0
    df["vol_shrinkage_pct"] = 100.0 * (1.0 - df["V_over_V0"])      # positive as it dries
    df["vol_strain_pct"] = 100.0 * (df["V_over_V0"] - 1.0)         # same thing, signed
    df["linear_strain_pct"] = 100.0 * (df["V_over_V0"] ** (1.0 / 3.0) - 1.0)

    # The two strains the bead actually has, measured rather than inferred.
    # linear_strain_pct above is the cube root of the volume ratio, which is a
    # real linear strain ONLY if the bead shrinks equally in every direction.
    # A sessile bead pinned to its mat does not: it collapses in height while
    # its footprint stays put.  Measured on one real run, -15.2% vertical
    # against -5.8% radial -- a factor of 2.6 -- where the cube root reported
    # -9.2% for both.  Reporting these two beside it is the difference between
    # a number that describes the bead and a number that assumes it away.
    h0 = float(df.loc[ref, "h_px"])
    a0 = float(df.loc[ref, "a_px"])
    df["height_strain_pct"] = 100.0 * (df["h_px"] / h0 - 1.0) if h0 else np.nan
    df["radial_strain_pct"] = 100.0 * (df["a_px"] / a0 - 1.0) if a0 else np.nan

    info = dict(cfg=cfg, baseline_y=y_base, baseline_source=src,
                baseline_conf=base_conf, baseline_notes=notes,
                failures=failures, n_input=len(paths), n_ok=len(good),
                reference_row=int(ref), V0_px3=V0,
                V0_mm3=(V0 / scale ** 3 / 1e9 if scale else np.nan),
                trust=TRUST_METHODS, deciders=deciders, med_conf=med_conf,
                reject_below=reject_below, workers=workers)
    return df, info


def _time_for(path, index, cfg):
    rx = cfg.get("TIME_REGEX")
    if rx:
        mm = re.search(rx, os.path.basename(path))
        if mm:
            return float(mm.group(1))
    iv = cfg.get("INTERVAL_S")
    if iv:
        return float(index) * float(iv)
    return float(index)


def warnings_for(df, info):
    """Set-level sanity checks, as a list of strings."""
    out = []
    cfg = info["cfg"]
    n = len(df)

    # A trusted estimator that was dropped for the whole run is the most
    # important thing on this list: it means nothing corroborated the answer.
    dropped = [k for k in info["trust"] if k not in info["deciders"]]
    if dropped:
        med = info["med_conf"]
        out.append("DROPPED from the decision for this whole run: "
                   + ", ".join(f"{k} (median confidence {med[k]:.2f} < {MIN_CONF})"
                               for k in dropped)
                   + ". " + ("The answer therefore rests on "
                             f"{' + '.join(info['deciders'])} alone, with nothing to "
                             "cross-check it against."
                             if len(info["deciders"]) < 2 else ""))
    if "V_disk" in dropped:
        out.append("V_disk was the dropped one, which specifically means the threshold never "
                   "reached the mat in this run: the silhouettes stop in the bead's own shadow. "
                   "Every volume here is an extrapolation of where the edges were heading, not "
                   "a measurement of where they ended. A backlight fixes this at the source.")

    # The systematic part of the disagreement: a caveat on the absolute volume,
    # not on the strain, which divides it out.
    _sp = df["spread_pct"].to_numpy(float)
    med_s = float(np.nanmedian(_sp[np.isfinite(_sp)])) if np.isfinite(_sp).any() else np.nan
    if np.isfinite(med_s) and med_s > 100 * LIKELY_TOL:
        out.append(f"{' and '.join(info['deciders'])} disagree by a median {med_s:.1f}% on EVERY "
                   f"frame. A disagreement that is the same throughout is a bias, not noise: it "
                   f"divides out of the strain (a ratio) but it does not divide out of the "
                   f"absolute volumes, so trust the strain figure here and treat the mm^3 "
                   f"numbers as uncertain to about that much.")

    bad = df[~df["use"]]
    if len(bad):
        why = []
        if df["broken"].any():
            why.append(f"{int(df['broken'].sum())} with a broken mask")
        if df["spread_outlier"].any():
            why.append(f"{int(df['spread_outlier'].sum())} disagreeing far more than the run's "
                       f"own median of {med_s:.1f}%")
        if not why:
            why.append("no estimator confident enough to decide")
        out.append(f"{len(bad)} of {n} image(s) quarantined ({'; '.join(why)}) and left out of "
                   f"the strain curve: " + ", ".join(bad["image"].head(8)) +
                   (" ..." if len(bad) > 8 else ""))
    # Quarantining shortens the span the strain is measured over.  Saying
    # "-5.7%" when the last frames were dropped, without saying so, is the
    # kind of quiet wrongness this whole verdict machinery exists to prevent.
    if len(bad):
        u = df[df["use"]]
        if len(u):
            if not bool(df["use"].iloc[-1]):
                out.append(f"the LAST frame of the run ({df['image'].iloc[-1]}) was quarantined, "
                           f"so the strain below is measured to {u['image'].iloc[-1]} "
                           f"(frame {int(u['index'].iloc[-1])} of {int(df['index'].iloc[-1])}) "
                           f"and spans LESS of the dry-down than the run does.")
            if not bool(df["use"].iloc[0]):
                out.append(f"the FIRST frame ({df['image'].iloc[0]}) was quarantined, so V0 is "
                           f"taken from {u['image'].iloc[0]} and the bead had already started "
                           f"shrinking before the reference.")
    if df["outlier"].any():
        k = int(df["outlier"].sum())
        out.append(f"{k} frame(s) had outlier brightness (exposure glitch / shadow); their "
                   f"volumes were interpolated from the neighbours.")

    clipped = df[df["clipped"].astype(bool)]
    if len(clipped):
        out.append(f"WARNING: {len(clipped)} of {n} silhouette(s) run into the edge of the crop "
                   f"({', '.join(sorted(set(clipped['clipped'])))}). The crop is cutting the bead "
                   f"off, so h / a / V describe the CROP, not the bead. Widen --roi and re-run.")

    if info["baseline_source"] == "auto":
        out.append("the mat row was inferred from the silhouettes themselves, so it can only be "
                   "as deep as the deepest one. Every confidence that compares a mask to the mat "
                   f"is therefore capped at {AUTO_BASELINE_CONF:.2f}. Crop so the mat's coloured "
                   "edge is inside the box, or pass --baseline <row>.")

    short = df["rows_short"].to_numpy(float)
    if n > 2 and np.ptp(short) > 10:
        out.append(f"V_disk's lower limit of integration moves {np.ptp(short):.0f} px across the "
                   f"run (masks end between {np.nanmin(short):+.0f} and {np.nanmax(short):+.0f} px "
                   f"of the mat). V_disk stops wherever the mask stops; V_extrap always reaches "
                   f"the mat.")
    if np.nanmedian(short) > 5:
        out.append(f"the silhouettes stop a median of {np.nanmedian(short):.0f} px short of the "
                   f"mat (worst {np.nanmax(short):.0f} px). Those missing rows are the widest part "
                   f"of the bead, so V_disk is an underestimate by more than the row count "
                   f"suggests. A backlight removes this entirely; failing that, lower "
                   f"--thresh-offset until the mask reaches the contact line.")

    ywide = df["Y_widest_full"].to_numpy(float)
    if n > 2 and np.ptp(ywide) > 20:
        out.append(f"the widest row moves {np.ptp(ywide):.0f} px across the run. With a fixed "
                   f"camera the mat cannot move, so either the bead really is changing shape at "
                   f"its base, or the base is being lost to shadow by a different amount in each "
                   f"frame.")

    h = df["h_px"].to_numpy(float)
    if n > 2 and np.ptp(h) == 0:
        out.append("WARNING: every image returned exactly the same height. A real bead never does "
                   "that - the apex is almost certainly the top edge of the crop.")

    tiers = df["tier"].value_counts().to_dict()
    if tiers.get("CERTIFIED", 0) == 0 and n >= 3:
        out.append("no frame reached CERTIFIED: V_disk and V_extrap never agreed to within "
                   f"{100*CERT_TOL:.0f}%, which means the mask never reached the mat on its own. "
                   "The volumes still stand, but they rest on the extrapolation rather than on "
                   "what the threshold actually saw.")
    return out
