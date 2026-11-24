#!/usr/bin/env python3
# ============================================================================
#  HUMIDITY CYCLES  |  analysis
#
#  A bead dried, then swollen at high humidity, dried again, ...  Two kinds of
#  set:
#
#    PHOTOS     a handful of before / after photos, one per step
#               (fresh -> after 72 % -> after 97 % -> after 72 % ...).
#               The bead can sit somewhere else in every photo, so every photo
#               has its own box and is measured to its own mat edge.
#
#    TIME-LAPSE one folder holding several recordings separated by long gaps
#               (time-lapse at 72 %, overnight at 97 %, time-lapse at 72 % ...).
#               The recordings ("blocks") are found from the gaps in the photos'
#               capture times.  Each block has its own box and its own mat
#               reading; every block votes with the first block's method so the
#               volumes stay comparable from block to block.
#
#  Every strain is against the FIRST photo / first frame (the fresh bead), and
#  the step-to-step change is given as well, so a swell shows as a rise and a
#  dry as a fall.  Times come from the camera's own capture time in the TIFF
#  (tag 306), not from the file name -- file names can be renamed out of order.
#
#  Writes <folder>/humidity_analysis/ :
#    results.csv  summary.txt  settings.json
#    strain_steps.png    volumetric / height / radius strain at every step
#    step_changes.png    what each step did (rise = swelling, fall = drying)
#    strain_vs_time.png  the same against real time (time-lapse: broken axis)
#    photo_sheet.png     each photo's crop, aligned on the bead: fresh outline
#                        cyan dashed, this one red dotted, no fill
#    outlines.png        every outline on one picture, aligned on the mat
#    blocks/<n>/height_track.png, radius_track.png, per_image.csv  (time-lapse)
#
#  Nothing in bead_pipeline / run_local / gui is changed by this file.
# ============================================================================
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bead_pipeline as bp   # noqa: E402
import run_local as rl       # noqa: E402  (palette, save_fig, track sheets)

OUT_NAME = "humidity_analysis"
GAP_MIN = 10.0               # a pause longer than this (min) starts a new block
SCALE = bp.DEFAULTS["SCALE_PX_PER_UM"]
TO_MM3 = 1.0 / SCALE ** 3 / 1e9
INK, INK2, SURFACE = rl.INK, rl.INK2, rl.SURFACE
COND = {"97": "#2a78d6", "72": "#eb6834", "fresh": "#52514e", "other": "#1baf7a"}
CYAN, RED, YEL, BLUE = "#22d3ee", "#ef4444", "#facc15", "#3b82f6"


# --------------------------------------------------------------- the photos
def capture_time(path):
    """The camera's capture time (TIFF DateTime, tag 306), else the file's
    modified time."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            tags = getattr(im, "tag_v2", None) or {}
            s = tags.get(306) or tags.get(36867)
            if s:
                return dt.datetime.strptime(str(s).strip()[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return dt.datetime.fromtimestamp(os.path.getmtime(path))


def name_rh(name):
    """The humidity written in a file name ('Image006(swell97%).tif' -> 97)."""
    m = re.search(r"(\d{2,3})\s*%", os.path.splitext(os.path.basename(name))[0])
    return int(m.group(1)) if m else None


def subfolders(folder):
    """Sub-folders holding photos (one level down), not the results folders."""
    out = []
    for d in sorted(os.listdir(folder), key=lambda s: bp.natural_key(s)):
        p = os.path.join(folder, d)
        if os.path.isdir(p) and not d.lower().startswith(("analysis", OUT_NAME)) \
                and bp.list_frames(p):
            out.append(p)
    return out


def list_photos(folder, order="time", subdirs=True):
    """Every image in the folder -- and, with subdirs, in its sub-folders one
    level down (a time-lapse split into 'timelapse 1', 'timelapse 2' ...) -- as
    dicts (path, name, time, rh, sub), in capture-time order ('time') or
    natural file-name order ('name').  sub is the sub-folder's name, or ''."""
    dirs = [(folder, "")] + ([(p, os.path.basename(p)) for p in subfolders(folder)]
                             if subdirs else [])
    out = [dict(path=p, name=os.path.basename(p), time=capture_time(p), rh=name_rh(p), sub=sub)
           for d, sub in dirs for p in bp.list_frames(d)]
    if order == "time":
        out.sort(key=lambda d: d["time"])
    return out


def order_clash(photos):
    """File names whose order differs from the capture-time order, as text."""
    by_name = [p["name"] for p in sorted(photos, key=lambda d: bp.natural_key(d["path"]))]
    by_time = [p["name"] for p in sorted(photos, key=lambda d: d["time"])]
    if by_name == by_time:
        return ""
    bad = [f"{a}  (taken {_clock(p)})" for a, b, p in
           zip(by_name, by_time, sorted(photos, key=lambda d: bp.natural_key(d["path"])))
           if a != b]
    return ("File-name order differs from the order the photos were taken:\n  "
            + "\n  ".join(bad))


def find_blocks(photos, gap_min=GAP_MIN):
    """Split time-ordered photos into recordings at every pause > gap_min, and
    wherever the sub-folder changes (each sub-folder is its own recording)."""
    blocks, cur = [], []
    for p in photos:
        if cur and ((p["time"] - cur[-1]["time"]).total_seconds() / 60.0 > gap_min
                    or p.get("sub", "") != cur[-1].get("sub", "")):
            blocks.append(cur)
            cur = []
        cur.append(p)
    if cur:
        blocks.append(cur)
    return blocks


def is_fresh(name):
    """A photo is the fresh bead only when its name says so: 'fresh' in it, or
    exactly Image000.  (A set can start after its first exposure.)"""
    stem = os.path.splitext(os.path.basename(name))[0].strip().lower()
    return "fresh" in stem or stem == "image000"


def _label(p, fallback):
    if is_fresh(p["name"]):
        return "fresh"
    rng = re.search(r"(\d{2,3})\s*-\s*(\d{2,3})\s*%", os.path.splitext(p["name"])[0])
    if rng:                                   # e.g. dry64-69%
        return f"after {rng.group(1)}-{rng.group(2)}% RH"
    return f"after {p['rh']}% RH" if p["rh"] else fallback


def default_photo_labels(photos):
    return [_label(p, f"photo {i + 1}") for i, p in enumerate(photos)]


def default_block_labels(blocks):
    out = []
    for i, b in enumerate(blocks):
        p = next((q for q in b[:3] if q["rh"] or is_fresh(q["name"])), b[0])
        sub = b[0].get("sub", "")
        if sub and any(x is not b and x[0].get("sub") == sub for x in blocks):
            sub = f"{sub} (part {sum(1 for x in blocks[:i + 1] if x[0].get('sub') == sub)})"
        out.append(_label(p, sub or f"block {i + 1}"))
    return out


def ref_text(t):
    """What step 1 is, for titles: 'the fresh bead' or 'step 1 (after 68% RH)'."""
    lab = str(t["label"].iloc[0])
    return "the fresh bead" if lab.lower().startswith("fresh") else f"step 1 ({lab})"


SWELL_RH = 90      # a step at or above this humidity is a swelling step (blue)


def colour_for(label):
    """Blue for a step at high humidity (>= 90 %), orange for a drying one
    (72 %, 68 %, 64-69 % ...), grey for fresh, green when no humidity is named."""
    s = str(label).lower()
    if "fresh" in s:
        return COND["fresh"]
    m = re.search(r"(\d{2,3})\s*%", s)
    if m:
        return COND["97"] if int(m.group(1)) >= SWELL_RH else COND["72"]
    return COND["other"]


def _clock(p):
    t = p["time"] if isinstance(p, dict) else p
    return t.strftime("%d-%m %H:%M")


def base_override(line):
    """Settings for a photo whose base was drawn by hand: the line is the floor,
    and nothing above it is thrown away for being blue (an out-of-focus mat
    smears blue into the bottom of the bead)."""
    return dict(BASE_LINE=tuple(float(v) for v in line), MAT_CHROMA_MIN=255)


def base_cfg(toff=0):
    return dict(SCALE_PX_PER_UM=SCALE, INVERT=False, MANUAL_THRESH=None,
                THRESH_OFFSET=int(toff), SEGMENT_METHOD="otsu",
                MAT_CHROMA_MIN=bp.DEFAULTS["MAT_CHROMA_MIN"], BASELINE="blue",
                BASELINE_TUNE=0.0, MAT_MODE="auto", CLIP_AT_BASELINE=True,
                INTERVAL_S=None, TIME_REGEX=None)


# ------------------------------------------------------------- measuring
def _volume(r):
    v = r.get("V_consensus", np.nan)
    if not np.isfinite(v):
        vs = [r.get(k, np.nan) for k in ("V_disk", "V_extrap")]
        v = float(np.nanmean(vs)) if np.isfinite(vs).any() else np.nan
    return float(v)


def _with_strains(t):
    """Strains vs the first row and changes vs the previous row (all %)."""
    def rel(col, ref):
        return 100.0 * (t[col] / ref - 1.0)
    for col, name in (("V_mm3", "vol"), ("h_um", "height"), ("a_um", "radius"),
                      ("V_disk_mm3", "V_disk"), ("V_extrap_mm3", "V_extrap")):
        t[f"{name}_strain_pct"] = rel(col, float(t[col].iloc[0]))
        t[f"{name}_change_pct"] = 100.0 * (t[col] / t[col].shift(1) - 1.0)
    return t


def analyse_photos(folder, items, toff=0, out_name=OUT_NAME, progress=None, workers=None):
    """items: list of dict(path, label, roi) in the order of the steps."""
    paths = [it["path"] for it in items]
    per = {os.path.basename(it["path"]): tuple(int(v) for v in it["roi"])
           for it in items if it.get("roi")}
    cfg = dict(base_cfg(toff), PAIR_MODE=True, ROI_PER_IMAGE=per,
               ROI=tuple(items[0]["roi"]) if items[0].get("roi") else None,
               PER_IMAGE={os.path.basename(it["path"]): base_override(it["base"])
                          for it in items if it.get("base")})
    df, info = bp.analyse_folder(paths, cfg, workers=workers, progress=progress)
    by = {r["image"]: r for _, r in df.iterrows()}
    t0 = capture_time(paths[0])
    rows, failed = [], []
    for i, it in enumerate(items):
        nm = os.path.basename(it["path"])
        r = by.get(nm)
        if r is None:
            failed.append(nm)
            continue
        tm = capture_time(it["path"])
        rows.append(dict(step=i + 1, label=it["label"], image=nm, path=it["path"],
                         taken=tm.strftime("%Y-%m-%d %H:%M:%S"),
                         hours=(tm - t0).total_seconds() / 3600.0,
                         V_mm3=_volume(r) * TO_MM3,
                         V_disk_mm3=float(r["V_disk"]) * TO_MM3,
                         V_extrap_mm3=float(r["V_extrap"]) * TO_MM3,
                         h_um=float(r["h_um"]), a_um=float(r["a_um"]),
                         contact_angle_deg=float(r.get("contact_angle_deg", np.nan)),
                         mat_y=float(r["baseline_y"]), tier=r.get("tier", ""),
                         use=bool(r.get("use", True)),
                         crop=",".join(str(v) for v in per.get(nm, ())) or "whole frame",
                         base="drawn" if it.get("base") else "mat edge"))
    if len(rows) < 2:
        raise RuntimeError("fewer than two photos could be measured: "
                           + ", ".join(failed or ["?"]))
    t = _with_strains(pd.DataFrame(rows))
    outdir = os.path.join(folder, out_name)
    os.makedirs(outdir, exist_ok=True)
    cfgs = {r["path"]: dict(info["cfg"]) for _, r in t.iterrows()}
    res = dict(mode="photos", folder=folder, outdir=outdir, table=t, failed=failed,
               deciders=list(info["deciders"]), toff=toff, notes=list(info.get("notes", [])))
    _write_all(res, t, cfgs, frames=None)
    with open(os.path.join(outdir, "settings.json"), "w") as f:
        json.dump(dict(mode="photos", thresh_offset=toff,
                       items=[dict(image=os.path.basename(it["path"]), label=it["label"],
                                   roi=list(it["roi"]) if it.get("roi") else None,
                                   base=list(it["base"]) if it.get("base") else None)
                              for it in items]), f, indent=2)
    return res


def analyse_blocks(folder, blocks, toff=0, out_name=OUT_NAME, progress=None, workers=None):
    """blocks: list of dict(paths, label, roi) -- the recordings, in time order."""
    total = sum(len(b["paths"]) for b in blocks)
    outdir = os.path.join(folder, out_name)
    os.makedirs(outdir, exist_ok=True)
    t0 = capture_time(blocks[0]["paths"][0])
    frames, first_info, done, cfgs = [], None, 0, {}
    for bi, b in enumerate(blocks):
        cfg = dict(base_cfg(toff), INTERVAL_S=60.0,
                   ROI=tuple(int(v) for v in b["roi"]) if b.get("roi") else None,
                   **(base_override(b["base"]) if b.get("base") else {}))
        if first_info is not None:
            # same estimators and weights as the first block: a volume from block
            # 3 must be the same kind of number as the fresh bead's
            cfg.update(FORCE_DECIDERS=list(first_info["deciders"]),
                       FORCE_WEIGHTS=dict(first_info["med_conf"]))
        off = done
        df, info = bp.analyse_folder(
            b["paths"], cfg, workers=workers,
            progress=(lambda i, n, off=off: progress(off + i, total)) if progress else None)
        done += len(b["paths"])
        if first_info is None:
            first_info = info
        times = {os.path.basename(p): capture_time(p) for p in b["paths"]}
        tb0 = min(times.values())
        df["t_s"] = [(times[nm] - tb0).total_seconds() for nm in df["image"]]
        # per block: the frames, and the usual height / radius tracking sheets
        bdir = os.path.join(outdir, "blocks", f"{bi + 1}")
        os.makedirs(bdir, exist_ok=True)
        df.to_csv(os.path.join(bdir, "per_image.csv"), index=False)
        try:
            rl.plot_track_sheets(df, dict(info, cfg=dict(info["cfg"], INTERVAL_S=1.0)), bdir)
        except Exception as e:          # a sheet is a picture, not a result
            with open(os.path.join(bdir, "sheet_error.txt"), "w") as f:
                f.write(f"{type(e).__name__}: {e}\n")
        for _, r in df.iterrows():
            tm = times[r["image"]]
            cfgs[r["path"]] = dict(info["cfg"])
            frames.append(dict(block=bi + 1, label=b["label"], image=r["image"],
                               path=r["path"], taken=tm.strftime("%Y-%m-%d %H:%M:%S"),
                               hours=(tm - t0).total_seconds() / 3600.0,
                               min_in_block=(tm - tb0).total_seconds() / 60.0,
                               V_mm3=_volume(r) * TO_MM3,
                               V_disk_mm3=float(r["V_disk"]) * TO_MM3,
                               V_extrap_mm3=float(r["V_extrap"]) * TO_MM3,
                               h_um=float(r["h_um"]), a_um=float(r["a_um"]),
                               mat_y=float(r["baseline_y"]), tier=r.get("tier", ""),
                               use=bool(r.get("use", True))))
    fr = pd.DataFrame(frames)
    fr = _with_strains(fr)
    fr.to_csv(os.path.join(outdir, "all_frames.csv"), index=False)

    # one row per block end: start and end of every recording (median of 3
    # frames, so one jittery frame does not set a step's value)
    rows = []
    for bi, b in enumerate(blocks):
        g = fr[(fr["block"] == bi + 1) & fr["use"]]
        g = g if len(g) else fr[fr["block"] == bi + 1]
        for where, sl in (("start", g.head(3)), ("end", g.tail(3))):
            r = sl.iloc[0 if where == "start" else -1]
            rows.append(dict(step=len(rows) + 1, label=f"{b['label']} - {where}",
                             block=bi + 1, image=r["image"], path=r["path"],
                             taken=r["taken"], hours=float(r["hours"]),
                             **{c: float(sl[c].median()) for c in
                                ("V_mm3", "V_disk_mm3", "V_extrap_mm3", "h_um", "a_um",
                                 "mat_y")},
                             tier=r["tier"], use=True, n_frames=int(len(g))))
    t = _with_strains(pd.DataFrame(rows))
    res = dict(mode="timelapse", folder=folder, outdir=outdir, table=t, frames=fr,
               blocks=[dict(label=b["label"], n=len(b["paths"]),
                            first=os.path.basename(b["paths"][0]),
                            last=os.path.basename(b["paths"][-1]),
                            roi=b.get("roi"), base=b.get("base")) for b in blocks],
               deciders=list(first_info["deciders"]), toff=toff, failed=[],
               notes=list(first_info.get("notes", [])))
    _write_all(res, t, cfgs, frames=fr)
    with open(os.path.join(outdir, "settings.json"), "w") as f:
        json.dump(dict(mode="timelapse", thresh_offset=toff, gap_min=GAP_MIN,
                       blocks=[dict(label=b["label"], first=os.path.basename(b["paths"][0]),
                                    last=os.path.basename(b["paths"][-1]), n=len(b["paths"]),
                                    roi=list(b["roi"]) if b.get("roi") else None,
                                    base=list(b["base"]) if b.get("base") else None)
                               for b in blocks]), f, indent=2)
    return res


# ----------------------------------------------------------------- writing
def _write_all(res, t, cfgs, frames=None):
    out = res["outdir"]
    t.drop(columns=["path"]).to_csv(os.path.join(out, "results.csv"), index=False)
    txt = summary_text(res)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(txt)
    failed = []
    jobs = [("strain_steps.png", lambda o: plot_steps(t, o, res)),
            ("step_changes.png", lambda o: plot_changes(t, o, res)),
            ("strain_vs_time.png", lambda o: (plot_timeline(frames, t, o, res)
                                              if frames is not None else
                                              plot_vs_hours(t, o, res))),
            ("change_steps.png", lambda o: plot_steps(t, o, res, kind="change")),
            ("change_vs_time.png", lambda o: (plot_timeline(vs_previous(frames), t, o, res,
                                                            kind="change")
                                              if frames is not None else
                                              plot_vs_hours(t, o, res, kind="change"))),
            ("photo_sheet.png + outlines.png", lambda o: plot_sheet(t, cfgs, out))]
    for name, draw in jobs:
        try:
            draw(os.path.join(out, name))
        except Exception as e:
            failed.append(f"{name}: {type(e).__name__}: {e}")
            plt.close("all")
    if failed:
        with open(os.path.join(out, "summary.txt"), "a") as f:
            f.write("\n  PLOTS THAT WOULD NOT SAVE (numbers above unaffected):\n"
                    + "\n".join("    - " + s for s in failed) + "\n")
    res["plot_errors"] = failed


def summary_text(res):
    t = res["table"]
    L = ["=" * 78,
         f" HUMIDITY CYCLES  {os.path.basename(res['folder'])}   "
         f"({'before / after photos' if res['mode'] == 'photos' else 'time-lapse blocks'})",
         "=" * 78, "",
         f"  strains are against step 1 ({t['label'].iloc[0]}, {t['image'].iloc[0]});"
         " 'change' is against the step before.",
         f"  volume decided by {' + '.join(res['deciders'])}"
         + ("  (every block votes the same way as block 1)" if res["mode"] == "timelapse"
            else "") + f";  threshold offset {res['toff']}", ""]
    if res["mode"] == "timelapse":
        L.append("  blocks (recordings found from the gaps in capture time):")
        for i, b in enumerate(res["blocks"]):
            L.append(f"    {i + 1}. {b['label']:<24s} {b['n']:4d} frames  {b['first']} -> "
                     f"{b['last']}   crop {rl._crop_txt(b['roi'])}"
                     + ("   base drawn by hand" if b.get("base") else ""))
        L += ["  start / end of each block = median of its first / last 3 frames", ""]
    hdr = (f"  {'#':>2s}  {'step':<28s} {'taken':<16s} {'h':>6s}  {'V mm3':>7s}  "
           f"{'vol %':>7s} {'chg %':>7s}  {'height %':>8s} {'chg %':>7s}  "
           f"{'radius %':>8s} {'chg %':>7s}")
    L += [hdr, "  " + "-" * (len(hdr) - 2)]

    def f(v, w=7):
        return f"{v:+{w}.2f}" if np.isfinite(v) else " " * (w - 1) + "-"
    for _, r in t.iterrows():
        L.append(f"  {int(r['step']):>2d}  {str(r['label'])[:28]:<28s} {r['taken'][5:16]:<16s} "
                 f"{r['hours']:6.1f}  {r['V_mm3']:7.3f}  {f(r['vol_strain_pct'])} "
                 f"{f(r['vol_change_pct'])}  {f(r['height_strain_pct'], 8)} "
                 f"{f(r['height_change_pct'])}  {f(r['radius_strain_pct'], 8)} "
                 f"{f(r['radius_change_pct'])}")
    L += ["", "  by method (volumetric strain vs step 1):"]
    for _, r in t.iterrows():
        L.append(f"    {int(r['step']):>2d}  {str(r['label'])[:28]:<28s} V_disk "
                 f"{f(r['V_disk_strain_pct'])}   V_extrap {f(r['V_extrap_strain_pct'])}"
                 + ("" if r.get("use", True) else "   (marked broken - check the sheet)"))
    swell = t[t["vol_change_pct"] > 0]
    if len(swell):
        L += ["", "  volume went UP at: " + ", ".join(
            f"step {int(r['step'])} ({r['label']}, {r['vol_change_pct']:+.2f} %)"
            for _, r in swell.iterrows())]
    if res.get("failed"):
        L += ["", "  could not be measured: " + ", ".join(res["failed"])]
    if "base" in t.columns and (t["base"] == "drawn").any():
        L += ["", "  base drawn by hand (measured down to the drawn line, blue smear above it "
              "counted as bead) on step(s): "
              + ", ".join(str(int(s)) for s in t.loc[t["base"] == "drawn", "step"])]
    mats = t["mat_y"].to_numpy(float)
    if np.ptp(mats) > 10:
        L += ["", f"  mat edge moves {np.ptp(mats):.0f} px between steps (Y "
              + " / ".join(f"{v:.0f}" for v in mats)
              + ") - each step is measured to its own mat edge."]
    L += ["", "  plots vs step 1:            strain_steps . strain_vs_time",
          "  plots vs the photo before:   change_steps . change_vs_time . step_changes (bars)",
          "  pictures:                    photo_sheet . outlines",
          f"  saved to {res['outdir']}", ""]
    return "\n".join(L)


# ------------------------------------------------------------------- plots
QTY = [("vol", "volumetric strain (%)"), ("height", "height strain (%)"),
       ("radius", "radius strain (%)")]


def _ticks(t):
    return [f"{int(r['step'])}. {r['label']}\n{r['taken'][5:16]}" for _, r in t.iterrows()]


def _label_points(ax, x, y, fmt="{:+.1f}"):
    """Value next to each point, above where the curve goes down after it and
    below where it goes up, so the line never runs through a number."""
    x = np.asarray(x, float)
    span = float(np.nanmax(x) - np.nanmin(x)) or 1.0
    for i, (xi, yi) in enumerate(zip(x, y)):
        if not np.isfinite(yi):
            continue
        nxt = y[i + 1] if i + 1 < len(y) else (y[i - 1] if i else yi)
        prv = y[i - 1] if i else nxt
        up = yi >= (nxt + prv) / 2.0
        # a neighbour this close in x: push the two numbers apart sideways
        near_next = i + 1 < len(x) and x[i + 1] - xi < 0.05 * span
        near_prev = i > 0 and xi - x[i - 1] < 0.05 * span
        dx, ha = (-6, "right") if near_next else (6, "left") if near_prev else (0, "center")
        ax.annotate(fmt.format(yi), (xi, yi), xytext=(dx, 9 if up else -9),
                    textcoords="offset points", ha=ha,
                    va="bottom" if up else "top", fontsize=8.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))


def _legend_conditions(fig, labels):
    from matplotlib.lines import Line2D
    seen, hs = [], []
    for lab in labels:
        c = colour_for(lab)
        if c in [s[0] for s in seen]:
            continue
        name = {COND["97"]: f"after / at >= {SWELL_RH} % RH (swelling)",
                COND["72"]: f"after / at < {SWELL_RH} % RH (drying)",
                COND["fresh"]: "fresh bead"}.get(c, "no humidity in the label")
        seen.append((c, name))
        hs.append(Line2D([], [], marker="o", ls="none", mfc=c, mec=SURFACE, ms=9, label=name))
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, fontsize=9)


TITLES = {
    "strain": ("Strain against the fresh bead at every step",
               "Strain against real time (hours since the fresh photo)"),
    "change": ("Change from the photo before, at every step  (up = swelled, down = shrank)",
               "Change from the photo before, against real time  (up = swelled, down = shrank)"),
}


def _y(t, q, kind):
    if kind == "change":
        y = t[f"{q}_change_pct"].to_numpy(float).copy()
        y[0] = 0.0                       # step 1 has nothing before it
        return y
    return t[f"{q}_strain_pct"].to_numpy(float)


def _ylab(yl, kind):
    return yl.replace("strain", "change from photo before") if kind == "change" else yl


def plot_steps(t, out, res, kind="strain"):
    n = len(t)
    fig, axes = plt.subplots(3, 1, figsize=(max(8.5, 1.6 * n + 3), 10), sharex=True)
    x = np.arange(n)
    cols = [colour_for(l) for l in t["label"]]
    for ax, (q, yl) in zip(axes, QTY):
        y = _y(t, q, kind)
        ax.axhline(0, lw=1, color="#c9c8c3", zorder=1)
        ax.plot(x, y, "-", color="#9a9994", lw=1.6, zorder=2)
        ax.scatter(x, y, c=cols, s=80, edgecolors=SURFACE, linewidths=1.8, zorder=3)
        _label_points(ax, x, y)
        rl.tidy(ax, None, None, _ylab(yl, kind))
        lo, hi = min(0.0, np.nanmin(y)), max(0.0, np.nanmax(y))
        pad = 0.18 * ((hi - lo) or 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    head = TITLES[kind][0].replace("the fresh bead", ref_text(t))
    axes[0].set_title(f"{head}  -  {os.path.basename(res['folder'])}", loc="left", pad=10)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(_ticks(t), fontsize=8)
    _legend_conditions(fig, t["label"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    rl.save_fig(fig, out)
    plt.close(fig)


def plot_changes(t, out, res):
    """Bars: what each step did to the bead, against the step before."""
    s = t.iloc[1:]
    n = len(s)
    fig, axes = plt.subplots(3, 1, figsize=(max(8.5, 1.6 * n + 3), 10), sharex=True)
    x = np.arange(n)
    cols = [colour_for(l) for l in s["label"]]
    for ax, (q, yl) in zip(axes, QTY):
        y = s[f"{q}_change_pct"].to_numpy(float)
        ax.axhline(0, lw=1, color="#8d8c87", zorder=1)
        ax.bar(x, y, color=cols, width=0.62, zorder=2)
        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                ax.annotate(f"{yi:+.2f}", (xi, yi), xytext=(0, 4 if yi >= 0 else -4),
                            textcoords="offset points", ha="center",
                            va="bottom" if yi >= 0 else "top", fontsize=8.5, color=INK)
        lo, hi = min(0.0, np.nanmin(y)), max(0.0, np.nanmax(y))
        pad = 0.2 * ((hi - lo) or 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        rl.tidy(ax, None, None, yl.replace("strain", "change"))
    axes[0].set_title("What each step did  (change from the step before: up = swelling, "
                      "down = shrinking)", loc="left", pad=10)
    prev = t["label"].to_numpy()[:-1]
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{a}\n-> {b}" for a, b in zip(prev, s["label"])], fontsize=8)
    _legend_conditions(fig, s["label"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    rl.save_fig(fig, out)
    plt.close(fig)


def plot_vs_hours(t, out, res, kind="strain"):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    x = t["hours"].to_numpy(float)
    cols = [colour_for(l) for l in t["label"]]
    for ax, (q, yl) in zip(axes, QTY):
        y = _y(t, q, kind)
        ax.axhline(0, lw=1, color="#c9c8c3", zorder=1)
        ax.plot(x, y, "-", color="#9a9994", lw=1.6, zorder=2)
        ax.scatter(x, y, c=cols, s=80, edgecolors=SURFACE, linewidths=1.8, zorder=3)
        _label_points(ax, x, y)
        lo, hi = min(0.0, np.nanmin(y)), max(0.0, np.nanmax(y))
        pad = 0.18 * ((hi - lo) or 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        rl.tidy(ax, None, None, _ylab(yl, kind))
    axes[0].set_title(TITLES[kind][1].replace("the fresh photo", "step 1"), loc="left", pad=10)
    axes[-1].set_xlabel("hours since step 1")
    _legend_conditions(fig, t["label"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    rl.save_fig(fig, out)
    plt.close(fig)


def vs_previous(fr):
    """Frames re-referenced to the photo before each exposure: block 1 to its
    own first frame, every later block to the END of the block before it
    (median of its last 3 frames).  Written into the *_strain_pct columns so the timeline
    plot draws it unchanged."""
    if fr is None:
        return None
    g = fr.copy()
    for bi in sorted(g["block"].unique()):
        cur = g["block"] == bi
        src = g[(g["block"] == (bi - 1 if bi > 1 else bi)) & g["use"]]
        if not len(src):
            src = g[g["block"] == (bi - 1 if bi > 1 else bi)]
        # block 1: its first frame (a fresh bead changes fast, so no averaging);
        # later blocks: the previous block's settled end, median of 3 frames
        ref = src.tail(3) if bi > 1 else src.head(1)
        for col, name in (("V_mm3", "vol"), ("h_um", "height"), ("a_um", "radius")):
            g.loc[cur, f"{name}_strain_pct"] = 100.0 * (g.loc[cur, col]
                                                        / float(ref[col].median()) - 1.0)
    return g


def plot_timeline(fr, t, out, res, kind="strain"):
    """Time-lapse: one column per block, width by its length, the gaps between
    them cut out (each gap's length written over the next block)."""
    nb = int(fr["block"].max())
    dur = [max(5.0, float(fr.loc[fr["block"] == b, "min_in_block"].max())) for b in
           range(1, nb + 1)]
    fig, axes = plt.subplots(3, nb, figsize=(max(10, 4.2 * nb), 10), sharey="row",
                             gridspec_kw=dict(width_ratios=dur, wspace=0.06),
                             squeeze=False)
    for bi in range(nb):
        g = fr[fr["block"] == bi + 1]
        u = g[g["use"]] if g["use"].any() else g
        lab = str(g["label"].iloc[0])
        c = colour_for(lab)
        when = g["taken"].iloc[0][5:16]
        if bi:
            prev_end = fr[fr["block"] == bi]["hours"].max()
            gap = float(g["hours"].min() - prev_end)
            head = f"{bi + 1}. {lab}\n{when}   (after a {gap:.1f} h gap)"
        else:
            head = f"{bi + 1}. {lab}\n{when}"
        for r, (q, yl) in enumerate(QTY):
            ax = axes[r, bi]
            ax.axhline(0, lw=1, color="#c9c8c3", zorder=1)
            y = u[f"{q}_strain_pct"].to_numpy(float)
            xm = u["min_in_block"].to_numpy(float)
            ax.plot(xm, y, "-", color=c, lw=2, zorder=3)
            bad = g[~g["use"]]
            if len(bad):
                ax.plot(bad["min_in_block"], bad[f"{q}_strain_pct"], "o", mfc="none",
                        mec=rl.STATUS["REJECT"], ms=7, zorder=4)
            for xx, yy, va, dy in ((xm[0], y[0], "bottom", 8), (xm[-1], y[-1], "top", -8)):
                ax.annotate(f"{yy:+.1f}", (xx, yy), xytext=(0, dy), textcoords="offset points",
                            ha="left" if xx == xm[0] else "right", va=va, fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none",
                                      alpha=0.85))
            rl.tidy(ax, None, None, _ylab(yl, kind) if bi == 0 else None)
            ax.margins(y=0.14)
            if bi:
                ax.spines["left"].set_linestyle((0, (3, 3)))
            if r == 0:
                ax.set_title(head, fontsize=9, loc="left")
            if r == 2:
                ax.set_xlabel("min in this block")
    what = (f"Strain against {ref_text(t)}, block by block" if kind == "strain" else
            "Change from the photo before each exposure (block 1: its own start; later blocks: "
            "the end of the block before)")
    fig.suptitle(f"{what}  -  {os.path.basename(res['folder'])}   (gaps between blocks cut out)",
                 x=0.01, ha="left", fontsize=11 if kind == "change" else 12)
    _legend_conditions(fig, t["label"])
    with warnings.catch_warnings():          # shared y across the blocks: harmless
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    rl.save_fig(fig, out)
    plt.close(fig)


def _outline(path, cfg, mat_y):
    """Full-frame grey image window data: (grey crop, mask, crop origin)."""
    m = bp.measure_one((path, cfg, True))
    if not m.get("ok"):
        return None
    x0, x1, y0, y1 = m["roi"]
    mask = m["mask"] > 0
    cut = int(round(mat_y - y0)) + 1
    if 0 <= cut < mask.shape[0]:
        mask[cut:] = False
    return dict(img=m["img"], mask=mask, x0=x0, y0=y0)


def plot_sheet(t, cfgs, outdir):
    """photo_sheet.png: one panel per step, every panel the same size and lined
    up on the bead (its middle at the mat), so growth and shrinkage between
    panels is real.  outlines.png: all outlines on one picture."""
    got = []
    for _, r in t.iterrows():
        o = _outline(r["path"], cfgs[r["path"]], float(r["mat_y"]))
        if o is None or not o["mask"].any():
            got.append(None)
            continue
        ys, xs = np.nonzero(o["mask"])
        o.update(cx=float(xs.min() + xs.max()) / 2.0, mat=float(r["mat_y"]) - o["y0"],
                 w=float(xs.max() - xs.min()), h=float(r["mat_y"] - o["y0"] - ys.min()))
        got.append(o)
    ok = [o for o in got if o]
    if not ok:
        return
    W = int(1.18 * max(o["w"] for o in ok)) // 2 * 2
    H = int(1.22 * max(o["h"] for o in ok))
    below = int(0.06 * H)

    def window(o, arr, fill):
        """W x (H+below) window of arr, bead middle at x = W/2, mat at row H."""
        out = np.full((H + below, W), fill, arr.dtype)
        xa = int(round(o["cx"])) - W // 2
        ya = int(round(o["mat"])) - H
        sy0, sx0 = max(0, ya), max(0, xa)
        sy1, sx1 = min(arr.shape[0], ya + H + below), min(arr.shape[1], xa + W)
        if sy1 > sy0 and sx1 > sx0:
            out[sy0 - ya:sy1 - ya, sx0 - xa:sx1 - xa] = arr[sy0:sy1, sx0:sx1]
        return out

    wins = [None if o is None else (window(o, o["img"], 0), window(o, o["mask"], False))
            for o in got]
    ref = next(w for w in wins if w is not None)[1]
    n = len(wins)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    pw = 4.6
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * pw, nrow * (pw * (H + below) / W + 0.8) + 0.4),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    um = 1.0 / SCALE
    for k, ((_, r), w) in enumerate(zip(t.iterrows(), wins)):
        ax = axes.flat[k]
        if w is None:
            ax.set_title(f"{int(r['step'])}. {r['label']}: not measured", fontsize=9)
            continue
        img, m = w
        # lines only, no fill: the bead's own edge must stay visible under them
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        if k and ref.any():
            ax.contour(ref, levels=[0.5], colors=[CYAN], linewidths=1.0, linestyles="--")
        if m.any():
            ax.contour(m, levels=[0.5], colors=[RED], linewidths=1.1, linestyles=":")
        ax.axhline(H, color="white", ls=":", lw=0.9)
        ax.set_title(f"{int(r['step'])}. {r['label']}   {r['taken'][5:16]}\n"
                     f"V {r['vol_strain_pct']:+.1f}%   h {r['h_um']:,.0f} um "
                     f"({r['height_strain_pct']:+.1f}%)   r {r['a_um']:,.0f} um "
                     f"({r['radius_strain_pct']:+.1f}%)", fontsize=8.5,
                     color=colour_for(r["label"]))
    # scale bar on the first panel: 500 um
    ax = axes.flat[0]
    bar = 500 / um
    ax.plot([10, 10 + bar], [H + below * 0.6] * 2, color="white", lw=3)
    ax.text(10 + bar / 2, H + below * 0.45, "500 um", color="white", ha="center",
            va="bottom", fontsize=7)
    fig.suptitle("Every photo lined up on the bead (middle) and the mat (white dotted).  "
                 "red dotted = the outline measured on this photo,  cyan dashed = step 1's "
                 "outline", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    rl.save_fig(fig, os.path.join(outdir, "photo_sheet.png"))
    plt.close(fig)

    # ---- all outlines together, in um from the bead's middle / the mat
    fig, ax = plt.subplots(figsize=(10, 5.6))
    cmap = plt.get_cmap("viridis")
    for k, ((_, r), w) in enumerate(zip(t.iterrows(), wins)):
        if w is None or not w[1].any():
            continue
        c = cmap(k / max(1, n - 1))
        xs = (np.arange(W) - W / 2) * um
        ys = (H - np.arange(H + below)) * um
        cs = ax.contour(xs, ys, w[1], levels=[0.5], colors=[c],
                        linewidths=2.0 if k in (0, n - 1) else 1.3,
                        linestyles="-" if colour_for(r["label"]) != COND["97"] else "--")
        ax.plot([], [], color=c, lw=2, ls="-" if colour_for(r["label"]) != COND["97"] else "--",
                label=f"{int(r['step'])}. {r['label']}  ({r['vol_strain_pct']:+.1f}%)")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_aspect("equal")
    ax.set_ylim(-below * um, None)
    rl.tidy(ax, "Every outline, lined up on the bead's middle and the mat "
                "(dashed = after 97 % RH)", "um from the bead's middle", "um above the mat")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    rl.save_fig(fig, os.path.join(outdir, "outlines.png"))
    plt.close(fig)


def results_lines(res):
    """Short text for the window."""
    return summary_text(res)
