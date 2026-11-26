#!/usr/bin/env python3
# ============================================================================
#  HUMIDITY CYCLES  |  analysis
#
#  A bead dried, then swollen at high humidity, dried again, ...: a handful of
#  BEFORE / AFTER PHOTOS, one per step (fresh -> after 72 % -> after 97 % ...).
#  The bead can sit somewhere else in every photo, so every photo has its own
#  box and is measured to its own mat edge.
#
#  Photos only, on purpose.  Time-lapses are run in Bead Shrinkage: logic made
#  for frames a minute apart (filling in a glitch from the neighbours, judging a
#  frame by the frames around it) is wrong for photos taken days apart.
#
#  Every strain is against the FIRST photo, and the change from the photo before
#  is given as well, so a swell shows as a rise and a dry as a fall.  Times come
#  from the camera's own clock in the TIFF (tag 306), not from the file name.
#
#  Writes <folder>/humidity_analysis/ :
#    results.csv  summary.txt  settings.json
#    strain_steps.png    volumetric / height / radius strain at every step
#    step_changes.png    what each step did (rise = swelling, fall = drying)
#    strain_vs_time.png  the same against real time
#    change_steps.png, change_vs_time.png   the same against the photo before
#    photo_sheet.png     each photo's crop, aligned on the bead: fresh outline
#                        cyan dashed, this one red dotted, no fill
#    outlines.png        every outline on one picture, aligned on the mat
#
#  Nothing in bead_pipeline / run_local / gui is changed by this file.
# ============================================================================
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bead_pipeline as bp   # noqa: E402
import run_local as rl       # noqa: E402  (palette, save_fig, track sheets)

OUT_NAME = "humidity_analysis"
MAX_PHOTOS = 40              # more than this in one folder is a time-lapse
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


def list_photos(folder, order="time"):
    """Every photo directly in the folder as dicts (path, name, time, rh), in
    capture-time order ('time') or natural file-name order ('name').
    Sub-folders are not read."""
    out = [dict(path=p, name=os.path.basename(p), time=capture_time(p), rh=name_rh(p),
                file_time=dt.datetime.fromtimestamp(os.path.getmtime(p)))
           for p in bp.list_frames(folder)]
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
                         odd_brightness=bool(r.get("outlier", False)),
                         base_filled_px=int(r.get("base_filled_px", 0) or 0),
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
    _write_all(res, t, cfgs)
    with open(os.path.join(outdir, "settings.json"), "w") as f:
        json.dump(dict(mode="photos", thresh_offset=toff,
                       items=[dict(image=os.path.basename(it["path"]), label=it["label"],
                                   roi=list(it["roi"]) if it.get("roi") else None,
                                   base=list(it["base"]) if it.get("base") else None)
                              for it in items]), f, indent=2)
    return res


# ----------------------------------------------------------------- writing
def _write_all(res, t, cfgs):
    out = res["outdir"]
    t.drop(columns=["path"]).to_csv(os.path.join(out, "results.csv"), index=False)
    txt = summary_text(res)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(txt)
    failed = []
    jobs = [("strain_steps.png", lambda o: plot_steps(t, o, res)),
            ("step_changes.png", lambda o: plot_changes(t, o, res)),
            ("strain_vs_time.png", lambda o: plot_vs_hours(t, o, res)),
            ("change_steps.png", lambda o: plot_steps(t, o, res, kind="change")),
            ("change_vs_time.png", lambda o: plot_vs_hours(t, o, res, kind="change")),
            ("volume_steps.png", lambda o: plot_volume_steps(t, o, res)),
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
         f" HUMIDITY CYCLES  {os.path.basename(res['folder'])}   (before / after photos)",
         "=" * 78, "",
         f"  strains are against step 1 ({t['label'].iloc[0]}, {t['image'].iloc[0]});"
         " 'change' is against the step before.",
         f"  volume decided by {' + '.join(res['deciders'])};  threshold offset {res['toff']}", ""]
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
                 + {"REJECT": "   (no method trusted on this photo - check the sheet)",
                    "CONFLICT": "   (volumes differ by more than 5 % - check the sheet)"}
                 .get(r.get("tier", ""), ""))
    swell = t[t["vol_change_pct"] > 0]
    if len(swell):
        L += ["", "  volume went UP at: " + ", ".join(
            f"step {int(r['step'])} ({r['label']}, {r['vol_change_pct']:+.2f} %)"
            for _, r in swell.iterrows())]
    if res.get("failed"):
        L += ["", "  could not be measured: " + ", ".join(res["failed"])]
    if "odd_brightness" in t.columns and t["odd_brightness"].any():
        L += ["", "  brightness differs from the other photos on step(s) "
              + ", ".join(str(int(s)) for s in t.loc[t["odd_brightness"], "step"])
              + " - measured as it is (nothing filled in); check its outline on photo_sheet"]
    if "base_filled_px" in t.columns and (t["base_filled_px"] > 0).any():
        L += ["", "  base carried down to the drawn line (too dark or blurred to see), rows: "
              + ", ".join(f"step {int(r.step)} {int(r.base_filled_px)} px"
                          for r in t.itertuples() if r.base_filled_px > 0)
              + "  (about 0.3 % of the volume per row; check the line on photo_sheet)"]
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


def plot_volume_steps(t, out, res):
    """Volume only: against the fresh bead (top) and against the photo before
    (bottom).  The whisker on each value spans its V_disk and V_extrap values,
    the minimum uncertainty of that step."""
    n = len(t)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(8.5, 1.5 * n + 3), 8.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1]))
    x = np.arange(n)
    cols = [colour_for(l) for l in t["label"]]

    y = t["vol_strain_pct"].to_numpy(float)
    lo = np.minimum(t["V_disk_strain_pct"], t["V_extrap_strain_pct"]).to_numpy(float)
    hi = np.maximum(t["V_disk_strain_pct"], t["V_extrap_strain_pct"]).to_numpy(float)
    a1.axhline(0, lw=1, color="#c9c8c3", zorder=1)
    a1.plot(x, y, "-", color="#9a9994", lw=1.6, zorder=2)
    a1.vlines(x, lo, hi, color=INK2, lw=1.4, zorder=2)
    a1.scatter(x, y, c=cols, s=85, edgecolors=SURFACE, linewidths=1.8, zorder=3)
    _label_points(a1, x, y)
    rl.tidy(a1, None, None, "volume vs fresh bead (%)")
    b, top = min(0.0, np.nanmin(lo)), max(0.0, np.nanmax(hi))
    a1.set_ylim(b - 0.18 * ((top - b) or 1), top + 0.18 * ((top - b) or 1))
    a1.set_title(f"Volumetric strain against {ref_text(t)}  -  {os.path.basename(res['folder'])}",
                 loc="left", pad=10)

    c = t["vol_change_pct"].to_numpy(float).copy()
    clo = np.minimum(t["V_disk_change_pct"], t["V_extrap_change_pct"]).to_numpy(float).copy()
    chi = np.maximum(t["V_disk_change_pct"], t["V_extrap_change_pct"]).to_numpy(float).copy()
    c[0] = clo[0] = chi[0] = np.nan                       # step 1 has nothing before it
    a2.axhline(0, lw=1, color="#8d8c87", zorder=1)
    a2.bar(x[1:], c[1:], color=cols[1:], width=0.6, zorder=2)
    a2.vlines(x[1:], clo[1:], chi[1:], color=INK, lw=1.4, zorder=3)
    for xi, yi in zip(x[1:], c[1:]):
        if np.isfinite(yi):
            a2.annotate(f"{yi:+.2f}", (xi, yi), xytext=(9, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8.5, color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    b, top = min(0.0, np.nanmin(clo)), max(0.0, np.nanmax(chi))
    a2.set_ylim(b - 0.15 * ((top - b) or 1), top + 0.15 * ((top - b) or 1))
    rl.tidy(a2, None, None, "change from photo before (%)")
    a2.set_title("Volume change from the photo before  (up = swelled, down = shrank)",
                 loc="left", pad=8, fontsize=10.5)
    a2.set_xticks(x)
    a2.set_xticklabels(_ticks(t), fontsize=8)
    _legend_conditions(fig, t["label"])
    fig.text(0.99, 0.005, "whiskers: from the V_disk to the V_extrap value of each step",
             ha="right", va="bottom", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
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
