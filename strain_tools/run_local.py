#!/usr/bin/env python3
# ============================================================================
#  BEAD VOLUME -> STRAIN  |  runner
#
#  Point it at a folder of side-view bead images.  It measures every image,
#  cross-checks four volume estimators against each other, quarantines the
#  frames they disagree on, and writes the table and the plots into
#
#      <folder>/analysis/
#
#  so the results live beside the data they came from.
#
#      python run_local.py /path/to/set --roi 800,2800,600,2000 --interval 30
#      python run_local.py /path/to/parent --each          # every subfolder
#
#  Nothing in here is specific to one set of images.  Every setting is a flag,
#  and whatever was used is written to analysis/settings.json beside the
#  results so a run can always be reproduced.
# ============================================================================
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")                 # headless: never pops a window, always saves
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bead_pipeline as bp

# --- palette -----------------------------------------------------------------
# Validated categorical slots, assigned in fixed order and never cycled, plus
# the reserved status colours for the verdict tiers.  A tier is never carried
# by colour alone: every tier also has its own marker and is named in the text.
SERIES = {"V_disk": "#2a78d6", "V_trunc": "#eb6834",
          "V_base": "#1baf7a", "V_extrap": "#eda100"}
STATUS = {"CERTIFIED": "#0ca30c", "LIKELY": "#fab219",
          "SINGLE": "#ec835a", "CONFLICT": "#ec835a", "REJECT": "#d03b3b"}
MARKER = {"CERTIFIED": "o", "LIKELY": "s", "SINGLE": "^", "CONFLICT": "X", "REJECT": "v"}
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d8d7d2", "grid.color": "#e8e7e2",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "figure.dpi": 110,
})


def place_end_labels(ax, items, pad_frac=0.052):
    """Direct-label each series at its right-hand end, pushed apart so they stay
    readable where the series converge -- which, for four estimators of the same
    volume, is most of the run.  A leader line keeps a nudged label tied to the
    point it belongs to."""
    if not items:
        return
    y0, y1 = ax.get_ylim()
    span = (y1 - y0) or 1.0
    items = sorted(items, key=lambda it: it[1])          # bottom to top
    placed = []
    for x, y, text, colour in items:
        yy = y
        if placed and yy - placed[-1] < pad_frac * span:
            yy = placed[-1] + pad_frac * span
        placed.append(yy)
        ax.annotate(text, xy=(x, yy), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=colour, fontweight="bold",
                    annotation_clip=False)
        if abs(yy - y) > 0.004 * span:                   # nudged: show where it belongs
            ax.plot([x, x], [y, yy], lw=0.8, color=colour, alpha=0.55, zorder=1)


def tidy(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", pad=10, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def add_image_number_axis(ax, df):
    """A second x axis across the top showing the image number.

    The same data on both axes -- time below, frame index above -- so a point
    can be read either way.  (Not a second y scale: those are never allowed.)
    """
    t = df["t_s"].to_numpy(float)
    i = df["index"].to_numpy(float)
    if len(t) < 2 or np.ptp(t) == 0:
        return
    # t and i are related by a straight line whenever the interval is constant,
    # which is the only case where a second axis is honest.
    a, b = np.polyfit(t, i, 1)
    sec = ax.secondary_xaxis("top", functions=(lambda x: a * x + b,
                                               lambda y: (y - b) / a))
    sec.set_xlabel("image number", color=INK2)
    sec.tick_params(colors=INK2)


# ------------------------------------------------------------------ plotting
def plot_strain(df, info, out):
    """THE plot: linear strain against time, with image number across the top."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    t = df["t_s"].to_numpy(float)
    y = df["linear_strain_pct"].to_numpy(float)
    use = df["use"].to_numpy(bool)

    ax.axhline(0, lw=1, color="#c9c8c3", zorder=1)
    ax.plot(t[use], y[use], "-", lw=2, color=SERIES["V_disk"], zorder=3,
            label="linear strain (consensus volume)")

    # Each point wears its verdict: colour AND marker AND the legend naming it.
    for tier in ("CERTIFIED", "LIKELY", "SINGLE", "CONFLICT", "REJECT"):
        sel = (df["tier"] == tier).to_numpy()
        if not sel.any():
            continue
        ax.plot(t[sel], y[sel], MARKER[tier], ms=8, mfc=STATUS[tier],
                mec=SURFACE, mew=2, ls="none", zorder=4,
                label=f"{tier}  ({int(sel.sum())})")

    if (~use).any():
        ax.plot(t[~use], y[~use], "o", ms=14, mfc="none", mec=STATUS["REJECT"],
                mew=1.4, ls="none", zorder=5, label="quarantined (left out)")

    xl = "time (s)" if info["cfg"].get("INTERVAL_S") or info["cfg"].get("TIME_REGEX") \
        else "image number"
    tidy(ax, "Linear shrinkage strain of the bead", xl, "linear strain (%)")
    if xl == "time (s)":
        add_image_number_axis(ax, df)

    fin = y[use][-1] if use.any() else np.nan
    ax.text(0.99, 0.97, f"final  {fin:+.2f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=13, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2,
              bbox_to_anchor=(0.0, -0.02))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_volumes(df, info, out):
    """The four estimators over the run, direct-labelled."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    t = df["t_s"].to_numpy(float)
    unit = "mm3" if "V_disk_mm3" in df.columns else None
    ends = []
    for k, c in SERIES.items():
        col = f"{k}_mm3" if unit else k
        v = df[col].to_numpy(float)
        trusted = k in info["trust"]
        ax.plot(t, v, "-", lw=2.2 if trusted else 1.4, color=c,
                alpha=1.0 if trusted else 0.55,
                label=k + (" (trusted)" if trusted else ""))
        ends.append((t[-1], v[-1], k, c))

    xl = "time (s)" if info["cfg"].get("INTERVAL_S") or info["cfg"].get("TIME_REGEX") \
        else "image number"
    tidy(ax, "Four volume estimators  (they differ only below the widest row)",
         xl, "volume (mm$^3$)" if unit else "volume (px$^3$)")
    if xl == "time (s)":
        add_image_number_axis(ax, df)
    ax.margins(x=0.10)
    # direct labels last, once the limits are final: the contrast WARN on two
    # of these four slots is relieved by a visible label, not by the legend
    place_end_labels(ax, ends)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_agreement(df, info, out):
    """How far apart the trusted estimators were, frame by frame -- the evidence
    behind each verdict."""
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    t = df["t_s"].to_numpy(float)
    s = df["spread_pct"].to_numpy(float)
    ax.axhspan(0, 100 * bp.CERT_TOL, color=STATUS["CERTIFIED"], alpha=0.10, zorder=0)
    ax.axhspan(100 * bp.CERT_TOL, 100 * bp.LIKELY_TOL, color=STATUS["LIKELY"],
               alpha=0.12, zorder=0)
    ax.axhline(100 * bp.CERT_TOL, lw=1, ls="--", color=STATUS["CERTIFIED"])
    ax.axhline(100 * bp.LIKELY_TOL, lw=1, ls="--", color=STATUS["LIKELY"])
    ax.text(t[0], 100 * bp.CERT_TOL, " CERTIFIED below", va="top", fontsize=8, color=INK2)
    ax.text(t[0], 100 * bp.LIKELY_TOL, " LIKELY below", va="top", fontsize=8, color=INK2)
    for tier in ("CERTIFIED", "LIKELY", "SINGLE", "CONFLICT", "REJECT"):
        sel = (df["tier"] == tier).to_numpy()
        if not sel.any():
            continue
        ax.plot(t[sel], s[sel], MARKER[tier], ms=7, mfc=STATUS[tier], mec=SURFACE,
                mew=1.5, ls="none", label=f"{tier} ({int(sel.sum())})")
    xl = "time (s)" if info["cfg"].get("INTERVAL_S") or info["cfg"].get("TIME_REGEX") \
        else "image number"
    tidy(ax, f"Disagreement between {' and '.join(info['deciders'])}", xl,
         "spread (% of volume)")
    ax.margins(y=0.18)
    # below the axes: inside, it lands on the band labels
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- report
def write_report(df, info, outdir):
    cfg = info["cfg"]
    L = []
    L.append("=" * 72)
    L.append(" BEAD VOLUME -> SHRINKAGE STRAIN")
    L.append("=" * 72)
    L.append(f"  images            {info['n_ok']} of {info['n_input']} measured"
             + (f"  ({len(info['failures'])} failed)" if info["failures"] else ""))
    L.append(f"  scale             {cfg['SCALE_PX_PER_UM']} px/um"
             f"   (1 px = {1/cfg['SCALE_PX_PER_UM']:.4f} um)")
    L.append(f"  crop              {cfg['ROI'] or 'whole frame'}")
    L.append(f"  mat row           Y = {info['baseline_y']:.0f}   "
             f"(source: {info['baseline_source']}, confidence {info['baseline_conf']:.2f})")
    L.append(f"  decided by        {' + '.join(info['deciders'])}"
             + (f"   (of {' + '.join(info['trust'])})"
                if list(info['deciders']) != list(info['trust']) else ""))
    L.append("")

    tiers = df["tier"].value_counts()
    L.append("  median confidence")
    for k in bp.METHODS:
        mark = "  <- decides" if k in info["deciders"] else ""
        L.append(f"    {k:<10s} {info['med_conf'][k]:.2f}{mark}")
    L.append("")
    L.append("  verdicts")
    for t in ("CERTIFIED", "LIKELY", "SINGLE", "CONFLICT", "REJECT"):
        if t in tiers:
            L.append(f"    {t:<10s} {tiers[t]:4d}")
    L.append("")

    u = df[df["use"]]
    if not len(u):
        L.append("  NO USABLE FRAME -- every image was quarantined, so there is no result.")
        L.append("  Read the notes below: they say which check failed and what to change.")
    if len(u):
        scale = cfg["SCALE_PX_PER_UM"]
        v0, v1 = u["V_consensus"].iloc[0], u["V_consensus"].iloc[-1]
        f0, f1 = u["image"].iloc[0], u["image"].iloc[-1]
        L.append(f"  RESULT  ({f0} -> {f1}; {len(u)} of {len(df)} frames used)")
        L.append(f"    initial volume    {v0:.4e} px^3   =  {v0/scale**3/1e9:.4f} mm^3")
        L.append(f"    final volume      {v1:.4e} px^3   =  {v1/scale**3/1e9:.4f} mm^3")
        L.append(f"    volume shrinkage  {u['vol_shrinkage_pct'].iloc[-1]:+.2f} %")
        L.append(f"    LINEAR STRAIN     {u['linear_strain_pct'].iloc[-1]:+.2f} %")
        L.append(f"    median spread     {u['spread_pct'].median():.2f} % between "
                 f"{' and '.join(info['trust'])}")
    L.append("")

    for n in info["baseline_notes"]:
        L.append("  [baseline] " + n)
    warns = bp.warnings_for(df, info)
    if warns:
        L.append("")
        L.append("  NOTES")
        for w in warns:
            L.append("    - " + w)
    if info["failures"]:
        L.append("")
        L.append("  FAILED")
        for p, e in info["failures"]:
            L.append(f"    - {os.path.basename(p)}: {e}")
    L.append("")
    L.append(f"  saved to  {outdir}")
    return "\n".join(L)


# ----------------------------------------------------------------------- run
def run_folder(folder, cfg, pattern="*", workers=None, quiet=False):
    paths = bp.list_frames(folder, pattern)
    if len(paths) < 2:
        raise SystemExit(f"{folder}: found {len(paths)} image(s); need at least 2")

    t0 = time.time()
    df, info = bp.analyse_folder(paths, cfg, workers=workers)
    secs = time.time() - t0

    outdir = os.path.join(folder, "analysis")
    os.makedirs(outdir, exist_ok=True)

    cols = (["index", "image", "t_s", "tier", "use", "outlier", "V_consensus",
             "spread_pct", "deciders"]
            + [f"{k}{s}" for k in bp.METHODS for s in ("", "_um3", "_mm3")
               if f"{k}{s}" in df.columns]
            + [f"conf_{k}" for k in bp.METHODS]
            + ["h_px", "a_px", "h_um", "a_um", "n_rows", "base_taper", "rows_short",
               "fill_px", "edge_holdout_px", "baseline_y", "Y_apex_full",
               "Y_widest_full", "Y_bottom_full", "clipped", "clipped_at_mat",
               "brightness", "V_over_V0", "vol_shrinkage_pct", "linear_strain_pct"])
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(os.path.join(outdir, "per_image.csv"), index=False)

    plot_strain(df, info, os.path.join(outdir, "linear_strain.png"))
    plot_volumes(df, info, os.path.join(outdir, "volumes.png"))
    plot_agreement(df, info, os.path.join(outdir, "agreement.png"))

    report = write_report(df, info, outdir)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(report + f"\n  run time  {secs:.1f} s for {info['n_ok']} images "
                         f"({secs/max(1,info['n_ok'])*1000:.0f} ms each, "
                         f"{info['workers']} workers)\n")
    with open(os.path.join(outdir, "settings.json"), "w") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in info["cfg"].items()}, f, indent=2)

    if not quiet:
        print(report)
        print(f"  run time  {secs:.1f} s for {info['n_ok']} images "
              f"({secs/max(1,info['n_ok'])*1000:.0f} ms each, {info['workers']} workers)")
    return df, info


def parse_roi(s):
    if not s:
        return None
    parts = [int(float(v)) for v in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi needs x0,x1,y0,y1")
    return tuple(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Measure bead volume and shrinkage strain from side-view images.")
    ap.add_argument("folder", help="folder of images (or a parent, with --each)")
    ap.add_argument("--each", action="store_true",
                    help="treat every immediate subfolder as its own experiment")
    ap.add_argument("--pattern", default="*", help="file pattern, e.g. '*.tif'")
    ap.add_argument("--roi", type=parse_roi, default=None, metavar="x0,x1,y0,y1",
                    help="crop, in full-image pixels; the same crop is used for every image")
    ap.add_argument("--scale", type=float, default=bp.DEFAULTS["SCALE_PX_PER_UM"],
                    help="px per um (default %(default)s)")
    ap.add_argument("--interval", type=float, default=None,
                    help="seconds between frames; without it the x axis is image number")
    ap.add_argument("--time-regex", default=None,
                    help=r"read the time out of the file name, e.g. '_(\d+)min'")
    ap.add_argument("--baseline", default="blue",
                    help="'blue' (read the mat off its colour), 'auto', or a row number")
    ap.add_argument("--baseline-tune", type=float, default=0.0)
    ap.add_argument("--thresh-offset", type=int, default=0)
    ap.add_argument("--manual-thresh", type=int, default=None)
    ap.add_argument("--method", default="otsu", choices=["otsu", "adaptive", "edges"])
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--chroma-min", type=int, default=bp.DEFAULTS["MAT_CHROMA_MIN"])
    ap.add_argument("--no-clip", action="store_true",
                    help="do not cut silhouettes off at the mat row")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel worker processes (default: one per core)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    baseline = a.baseline
    if baseline not in ("blue", "auto"):
        baseline = float(baseline)

    cfg = dict(SCALE_PX_PER_UM=a.scale, ROI=a.roi, INVERT=a.invert,
               MANUAL_THRESH=a.manual_thresh, THRESH_OFFSET=a.thresh_offset,
               SEGMENT_METHOD=a.method, MAT_CHROMA_MIN=a.chroma_min,
               BASELINE=baseline, BASELINE_TUNE=a.baseline_tune,
               CLIP_AT_BASELINE=not a.no_clip, INTERVAL_S=a.interval,
               TIME_REGEX=a.time_regex)

    if a.each:
        subs = sorted(d.path for d in os.scandir(a.folder)
                      if d.is_dir() and d.name != "analysis")
        if not subs:
            raise SystemExit(f"no subfolders in {a.folder}")
        for s in subs:
            print(f"\n######  {os.path.basename(s)}")
            try:
                run_folder(s, cfg, a.pattern, a.workers, a.quiet)
            except SystemExit as e:
                print(f"  skipped: {e}")
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
    else:
        run_folder(a.folder, cfg, a.pattern, a.workers, a.quiet)


if __name__ == "__main__":
    main()
