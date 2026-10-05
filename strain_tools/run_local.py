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


def save_fig(fig, out):
    """Render to a temp file on the local disk, then copy the finished file
    into place.

    matplotlib writes a PNG to its destination in many small writes.  Onto a
    removable drive that is many chances to fail, and one did: a run died with
    OSError 22 on the fifth of seven figures, in a folder where the previous
    four had just saved.  Rendering locally and copying one complete file is a
    single sequential write, and it also makes a half-written figure on the
    drive impossible.  Falls back to writing straight to `out` if no temp file
    can be made at all.
    """
    import shutil
    import tempfile
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    except Exception:
        fig.savefig(out, bbox_inches="tight")
        return
    try:
        fig.savefig(tmp, bbox_inches="tight")
        shutil.copyfile(tmp, out)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def time_axis(df, info):
    """x values and their label: seconds, minutes or hours, whichever reads best.

    Returns (values, label, is_time).  A 77-frame run at one frame a minute is
    4560 s, which nobody reads as "an hour and a quarter".  Only a run timed by
    --interval is converted: with --time-regex the number came out of the file
    name and its unit is whatever you named it, so it is left exactly as read.
    """
    t = df["t_s"].to_numpy(float)
    cfg = info["cfg"]
    if cfg.get("TIME_REGEX"):
        return t, "time (from file name)", True
    if not cfg.get("INTERVAL_S"):
        return t, "image number", False
    span = float(np.nanmax(t) - np.nanmin(t)) if len(t) else 0.0
    if span >= 7200:
        return t / 3600.0, "time (h)", True
    if span >= 180:
        return t / 60.0, "time (min)", True
    return t, "time (s)", True


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


def add_image_number_axis(ax, df, t=None):
    """A second x axis across the top showing the image number.

    The same data on both axes -- time below, frame index above -- so a point
    can be read either way.  (Not a second y scale: those are never allowed.)
    """
    t = df["t_s"].to_numpy(float) if t is None else np.asarray(t, float)
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
def plot_vs_time(df, info, out, col, title, ylabel, headline="final  {v:+.2f}%",
                 colour=None, zero_line=True):
    """One measured quantity against time, every point wearing its verdict.

    Shared by the strain and the geometry plots so they read as one family and
    a quarantined frame is marked the same way everywhere.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    t, xl, is_time = time_axis(df, info)
    y = df[col].to_numpy(float)
    use = df["use"].to_numpy(bool)

    if zero_line:
        ax.axhline(0, lw=1, color="#c9c8c3", zorder=1)
    ax.plot(t[use], y[use], "-", lw=2, color=colour or SERIES["V_disk"], zorder=3,
            label=ylabel)

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

    tidy(ax, title, xl, ylabel)
    if is_time:
        add_image_number_axis(ax, df, t)
    if headline:
        fin = y[use][-1] if use.any() and np.isfinite(y[use]).any() else np.nan
        ax.text(0.99, 0.97, headline.format(v=fin), transform=ax.transAxes,
                ha="right", va="top", fontsize=13, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2,
              bbox_to_anchor=(0.0, -0.02))
    fig.tight_layout()
    save_fig(fig, out)
    plt.close(fig)


def plot_shape_strain(df, info, out):
    """Height, footprint and the cube-root number on one axis, so it is obvious
    whether the bead is shrinking the same way in every direction.

    All three are percentages of the same reference frame, which is why they
    can share an axis -- and sharing it is the point: the gap between the top
    and bottom curves IS the anisotropy.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    t, xl, is_time = time_axis(df, info)
    use = df["use"].to_numpy(bool)
    series = [("height_strain_pct", "vertical", SERIES["V_disk"]),
              ("radial_strain_pct", "radial (base)", SERIES["V_extrap"]),
              ("linear_strain_pct", "isotropic equivalent", SERIES["V_base"])]
    ax.axhline(0, lw=1, color="#c9c8c3", zorder=1)
    ends = []
    for col, lab, c in series:
        if col not in df.columns:
            continue
        y = df[col].to_numpy(float)
        ax.plot(t[use], y[use], "-", lw=2.2, color=c, zorder=3, label=lab)
        ends.append((t[use][-1], y[use][-1], lab, c))
    tidy(ax, "Is the bead shrinking the same way in every direction?", xl, "strain (%)")
    if is_time:
        add_image_number_axis(ax, df, t)
    # room on the right for the direct labels, none on the left: padding there
    # just puts negative time on the axis
    span = (t.max() - t.min()) or 1.0
    ax.set_xlim(t.min() - 0.02 * span, t.max() + 0.26 * span)
    place_end_labels(ax, ends)

    if {"height_strain_pct", "radial_strain_pct"} <= set(df.columns) and use.any():
        ez = df["height_strain_pct"].to_numpy(float)[use][-1]
        er = df["radial_strain_pct"].to_numpy(float)[use][-1]
        if np.isfinite(ez) and np.isfinite(er) and abs(er) > 1e-9:
            note = (f"vertical {ez:+.1f}%  vs  radial {er:+.1f}%   =  {ez/er:.1f}x"
                    if ez / er > 1.25 else
                    f"vertical {ez:+.1f}%  vs  radial {er:+.1f}%   (near isotropic)")
            ax.text(0.01, 0.04, note, transform=ax.transAxes, fontsize=11,
                    fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    save_fig(fig, out)
    plt.close(fig)


def plot_volumes(df, info, out):
    """The four estimators over the run, direct-labelled."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    t, xl, is_time = time_axis(df, info)
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

    tidy(ax, "Four volume estimators  (they differ only below the widest row)",
         xl, "volume (mm$^3$)" if unit else "volume (px$^3$)")
    if is_time:
        add_image_number_axis(ax, df, t)
    span = (t.max() - t.min()) or 1.0
    ax.set_xlim(t.min() - 0.02 * span, t.max() + 0.16 * span)
    # direct labels last, once the limits are final: the contrast WARN on two
    # of these four slots is relieved by a visible label, not by the legend
    place_end_labels(ax, ends)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    save_fig(fig, out)
    plt.close(fig)


def plot_agreement(df, info, out):
    """How far apart the trusted estimators were, frame by frame -- the evidence
    behind each verdict."""
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    t, xl, is_time = time_axis(df, info)
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
    tidy(ax, f"Disagreement between {' and '.join(info['deciders'])}", xl,
         "spread (% of volume)")
    ax.margins(y=0.18)
    # below the axes: inside, it lands on the band labels
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    save_fig(fig, out)
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
    L.append(f"    (a frame is called broken below {info['reject_below']:.2f})")
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
        L.append(f"    VOLUMETRIC STRAIN {u['vol_strain_pct'].iloc[-1]:+.2f} %"
                 f"   (shrinkage {u['vol_shrinkage_pct'].iloc[-1]:+.2f} %)")
        L.append(f"    linear strain     {u['linear_strain_pct'].iloc[-1]:+.2f} %"
                 f"   (cube root of the volume ratio)")
        if {"height_strain_pct", "radial_strain_pct"} <= set(u.columns):
            ez = float(u["height_strain_pct"].iloc[-1])
            er = float(u["radial_strain_pct"].iloc[-1])
            L.append(f"      vertical        {ez:+.2f} %   "
                     f"height {u['h_um'].iloc[0]:,.0f} -> {u['h_um'].iloc[-1]:,.0f} um"
                     if "h_um" in u.columns else f"      vertical        {ez:+.2f} %")
            if "contact_angle_deg" in u.columns:
                L.append(f"      drop shape      h/a {u['aspect_h_over_a'].iloc[0]:.3f} "
                         f"-> {u['aspect_h_over_a'].iloc[-1]:.3f}, contact angle "
                         f"{u['contact_angle_deg'].iloc[0]:.0f} -> "
                         f"{u['contact_angle_deg'].iloc[-1]:.0f} deg")
            L.append(f"      radial          {er:+.2f} %   "
                     f"widest r {u['a_um'].iloc[0]:,.0f} -> {u['a_um'].iloc[-1]:,.0f} um"
                     if "a_um" in u.columns else f"      radial          {er:+.2f} %")
            if "radial_contact_pct" in u.columns:
                cp = u["radial_contact_pct"].dropna()
                if len(cp) > 1:
                    good = u.loc[cp.index]
                    L.append(f"      contact line    {cp.iloc[-1]:+.2f} %   "
                             f"over the {len(cp)} frame(s) with a measurable base "
                             f"({good['image'].iloc[0]} -> {good['image'].iloc[-1]})")
            if abs(er) > 1e-9 and ez / er > 1.25:
                L.append(f"      -> this bead shrinks {ez/er:.1f}x more vertically than it does")
                L.append( "         radially, so the linear strain above is a geometric mean")
                L.append( "         and not the strain in any direction the bead actually has.")
                L.append( "         A footprint held back while the height collapses means the")
                L.append( "         contact line is resisting, which puts the material in radial")
                L.append( "         tension as it dries. Read base_radius.png to see whether it")
                L.append( "         is fully pinned (flat) or receding slowly (sloping) -- those")
                L.append( "         are different mechanisms and this ratio alone cannot tell")
                L.append( "         them apart.")
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
            + ["h_px", "a_px", "a_contact_px", "h_um", "a_um", "a_contact_um",
               "base_lost", "radial_contact_pct", "aspect_h_over_a", "contact_angle_deg",
               "n_rows", "base_taper", "rows_short",
               "fill_px", "edge_holdout_px", "baseline_y", "Y_apex_full",
               "Y_widest_full", "Y_bottom_full", "clipped", "clipped_at_mat",
               "brightness", "V_over_V0", "vol_shrinkage_pct", "vol_strain_pct",
               "linear_strain_pct", "height_strain_pct", "radial_strain_pct"])
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(os.path.join(outdir, "per_image.csv"), index=False)

    # The numbers go out BEFORE the pictures.  A figure that will not save --
    # a removable drive hiccuping mid-run is enough -- used to take summary.txt
    # down with it and leave a run with no readable result at all.
    report = write_report(df, info, outdir)
    timing = (f"\n  run time  {secs:.1f} s for {info['n_ok']} images "
              f"({secs / max(1, info['n_ok']) * 1000:.0f} ms each, "
              f"{info['workers']} workers)\n")
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(report + timing)

    um = "h_um" in df.columns
    plots = [
        ("linear_strain.png", lambda o: plot_vs_time(
            df, info, o, "linear_strain_pct", "Linear shrinkage strain of the bead",
            "linear strain (%)")),
        ("volumetric_strain.png", lambda o: plot_vs_time(
            df, info, o, "vol_strain_pct", "Volumetric strain of the bead",
            "volumetric strain (%)", colour=SERIES["V_extrap"])),
        ("height.png", lambda o: plot_vs_time(
            df, info, o, "h_um" if um else "h_px", "Bead height",
            "height (um)" if um else "height (px)",
            headline="final  {v:,.0f}", colour=SERIES["V_disk"], zero_line=False)),
        ("base_radius.png", lambda o: plot_vs_time(
            df, info, o, "a_um" if um else "a_px", "Bead base radius",
            "base radius (um)" if um else "base radius (px)",
            headline="final  {v:,.0f}", colour=SERIES["V_extrap"], zero_line=False)),
        ("shape_strain.png", lambda o: plot_shape_strain(df, info, o)),
        ("volumes.png", lambda o: plot_volumes(df, info, o)),
        ("agreement.png", lambda o: plot_agreement(df, info, o)),
    ]
    failed = []
    for name, draw in plots:
        try:
            draw(os.path.join(outdir, name))
        except Exception as e:                  # one bad figure, not a lost run
            failed.append(f"{name}: {type(e).__name__}: {e}")
            plt.close("all")
    if failed:
        msg = ("\n  PLOTS THAT WOULD NOT SAVE (the numbers above are unaffected):\n"
               + "\n".join("    - " + f for f in failed)
               + "\n    An OSError here is usually the drive, not the data. Copy the\n"
                 "    folder to a local disk and re-run; results are written next to\n"
                 "    the images, and a removable drive can refuse a write mid-run.\n")
        with open(os.path.join(outdir, "summary.txt"), "a") as f:
            f.write(msg)
        if not quiet:
            print(msg)

    with open(os.path.join(outdir, "settings.json"), "w") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in info["cfg"].items()}, f, indent=2)

    if not quiet:
        print(report + timing)
    return df, info


# ------------------------------------------------------------- interactive
# Run with no folder and it asks, the way the cantilever app does: choose the
# folder, confirm the interval, drag the crop.  The answers are remembered in
# app_settings.json beside this script so the next run starts where this one
# left off.
SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_settings.json")


def load_settings():
    try:
        with open(SETTINGS) as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d):
    try:
        with open(SETTINGS, "w") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass            # a read-only folder must not take the run down with it


def _tk_root():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)      # else the dialog hides behind the console
    root.update()                          # without this the attribute is never applied
    return root                            # and the dialog opens behind, or not at all


def _ask_console(prompt):
    """Last resort when no dialog appears: take the answer on the console."""
    try:
        return input(prompt).strip().strip('"').strip("'")
    except (EOFError, KeyboardInterrupt):
        return ""


def choose_folder(start=None):
    """Folder picker, falling back to a typed path.  None if given up on."""
    print("  opening a folder picker - if you cannot see it, look in the taskbar or "
          "press Alt+Tab (it can open behind this window)", flush=True)
    try:
        from tkinter import filedialog
        root = _tk_root()
        try:
            d = filedialog.askdirectory(
                title="Choose the folder of bead images",
                initialdir=start if start and os.path.isdir(start) else os.path.expanduser("~"),
                mustexist=True, parent=root)
        finally:
            root.destroy()
        if d:
            return d
        print("  nothing chosen in the picker.")
    except Exception as e:
        print(f"  the folder picker could not open ({type(e).__name__}: {e}).")
    d = _ask_console("  type or paste the folder path, then Enter (blank to quit):\n  folder> ")
    return d or None


def ask_interval(default=60.0):
    """Seconds between frames, falling back to the console.  None = image number.

    `default` is whatever was remembered last time, and last time may have been
    "no interval", which comes back as None -- so it cannot be formatted or
    handed to a spinbox until it has been given a number to fall back on.
    """
    try:
        default = 60.0 if default is None else float(default)
    except (TypeError, ValueError):
        default = 60.0
    try:
        from tkinter import simpledialog
        root = _tk_root()
        try:
            v = simpledialog.askfloat(
                "Time between images",
                "Seconds between frames.\n\n"
                "Cancel = plot against image number instead of time.",
                initialvalue=default, minvalue=0.0, parent=root)
        finally:
            root.destroy()
        return v
    except Exception:
        pass
    raw = _ask_console(f"  seconds between frames [{default:g}] "
                       f"(or 'n' to use image number): ")
    if raw.lower().startswith("n"):
        return None
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def looks_like_parent(folder, pattern="*"):
    """True when the folder holds no images itself but its subfolders do."""
    if bp.list_frames(folder, pattern):
        return False
    try:
        subs = [d.path for d in os.scandir(folder) if d.is_dir() and d.name != "analysis"]
    except OSError:
        return False
    return any(bp.list_frames(sd, pattern) for sd in subs)


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
    ap.add_argument("folder", nargs="?", default=None,
                    help="folder of images (or a parent, with --each). "
                         "Leave it out and you will be asked.")
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
    ap.add_argument("--pick-roi", action="store_true",
                    help="drag the crop on the first image, then run with it")
    ap.add_argument("--one-roi", action="store_true",
                    help="with --each, draw ONE crop and use it for every subfolder "
                         "(default: you are asked for each, since the bead moves)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    cfgsave = load_settings()

    # ---- no folder given: ask for everything, like the cantilever app ----
    interactive = a.folder is None
    if interactive:
        a.folder = choose_folder(cfgsave.get("last_dir"))
        if not a.folder:
            raise SystemExit("no folder chosen - nothing to do")
        if not os.path.isdir(a.folder):
            raise SystemExit(f"not a folder: {a.folder}")
        print(f"folder: {a.folder}")
        if a.interval is None and not a.time_regex:
            a.interval = ask_interval(cfgsave.get("interval", 60.0))
            print(f"interval: {a.interval} s" if a.interval else
                  "interval: none - plotting against image number")
        if not a.each and looks_like_parent(a.folder, a.pattern):
            a.each = True
            print("no images here but the subfolders have some - treating each as its "
                  "own experiment")
        a.pick_roi = a.pick_roi or a.roi is None

    baseline = a.baseline
    if baseline not in ("blue", "auto"):
        baseline = float(baseline)

    cfg = dict(SCALE_PX_PER_UM=a.scale, ROI=a.roi, INVERT=a.invert,
               MANUAL_THRESH=a.manual_thresh, THRESH_OFFSET=a.thresh_offset,
               SEGMENT_METHOD=a.method, MAT_CHROMA_MIN=a.chroma_min,
               BASELINE=baseline, BASELINE_TUNE=a.baseline_tune,
               CLIP_AT_BASELINE=not a.no_clip, INTERVAL_S=a.interval,
               TIME_REGEX=a.time_regex)

    def ask_roi(folder, seed=None):
        """Draw the crop for one folder, falling back to a typed one."""
        import pick_roi
        print(f"  opening the first image of {os.path.basename(folder)} to draw the "
              f"crop on - check the taskbar if you cannot see it", flush=True)
        try:
            roi = pick_roi.pick_for_folder(folder, a.pattern, seed)
        except Exception as e:
            print(f"  the crop picker could not open ({e})")
            raw = _ask_console("  type the crop as x0,x1,y0,y1, or Enter for the "
                               "whole frame:\n  roi> ")
            roi = parse_roi(raw) if raw else None
        if roi:
            print(f"  using --roi {roi[0]},{roi[1]},{roi[2]},{roi[3]}")
        else:
            print("  no box drawn - using the whole frame")
        return roi

    if a.each:
        subs = sorted(d.path for d in os.scandir(a.folder)
                      if d.is_dir() and d.name != "analysis")
        if not subs:
            raise SystemExit(f"no subfolders in {a.folder}")

        # One crop per experiment, because the bead is not in the same place in
        # every set -- a crop carried over from the previous folder would be
        # wrong by however far the sample moved, and wrong quietly: it would
        # clip the bead rather than fail.  --one-roi is for a campaign that
        # really was framed identically throughout.
        per_folder = a.pick_roi and not a.one_roi and cfg["ROI"] is None
        shared = cfg["ROI"]
        if a.pick_roi and not per_folder and shared is None:
            shared = ask_roi(subs[0], cfgsave.get("roi") and tuple(cfgsave["roi"]))
        if per_folder:
            print(f"\n{len(subs)} experiment(s) - you will be asked for a crop on each "
                  f"one in turn (pass --one-roi to draw it once for all of them)")

        save_settings(dict(cfgsave, last_dir=a.folder, interval=a.interval))
        for sub in subs:
            print(f"\n######  {os.path.basename(sub)}")
            cfg_sub = dict(cfg)
            if per_folder:
                # no seed from the previous folder: the bead has moved, and a
                # box already drawn around where it used to be invites OK
                cfg_sub["ROI"] = ask_roi(sub)
            elif shared is not None:
                cfg_sub["ROI"] = shared
            try:
                run_folder(sub, cfg_sub, a.pattern, a.workers, a.quiet)
            except SystemExit as e:
                print(f"  skipped: {e}")
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
    else:
        if a.pick_roi and cfg["ROI"] is None:
            seed = cfgsave.get("roi") and tuple(cfgsave["roi"])
            cfg["ROI"] = ask_roi(a.folder, seed)
        save_settings(dict(cfgsave, last_dir=a.folder, interval=a.interval,
                           roi=list(cfg["ROI"]) if cfg["ROI"] else None))
        run_folder(a.folder, cfg, a.pattern, a.workers, a.quiet)


if __name__ == "__main__":
    main()
