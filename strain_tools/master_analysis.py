#!/usr/bin/env python3
# ============================================================================
#  MASTER ANALYSIS  |  every bead set side by side
#
#  The bead counterpart of the Stoney setup's master analysis: reads the
#  results the app already wrote for each set (analysis/summary.txt,
#  per_image.csv, settings.json, results.txt) and writes one folder with
#
#      comparative_master.csv   one row per set, recipe parsed from the name
#      findings.txt             repeatability, factor effects, anything unusual
#      01_..png - 10_..png      the plots listed at the end of findings.txt
#
#  Nothing is re-measured and no set's results are changed.
#
#      pythonw master_analysis.py                     (or Master Analysis.bat)
#      python  master_analysis.py <data folder> [<data folder> ...] [--out DIR]
#
#  Folder names are read as  [f]<conc>%_<soak>hr_[remarks_]<replicate> (<yyyymmdd>)
#  e.g. "f90%_8hr_3 (20261022)":  f = fresh set with the mat taped.
#  Sets to leave out: put "discarded" in the folder name, or list folder names,
#  one per line, in exclude.txt in the output folder (created the first time).
#  Other remarks in the name (nextday, outlier, thick...) are kept as notes; the
#  recipe that replicates are grouped by is concentration + soak time only.
# ============================================================================
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Runs written before these fixes can differ from a rerun today; flagged, not dropped.
FIXES = [(dt.datetime(2026, 10, 12, 17, 30), "first + last uses the time-lapse floor"),
         (dt.datetime(2026, 10, 15, 18, 0), "tilt-proof base check")]

INK, GRID = "#0b0b0b", "#e6e6e6"
SERIES_STYLE = {"fresh (taped mat)": dict(color="#1f77b4", marker="o"),
                "earlier": dict(color="#ff7f0e", marker="s")}


# ------------------------------------------------------------------ reading
def parse_name(name):
    """Recipe fields from a set's folder name."""
    s = name.strip()
    date = re.search(r"\((\d{8})\)", s)
    core = re.sub(r"\(\d{8}\)", "", s).strip(" _")
    fresh = bool(re.match(r"^f\d", core, re.I))
    if fresh:
        core = core[1:]
    conc = re.search(r"(\d+(?:\.\d+)?)\s*%", core)
    soak = re.search(r"(\d+(?:\.\d+)?)\s*hr", core, re.I)
    rest = core
    for m in (conc, soak):
        if m:
            rest = rest.replace(m.group(0), " ")
    toks = [t for t in re.split(r"[_\s]+", rest) if t]
    rep = None
    for t in reversed(toks):
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            rep = t
            toks.remove(t)
            break
    remarks = "_".join(toks)
    c = float(conc.group(1)) if conc else np.nan
    k = float(soak.group(1)) if soak else np.nan
    recipe = (f"{c:g}%" if np.isfinite(c) else "?") + (f"_{k:g}hr" if np.isfinite(k) else "")
    return dict(conc_pct=c, soak_hr=k, replicate=rep, remarks=remarks, fresh=fresh,
                series="fresh (taped mat)" if fresh else "earlier", recipe=recipe,
                date=dt.datetime.strptime(date.group(1), "%Y%m%d").date() if date else None)


def grab(text, pat, cast=str, default=np.nan):
    m = re.search(pat, text)
    if not m:
        return default
    try:
        return cast(m.group(1).replace(",", ""))
    except ValueError:
        return default


def read_set(folder):
    """One row for one set, or None if it has no analysis."""
    an = os.path.join(folder, "analysis")
    sp, cp, jp = (os.path.join(an, f) for f in ("summary.txt", "per_image.csv", "settings.json"))
    if not (os.path.isfile(sp) and os.path.isfile(cp)):
        return None, None
    t = open(sp, encoding="utf-8", errors="replace").read()
    df = pd.read_csv(cp)
    cfg = json.load(open(jp)) if os.path.isfile(jp) else {}
    name = os.path.basename(os.path.normpath(folder))
    row = dict(experiment=name, folder=folder.replace("\\", "/"))
    row.update(parse_name(name))
    run = dt.datetime.fromtimestamp(os.path.getmtime(sp))
    row["run_time"] = run.isoformat(timespec="seconds")
    stale = [what for when, what in FIXES if run < when]
    row["stale_code"] = "; ".join(stale)

    u = df[df["use"]] if "use" in df.columns and df["use"].any() else df
    scale = float(cfg.get("SCALE_PX_PER_UM") or 0.256)
    to_mm3 = 1.0 / scale ** 3 / 1e9
    row.update(
        n_frames=len(df), n_used=len(u), interval_s=cfg.get("INTERVAL_S"),
        total_time_min=(u["t_s"].iloc[-1] - u["t_s"].iloc[0]) / 60.0 if len(u) > 1 else np.nan,
        um_per_px=1.0 / scale, crop=str(cfg.get("ROI")), thresh_offset=cfg.get("THRESH_OFFSET", 0),
        method=cfg.get("SEGMENT_METHOD", "otsu"),
        V0_mm3=float(u["V_consensus"].iloc[0]) * to_mm3 if len(u) else np.nan,
        V1_mm3=float(u["V_consensus"].iloc[-1]) * to_mm3 if len(u) else np.nan,
        vol_strain_pct=grab(t, r"VOLUMETRIC STRAIN ([-+0-9.]+) %", float),
        V_disk_strain_pct=grab(t, r"V_disk +[0-9.]+ -> +[0-9.]+ mm\^3 +([-+0-9.]+) %", float),
        V_extrap_strain_pct=grab(t, r"V_extrap +[0-9.]+ -> +[0-9.]+ mm\^3 +([-+0-9.]+) %", float),
        linear_strain_pct=grab(t, r"linear strain +([-+0-9.]+) %", float),
        vertical_strain_pct=grab(t, r"vertical +([-+0-9.]+) %", float),
        radial_strain_pct=grab(t, r"radial +([-+0-9.]+) %", float),
        contact_angle_start=grab(t, r"contact angle (\d+) ->", float),
        contact_angle_end=grab(t, r"contact angle \d+ -> (\d+) deg", float),
        deciding_methods=grab(t, r"decided by +([^\n(]+)", str, "").strip(),
        median_spread_pct=grab(t, r"median spread +([0-9.]+) %", float),
        drying=grab(t, r"drying +([A-Z][A-Z ]+?)(?: --|$| at)", str, "?").strip(),
        mat_rise_um=grab(t, r"mat height +rises [-+0-9.]+ px = ([-+0-9,]+) um", float),
        mat_rise_px=grab(t, r"mat height +rises ([-+0-9.]+) px", float),
        mat_reading=grab(t, r"moving mat +read as: ([^(]+)\(", str, "").strip(),
    )
    for tier in ("CERTIFIED", "LIKELY", "SINGLE", "CONFLICT", "REJECT"):
        row[f"n_{tier.lower()}"] = int(grab(t, rf"\n    {tier} +(\d+)", int, 0))
    row["frac_cross_checked"] = (row["n_certified"] + row["n_likely"]) / max(1, row["n_frames"])
    row["first_frame_dropped"] = "the FIRST frame" in t

    rp = os.path.join(an, "results.txt")
    if os.path.isfile(rp):
        r = open(rp, encoding="utf-8", errors="replace").read()
        sec = r.split("3. FIRST + LAST IMAGE ONLY", 1)[-1] if "3. FIRST" in r else ""
        if sec and "not run" not in sec.split("\n\n")[0]:
            row["first_last_strain_pct"] = grab(sec, r"volumetric strain +([-+0-9.]+) %", float)
            row["first_last_to"] = grab(sec, r"-> ([^\n]+)", str, "").strip()
    row.setdefault("first_last_strain_pct", np.nan)
    row["has_final_photo"] = any("last image" in f.lower() for f in os.listdir(folder))

    curve = pd.DataFrame(dict(t_min=(u["t_s"] - u["t_s"].iloc[0]) / 60.0,
                              strain=u["vol_strain_pct"]))
    # time to reach 90 % of the final strain: how fast the bead dries
    fin = row["vol_strain_pct"]
    if np.isfinite(fin) and fin < 0 and len(curve) > 2:
        k = np.flatnonzero(curve["strain"].to_numpy() <= 0.9 * fin)
        row["t90_min"] = float(curve["t_min"].iloc[k[0]]) if k.size else np.nan
    else:
        row["t90_min"] = np.nan
    return row, curve


def find_sets(roots):
    out = []
    for root in roots:
        if os.path.isdir(os.path.join(root, "analysis")):
            out.append(root)
            continue
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if os.path.isdir(os.path.join(p, "analysis")):
                out.append(p)
    return out


# ------------------------------------------------------------------ plots
def style(ax, title, xl=None, yl=None):
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK)
    if xl:
        ax.set_xlabel(xl)
    if yl:
        ax.set_ylabel(yl)
    ax.grid(True, color=GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def recipe_colours(recipes):
    cm = plt.get_cmap("viridis")
    rs = sorted(recipes)
    return {r: cm(i / max(1, len(rs) - 1) * 0.9) for i, r in enumerate(rs)}


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_plots(m, curves, out):
    names = []
    key = m["series"] + " | " + m["recipe"]
    col = recipe_colours(set(key))

    # 01 repeatability
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * key.nunique() + 4), 5.2))
    groups = sorted(key.unique())
    for i, g in enumerate(groups):
        sub = m[key == g]
        y = sub["vol_strain_pct"].to_numpy(float)
        ax.scatter(np.full(len(y), i), y, s=70, color=col[g], edgecolor=INK, zorder=3)
        for yy, r in zip(y, sub["replicate"]):
            if r is None or (isinstance(r, float) and np.isnan(r)):
                continue
            ax.annotate(f"r{r}", (i, yy), xytext=(6, 0), textcoords="offset points", fontsize=8,
                        va="center")
        if len(y) > 1:
            mu, sd = np.nanmean(y), np.nanstd(y, ddof=1)
            ax.errorbar(i, mu, yerr=sd, color=INK, capsize=6, lw=1.6, zorder=2)
            ax.annotate(f"CV {abs(sd / mu) * 100:.0f}%", (i, mu), xytext=(14, 0),
                        textcoords="offset points", color="#b03a2e", fontweight="bold",
                        fontsize=10, va="center")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace(" | ", "\n") for g in groups], rotation=0, fontsize=8)
    style(ax, "Repeatability - replicates of each recipe (CV = spread / mean)", None,
          "Volumetric strain (%)")
    save(fig, out, "01_repeatability.png"); names.append("01 repeatability")

    # 02 concentration
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (ser, soak), sub in m.groupby(["series", "soak_hr"]):
        g = sub.groupby("conc_pct")["vol_strain_pct"].agg(["mean", "std", "count"]).reset_index()
        st = SERIES_STYLE.get(ser, dict(color="grey", marker="^"))
        ax.errorbar(g["conc_pct"], g["mean"], yerr=g["std"].fillna(0), capsize=5, lw=2.2,
                    label=f"{ser}, {soak:g} h soak", **st)
        for _, r in g.iterrows():
            ax.annotate(f"n={int(r['count'])}", (r["conc_pct"], r["mean"]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color=st["color"])
    style(ax, "Effect of concentration  (mean +/- SD of replicates)",
          "Chalk/glue concentration (%)", "Volumetric strain (%)")
    ax.legend(frameon=False)
    save(fig, out, "02_concentration.png"); names.append("02 concentration")

    # 03 every strain curve
    fig, ax = plt.subplots(figsize=(11, 6))
    for _, r in m.iterrows():
        c = curves[r["experiment"]]
        ax.plot(c["t_min"], c["strain"], lw=1.6, color=col[r["series"] + " | " + r["recipe"]],
                ls="-" if r["fresh"] else "--", alpha=0.9)
        ax.annotate(r["experiment"].split(" (")[0], (c["t_min"].iloc[-1], c["strain"].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", fontsize=7, va="center")
    style(ax, "Every drying curve  (solid = fresh/taped, dashed = earlier)", "time (min)",
          "Volumetric strain (%)")
    save(fig, out, "03_curves.png"); names.append("03 curves")

    # 04 mean curve per recipe
    fig, ax = plt.subplots(figsize=(10, 5.8))
    grid = np.arange(0, 181, 2.0)
    for g in groups:
        sub = m[key == g]
        ys = []
        for e in sub["experiment"]:
            c = curves[e]
            y = np.interp(grid, c["t_min"], c["strain"], right=np.nan)
            ys.append(y)
        ys = np.array(ys)
        # only where at least half the replicates still have data: a mean over
        # whichever runs happen to be longest jumps each time one of them ends
        ok = np.sum(np.isfinite(ys), axis=0) >= max(1, int(np.ceil(len(sub) / 2)))
        mu = np.nanmean(ys[:, ok], axis=0)
        ax.plot(grid[ok], mu, lw=2.4, color=col[g], label=f"{g}  (n={len(sub)})")
        if len(sub) > 1:
            sd = np.nanstd(ys[:, ok], axis=0)
            ax.fill_between(grid[ok], mu - sd, mu + sd, color=col[g], alpha=0.15)
    style(ax, "Mean drying curve per recipe  (band = +/- SD)", "time (min)",
          "Volumetric strain (%)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out, "04_mean_curves.png"); names.append("04 mean curves")

    # 05 shape: vertical vs radial
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for _, r in m.iterrows():
        ax.scatter(r["radial_strain_pct"], r["vertical_strain_pct"], s=70, edgecolor=INK,
                   color=col[r["series"] + " | " + r["recipe"]], zorder=3)
        ax.annotate(r["experiment"].split(" (")[0], (r["radial_strain_pct"],
                    r["vertical_strain_pct"]), xytext=(5, 3), textcoords="offset points",
                    fontsize=7)
    lim = np.nanmin(m[["radial_strain_pct", "vertical_strain_pct"]].to_numpy()) * 1.1
    ax.plot([0, lim], [0, lim], color="grey", ls=":", label="shrinks the same both ways")
    style(ax, "How the bead shrinks: height vs width", "radial strain (%)",
          "vertical strain (%)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out, "05_shape.png"); names.append("05 shape")

    # 06 bead size
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for _, r in m.iterrows():
        ax.scatter(r["V0_mm3"], r["vol_strain_pct"], s=70, edgecolor=INK,
                   color=col[r["series"] + " | " + r["recipe"]],
                   marker=SERIES_STYLE.get(r["series"], {}).get("marker", "o"), zorder=3)
        ax.annotate(r["experiment"].split(" (")[0], (r["V0_mm3"], r["vol_strain_pct"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=7)
    style(ax, "Does bead size matter?", "starting volume (mm$^3$)", "Volumetric strain (%)")
    save(fig, out, "06_bead_size.png"); names.append("06 bead size")

    # 07 drying speed
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, g in enumerate(groups):
        y = m.loc[key == g, "t90_min"].to_numpy(float)
        ax.scatter(np.full(len(y), i), y, s=70, color=col[g], edgecolor=INK, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace(" | ", "\n") for g in groups], fontsize=8)
    style(ax, "Drying speed - time to reach 90 % of the final strain", None, "minutes")
    save(fig, out, "07_drying_speed.png"); names.append("07 drying speed")

    # 08 mat rise vs strain
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for _, r in m.iterrows():
        ax.scatter(r["mat_rise_um"], r["vol_strain_pct"], s=70, edgecolor=INK,
                   color=col[r["series"] + " | " + r["recipe"]], zorder=3)
        ax.annotate(r["experiment"].split(" (")[0], (r["mat_rise_um"], r["vol_strain_pct"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=7)
    style(ax, "Does a rising mat bias the result?", "mat edge rise (um)",
          "Volumetric strain (%)")
    save(fig, out, "08_mat_rise.png"); names.append("08 mat rise")

    # 09 time-lapse vs first + last / next day
    fl = m[np.isfinite(m["first_last_strain_pct"])]
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(fl):
        ax.scatter(fl["vol_strain_pct"], fl["first_last_strain_pct"], s=80, edgecolor=INK,
                   color=[col[a + " | " + b] for a, b in zip(fl["series"], fl["recipe"])],
                   zorder=3)
        for _, r in fl.iterrows():
            ax.annotate(r["experiment"].split(" (")[0],
                        (r["vol_strain_pct"], r["first_last_strain_pct"]), xytext=(5, 3),
                        textcoords="offset points", fontsize=7)
        lo = np.nanmin(fl[["vol_strain_pct", "first_last_strain_pct"]].to_numpy()) - 3
        ax.plot([lo, 0], [lo, 0], color="grey", ls=":", label="same value")
        ax.legend(frameon=False, fontsize=8)
    style(ax, "Time-lapse vs first + last (final photo)", "time-lapse strain (%)",
          "first + last strain (%)")
    save(fig, out, "09_final_photo.png"); names.append("09 final photo")

    # 10 quality
    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(m) + 3), 5.2))
    xs = np.arange(len(m))
    ax.bar(xs, m["frac_cross_checked"] * 100, color=["#2a9d8f" if f > 0.8 else "#e9c46a"
                                                     if f > 0.3 else "#e76f51"
                                                     for f in m["frac_cross_checked"]])
    for x, (_, r) in zip(xs, m.iterrows()):
        flags = []
        if r["thresh_offset"]:
            flags.append(f"off {int(r['thresh_offset'])}")
        if r["stale_code"]:
            flags.append("old code")
        if r["drying"] not in ("DRY",):
            flags.append(r["drying"].split()[0].lower())
        ax.annotate("\n".join(flags), (x, 2), ha="center", fontsize=6.5, va="bottom",
                    rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels([e.split(" (")[0] for e in m["experiment"]], rotation=60, ha="right",
                       fontsize=8)
    style(ax, "Quality - share of frames where two methods cross-checked", None,
          "CERTIFIED + LIKELY frames (%)")
    save(fig, out, "10_quality.png"); names.append("10 quality")
    return names


# ------------------------------------------------------------------ findings
def findings(m, excluded, names):
    L = ["MASTER FINDINGS  (auto-generated - read alongside the plots)", "=" * 60]
    L.append(f"sets: {len(m)}   recipes: {m['recipe'].nunique()}   concentrations: "
             f"{sorted(m['conc_pct'].dropna().unique().tolist())}   soak times: "
             f"{sorted(m['soak_hr'].dropna().unique().tolist())} h")
    if excluded:
        L.append(f"left out (exclude.txt): {', '.join(excluded)}")
    L.append("")
    L.append("REPEATABILITY (replicates of the same recipe, volumetric strain):")
    for (ser, rec), sub in m.groupby(["series", "recipe"]):
        y = sub["vol_strain_pct"].dropna()
        if len(y) > 1:
            L.append(f"  {rec:18s} {ser:18s} n={len(y)}  mean {y.mean():+.1f} %  SD {y.std(ddof=1):.1f}"
                     f"  CV {abs(y.std(ddof=1) / y.mean()) * 100:.0f}%   "
                     f"[{', '.join(f'{v:+.1f}' for v in y)}]")
        elif len(y):
            L.append(f"  {rec:18s} {ser:18s} n=1  {y.iloc[0]:+.1f} %")
    L.append("")
    L.append("FACTOR EFFECTS:")
    for ser, sub in m.groupby("series"):
        s = sub.dropna(subset=["conc_pct", "vol_strain_pct"])
        if s["conc_pct"].nunique() > 1:
            r = np.corrcoef(s["conc_pct"], s["vol_strain_pct"])[0, 1]
            slope = np.polyfit(s["conc_pct"], s["vol_strain_pct"], 1)[0]
            L.append(f"  concentration ({ser}): shrinkage "
                     f"{'falls' if slope > 0 else 'rises'} with concentration, "
                     f"{slope:+.2f} points per % (r={r:+.2f}, n={len(s)})")
    s = m.dropna(subset=["V0_mm3", "vol_strain_pct"])
    if len(s) > 2:
        L.append(f"  bead size: r={np.corrcoef(s['V0_mm3'], s['vol_strain_pct'])[0, 1]:+.2f} between "
                 f"starting volume and strain (n={len(s)})")
    for rec, sub in m.groupby("recipe"):
        if sub["series"].nunique() > 1:
            a = sub.groupby("series")["vol_strain_pct"].mean()
            L.append(f"  {rec}: " + "  vs  ".join(f"{k} {v:+.1f} %" for k, v in a.items()))
    L.append("")
    fl = m[np.isfinite(m["first_last_strain_pct"]) & m["has_final_photo"]]
    if len(fl):
        L.append("FINAL PHOTO (start -> photo taken after the run):")
        for _, r in fl.iterrows():
            L.append(f"  {r['experiment']:28s} time-lapse {r['vol_strain_pct']:+.2f} %   "
                     f"final photo {r['first_last_strain_pct']:+.2f} %   "
                     f"(diff {r['first_last_strain_pct'] - r['vol_strain_pct']:+.2f})")
        L.append("")
    L.append("ANYTHING UNUSUAL:")
    for _, r in m.iterrows():
        e = r["experiment"]
        if r["remarks"]:
            L.append(f"  remark: {e}  ({r['remarks']})")
        if r["thresh_offset"]:
            L.append(f"  threshold offset {int(r['thresh_offset'])}: {e}  (dark wet base)")
        if r["stale_code"]:
            L.append(f"  run before a fix ({r['stale_code']}): {e}  - rerun to bring it up to date")
        if r["first_frame_dropped"]:
            L.append(f"  first frame left out: {e}  (start taken later -> strain a little low)")
        if np.isfinite(r["mat_rise_px"]) and abs(r["mat_rise_px"]) >= 20:
            L.append(f"  mat rose {r['mat_rise_px']:+.0f} px: {e}  ({r['mat_reading'] or 'rebuilt'})")
    L.append("")
    nd = m[~m["drying"].isin(["DRY"])]
    L.append(f"NOT CONFIRMED DRY ({len(nd)} of {len(m)} sets): the strain may still grow a little.")
    for _, r in nd.iterrows():
        L.append(f"  {r['experiment']:28s} {r['drying']:14s} after {r['total_time_min']:.0f} min")
    L.append("")
    one = m[m["frac_cross_checked"] < 0.3]
    L.append(f"NO CROSS-CHECK ({len(one)} of {len(m)} sets): one method carried most frames "
             "(usually V_extrap with a rising mat), so nothing corroborated it.")
    for _, r in one.iterrows():
        L.append(f"  {r['experiment']:28s} (decided by {r['deciding_methods']})")
    L.append("")
    L.append("Plots: " + " . ".join(names))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main
def run(roots, out):
    os.makedirs(out, exist_ok=True)
    ex_path = os.path.join(out, "exclude.txt")
    if not os.path.isfile(ex_path):
        with open(ex_path, "w") as f:
            f.write("# one set folder name per line to leave out of the master analysis\n"
                    "# e.g.\n# f90%_8hr_1 (20261021)\n")
    excluded = [l.strip() for l in open(ex_path) if l.strip() and not l.startswith("#")]
    rows, curves = [], {}
    for folder in find_sets(roots):
        name = os.path.basename(os.path.normpath(folder))
        if name in excluded or os.path.normpath(folder) == os.path.normpath(out):
            continue
        if "discard" in parse_name(name)["remarks"].lower():
            excluded.append(name + " (named 'discarded')")
            continue
        r, c = read_set(folder)
        if r is None:
            continue
        rows.append(r)
        curves[name] = c
    if not rows:
        raise SystemExit("no analysed sets found under: " + ", ".join(roots))
    m = pd.DataFrame(rows).sort_values(["series", "conc_pct", "soak_hr", "replicate"],
                                       na_position="last").reset_index(drop=True)
    m.to_csv(os.path.join(out, "comparative_master.csv"), index=False)
    names = make_plots(m, curves, out)
    txt = findings(m, excluded, names)
    with open(os.path.join(out, "findings.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    return m, txt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Every bead set side by side.")
    ap.add_argument("roots", nargs="*", help="data folders holding the set folders")
    ap.add_argument("--out", default=None, help="output folder (default: 'Master Analysis' "
                    "inside the first data folder)")
    a = ap.parse_args(argv)
    roots = a.roots
    if not roots:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        tk.Tk().withdraw()
        roots = []
        while True:
            d = filedialog.askdirectory(title="Choose a data folder (Cancel when done)" if roots
                                        else "Choose the data folder holding your set folders",
                                        mustexist=True)
            if not d:
                break
            roots.append(d)
            if not messagebox.askyesno("Master analysis", f"Added:\n{d}\n\nAdd another data "
                                       "folder (e.g. one with older sets)?"):
                break
        if not roots:
            return
    out = a.out or os.path.join(roots[0], "Master Analysis")
    m, txt = run(roots, out)
    print(txt)
    print(f"written to {out}")
    if not a.roots:
        try:
            os.startfile(out)
        except Exception:
            pass


if __name__ == "__main__":
    main()
