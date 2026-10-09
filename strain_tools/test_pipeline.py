#!/usr/bin/env python3
# ============================================================================
#  REGRESSION TESTS
#
#      python test_pipeline.py
#
#  Every test here is a bug that reached a real run and produced a wrong
#  number that looked right. They are written as the failure, not as the fix,
#  so they stay meaningful if the implementation changes.
#
#  Two mistakes account for most of them and are worth naming, because they
#  are easy to make again:
#
#    1. Judging a per-frame quantity against the WHOLE RUN. Everything here
#       drifts -- the lamp dims, the base goes into shadow, the bead shrinks --
#       so a global median plus MAD does not find outliers, it finds the ends
#       of the run. Compare a frame with its neighbours.
#
#    2. Writing a value that was not measured. Interpolation, clamping and
#       fallbacks all put numbers in the table that no image produced, and
#       they do not look any different from the real ones afterwards.
# ============================================================================
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bead_pipeline as bp

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def fake_run(n=99, bright=(66.6, 52.5), glitches=(), vol=(20.0, 15.8)):
    """A run of n frames with drifting illumination, as every real run has."""
    b = np.linspace(*bright, n)
    V = np.linspace(*vol, n) * 1e8
    for i, mult in glitches:
        b[i] = 30.0 if mult < 1 else 85.0
        V[i] *= mult
    return pd.DataFrame({
        "image": [f"f{i:03d}.png" for i in range(n)],
        "brightness": b, "V_disk": V, "V_trunc": V, "V_base": V,
        "V_extrap": V, "V_consensus": V.copy(),
        "h_px": np.linspace(450, 380, n), "a_px": np.full(n, 600.0),
    })


# ---------------------------------------------------------------------------
print("\nbrightness outliers -- a drifting lamp is not 34 outliers")
df = fake_run(glitches=[(40, 1.4), (77, 0.6)])
raw = df["V_consensus"].to_numpy().copy()
out = bp.mark_brightness_outliers(df, 2.5)
flagged = np.flatnonzero(out["outlier"].to_numpy())
check("only the real glitches are flagged", set(flagged) == {40, 77},
      f"flagged {flagged.tolist()}")
check("the bright start is left alone", not out["outlier"].to_numpy()[:20].any())
check("a glitch is repaired from its neighbours",
      abs(out["V_consensus"][40] - np.interp(40, [39, 41], [raw[39], raw[41]])) < 1e-6)

print("\nno value is ever extrapolated beyond the measured range")
df = fake_run(n=40)
df.loc[:9, "brightness"] = 200.0                    # first 10 frames unusable
raw = df["V_consensus"].to_numpy().copy()
out = bp.mark_brightness_outliers(df, 2.5)
check("leading flagged frames keep their measured values",
      np.allclose(out["V_consensus"].to_numpy()[:10], raw[:10]))
check("they are still marked unmeasured-or-flagged", out["outlier"].to_numpy()[:10].any())

print("\nfrozen data is caught -- the check nothing else can do")
df = fake_run(n=50)
df.loc[:33, "V_disk"] = df["V_disk"][34]            # exactly what the real bug produced
df.loc[:33, "V_extrap"] = df["V_extrap"][34]
fr = bp.find_frozen(df)
check("a 34-frame freeze is found", any(r >= 34 for _, _, r in fr), f"{fr[:2]}")
check("a clean run reports nothing frozen", bp.find_frozen(fake_run(n=50)) == [])

print("\nestimator spread -- a trend is not a pile of outliers")
n = 60
s = np.concatenate([np.full(30, 0.3), np.linspace(0.3, 2.0, 30)])   # drifts upward
loc = pd.Series(s).rolling(9, center=True, min_periods=3).median().to_numpy()
r = s - loc
mad = float(np.nanmedian(np.abs(r - np.nanmedian(r)))) * 1.4826
local_flags = int((r > max(3 * mad, 1.0)).sum())
g = float(np.median(s))
gmad = float(np.median(np.abs(s - g))) * 1.4826
global_flags = int((s > g + max(3 * gmad, 1.0)).sum())
check("a local test ignores the drift", local_flags == 0,
      f"local {local_flags}, a global one would flag {global_flags}")

print("\ndrying -- a run stopped just after it levels off is not proven dry")
t = np.arange(0, 140, 1.0)
v = 100 - 40 * (1 - np.exp(-t / 15.0))               # dries out, then sits still
r = bp.drying_status(t, v)
check("a long flat tail is called DRY", r["state"] == "DRY",
      f"{r['state']}, flat {r['flat_for_min']:.0f} min")
r = bp.drying_status(t[:70], v[:70])                  # same bead, stopped at 70 min
check("the same curve cut short is not", r["state"] != "DRY",
      f"{r['state']}, flat {r['flat_for_min']:.0f} min")
check("a bead still losing water is DRYING",
      bp.drying_status(t, 100 - 0.1 * t)["state"] == "DRYING")
check("a volume that climbs at the end is not called drying",
      bp.drying_status(t, np.maximum(100 - 0.3 * t, 70) + np.clip(t - 110, 0, None) * 0.05)["state"]
      == "RISING")
rng = np.random.default_rng(0)
r = bp.drying_status(t, v + rng.normal(0, 0.06, t.size))   # worst real run: ~0.05 %
check("frame noise at the measured level does not hide a real plateau",
      r["state"] == "DRY", f"{r['state']}, rate known to {r['rate_noise']:.4f} %/min")
r = bp.drying_status(t, v + rng.normal(0, 0.3, t.size))    # 6x worse than any real run
check("noise too large to judge is said so, not reported as drying",
      r["state"] == "NOISY", f"{r['state']}, rate known to {r['rate_noise']:.4f} %/min")

print("\nthe mat can move -- on 88%_8hr_4 it rose 155 px; test frames like it read +5% for -30%")
def mat_frames(ys, ride=True):
    """Frames whose mat edge reads `ys`. With ride=True the bead sits on the
    mat and its top moves with it; with ride=False the top stays put."""
    out = []
    m = [y for y in ys if y is not None]
    for i, y in enumerate(ys):
        ref = m[min(i, len(m) - 1)] if y is None else y
        apex = int(round((ref if ride else 1900) - 600 + 0.8 * i))   # + drying
        out.append({"mat_y": (None if y is None else float(y)), "mat_tilt_deg": 0.0,
                    "Y": np.arange(apex, apex + 500)})
    return out
cfg_blue = {"BASELINE": "blue"}
n = 60
k = np.arange(n) / (n - 1)
true = 1900 - 160 * (0.55 * np.clip((k - .12) / .13, 0, 1) + 0.45 * np.clip((k - .5) / .12, 0, 1))
noisy = true + np.random.default_rng(1).normal(0, 1.5, n)
yb, src, conf, notes = bp.resolve_baseline(mat_frames(noisy), cfg_blue)
check("a smoothly rising mat is followed frame by frame",
      np.ndim(yb) == 1 and src.startswith("blue"), f"source {src}")
check("...without lagging at the ends of the run",
      np.ndim(yb) == 1 and abs(yb[0] - true[0]) < 3 and abs(yb[-1] - true[-1]) < 3,
      f"ends off by {yb[0]-true[0]:+.1f}, {yb[-1]-true[-1]:+.1f} px" if np.ndim(yb) == 1 else "")
bow = 1900 - 30 * np.clip((k - .3) / .3, 0, 1)    # inside the old 30 px "fixed" tolerance
yb, src, _, _ = bp.resolve_baseline(mat_frames(bow), cfg_blue)
check("a 30 px drift is not averaged into one row", np.ndim(yb) == 1, f"source {src}")
still = 1900 + np.random.default_rng(2).normal(0, 2.0, n)   # jittery reads, mat not moving
yb, src, _, _ = bp.resolve_baseline(mat_frames(still), cfg_blue)
check("a mat that does not move keeps a single row", np.ndim(yb) == 0, f"source {src}")
jumpy = 1900 + np.random.default_rng(3).choice([-80, 0, 70], n)
yb, src, _, _ = bp.resolve_baseline(mat_frames(jumpy), cfg_blue)
check("an edge that jumps about is still refused", src == "auto", f"source {src}")
yb, src, _, _ = bp.resolve_baseline(mat_frames(noisy, ride=False), cfg_blue)
check("an edge rising IN FRONT of a bead that stays put is not followed (88%_8hr_1)",
      np.ndim(yb) == 0 and "front" in src and abs(yb - true.max()) < 3,
      f"source {src}, floor {float(np.max(yb)):.0f} vs {true.max():.0f}")
small = 1900 - 7 * k + np.random.default_rng(4).normal(0, 0.8, n)   # 88%_8hr_3: 7 px
yb, src, _, _ = bp.resolve_baseline(mat_frames(small, ride=False), cfg_blue)
check("a 7 px drift is a flat mat: one row, V_disk keeps its vote (88%_8hr_3)",
      np.ndim(yb) == 0 and "front" not in src, f"source {src}")
check("the apex test decides the real runs",
      bp.mat_reading(-1.00, 0.26) == "fixed" and bp.mat_reading(0.15, 0.07) == "fixed"
      and bp.mat_reading(-0.37, 0.05) == "fixed" and bp.mat_reading(1.20, 0.03) == "follow"
      and bp.mat_reading(0.6, 0.4) is None,
      "set 2 -1.00+/-0.26, set 1 +0.15, set 4 -0.37, riding +1.20, unclear +0.6+/-0.4")
yb, src, _, _ = bp.resolve_baseline(mat_frames([1971.0, 1899.0], ride=False),
                                    dict(cfg_blue, PAIR_MODE=True))
check("two hand-picked photos are each measured to their own mat edge",
      np.ndim(yb) == 1 and list(yb) == [1971.0, 1899.0], f"source {src}")
gaps = [None if i % 4 == 0 else y for i, y in enumerate(noisy)]
gaps[0] = noisy[0]
yb, src, _, _ = bp.resolve_baseline(mat_frames(gaps), cfg_blue)
check("frames with no read take the trend, never a made-up end",
      np.ndim(yb) == 1 and np.all(np.isfinite(yb)) and abs(yb[4] - true[4]) < 4,
      f"frame 4 off by {yb[4]-true[4]:+.1f} px" if np.ndim(yb) == 1 else f"source {src}")

print("\nconfidence thresholds must not sit where confidences cluster")
check("the auto-baseline cap is clear of the decision gate",
      bp.AUTO_BASELINE_CONF > bp.MIN_CONF * 1.5,
      f"cap {bp.AUTO_BASELINE_CONF} vs gate {bp.MIN_CONF}")
check("a frame is judged broken relative to its run, not a constant",
      0 < bp.FRAME_REJECT_FRAC < 1 and bp.FRAME_REJECT_ABS < bp.MIN_CONF)

print("\nend-to-end against images whose true volume is known")
here = os.path.dirname(os.path.abspath(__file__))
gt = os.environ.get("BEAD_GT_DIR", os.path.join(here, "..", "testdata", "gt40"))
if os.path.isdir(gt) and bp.list_frames(gt):
    d, info = bp.analyse_folder(bp.list_frames(gt), {"INTERVAL_S": 60}, workers=2)
    u = d[d["use"]]
    v = u["V_consensus"].to_numpy()
    strain = 100 * ((v[-1] / v[0]) ** (1 / 3) - 1)
    check("strain within 0.3 points of the true -11.199%", abs(strain + 11.199) < 0.3,
          f"{strain:+.3f}%")
    check("nothing frozen", info["frozen"] == [])
    check("vertical and radial agree on an isotropic bead",
          abs(u["height_strain_pct"].iloc[-1] - u["radial_strain_pct"].iloc[-1]) < 0.5)
else:
    print(f"  [skip] no ground-truth images at {gt}")
    print("         set BEAD_GT_DIR to a folder with a known answer to enable")

# ---------------------------------------------------------------------------
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  failed:", f)
sys.exit(1 if FAIL else 0)
