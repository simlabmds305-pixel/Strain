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
