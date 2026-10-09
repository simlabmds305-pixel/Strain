# Bead volume → shrinkage strain

A fast, scriptable version of the bead-volume notebook, built on the structure of
the cantilever/Stoney tool: measure a folder of images, cross-check several
estimators against each other, stamp a verdict on every frame, and save
everything into `<folder>/analysis/` beside the data.

The notebook (`../bead_volume_batch_multi_image.ipynb`) is unchanged and still
works. This is the same physics, restructured.

```
python run_local.py                                 # asks: folder, interval, crop
python run_local.py /path/to/set --pick-roi --interval 30
python run_local.py /path/to/set --roi 1050,1950,1350,1720 --interval 30
python run_local.py /path/to/parent --each          # every subfolder, in one go
```

## Running it on your own machine

```bash
git clone https://github.com/simlabmds305-pixel/Strain.git
cd Strain
git checkout claude/bead-volume-multi-image-4b9d1q
cd strain_tools
pip install -r requirements.txt
```

Four packages, no scipy. If you already have the `cantilever` conda
environment, `conda activate cantilever` then `pip install -r
requirements.txt` into it — everything here is a subset of what that
environment already carries.

Check it works on anything you have:

```bash
python run_local.py "C:/path/to/a/set" --pick-roi --interval 30
```

Or run it with no arguments at all and it asks — folder picker, then the
interval, then the crop — remembering all three in `app_settings.json` beside
the script so the next run starts where the last one left off. If the folder
you choose has no images but its subfolders do, it treats each subfolder as
its own experiment without being told.

`--pick-roi` opens the first image; drag a box round the bead, close the
window, and it runs with that crop and prints the flag so you can skip the
picker next time.

Under `--each` you are asked for **a crop per experiment**, because the bead
is not in the same place in every set and a crop carried over from the
previous folder would be wrong by however far the sample moved — and wrong
quietly, clipping the bead rather than failing. Each folder's own crop is
written to its `analysis/settings.json`. Pass `--one-roi` to draw it once for
all of them when a campaign really was framed identically, or `--roi` to give
the same box on the command line. Quote Windows paths — they contain backslashes and often
spaces. Results appear in `<that folder>/analysis/`.

Two platform notes:

- **On Windows the parallelism needs the `if __name__ == "__main__"` guard**,
  which `run_local.py` has. Calling `analyse_folder` directly from a Jupyter
  notebook on Windows will hang instead, because notebooks have no such guard —
  pass `workers=1` there, or just use the command line.
- `--pick-roi` needs a GUI, so it will not work over plain SSH or in a headless
  container. Everything else runs headless (matplotlib is forced to Agg).

Outputs, written into `<folder>/analysis/`:

| file | what it is |
|---|---|
| `linear_strain.png` | linear strain vs **time**, with **image number** across the top |
| `volumetric_strain.png` | volumetric strain vs time |
| `height.png` | bead height (um) vs time |
| `base_radius.png` | bead base radius (um) vs time |
| `shape_strain.png` | vertical, radial and isotropic-equivalent strain on one axis |
| `volumes.png` | all four estimators over the run |
| `agreement.png` | how far apart the deciding estimators were, frame by frame |
| `per_image.csv` | every number for every frame |
| `summary.txt` | the headline, the verdict counts, and the warnings |
| `settings.json` | exactly what was used, so a run can be reproduced |

## Speed

40 frames at 3088×2076, 4 cores:

| | whole frame | with `--roi` |
|---|---|---|
| notebook | 65.3 s (1632 ms/frame) | — |
| this, 1 worker | 28.0 s (699 ms/frame) | 6.8 s (170 ms/frame) |
| this, 4 workers | **9.2 s (229 ms/frame)** | **2.0 s (51 ms/frame)** |

Three changes account for it:

1. **One decode per image.** The notebook read every file three times — once
   greyscale, once in colour for the chroma gate, once more for the mat-edge
   fit. Here the colour frame is decoded once and the grey and chroma images
   are derived from it. Decoding alone is 154 ms/frame at this size.
2. **A vectorised row profile.** `xs[ys == y]` per row rescans the whole
   coordinate list once per row — quadratic in the silhouette, ~10⁹
   comparisons on a 1000-row bead, and paid again for every candidate blob.
   `argmax` on a boolean array does it in one pass.
3. **Parallel across images**, one worker per core.

Clipping a mask at the mat is also now a truncation of the profile rather than
a full re-segmentation of the image, which is exactly equivalent (the clip was
always applied after the blob was chosen) and free.

One cost, stated plainly: deriving grey with `cvtColor` instead of
`IMREAD_GRAYSCALE` differs by a rounding LSB, which can flip a pixel at the
threshold. Measured against the notebook on 104 images, mean V_disk difference
**+0.017%**, worst single frame 0.66%.

## The four estimators

All four integrate the silhouette as a stack of discs, `V = Σ π r² dz`. They
differ **only** in what they do with the rows between the bead's widest row and
the mat — the part front lighting loses to the bead's own shadow.

| | what it does | fails when |
|---|---|---|
| `V_disk` | integrates exactly the rows the threshold found | the mask stops short of the mat |
| `V_trunc` | stops at the widest row | anything below the widest row is real |
| `V_base` | fills to the mat at constant radius | the fill is long |
| `V_extrap` | continues each side to the mat along the slope it had where the edge was crisp | the edges are not straight that far |

**`V_disk` and `V_extrap` decide.** They bracket the truth, and — the useful
part — they *agree only when the mask already reached the mat*, which is
precisely the condition under which the measurement can be trusted. Their
spread is a real test, not a formality.

## Confidences

Each estimator carries a confidence, scored on evidence **independent of its own
value**:

```
conf = (measured share) + (invented share) × (is the invention sound?)
```

so an estimator that invents nothing scores 1, and one that invents half its
volume scores at most 0.5 + 0.5 × soundness. The inputs are mask-versus-mat
geometry (how far short the mask stopped, whether the silhouette was still
widening when it ended) and **held-out** edge-prediction error — the fit is
scored on rows it never saw, over a window sized to match how far it has to
reach. Scoring a fit on the points it was fitted to measures nothing, the way a
template matched against a copy of itself always wins.

Three places where the obvious design is wrong, each found by testing against a
synthetic run with a known answer:

- **Deciders are chosen once per run, not per frame.** A hard per-frame gate
  puts one frame at 0.36 and its neighbour at 0.34 and gives them opposite
  verdicts. Worse, it inverts the meaning: a frame whose `V_disk` is slightly
  *better* clears the gate, meets `V_extrap`, disagrees with it and is thrown
  out, while its neighbour with a worse `V_disk` never faces the comparison and
  survives.
- **Consensus weights are the run's median confidences, not each frame's.**
  Per-frame weights drift over a run — as the bead shrinks, a mask that stops a
  fixed 16 px short of the mat loses a growing *fraction* of it, so `V_disk`'s
  confidence decays — the blend slides from one biased estimator toward
  another, and that slide enters the strain as shrinkage that never happened.
  Measured: per-frame weights gave **−10.05%** against a true **−11.199%**,
  *worse than any of the four estimators alone*. Fixed weights gave −11.29%.
- **A baseline inferred from the masks ("auto") cannot judge those same masks.**
  That is circular, and a circular score reads high exactly when it deserves it
  least, so it is capped — but the cap must sit clearly *above* the decision
  gate, not on it. Set equal to it (both were 0.35) every greyscale run landed
  exactly on the gate, confidences of 0.345/0.348/0.350 flipped frames between
  SINGLE and REJECT on rounding noise, and a 4-frame set was thrown out whole
  with masks that were in fact perfect. The cap is 0.65.

## Verdicts and quarantine

Per frame, from the spread between the deciders, relative to the consensus
(volumes run to ~10⁸ px³, so an absolute tolerance would mean nothing):

`CERTIFIED` ≤2% · `LIKELY` ≤5% · `CONFLICT` >5% · `SINGLE` one decider · `REJECT` none

**What gets quarantined is not simply CONFLICT.** The strain is a ratio, so a
bias that is the same in every frame divides out of it exactly: if `V_disk` and
`V_extrap` disagree by 12% on all 24 frames, that is a statement about the
absolute volume (the base is lost to shadow), not a reason to discard the run —
discarding it would leave a biased subset and a strain measured over a shorter
span, which is worse than the disagreement was. What *does* corrupt a strain is
a frame that disagrees much more than the run normally does, or one whose mask
is broken. So frames are quarantined on **outlier** disagreement (>3 MAD above
the run's own median) or a broken mask, and the systematic part is reported as a
caveat on the absolute volume.

If quarantining shortens the span, the report says so — a strain quoted over
frames 2→14 of a 24-frame run, without that being stated, is the exact kind of
quiet wrongness this machinery exists to prevent.

Brightness outliers (exposure glitch, a shadow crossing the frame) are flagged
by MAD on each frame's mean brightness and their volumes interpolated from the
neighbours.

## Three strains, not one

`linear_strain_pct` is the cube root of the volume ratio. That is a real linear
strain **only if the bead shrinks equally in every direction**, and a sessile
bead pinned to its mat does not: it collapses in height while its footprint
stays put. So the pipeline also measures the two strains the bead actually has,
from the height and base radius it already records every frame:

| column | what it is |
|---|---|
| `vol_strain_pct` | ΔV/V₀, signed (`vol_shrinkage_pct` is the same number, positive) |
| `linear_strain_pct` | (V/V₀)^⅓ − 1 — the isotropic equivalent |
| `height_strain_pct` | h/h₀ − 1 — vertical |
| `radial_strain_pct` | a/a₀ − 1 — radial |

When the last two differ by more than 25%, the summary says so and names the
likely cause. On a real 81-frame run they came out **−15.2% vertical against
−5.8% radial**, a factor of 2.6, where the cube root reported −9.2% for both.
A footprint held back while the height collapses means the contact line is
resisting, which puts the material in radial tension as it dries — the thing
worth measuring, and the thing a single cube-rooted number hides. Whether it
is fully pinned or receding slowly is a different question, and the ratio
alone cannot answer it: read `base_radius.png`, where a pinned line is flat
and a receding one slopes. On the real run above it recedes 133 um over the
first 44 minutes and then arrests, so it is neither.

Validated both ways. On a synthetic cap built to shrink **isotropically**, the
three agree (vertical −10.93%, radial −11.09%, linear −11.13%), so the measure
does not invent anisotropy. On one built with a pinned base — true vertical
−15.00%, true radial −6.00% — it reads −15.52% and −6.46%, both within half a
point, and the warning fires.

## A mat that moves

The mat is found in every frame from its colour, beside the bead and under it.
On every real run so far but one, its edge moved during the run (up to 155 px).
A blue edge can rise for two reasons, and they need opposite handling:

| what is happening | how it is measured |
|---|---|
| the surface under the bead rises and **carries the bead up** | each frame to its own mat row |
| the mat's **front edge rises in front of the bead** and hides its bottom | every frame to where the bead sits (the deepest the edge was seen); the hidden rows are rebuilt by continuing the bead's sides down (V_extrap), and V_disk -- which only sees what is above the edge -- does not vote |

The code tells them apart from the TOP of the bead: fitted as a smooth drying
trend plus beta x the mat row, beta is ~1 if the bead rides on the mat and ~0
if the edge is sliding up in front of it. That needs the mat to move unevenly
(in bursts, as it has on every real run). When it moves too evenly to tell,
physics decides if it can -- a bead whose floor stayed put cannot grow taller
while it dries -- and otherwise the summary says it could not tell and assumes
the edge is in front. Override with `--mat follow` or `--mat fixed`.

Either way the summary prints the strain under BOTH readings, so you can see
how much rides on the choice.

On `88%_8hr_1` beta was +0.14 +/- 0.07 and the volume lost an extra 0.25 % of V0
for every px the edge rose -- exactly what hiding the bead's widest rows does.
Following the edge read -37.5 %; with the floor fixed it is about -23 %.

On test frames (true volumetric strain -30.0 %):

| case | old code | follow every edge | now |
|---|---|---|---|
| bead rides a mat rising in bursts | | -29.5 % | -29.5 % |
| edge rises in front of a still bead | | -42.4 % | -28.9 % |
| bead rides a mat rising steadily | +5.0 % | -29.0 % | -29.0 % |

## Two-photo measurement when the mat moves

The tracked run is always done and saved as before (`analysis/`). If the mat
moved 20 px (~80 um) or more, part of the bead was hidden behind its edge, so
the run then says so and asks for two photos in which the mat is flat:

1. the INITIAL image (normally the first one of the run), and
2. a FINAL image -- for example one taken after the bead is dry and the mat
   has been pressed flat (same zoom, same focus, stage not moved).

Those two are measured on their own, each to its own mat edge, with nothing
rebuilt, and saved beside the tracked results in `analysis_pair/`:
`summary.txt`, `per_image.csv`, `settings.json` and `pair.png` (both photos
with the outline measured and the mat row it was measured to -- check the bead
did not tilt). The tracked `summary.txt` gets a line with the two-photo result.

Without the prompt: `python run_local.py <folder> --pair INITIAL FINAL`.
To never be asked: `--no-pair`.

On test frames where the mat edge rose 73 px in front of the bead (true
volumetric strain -29.99 %), the tracked run read -28.85 %; the two-photo
measurement read -29.95 %, both photos CERTIFIED.

## Did the bead finish drying?

`summary.txt` answers this on its own line, `drying`. The local drying rate is
the slope of V/V0 over the trailing 15 minutes; the bead is flat once that
rate stays under 0.02 %/min, and **DRY** once it has stayed flat for 20 minutes.

| verdict        | meaning                                                         |
|----------------|-----------------------------------------------------------------|
| DRY            | the strain reported is the bead's final value                   |
| LEVELLING OFF  | flat, but not for long enough to be sure -- run it longer       |
| STILL DRYING   | still losing water at the last frame; the strain is not final   |
| NOT JUDGED     | frames too noisy for a 0.02 %/min rate to be told from zero     |

It never compares a frame with the last frame. "Within x % of the final value"
is circular: a run stopped soon after the curve levels off has few frames left
to fail the test, so it passes almost for free. On the first three 88 % runs
that test said all three had plateaued; this one says set 3 was levelling off
(15 of the 20 minutes), set 1 had just started to (2 minutes), and set 2 never
got below 0.028 %/min. Needs `--interval`, since it works in minutes.

## Checks and tests

`python test_pipeline.py` runs the regression suite. Every test in it is a bug
that reached a real run and produced a wrong number that looked right, written
as the failure rather than the fix. Point it at a folder with a known answer
to include the end-to-end check:

```bash
BEAD_GT_DIR=/path/to/known/set python test_pipeline.py
```

Two mistakes account for most of that list and are worth naming, because they
are easy to make again:

- **Judging a per-frame quantity against the whole run.** Everything drifts —
  the lamp dims, the base goes into shadow, the bead shrinks — so a global
  median plus MAD does not find outliers, it finds the *ends of the run*. On
  one 99-frame set it called the whole bright first half-hour outliers. Compare
  a frame with its neighbours. But a purely local test has the opposite blind
  spot: a long enough *block* of bad frames drags the local median with it, so
  there is also a gross check against the run's own level.
- **Writing a value that was not measured.** Interpolation, clamping and
  fallbacks put numbers in the table that no image produced, and afterwards
  they look exactly like the real ones. `np.interp` clamping past the end of
  its range once handed 34 frames a copy of frame 34 — a perfectly flat
  half-hour that read as a bead sitting still, and became V₀.

`find_frozen` is the guard for the second. It looks for runs of *exactly*
equal volumes, which a sum over tens of thousands of pixels cannot produce
twice, and refuses the run out loud when it finds them. Nothing else could:
the estimators agreed with each other (they were copies of the same frame),
every verdict came out CERTIFIED, and the plots looked clean. Cross-checking
estimates against each other cannot catch data duplicated before the estimates
were made. The `measured` column marks any frame whose volume was filled in.

## Accuracy

Against synthetic runs whose true disc-integral volume is known exactly
(spherical cap on a blue mat, front-lit so the base fades, true linear strain
−11.199%):

| set | what it tests | verdicts | measured strain | error |
|---|---|---|---|---|
| `gt40` | clean, 40 frames | 40 CERTIFIED | −11.133% | +0.066 pts |
| `gt_shadow` | deep shadow at the base | 24 CERTIFIED | −11.146% | +0.053 pts |
| `gt_mixed` | 1-in-4 bad frames + 2 exposure glitches | 17 CERTIFIED, 5 LIKELY, 2 REJECT | −11.057% | +0.142 pts |
| `gt_lost` | base genuinely below the threshold | 24 CONFLICT (all flagged) | −11.195% | +0.004 pts |
| `gt40` tight crop | crop shifts Otsu, mask stops 16 px short | 40 CONFLICT | −11.292% | −0.093 pts |

`gt_lost` is the one worth reading twice: `V_disk` is **−10.7%** and `V_extrap`
**+2.0%** on every frame, the tool says CONFLICT on all 24 and explains why —
and the strain still comes out right, because the bias is the same at both ends
of the ratio.

On the real-image set with an independent ground truth (`bluemat`): true strain
−15.000%, measured **−14.902%**, which reproduces the notebook's own result.

It also refuses when it should: on a set whose crop cuts the bead off at both
sides, all three frames come back REJECT, no strain is reported, and the
summary says the crop is the problem.

## Settings

Everything is a flag; nothing is baked in for one set of images.

```
--roi x0,x1,y0,y1     crop, same for every image (big speed win, and usually
                      needed to keep other bright objects out)
--scale 0.256         px per um. 0.256 = Olympus SZX16 at 1x, 3088 px wide
--interval 30         seconds between frames; without it the x axis is frame number
--time-regex '_(\d+)min'   or read the time out of the file name
--baseline blue       read the mat off its own colour (default) | auto | a row number
--thresh-offset -20   lower the threshold if the mask will not reach the mat
--method otsu         otsu | adaptive | edges
--workers 4           default: one per core
--each                treat every subfolder of the given folder as its own experiment
                      (asks for a crop on each one)
--one-roi             with --each, draw one crop and use it for every subfolder
```

## What this does not fix

The base is lost to shadow because the bead is lit from the front, so its
underside sits in its own shadow while the mat scatters light back. At the
contact line the real images read bead 99, mat 83 — a 16-level difference where
mid-height contrast is 226. No threshold separates that, and `V_extrap` is an
estimate of what the threshold could not see.

**A backlight removes the problem at source** rather than estimating around it.
Everything above is what to do until then.
