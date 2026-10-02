# ageingfish

Estimating fish age by counting annuli in otolith images.

- `DataExploration_FishImages.ipynb`: early exploration on three walleye otolith sections (`*.jpg`).
- `ageingfish/thuenen.py`: loader for the Thünen Institute's labelled otolith datasets.
- `ageingfish/scoring.py`: scores predicted ages against reference ages.
- `ageingfish/ringcount.py`: classical annulus counter (no machine learning) that writes predicted ages.
- `ageingfish/tune.py`: picks the counter's settings and age calibration on a development split.
- `ageingfish/ringclass.py`: learned ring detector trained on Thünen's ring annotations, combined
  with the classical count (best results so far; model in `models/ringclass.joblib`).

```
pip install -r requirements.txt
python -m pytest
```

## Labelled test data: Thünen otolith datasets

The ages ship with this repo (`data/thuenen/`, see the README there). The images are downloaded
separately from Zenodo:

| Dataset | Images | Species | Ages |
|---|---|---|---|
| North Sea, [10.5281/zenodo.8341092](https://doi.org/10.5281/zenodo.8341092) | 660 | cod 194, saithe 351, haddock 78, whiting 37 | 1–11 |
| Baltic Sea, [10.5281/zenodo.8341149](https://doi.org/10.5281/zenodo.8341149) | 1,155 | Baltic cod | 1–5 (validated with tetracycline marks) |

Unzip anywhere; images are found by filename, recursively.

```
# summarise a dataset; check how many images are on disk; export labels (with image paths)
python -m ageingfish.thuenen north --images ~/data/datasets_north --out north_labels.csv
```

```python
from ageingfish.thuenen import load_dataset, load_ring_annotations
records = load_dataset("baltic", image_root="~/data/datasets_baltic", require_images=True)
for r in records:
    r.image, r.age, r.species, r.path
```

`load_ring_annotations` reads Thünen's VIA polygon annotations from a clone of
[arjaycc/ai_otolith](https://github.com/arjaycc/ai_otolith), for testing ring detection directly
rather than only the final age.

## Scoring predicted ages

Write predictions as a CSV with columns `image,predicted_age` (fractional ages are rounded half up
for the agreement metrics), then:

```
python -m ageingfish.scoring predictions.csv --dataset north --by-species --plot age_bias.png
python -m ageingfish.scoring predictions.csv --truth labels.csv          # your own image,age[,species] CSV
```

Reported: n, exact agreement, agreement within ±1 year, mean bias, MAE/RMSE, APE and CV,
Bowker's symmetry test (systematic over/under-ageing), an "always predict the most common age"
baseline, and a per-age table of mean predicted age with 95% CI (also drawn as an age-bias plot).

## Ring counter

`ageingfish/ringcount.py` segments the otolith from the dark background, rotates it so the long
axis is horizontal, and reads brightness along five lines parallel to the midline from tip to tip.
Annuli cross these lines as alternating bands, so it counts peaks on both sides of the core,
averages the count over the lines, and converts it to an age with a linear calibration.
All settings are relative to otolith length, so image resolution does not matter.

```
# choose settings + calibration on the dev split (~30% of images, fixed by filename hash)
python -m ageingfish.tune --dataset north --images ~/data/datasets_north --out configs/north.json
# predict the held-out test split and score it
python -m ageingfish.ringcount --dataset north --split test --images ~/data/datasets_north \
    --config configs/north.json --out results/north_test_predictions.csv --plots results/plots
python -m ageingfish.scoring results/north_test_predictions.csv --dataset north --split test --by-species
```

For your own images (e.g. walleye), tune with `--truth labels.csv` (columns `image,age`) on a set
of labelled fish, then run `ringcount --images DIR --config ...` on the rest.

### Results on held-out images

Settings and calibration were chosen on the dev split only (`configs/`); these numbers are on the
test split, which was never looked at during development. Full reports and age-bias plots are in
`results/` (`*_ringclass*` files are the learned detector below).

| Test split | n | Classical: exact / ±1 y / MAE | + ring classifier: exact / ±1 y / MAE | Baseline exact |
|---|---|---|---|---|
| North Sea, all | 462 | 31.8% / 76.6% / 1.07 y | 30.7% / 80.7% / 1.00 y | 13.6% |
| saithe | 253 | 36.4% / 85.4% / 0.86 y | 34.8% / 87.0% / 0.83 y | 16.6% |
| cod | 131 | 30.5% / 69.5% / 1.24 y | 29.8% / 74.8% / 1.16 y | 16.0% |
| haddock | 54 | 18.5% / 63.0% / 1.44 y | 22.2% / 70.4% / 1.28 y | 18.5% |
| whiting | 24 | 20.8% / 54.2% / 1.43 y | 12.5% / 70.8% / 1.23 y | 20.8% |
| Baltic cod | 804 | 52.5% / 92.4% / 0.64 y | **61.2% / 95.6% / 0.50 y** | 28.2% |

Baseline: always predict the most common age.
Dev-split results were similar (North MAE 1.06, Baltic 0.60), so the tuning did not overfit.
For comparison, Thünen's deep-learning models reached about 72% mean accuracy on the Baltic set
(Sigurðardóttir et al. 2024), under a different evaluation setup.

Known weaknesses, visible in `results/examples/` and the age-bias plots:

- **Pull toward the middle.** The calibration shrinks toward the mean age: Baltic 1-year-olds are
  predicted about 2.0 and 5-year-olds about 3.7 (Bowker's test p < 1e-30). Better ring evidence,
  not a different calibration, is the fix.
- **Core position matters little for this counter.** Using Thünen's ring annotations on dev images
  (235 otoliths whose annotation count matched 2 × age), the thickest-point core guess was off by a
  median 10% of otolith length, and simply taking the midpoint of the long axis was off by 2–4%.
  But plugging in the *annotated* core improved dev MAE only from 0.85 to 0.80 years (North) and
  0.59 to 0.57 (Baltic), because rings are counted tip to tip and the core only decides which side
  a ring falls on. `tune` now chooses between the `thickness` and `midpoint` core rules; it kept
  `thickness` for the North Sea and picked `midpoint` for the Baltic (held-out change: 52.0% to
  52.5% exact). `load_profile(path, core=(x, y))` accepts a known core for experiments.
- **Fine rings near the core and compressed rings at the edge of old fish** are lost by a single
  smoothing scale.

## Learned ring detector

`ageingfish/ringclass.py` classifies every position along each reading line as inside an annulus
or not, using 19 scale-free features (multi-scale band contrast and curvature, agreement with the
parallel lines, distance from core and tip, section thickness). Labels come from Thünen's ring
polygons, using only dev-split images with complete annotations (exactly 2 × age polygons) and
only reading lines that cross every one of them. Predicted ring segments are counted per line and
averaged, and each dataset gets a calibration
`age = c0 + c1 × classical count + c2 × learned count`.

The counting rule and calibration are chosen from out-of-fold probabilities (5 folds grouped by
image), so they reflect unseen images; the final classifier is then trained on all dev labels.
Position-level out-of-fold AUC was about 0.88.

```
git clone https://github.com/arjaycc/ai_otolith   # ring annotations
python -m ageingfish.ringclass train --images-north ~/data/datasets_north \
    --images-baltic ~/data/datasets_baltic --annotations ai_otolith --out models/ringclass.joblib
python -m ageingfish.ringclass predict --model models/ringclass.joblib --dataset baltic \
    --split test --images ~/data/datasets_baltic --out results/baltic_test_predictions_ringclass.csv
```

On dev (cross-validated), the classical count alone gave MAE 1.07 / 0.61 years (North / Baltic),
the learned count alone 1.25 / 0.50, and the combination 1.05 / 0.49, which is why both are kept.
The learned detector helps most on Baltic cod and reduces the pull toward the middle there
(1-year-olds predicted 1.6 instead of 2.0), but the North Sea gain is small; old fish are still
under-aged (age 11 predicted about 9.2).
