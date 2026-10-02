# ageingfish

Estimating fish age by counting annuli in otolith images.

- `DataExploration_FishImages.ipynb`: early exploration on three walleye otolith sections (`*.jpg`).
- `ageingfish/thuenen.py`: loader for the Thünen Institute's labelled otolith datasets.
- `ageingfish/scoring.py`: scores predicted ages against reference ages.
- `ageingfish/ringcount.py`: classical annulus counter (no machine learning) that writes predicted ages.
- `ageingfish/tune.py`: picks the counter's settings and age calibration on a development split.

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
`results/`.

| Test split | n | Exact | Within ±1 year | MAE | Mean bias | Baseline exact (always modal age) |
|---|---|---|---|---|---|---|
| North Sea, all | 462 | 31.8% | 76.6% | 1.07 y | −0.01 y | 13.6% |
| saithe | 253 | 36.4% | 85.4% | 0.86 y | −0.10 y | 16.6% |
| cod | 131 | 30.5% | 69.5% | 1.24 y | −0.16 y | 16.0% |
| haddock | 54 | 18.5% | 63.0% | 1.44 y | +0.43 y | 18.5% |
| whiting | 24 | 20.8% | 54.2% | 1.43 y | +0.91 y | 20.8% |
| Baltic cod | 804 | 52.0% | 92.3% | 0.64 y | +0.00 y | 28.2% |

Dev-split results were similar (North MAE 1.06, Baltic 0.60), so the tuning did not overfit.
For comparison, Thünen's deep-learning models reached about 72% mean accuracy on the Baltic set
(Sigurðardóttir et al. 2024), under a different evaluation setup.

Known weaknesses, visible in `results/examples/` and the age-bias plots:

- **Pull toward the middle.** The calibration shrinks toward the mean age: Baltic 1-year-olds are
  predicted about 2.0 and 5-year-olds about 3.7 (Bowker's test p < 1e-30). Better ring evidence,
  not a different calibration, is the fix.
- **Core position.** The core is taken as the thickest part of the section, which can miss the true core
  (see `results/examples/north_example.png`), and the midline can pass beside the core rather than through it.
- **Fine rings near the core and compressed rings at the edge of old fish** are lost by a single
  smoothing scale.
