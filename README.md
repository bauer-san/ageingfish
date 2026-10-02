# ageingfish

Estimating fish age by counting annuli in otolith images.

- `DataExploration_FishImages.ipynb`: early exploration on three walleye otolith sections (`*.jpg`).
- `ageingfish/thuenen.py`: loader for the Thünen Institute's labelled otolith datasets.
- `ageingfish/scoring.py`: scores predicted ages against reference ages.

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
