"""Learned ring detector: classify each position along a reading line as inside
an annulus or not, then count the predicted ring segments.

Trained on Thünen's ring annotations (closed polygons around each annotated
zone) for development-split images. Features are scale-free, built from the
same reading-line profiles as ageingfish.ringcount.

    python -m ageingfish.ringclass train --images-north DIR --images-baltic DIR \
        --annotations AI_OTOLITH_CLONE --out models/ringclass.joblib
    python -m ageingfish.ringclass predict --model models/ringclass.joblib --dataset north \
        --split test --images DIR --out predictions.csv
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import ndimage

from .ringcount import OFFSETS, Profile, filtered_signal, load_profile

SCALES = (0.002, 0.004, 0.008, 0.016)  # smoothing sigmas, fraction of otolith length
DETREND = 0.1
DATASET_CODES = {"north": 0, "baltic": 1}
FEATURE_NAMES = (
    [f"signal_s{s}" for s in SCALES]
    + [f"d1_s{s}" for s in SCALES[1:3]] + [f"d2_s{s}" for s in SCALES[1:3]]
    + ["peakness_w02", "peakness_w04", "raw", "other_lines", "other_lines_d2",
       "dist_core", "dist_tip", "rel_pos", "thickness", "line_offset", "dataset"]
)


def _norm(signal: np.ndarray) -> np.ndarray:
    spread = 1.4826 * np.median(np.abs(signal - np.median(signal))) or 1.0
    return (signal - np.median(signal)) / spread


def features(profile: Profile, dataset: str) -> np.ndarray:
    """Per-position features, shape (lines, positions, len(FEATURE_NAMES))."""
    k, n = profile.values.shape
    smoothed = np.array([[filtered_signal(row, 1, DETREND, s)[0] for s in SCALES] for row in profile.values])
    smoothed = np.array([[_norm(sig) for sig in line] for line in smoothed])  # (k, scales, n)
    pos = np.arange(n)
    core = profile.core_index
    dist_core = np.abs(pos - core) / n
    side_len = np.where(pos < core, core, n - 1 - core).clip(min=1)
    rel = np.abs(pos - core) / side_len
    dist_tip = np.minimum(pos, n - 1 - pos) / n
    thickness = profile.thickness / (profile.thickness.max() or 1)
    out = np.zeros((k, n, len(FEATURE_NAMES)))
    mid_scale = smoothed[:, 1]
    for li in range(k):
        s = smoothed[li]
        f = list(s)
        for si in (1, 2):
            f.append(np.gradient(s[si]) * n * SCALES[si])
        for si in (1, 2):
            f.append(np.gradient(np.gradient(s[si])) * (n * SCALES[si]) ** 2)
        for w in (0.02, 0.04):
            size = max(3, int(w * n))
            lo = ndimage.minimum_filter1d(s[1], size)
            hi = ndimage.maximum_filter1d(s[1], size)
            f.append((s[1] - lo) / ((hi - lo) + 1e-6))
        f.append(_norm(profile.values[li]))
        others = np.delete(mid_scale, li, axis=0)
        f.append(others.mean(0))
        f.append(np.gradient(np.gradient(others.mean(0))) * (n * SCALES[1]) ** 2)
        f += [dist_core, dist_tip, rel, thickness, np.full(n, abs(OFFSETS[li])),
              np.full(n, DATASET_CODES.get(dataset, -1))]
        out[li] = np.column_stack(f)
    return out


def ring_labels(profile: Profile, polygons: Sequence[Sequence[tuple]]):
    """Annotated-ring id (1-based, 0 = none) at each position of each line, and
    the number of distinct rings each line crosses. Polygons are in original
    image pixels."""
    from matplotlib.path import Path
    labels = np.zeros(profile.values.shape, dtype=int)
    for ring_id, poly in enumerate(polygons, start=1):
        px = np.array([p[0] for p in poly], float)
        py = np.array([p[1] for p in poly], float)
        wx, wy = profile.work.to_work(px, py)
        path = Path(np.column_stack([wx, wy]))
        for li, ys in enumerate(profile.lines):
            labels[li][path.contains_points(np.column_stack([profile.x, ys]))] = ring_id
    crossed = np.array([len(set(row[row > 0])) for row in labels])
    return labels, crossed


@dataclass
class CountParams:
    threshold: float = 0.5   # probability above which a position is "in a ring"
    min_len: float = 0.004   # shortest ring segment, fraction of length
    merge_gap: float = 0.004 # segments closer than this merge, fraction of length
    smooth: float = 0.002    # Gaussian sigma applied to probabilities, fraction of length


def count_segments(prob: np.ndarray, p: CountParams) -> float:
    """Ring segments per line averaged over lines; prob has shape (lines, positions)."""
    counts = []
    for row in prob:
        n = len(row)
        row = ndimage.gaussian_filter1d(row, max(p.smooth * n, 0.5))
        on = row > p.threshold
        gap = int(p.merge_gap * n)
        if gap > 0:
            on = ndimage.binary_closing(on, structure=np.ones(2 * gap + 1))
        labels, k = ndimage.label(on)
        if k == 0:
            counts.append(0)
            continue
        sizes = ndimage.sum(on, labels, range(1, k + 1))
        counts.append(int(np.sum(sizes >= max(1, p.min_len * n))))
    return float(np.mean(counts))


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------

COUNT_GRID = {
    "threshold": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    "min_len": [0.002, 0.005],
    "merge_gap": [0.0, 0.005, 0.01],
    "smooth": [0.001, 0.003],
}
FEATURE_DETECTOR = "midpoint"  # core rule used for the learned features


def _classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=0)


def _training_rows(items, step=3):
    """Positions from lines whose annotation is complete: the image has exactly
    2 x age polygons and the line crosses all of them."""
    X, y = [], []
    for it in items:
        if it["labels"] is None or it["n_poly"] != 2 * it["age"]:
            continue
        for li, crossed in enumerate(it["crossed"]):
            if crossed == 2 * it["age"]:
                X.append(it["F"][li, ::step])
                y.append(it["labels"][li, ::step] > 0)
    if not X:
        raise ValueError("no completely annotated reading lines to train on")
    return np.concatenate(X), np.concatenate(y)


def _probabilities(model, F):
    return model.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(F.shape[:2])


def _image_item(job):
    path, image, dataset, age, polygons, classical_params = job
    try:
        profile = load_profile(path, detector=FEATURE_DETECTOR)
        F = features(profile, dataset).astype(np.float32)
        labels, crossed = ring_labels(profile, polygons) if polygons else (None, None)
        classical = _classical_count(path, profile, classical_params)
    except Exception as e:
        return dict(image=image, error=repr(e))
    return dict(image=image, path=path, ds=dataset, age=age, F=F, labels=labels, crossed=crossed,
                n_poly=len(polygons or []), classical=classical)


def _classical_count(path, profile, params):
    from .ringcount import ring_count
    if params is None:
        return float("nan")
    if params.core_detector != FEATURE_DETECTOR:
        profile = load_profile(path, detector=params.core_detector)
    return ring_count(profile, params)


def _load_items(jobs, workers=0):
    from multiprocessing import Pool
    import sys
    with Pool(workers or os.cpu_count() or 1) as pool:
        items = pool.map(_image_item, jobs, chunksize=4)
    for it in items:
        if "error" in it:
            print("skipped {}: {}".format(it["image"], it["error"]), file=sys.stderr)
    return [it for it in items if "error" not in it]


def _fit_stack(classical, learned, ages):
    X = np.column_stack([np.ones(len(ages)), classical, learned])
    coef = np.linalg.lstsq(X, ages, rcond=None)[0]
    return [float(c) for c in coef]


def train(items: List[dict], folds: int = 5) -> dict:
    """Fit the ring classifier and, per dataset, the counting rule and the age
    calibration age = c0 + c1 * classical_count + c2 * learned_count.

    Counting rule and calibration are chosen from out-of-fold probabilities
    (folds grouped by image), so they reflect performance on unseen images.
    """
    import itertools
    from sklearn.model_selection import GroupKFold

    order = np.arange(len(items))
    oof = [None] * len(items)
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(order, groups=order):
        model = _classifier().fit(*_training_rows([items[i] for i in train_idx]))
        for i in test_idx:
            oof[i] = _probabilities(model, items[i]["F"])

    per_dataset = {}
    for ds in sorted({it["ds"] for it in items}):
        sel = [i for i, it in enumerate(items) if it["ds"] == ds]
        ages = np.array([items[i]["age"] for i in sel], float)
        classical = np.array([items[i]["classical"] for i in sel])
        best = None
        for values in itertools.product(*COUNT_GRID.values()):
            cp = CountParams(**dict(zip(COUNT_GRID, values)))
            learned = np.array([count_segments(oof[i], cp) for i in sel])
            if learned.std() == 0:
                continue
            coef = _fit_stack(classical, learned, ages)
            pred = coef[0] + coef[1] * classical + coef[2] * learned
            mae = float(np.mean(np.abs(pred - ages)))
            if best is None or mae < best[0]:
                best = (mae, float(np.mean(np.floor(pred + 0.5) == ages)), cp, coef)
        mae, exact, cp, coef = best
        per_dataset[ds] = {"count": vars(cp), "stack": coef,
                           "dev_oof": {"n": len(sel), "mae": round(mae, 3), "exact_pct": round(100 * exact, 1)}}

    model = _classifier().fit(*_training_rows(items))
    return {"model": model, "datasets": per_dataset, "feature_names": list(FEATURE_NAMES),
            "feature_detector": FEATURE_DETECTOR}


def predict_age(bundle: dict, item: dict) -> float:
    cfg = bundle["datasets"][item["ds"]]
    learned = count_segments(_probabilities(bundle["model"], item["F"]), CountParams(**cfg["count"]))
    c0, c1, c2 = cfg["stack"]
    return c0 + c1 * item["classical"] + c2 * learned


def main(argv=None) -> int:
    import argparse
    import csv
    import json
    import sys

    import joblib

    from . import thuenen
    from .ringcount import Params

    parser = argparse.ArgumentParser(description="Train or apply the learned ring detector.")
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="train on the dev split of the Thünen datasets")
    tr.add_argument("--images-north", help="folder with the North Sea images")
    tr.add_argument("--images-baltic", help="folder with the Baltic images")
    tr.add_argument("--annotations", required=True, help="clone of github.com/arjaycc/ai_otolith")
    tr.add_argument("--configs", default="configs", help="folder with classical-counter configs <dataset>.json")
    tr.add_argument("--out", required=True, help="model file to write (.joblib)")
    tr.add_argument("--workers", type=int, default=0)
    pr = sub.add_parser("predict", help="write predicted ages for a Thünen dataset")
    pr.add_argument("--model", required=True)
    pr.add_argument("--dataset", required=True, choices=thuenen.DATASETS)
    pr.add_argument("--split", choices=["dev", "test", "all"], default="test")
    pr.add_argument("--images", required=True)
    pr.add_argument("--configs", default="configs")
    pr.add_argument("--out", required=True)
    pr.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)

    if args.command == "train":
        roots = {"north": args.images_north, "baltic": args.images_baltic}
        annotations = thuenen.load_ring_annotations(args.annotations)
        jobs = []
        for ds, root in roots.items():
            if not root:
                continue
            params = Params.load(os.path.join(args.configs, ds + ".json"))
            for r in thuenen.load_dataset(ds, root, require_images=True):
                if thuenen.split_of(r.image) == "dev":
                    jobs.append((r.path, r.image, ds, r.age, annotations.get(r.image), params))
        if not jobs:
            parser.error("give --images-north and/or --images-baltic")
        print("loading {} dev images...".format(len(jobs)))
        bundle = train(_load_items(jobs, args.workers))
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        joblib.dump(bundle, args.out, compress=3)
        print(json.dumps(bundle["datasets"], indent=2))
        print("wrote", args.out)
        return 0

    bundle = joblib.load(args.model)
    if args.dataset not in bundle["datasets"]:
        parser.error("model was not trained for dataset {}".format(args.dataset))
    params = Params.load(os.path.join(args.configs, args.dataset + ".json"))
    records = thuenen.load_dataset(args.dataset, args.images, require_images=True)
    if args.split != "all":
        records = [r for r in records if thuenen.split_of(r.image) == args.split]
    items = _load_items([(r.path, r.image, args.dataset, r.age, None, params) for r in records], args.workers)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "predicted_age"])
        for it in items:
            writer.writerow([it["image"], "{:.3f}".format(predict_age(bundle, it))])
    print("wrote {} predictions to {}".format(len(items), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
