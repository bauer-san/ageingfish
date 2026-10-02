"""Choose ring-counter settings and the age calibration on a development split.

For every combination in GRID, rings are counted on each development image,
a line age = intercept + slope * count is fitted by least squares, and the
combination with the lowest mean absolute error wins. Only "dev" images are
used, so scoring on the "test" split stays an honest held-out estimate.

    python -m ageingfish.tune --dataset north --images DIR --out configs/north.json
    python -m ageingfish.tune --truth labels.csv --images DIR --out configs/walleye.json
"""
import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from . import thuenen
from .ringcount import Params, filtered_signal, find_rings, load_profile, split_sides

GRID = {
    "core_detector": ["thickness", "midpoint"],
    "polarity": [1, -1],
    "detrend": [0.06, 0.1, 0.15, 0.2],
    "smooth": [0.004, 0.008, 0.012],
    "min_gap": [0.015, 0.03],
    "prominence": [0.25, 0.5, 1.0],
    "edge_trim": [0.05, 0.08, 0.12],
    "core_trim": [0.0, 0.03, 0.06],
}
_PROFILE_KEYS = ("core_detector",)
_SIGNAL_KEYS = ("polarity", "detrend", "smooth")
_PEAK_KEYS = ("min_gap", "prominence")
_SIDE_KEYS = ("edge_trim", "core_trim")


def _grid(keys):
    return list(itertools.product(*(GRID[k] for k in keys)))


def _counts(path):
    """Tip-to-tip ring count, averaged over reading lines, for every grid point.

    Shape (profile combos, signal combos, peak combos, side combos). Profiles
    and filtering are the slow parts, so the cheap steps loop inside them.
    """
    profiles = _grid(_PROFILE_KEYS)
    signals, peaks_grid, sides = _grid(_SIGNAL_KEYS), _grid(_PEAK_KEYS), _grid(_SIDE_KEYS)
    out = np.zeros((len(profiles), len(signals), len(peaks_grid), len(sides)))
    for di, (detector,) in enumerate(profiles):
        try:
            profile = load_profile(path, detector=detector)
        except Exception as e:
            return None, repr(e)
        for row in profile.values:
            n = len(row)
            for si, (polarity, detrend, smooth) in enumerate(signals):
                signal, spread = filtered_signal(row, polarity, detrend, smooth)
                for pi, (min_gap, prominence) in enumerate(peaks_grid):
                    peaks = find_rings(signal, spread, min_gap, prominence)
                    for ei, (edge_trim, core_trim) in enumerate(sides):
                        left, right = split_sides(peaks, n, profile.core_index, edge_trim, core_trim)
                        out[di, si, pi, ei] += len(left) + len(right)
        out[di] /= len(profile.values)
    return out, None


def tune(paths, ages, workers=0):
    """Best Params (with calibration) and its development-set metrics."""
    with Pool(workers or os.cpu_count() or 1) as pool:
        results = pool.map(_counts, paths, chunksize=4)
    keep = [i for i, (c, err) in enumerate(results) if c is not None]
    for i, (c, err) in enumerate(results):
        if err:
            print("skipped {}: {}".format(paths[i], err), file=sys.stderr)
    counts = np.array([results[i][0] for i in keep])
    ages = np.asarray(ages, dtype=float)[keep]

    best = None
    profiles = _grid(_PROFILE_KEYS)
    signals, peaks_grid, sides = _grid(_SIGNAL_KEYS), _grid(_PEAK_KEYS), _grid(_SIDE_KEYS)
    for di, si, pi, ei in itertools.product(range(len(profiles)), range(len(signals)),
                                            range(len(peaks_grid)), range(len(sides))):
        total = counts[:, di, si, pi, ei]
        if total.std() == 0:
            continue
        slope, intercept = np.polyfit(total, ages, 1)
        predicted = intercept + slope * total
        mae = float(np.mean(np.abs(predicted - ages)))
        if best is None or mae < best[0]:
            exact = float(np.mean(np.floor(predicted + 0.5) == ages))
            values = dict(zip(_PROFILE_KEYS, profiles[di]), **dict(zip(_SIGNAL_KEYS, signals[si])),
                          **dict(zip(_PEAK_KEYS, peaks_grid[pi])),
                          **dict(zip(_SIDE_KEYS, sides[ei])))
            best = (mae, exact, float(np.corrcoef(total, ages)[0, 1]),
                    Params(intercept=float(intercept), slope=float(slope), **values))
    if best is None:
        raise ValueError("no grid point gave varying ring counts; check the images")
    mae, exact, r, params = best
    return params, {"n": len(keep), "mae": round(mae, 3), "exact_pct": round(100 * exact, 1),
                    "count_age_correlation": round(r, 3)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tune the ring counter on a development split.")
    parser.add_argument("--images", required=True, help="folder of otolith images (searched recursively)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=thuenen.DATASETS, help="tune on this Thünen dataset's dev split")
    source.add_argument("--truth", help="CSV with columns image,age; every listed image is used for tuning")
    parser.add_argument("--out", required=True, help="config JSON to write")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)

    if args.dataset:
        records = [r for r in thuenen.load_dataset(args.dataset, args.images, require_images=True)
                   if thuenen.split_of(r.image) == "dev"]
        paths, ages = [r.path for r in records], [r.age for r in records]
        source = {"dataset": args.dataset, "split": "dev"}
    else:
        from .scoring import read_csv_column
        truth = read_csv_column(args.truth, "age")
        found = thuenen.find_images(args.images)
        names = sorted(n for n in truth if n in found)
        paths, ages = [found[n] for n in names], [int(float(truth[n])) for n in names]
        source = {"truth": os.path.basename(args.truth)}
    if not paths:
        print("error: no labelled images found", file=sys.stderr)
        return 1

    print("tuning on {} images over {} settings...".format(
        len(paths), int(np.prod([len(v) for v in GRID.values()]))))
    params, metrics = tune(paths, ages, args.workers)
    config = {"source": source, "dev_metrics": metrics, "params": vars(params), "grid": GRID}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps({"dev_metrics": metrics, "params": vars(params)}, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
