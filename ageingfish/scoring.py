"""Score predicted ages against reference ages.

Metrics follow common fisheries age-precision practice:
  * exact agreement and agreement within +/-1 year
  * mean bias (predicted - reference), MAE, RMSE
  * APE (Beamish & Fournier 1981) and CV (Chang 1982), treating the prediction
    and the reference as two readings of the same fish
  * an age-bias table: mean predicted age and 95% CI for each reference age
  * Bowker's test of symmetry (Hoenig et al. 1995): small p means disagreements
    lean one way, i.e. systematic over- or under-ageing
"""
import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats


def round_age(value: float) -> int:
    """Round half up, so a prediction of 2.5 counts as 3."""
    return int(math.floor(value + 0.5))


def ape_cv(reference: Sequence[float], predicted: Sequence[float]):
    """Mean APE and CV (%) across fish for two readings per fish."""
    readings = np.column_stack([reference, predicted]).astype(float)
    mean = readings.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ape = np.abs(readings - mean[:, None]).mean(axis=1) / mean
        cv = readings.std(axis=1, ddof=1) / mean
    # Both readings 0 means perfect agreement on an age-0 fish.
    ape = np.where(mean == 0, 0.0, ape)
    cv = np.where(mean == 0, 0.0, cv)
    return 100 * ape.mean(), 100 * cv.mean()


def bowker_test(reference: Sequence[int], predicted: Sequence[int]):
    """Bowker's symmetry test on the reference x predicted age table.

    Returns (chi2, df, p); (nan, 0, nan) when there are no disagreements.
    """
    counts = Counter(zip(reference, predicted))
    ages = sorted(set(reference) | set(predicted))
    chi2, df = 0.0, 0
    for i, a in enumerate(ages):
        for b in ages[i + 1:]:
            n_ab, n_ba = counts[(a, b)], counts[(b, a)]
            if n_ab + n_ba > 0:
                chi2 += (n_ab - n_ba) ** 2 / (n_ab + n_ba)
                df += 1
    if df == 0:
        return float("nan"), 0, float("nan")
    return chi2, df, float(stats.chi2.sf(chi2, df))


def age_bias_table(reference: Sequence[int], predicted: Sequence[float]) -> List[dict]:
    """Per reference age: n, mean predicted age, SD and 95% t-interval."""
    by_age = defaultdict(list)
    for ref, pred in zip(reference, predicted):
        by_age[ref].append(pred)
    rows = []
    for age in sorted(by_age):
        values = np.asarray(by_age[age], dtype=float)
        n = len(values)
        mean = values.mean()
        sd = values.std(ddof=1) if n > 1 else float("nan")
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
        rows.append({"age": age, "n": n, "mean_pred": mean, "sd": sd,
                     "ci_low": mean - half, "ci_high": mean + half})
    return rows


def score(reference: Sequence[int], predicted: Sequence[float]) -> dict:
    """All summary metrics for paired reference and predicted ages."""
    if len(reference) == 0:
        raise ValueError("nothing to score: no images have both a reference and a predicted age")
    ref = np.asarray(reference, dtype=int)
    pred = np.asarray(predicted, dtype=float)
    pred_int = np.array([round_age(p) for p in pred])
    diff = pred - ref
    ape, cv = ape_cv(ref, pred)
    chi2, df, p = bowker_test(list(ref), list(pred_int))
    majority_age, majority_n = Counter(ref.tolist()).most_common(1)[0]
    return {
        "n": len(ref),
        "exact_pct": 100 * np.mean(pred_int == ref),
        "within1_pct": 100 * np.mean(np.abs(pred_int - ref) <= 1),
        "mean_bias": diff.mean(),
        "mae": np.abs(diff).mean(),
        "rmse": math.sqrt(np.mean(diff ** 2)),
        "ape_pct": ape,
        "cv_pct": cv,
        "bowker_chi2": chi2,
        "bowker_df": df,
        "bowker_p": p,
        # Always predicting the most common reference age; computed on the scored
        # set itself, so it is an optimistic floor that any method should beat.
        "majority_age": majority_age,
        "majority_exact_pct": 100 * majority_n / len(ref),
        "age_bias": age_bias_table(ref.tolist(), pred.tolist()),
    }


def read_csv_column(path: str, value_column: str, key_column: str = "image") -> Dict[str, str]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = {key_column, value_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError("{} is missing column(s): {}".format(path, ", ".join(sorted(missing))))
        return {row[key_column]: row[value_column] for row in reader if row[value_column] not in ("", None)}


def format_report(result: dict, title: str = "") -> str:
    lines = []
    if title:
        lines += [title, "-" * len(title)]
    lines += [
        "n                    {}".format(result["n"]),
        "exact agreement      {:.1f}%".format(result["exact_pct"]),
        "within +/-1 year     {:.1f}%".format(result["within1_pct"]),
        "mean bias (pred-ref) {:+.2f} years".format(result["mean_bias"]),
        "MAE / RMSE           {:.2f} / {:.2f} years".format(result["mae"], result["rmse"]),
        "APE / CV             {:.1f}% / {:.1f}%".format(result["ape_pct"], result["cv_pct"]),
    ]
    if result["bowker_df"]:
        lines.append("Bowker symmetry      chi2={:.2f}, df={}, p={:.3g}".format(
            result["bowker_chi2"], result["bowker_df"], result["bowker_p"]))
    else:
        lines.append("Bowker symmetry      no disagreements")
    lines.append("baseline: always {}  {:.1f}% exact".format(result["majority_age"], result["majority_exact_pct"]))
    lines += ["", "age   n   mean pred   95% CI"]
    for row in result["age_bias"]:
        ci = "" if math.isnan(row["ci_low"]) else "[{:.2f}, {:.2f}]".format(row["ci_low"], row["ci_high"])
        lines.append("{:>3} {:>3}   {:>9.2f}   {}".format(row["age"], row["n"], row["mean_pred"], ci))
    return "\n".join(lines)


def plot_age_bias(result: dict, path: str, title: str = "Age-bias plot") -> None:
    """Mean predicted age (95% CI) against reference age, with a 1:1 line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["age_bias"]
    ages = [r["age"] for r in rows]
    means = [r["mean_pred"] for r in rows]
    err = [[0 if math.isnan(r["ci_low"]) else r["mean_pred"] - r["ci_low"] for r in rows],
           [0 if math.isnan(r["ci_high"]) else r["ci_high"] - r["mean_pred"] for r in rows]]
    hi = max(max(ages), max(r["ci_high"] if not math.isnan(r["ci_high"]) else r["mean_pred"] for r in rows)) + 1
    lo = min(0, min(ages) - 1)

    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot([lo, hi], [lo, hi], color="#a8a7a2", lw=1, ls="--", zorder=1)
    ax.errorbar(ages, means, yerr=err, fmt="o", color="#2a78d6", ms=6, lw=2, capsize=0, zorder=2)
    for age, mean, row in zip(ages, means, rows):
        low = mean if math.isnan(row["ci_low"]) else row["ci_low"]
        ax.annotate("n={}".format(row["n"]), (age, low), xytext=(0, -6), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color="#52514e")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Reference age (years)", color="#0b0b0b")
    ax.set_ylabel("Predicted age (years), mean and 95% CI", color="#0b0b0b")
    ax.set_title("{}\nexact {:.0f}%, within 1 year {:.0f}%, bias {:+.2f}".format(
        title, result["exact_pct"], result["within1_pct"], result["mean_bias"]), fontsize=10, color="#0b0b0b")
    ax.grid(color="#e8e7e3", lw=0.6)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score predicted ages (CSV with columns image,predicted_age) against reference ages.")
    parser.add_argument("predictions", help="CSV with columns image,predicted_age")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=["north", "baltic"], help="use Thünen reference ages")
    source.add_argument("--truth", help="CSV with columns image,age (e.g. your own labels.csv)")
    parser.add_argument("--species", help="score only this species (north: cod, saithe, haddock, whiting)")
    parser.add_argument("--split", choices=["dev", "test"],
                        help="with --dataset: score only this split (see thuenen.split_of)")
    parser.add_argument("--by-species", action="store_true", help="also report each species separately")
    parser.add_argument("--plot", help="write an age-bias plot to this PNG")
    args = parser.parse_args(argv)
    if args.split and not args.dataset:
        parser.error("--split needs --dataset")

    predictions = {k: float(v) for k, v in read_csv_column(args.predictions, "predicted_age").items()}
    if args.dataset:
        from .thuenen import load_dataset, split_of
        records = load_dataset(args.dataset, species=args.species)
        if args.split:
            records = [r for r in records if split_of(r.image) == args.split]
        truth = {r.image: r.age for r in records}
        species = {r.image: r.species for r in records}
    else:
        rows = read_csv_column(args.truth, "age")
        truth = {k: int(float(v)) for k, v in rows.items()}
        species = {}
        if args.species or args.by_species:
            try:
                species = read_csv_column(args.truth, "species")
            except ValueError:
                parser.error("--species/--by-species need a 'species' column in --truth")
            if args.species:
                truth = {k: v for k, v in truth.items() if species.get(k) == args.species}

    images = sorted(set(truth) & set(predictions))
    unscored = len(set(predictions) - set(truth))
    unpredicted = len(set(truth) - set(predictions))
    if unscored:
        print("note: {} predicted image(s) have no reference age and were skipped".format(unscored), file=sys.stderr)
    if unpredicted:
        print("note: {} reference image(s) have no prediction".format(unpredicted), file=sys.stderr)
    if not images:
        print("error: no image names in common between predictions and reference ages", file=sys.stderr)
        return 1

    result = score([truth[i] for i in images], [predictions[i] for i in images])
    title = "Thünen {}".format(args.dataset) if args.dataset else "reference: {}".format(args.truth)
    if args.dataset and args.split:
        title += " {} split".format(args.split)
    if args.species:
        title += " ({})".format(args.species)
    print(format_report(result, title))

    if args.by_species and species:
        for sp in sorted({species[i] for i in images if i in species}):
            subset = [i for i in images if species.get(i) == sp]
            print()
            print(format_report(score([truth[i] for i in subset], [predictions[i] for i in subset]), sp))

    if args.plot:
        plot_age_bias(result, args.plot, title)
        print("\nwrote {}".format(args.plot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
