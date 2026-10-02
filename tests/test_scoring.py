import csv
import math

import pytest

from ageingfish import scoring


def test_perfect_agreement():
    r = scoring.score([1, 2, 3, 3], [1, 2, 3, 3])
    assert r["exact_pct"] == 100 and r["within1_pct"] == 100
    assert r["mean_bias"] == 0 and r["ape_pct"] == 0 and r["cv_pct"] == 0
    assert r["bowker_df"] == 0
    assert r["majority_age"] == 3 and r["majority_exact_pct"] == 50


def test_agreement_bias_and_rounding():
    # 2.5 rounds up to 3 (exact), 4.4 rounds to 4 (off by one), 7 is off by 2.
    r = scoring.score([3, 5, 5], [2.5, 4.4, 7])
    assert r["exact_pct"] == pytest.approx(100 / 3)
    assert r["within1_pct"] == pytest.approx(200 / 3)
    assert r["mean_bias"] == pytest.approx((-0.5 - 0.6 + 2) / 3)
    assert r["mae"] == pytest.approx((0.5 + 0.6 + 2) / 3)


def test_ape_cv_two_readings():
    # For two readings a, b: APE = |a-b|/(a+b), CV = sqrt(2)|a-b|/(a+b).
    ape, cv = scoring.ape_cv([2, 0], [4, 0])
    assert ape == pytest.approx(100 * (2 / 6) / 2)
    assert cv == pytest.approx(100 * (math.sqrt(2) * 2 / 6) / 2)


def test_bowker_detects_one_sided_disagreement():
    ref = [3] * 20
    chi2, df, p = scoring.bowker_test(ref, [4] * 10 + [3] * 10)
    assert df == 1 and chi2 == pytest.approx(10) and p < 0.01
    chi2, df, p = scoring.bowker_test([3] * 5 + [4] * 5, [4] * 5 + [3] * 5)
    assert chi2 == 0 and p == pytest.approx(1)


def test_age_bias_table():
    rows = scoring.age_bias_table([1, 1, 2], [1, 2, 2])
    assert rows[0]["age"] == 1 and rows[0]["n"] == 2 and rows[0]["mean_pred"] == 1.5
    assert rows[0]["ci_low"] < 1.5 < rows[0]["ci_high"]
    assert math.isnan(rows[1]["sd"])


def test_empty_raises():
    with pytest.raises(ValueError):
        scoring.score([], [])


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_cli_with_truth_csv(tmp_path, capsys):
    write_csv(tmp_path / "truth.csv", ["image", "age", "species"],
              [["a.jpg", 2, "walleye"], ["b.jpg", 4, "walleye"], ["c.jpg", 5, "perch"]])
    write_csv(tmp_path / "pred.csv", ["image", "predicted_age"],
              [["a.jpg", 2], ["b.jpg", 5], ["c.jpg", 5], ["extra.jpg", 1]])
    plot = tmp_path / "bias.png"
    code = scoring.main([str(tmp_path / "pred.csv"), "--truth", str(tmp_path / "truth.csv"),
                         "--by-species", "--plot", str(plot)])
    out = capsys.readouterr()
    assert code == 0
    assert "exact agreement      66.7%" in out.out
    assert "walleye" in out.out and "perch" in out.out
    assert "1 predicted image(s) have no reference age" in out.err
    assert plot.stat().st_size > 0


def test_cli_with_thuenen_dataset(tmp_path, capsys):
    from ageingfish.thuenen import load_dataset
    records = load_dataset("north", species="whiting")
    write_csv(tmp_path / "pred.csv", ["image", "predicted_age"], [[r.image, r.age] for r in records])
    assert scoring.main([str(tmp_path / "pred.csv"), "--dataset", "north", "--species", "whiting"]) == 0
    assert "n                    37" in capsys.readouterr().out


def test_cli_no_overlap(tmp_path):
    write_csv(tmp_path / "truth.csv", ["image", "age"], [["a.jpg", 2]])
    write_csv(tmp_path / "pred.csv", ["image", "predicted_age"], [["z.jpg", 2]])
    assert scoring.main([str(tmp_path / "pred.csv"), "--truth", str(tmp_path / "truth.csv")]) == 1
