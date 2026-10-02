import csv
import json

import numpy as np
import pytest
from PIL import Image

from ageingfish import ringcount, tune


def synthetic_otolith(path, rings, size=(900, 400), seed=0):
    """Elliptical section on a dark background with `rings` dark annuli per side."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (0.42 * w)) ** 2 + ((yy - h / 2) / (0.38 * h)) ** 2)
    image = np.where(r <= 1, 120 + 60 * np.cos(2 * np.pi * rings * r), 8)
    image = image + np.random.default_rng(seed).normal(0, 4, (h, w))
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert("RGB").save(path)
    return str(path)


@pytest.mark.parametrize("rings", [2, 4, 7])
def test_midline_counts_each_annulus_once_per_side(tmp_path, rings):
    profile = ringcount.load_profile(synthetic_otolith(tmp_path / "o.png", rings))
    left, right = ringcount.ring_peaks(profile, ringcount.Params(polarity=-1, edge_trim=0.02, core_trim=0.0))
    assert (len(left), len(right)) == (rings, rings)


def test_core_found_near_centre(tmp_path):
    profile = ringcount.load_profile(synthetic_otolith(tmp_path / "o.png", 3))
    assert abs(profile.core_index - len(profile.values[0]) / 2) < 0.05 * len(profile.values[0])


def test_resolution_independent(tmp_path):
    small = ringcount.load_profile(synthetic_otolith(tmp_path / "s.png", 4, size=(600, 270)))
    large = ringcount.load_profile(synthetic_otolith(tmp_path / "l.png", 4, size=(2400, 1070)))
    p = ringcount.Params(polarity=-1, edge_trim=0.02, core_trim=0.0)
    assert abs(len(small.values[0]) - len(large.values[0])) < 20
    assert ringcount.ring_count(small, p) == pytest.approx(ringcount.ring_count(large, p), abs=1)


def test_estimate_age_applies_calibration(tmp_path):
    profile = ringcount.load_profile(synthetic_otolith(tmp_path / "o.png", 4))
    p = ringcount.Params(polarity=-1, edge_trim=0.02, core_trim=0.0, intercept=1.0, slope=0.25)
    assert ringcount.estimate_age(profile, p) == pytest.approx(1.0 + 0.25 * ringcount.ring_count(profile, p))


def test_no_otolith_raises():
    with pytest.raises(ValueError):
        ringcount.segment(np.zeros((50, 50)))


def test_params_load(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"params": {"polarity": -1, "slope": 0.9, "intercept": -2}}))
    p = ringcount.Params.load(str(path))
    assert (p.polarity, p.slope, p.intercept, p.detrend) == (-1, 0.9, -2, ringcount.Params().detrend)


def test_cli_writes_predictions_and_skips_bad_images(tmp_path, capsys):
    images = tmp_path / "imgs"
    images.mkdir()
    synthetic_otolith(images / "a.png", 3)
    Image.fromarray(np.zeros((40, 40), dtype=np.uint8)).save(images / "blank.png")
    out = tmp_path / "pred.csv"
    assert ringcount.main(["--images", str(images), "--out", str(out), "--workers", "1",
                           "--plots", str(tmp_path / "plots"), "--n-plots", "1"]) == 0
    rows = list(csv.DictReader(open(out)))
    assert [r["image"] for r in rows] == ["a.png"]
    assert float(rows[0]["ring_count"]) > 0
    assert "skipped" in capsys.readouterr().err
    assert (tmp_path / "plots" / "a_rings.png").exists()


def test_tune_recovers_ring_age_relation(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "GRID", {"core_detector": ["midpoint"], "polarity": [1, -1], "detrend": [0.1], "smooth": [0.004],
                                       "min_gap": [0.015], "prominence": [0.5], "edge_trim": [0.02],
                                       "core_trim": [0.0]})
    ages = [2, 3, 4, 5, 6]
    paths = [synthetic_otolith(tmp_path / "o{}.png".format(a), a, seed=a) for a in ages]
    params, metrics = tune.tune(paths, ages, workers=1)
    assert params.slope > 0
    assert metrics["mae"] < 0.3 and metrics["exact_pct"] == 100


def test_to_work_maps_original_points_through_resize_and_rotation(tmp_path):
    # A tilted bright ellipse with a dark spot; the spot must land on the spot after prepare().
    h, w = 500, 1200
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.radians(20)
    u = (xx - 600) * np.cos(a) + (yy - 250) * np.sin(a)
    v = -(xx - 600) * np.sin(a) + (yy - 250) * np.cos(a)
    image = np.where((u / 450) ** 2 + (v / 150) ** 2 <= 1, 200.0, 5.0)
    spot = (780, 300)
    image[(xx - spot[0]) ** 2 + (yy - spot[1]) ** 2 <= 36] = 0
    path = tmp_path / "tilted.png"
    Image.fromarray(image.astype(np.uint8)).save(path)

    work = ringcount.prepare(str(path))
    x, y = work.to_work(*spot)
    inside = ndimage_dark_centre(work.gray, work.mask)
    assert abs(x - inside[0]) < 3 and abs(y - inside[1]) < 3


def ndimage_dark_centre(gray, mask):
    from scipy import ndimage
    dark = (gray < 100) & ndimage.binary_erosion(mask, iterations=3)
    ys, xs = np.nonzero(dark)
    return xs.mean(), ys.mean()


def test_known_core_overrides_detector(tmp_path):
    path = synthetic_otolith(tmp_path / "o.png", 3)
    auto = ringcount.load_profile(path, detector="midpoint")
    # 150 px from the centre in original pixels; the image is resized ~1000/756
    # and may be rotated 180 degrees, so check the distance moved, not the side.
    moved = ringcount.load_profile(path, core=(300, 200))
    assert 170 < abs(moved.core_index - auto.core_index) < 230
