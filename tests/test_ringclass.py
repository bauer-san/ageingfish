import numpy as np
import pytest

from ageingfish import ringclass, ringcount
from test_ringcount import synthetic_otolith


def ring_boxes(rings, size=(900, 400), half_width=6):
    """Polygons around each dark annulus where it crosses the long axis of synthetic_otolith."""
    w, h = size
    boxes = []
    for j in range(rings):
        for sign in (-1, 1):
            x = w / 2 + sign * (j + 0.5) / rings * 0.42 * w
            boxes.append([(x - half_width, h / 2 - 70), (x + half_width, h / 2 - 70),
                          (x + half_width, h / 2 + 70), (x - half_width, h / 2 + 70)])
    return boxes


def test_count_segments_counts_and_merges():
    prob = np.zeros((2, 1000))
    for centre in (200, 500, 800):
        prob[:, centre - 10:centre + 10] = 0.9
    assert ringclass.count_segments(prob, ringclass.CountParams(threshold=0.5, merge_gap=0.0)) == 3
    prob[:, 515:535] = 0.9  # a second segment 5 px after the one at 500
    assert ringclass.count_segments(prob, ringclass.CountParams(threshold=0.5, merge_gap=0.0, smooth=0.0005)) == 4
    assert ringclass.count_segments(prob, ringclass.CountParams(threshold=0.5, merge_gap=0.01, smooth=0.0005)) == 3


def test_features_shape_and_finite(tmp_path):
    profile = ringcount.load_profile(synthetic_otolith(tmp_path / "o.png", 3), detector="midpoint")
    F = ringclass.features(profile, "north")
    assert F.shape == profile.values.shape + (len(ringclass.FEATURE_NAMES),)
    assert np.isfinite(F).all()


def test_ring_labels_find_each_annotated_ring(tmp_path):
    profile = ringcount.load_profile(synthetic_otolith(tmp_path / "o.png", 3), detector="midpoint")
    labels, crossed = ringclass.ring_labels(profile, ring_boxes(3))
    assert crossed[len(crossed) // 2] == 6
    assert set(np.unique(labels)) == set(range(7))


def test_train_and_predict_on_synthetic_otoliths(tmp_path):
    jobs = []
    for i, age in enumerate([2, 3, 4, 5, 6, 2, 3, 4, 5, 6]):
        path = synthetic_otolith(tmp_path / "o{}.png".format(i), age, seed=i)
        params = ringcount.Params(polarity=-1, edge_trim=0.02, core_trim=0.0, core_detector="midpoint")
        jobs.append((path, "o{}.png".format(i), "north", age, ring_boxes(age), params))
    items = [ringclass._image_item(job) for job in jobs]
    bundle = ringclass.train(items, folds=2)
    assert bundle["datasets"]["north"]["dev_oof"]["mae"] < 0.5
    predicted = [ringclass.predict_age(bundle, it) for it in items]
    assert np.mean(np.abs(np.array(predicted) - [it["age"] for it in items])) < 0.5
