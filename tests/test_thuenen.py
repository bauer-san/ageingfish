import json

import pytest

from ageingfish import thuenen


def test_north_labels_match_published_counts():
    records = thuenen.load_labels("north")
    assert len(records) == 660
    counts = {sp: sum(r.species == sp for r in records) for sp in ("cod", "saithe", "haddock", "whiting")}
    assert counts == {"cod": 194, "saithe": 351, "haddock": 78, "whiting": 37}
    assert {r.age for r in records} == set(range(1, 12))


def test_baltic_labels_match_published_counts():
    records = thuenen.load_labels("baltic")
    assert len(records) == 1155
    assert {r.age for r in records} == {1, 2, 3, 4, 5}
    assert {r.species for r in records} == {"cod"}


def test_age_from_north_filename():
    assert thuenen.age_from_north_filename("20160621victoria_q2_4_st1_id1_age_2.png") == 2
    assert thuenen.age_from_north_filename("some/dir/x_age_11.png") == 11
    with pytest.raises(ValueError):
        thuenen.age_from_north_filename("16_716_104.png")


def test_unknown_dataset():
    with pytest.raises(ValueError):
        thuenen.load_labels("irish")


def test_images_found_recursively_and_filtered(tmp_path):
    name = thuenen.load_labels("north")[0].image
    nested = tmp_path / "datasets_north" / "images"
    nested.mkdir(parents=True)
    (nested / name).write_bytes(b"")
    records = thuenen.load_dataset("north", str(tmp_path))
    found = [r for r in records if r.path]
    assert [r.image for r in found] == [name]
    assert len(thuenen.load_dataset("north", str(tmp_path), require_images=True)) == 1


def test_species_filter():
    assert len(thuenen.load_dataset("north", species="haddock")) == 78


def test_ring_annotations_handle_list_and_dict_regions(tmp_path):
    region = {"shape_attributes": {"name": "polyline", "all_points_x": [1, 2, 3], "all_points_y": [4, 5, 6]},
              "region_attributes": {}}
    (tmp_path / "a.json").write_text(json.dumps({
        "x": {"filename": "a.png", "regions": [region, region]},
        "y": {"filename": "b.png", "regions": {"0": region}},
        "z": {"filename": "c.png", "regions": []},
    }))
    (tmp_path / "not_via.json").write_text(json.dumps([1, 2]))
    (tmp_path / "broken.json").write_text("{")
    ann = thuenen.load_ring_annotations(str(tmp_path))
    assert ann == {"a.png": [[(1, 4), (2, 5), (3, 6)]] * 2, "b.png": [[(1, 4), (2, 5), (3, 6)]]}


def test_write_labels_csv(tmp_path):
    out = tmp_path / "labels.csv"
    thuenen.write_labels_csv(thuenen.load_dataset("north", species="whiting"), str(out))
    lines = out.read_text().splitlines()
    assert lines[0] == "image,age,species,dataset,path"
    assert len(lines) == 38
