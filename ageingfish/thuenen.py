"""Loader for the Thünen Institute otolith datasets (North Sea and Baltic Sea).

Ages come from the label files vendored in ``data/thuenen`` (North Sea ages are
also encoded in the filenames), so labels are available without the images.
Images are matched by filename anywhere under the folder you unzipped them to.
"""
import csv
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "thuenen")

DATASETS = ("north", "baltic")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# The source labels use German common names for two species.
SPECIES_NAMES = {"cod": "cod", "saithe": "saithe", "schellfisch": "haddock", "wittling": "whiting"}

_NORTH_AGE = re.compile(r"_age_(\d+)\.[A-Za-z]+$")


@dataclass
class OtolithRecord:
    image: str                  # filename, the key used everywhere else
    age: int                    # age read by Thünen readers
    dataset: str                # "north" or "baltic"
    species: str                # "cod", "saithe", "haddock", "whiting"
    path: Optional[str] = None  # full path if the image was found on disk
    rings: List = field(default_factory=list)  # optional annotation polygons


def age_from_north_filename(filename: str) -> int:
    match = _NORTH_AGE.search(os.path.basename(filename))
    if not match:
        raise ValueError("no '_age_<N>' in North Sea filename: {}".format(filename))
    return int(match.group(1))


def load_labels(dataset: str) -> List[OtolithRecord]:
    """Return every labelled image in a dataset, without touching the images."""
    if dataset == "north":
        with open(os.path.join(DATA_DIR, "north_species_map.json")) as f:
            species_map = json.load(f)
        records = [OtolithRecord(name, age_from_north_filename(name), "north", SPECIES_NAMES[sp])
                   for name, sp in species_map.items()]
    elif dataset == "baltic":
        with open(os.path.join(DATA_DIR, "baltic_age_map.json")) as f:
            age_map = json.load(f)
        # The Baltic set is all cod (Thünen Institute of Baltic Sea Fisheries).
        records = [OtolithRecord(name, int(age), "baltic", "cod") for name, age in age_map.items()]
    else:
        raise ValueError("dataset must be one of {}, got {!r}".format(DATASETS, dataset))
    return sorted(records, key=lambda r: r.image)


def split_of(image: str) -> str:
    """Fixed ~30/70 split by filename hash: "dev" (tune here) or "test" (report here).

    Hashing keeps the split stable as files are added and independent of order.
    All development of ageingfish.ringcount looked only at "dev" images.
    """
    bucket = int(hashlib.md5(os.path.basename(image).encode()).hexdigest(), 16) % 10
    return "dev" if bucket < 3 else "test"


def find_images(image_root: str) -> Dict[str, str]:
    """Map filename -> path for every image under image_root (searched recursively)."""
    found = {}
    for path in glob.glob(os.path.join(image_root, "**", "*"), recursive=True):
        if path.lower().endswith(IMAGE_EXTENSIONS):
            found.setdefault(os.path.basename(path), path)
    return found


def load_dataset(dataset: str, image_root: Optional[str] = None, species: Optional[str] = None,
                 require_images: bool = False) -> List[OtolithRecord]:
    """Labels for a dataset, with image paths filled in when image_root is given.

    species filters by common name ("cod", "saithe", "haddock", "whiting").
    require_images drops records whose image was not found under image_root.
    """
    records = load_labels(dataset)
    if species is not None:
        records = [r for r in records if r.species == species]
    if image_root is not None:
        paths = find_images(image_root)
        for r in records:
            r.path = paths.get(r.image)
    if require_images:
        records = [r for r in records if r.path is not None]
    return records


def load_ring_annotations(annotation_root: str) -> Dict[str, List[List[tuple]]]:
    """Read the Thünen VIA annotation JSON files under annotation_root.

    Returns filename -> list of polygons, each a list of (x, y) points. These are
    the closed polylines Thünen used as segmentation targets; an image can hold
    two otoliths, so the polygon count is not the age. When a filename appears
    in several files (the training splits repeat images), the first is kept.
    """
    annotations = {}
    for path in sorted(glob.glob(os.path.join(annotation_root, "**", "*.json"), recursive=True)):
        try:
            with open(path) as f:
                content = json.load(f)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(content, dict):
            continue
        for entry in content.values():
            if not isinstance(entry, dict) or "filename" not in entry or "regions" not in entry:
                continue
            regions = entry["regions"]
            regions = list(regions.values()) if isinstance(regions, dict) else regions
            polygons = [list(zip(r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"]))
                        for r in regions if "all_points_x" in r.get("shape_attributes", {})]
            if polygons and entry["filename"] not in annotations:
                annotations[entry["filename"]] = polygons
    return annotations


def write_labels_csv(records: List[OtolithRecord], path: str) -> None:
    """Write records as image,age,species,dataset,path for use with the scoring script."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "age", "species", "dataset", "path"])
        for r in records:
            writer.writerow([r.image, r.age, r.species, r.dataset, r.path or ""])


def main(argv=None):
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description="Summarise a Thünen dataset and optionally export its labels.")
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--images", help="folder the Zenodo zip was unzipped to")
    parser.add_argument("--species", help="cod, saithe, haddock or whiting")
    parser.add_argument("--out", help="write image,age,species,dataset,path CSV here")
    args = parser.parse_args(argv)

    records = load_dataset(args.dataset, args.images, args.species)
    print("{} labelled images".format(len(records)))
    print("by species:", dict(sorted(Counter(r.species for r in records).items())))
    print("by age:    ", dict(sorted(Counter(r.age for r in records).items())))
    if args.images:
        print("images found on disk: {} of {}".format(sum(r.path is not None for r in records), len(records)))
    if args.out:
        write_labels_csv(records, args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
