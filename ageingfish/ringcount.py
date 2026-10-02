"""Classical annulus counter for sectioned otolith images.

Pipeline, per image:
  1. resize so the otolith is about WORK_LENGTH pixels long, making every
     setting relative to otolith size rather than to camera resolution
  2. segment the otolith from the dark background (Otsu threshold, largest blob)
  3. rotate so the long axis is horizontal
  4. follow the midline between the upper and lower edges, tip to tip, and
     average brightness in a thin band around it: annuli cross the midline
     roughly at right angles, so they show up as alternating bands
  5. remove slow shading, smooth, and count peaks on each side of the core

Steps 4-5 run on several reading lines parallel to the midline; the tip-to-tip
count is averaged over them and mapped to an age by a linear calibration
fitted on a development split (see ageingfish.tune).

    python -m ageingfish.ringcount --images DIR --config configs/north.json --out predictions.csv
"""
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks

WORK_LENGTH = 1000  # otolith length in pixels after resizing
# Parallel reading lines, as fractions of the local half-thickness above and
# below the midline. Counting each and taking the median damps false peaks
# from bubbles, cracks and noise that only cross one line.
OFFSETS = (-0.3, -0.15, 0.0, 0.15, 0.3)


@dataclass
class Profile:
    """Brightness along reading lines through one otolith, tip to tip."""
    values: np.ndarray      # brightness along each reading line, shape (len(OFFSETS), length)
    x: np.ndarray           # column of each value in the rotated working image
    midline: np.ndarray     # row of the midline at each column
    core_index: int         # index into values of the estimated core
    thickness: np.ndarray   # smoothed top-to-bottom height of the section per position
    gray: np.ndarray        # rotated working image (for plotting)
    mask: np.ndarray        # rotated otolith mask (for plotting)


@dataclass
class Params:
    polarity: int = 1          # +1 counts bright bands, -1 counts dark bands
    detrend: float = 0.06      # Gaussian sigma for shading removal, fraction of length
    smooth: float = 0.004      # Gaussian sigma for noise, fraction of length
    min_gap: float = 0.015     # minimum distance between rings, fraction of length
    prominence: float = 0.5    # minimum peak prominence, in robust SDs of the profile
    edge_trim: float = 0.02    # ignore this fraction of length at each tip
    core_trim: float = 0.03    # ignore this fraction of length either side of the core
    intercept: float = 0.0     # age = intercept + slope * tip-to-tip ring count
    slope: float = 0.5
    core_detector: str = "thickness"  # key of CORE_DETECTORS

    @classmethod
    def load(cls, path: str) -> "Params":
        with open(path) as f:
            config = json.load(f)
        return cls(**config["params"])


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(hist)
    mean = np.cumsum(hist * centers)
    total, total_mean = weight[-1], mean[-1]
    w_bg, w_fg = weight[:-1], total - weight[:-1]
    valid = (w_bg > 0) & (w_fg > 0)
    between = np.zeros_like(w_bg, dtype=float)
    m_bg = mean[:-1][valid] / w_bg[valid]
    m_fg = (total_mean - mean[:-1][valid]) / w_fg[valid]
    between[valid] = w_bg[valid] * w_fg[valid] * (m_bg - m_fg) ** 2
    return centers[np.argmax(between)]


def segment(gray: np.ndarray) -> np.ndarray:
    """Mask of the largest bright object (the otolith) on a dark background."""
    if np.ptp(gray) < 1:
        raise ValueError("blank image")
    blurred = ndimage.gaussian_filter(gray, 3)
    # Otsu alone cuts dark (translucent) zones out of the otolith, so use a
    # threshold halfway between the background level and Otsu's.
    otsu = _otsu(blurred)
    background = np.median(blurred[blurred <= otsu])
    mask = blurred > (background + otsu) / 2
    mask = ndimage.binary_opening(mask, iterations=3)
    labels, n = ndimage.label(mask)
    if n == 0:
        raise ValueError("no otolith found")
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    mask = labels == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=5))


def _long_axis_angle(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    coords = np.stack([xs - xs.mean(), ys - ys.mean()])
    eigvals, eigvecs = np.linalg.eigh(np.cov(coords))
    vx, vy = eigvecs[:, np.argmax(eigvals)]
    return math.degrees(math.atan2(vy, vx))


@dataclass
class Working:
    """An otolith resized to WORK_LENGTH and rotated so its long axis is horizontal."""
    gray: np.ndarray
    mask: np.ndarray
    scale: float
    angle: float
    resized_shape: Tuple[int, int]

    def to_work(self, x: float, y: float) -> Tuple[float, float]:
        """Map a point in original image pixels to working (resized, rotated) pixels."""
        x, y = x * self.scale, y * self.scale
        a = math.radians(self.angle)
        cx, cy = (self.resized_shape[1] - 1) / 2, (self.resized_shape[0] - 1) / 2
        ox, oy = (self.gray.shape[1] - 1) / 2, (self.gray.shape[0] - 1) / 2
        dx, dy = x - cx, y - cy
        return ox + math.cos(a) * dx + math.sin(a) * dy, oy - math.sin(a) * dx + math.cos(a) * dy


def prepare(path: str) -> Working:
    image = Image.open(path).convert("L")
    # A first pass at low resolution finds the otolith size for rescaling.
    small = image.copy()
    small.thumbnail((800, 800))
    small_mask = segment(np.asarray(small, dtype=float))
    ys, xs = np.nonzero(small_mask)
    extent = max(np.ptp(xs), np.ptp(ys)) * image.width / small.width
    scale = WORK_LENGTH / max(extent, 1)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.BILINEAR)
    scale = size[0] / Image.open(path).width

    gray = np.asarray(image, dtype=float)
    mask = segment(gray)
    angle = _long_axis_angle(mask)
    rotated = ndimage.rotate(gray, angle, reshape=True, order=1)
    rotated_mask = ndimage.rotate(mask.astype(float), angle, reshape=True, order=0) > 0.5
    return Working(rotated, rotated_mask, scale, angle, gray.shape)


def _edges(mask: np.ndarray):
    """Columns spanned by the otolith and its top and bottom edge in each."""
    cols = np.nonzero(mask.any(axis=0))[0]
    x = np.arange(cols.min(), cols.max() + 1)
    top = np.array([np.argmax(mask[:, c]) if mask[:, c].any() else np.nan for c in x], dtype=float)
    bottom = np.array([mask.shape[0] - 1 - np.argmax(mask[::-1, c]) if mask[:, c].any() else np.nan
                       for c in x], dtype=float)
    return x, top, bottom


def core_by_thickness(work: Working) -> Tuple[float, float]:
    """Core guess: the thickest point of the section (top-to-bottom height),
    restricted to the middle half so a bulging tip cannot win. The row is NaN,
    meaning "on the midline"."""
    x, top, bottom = _edges(work.mask)
    thickness = ndimage.gaussian_filter1d(np.nan_to_num(bottom - top), 0.03 * len(x))
    lo, hi = len(x) // 4, 3 * len(x) // 4
    i = lo + int(np.argmax(thickness[lo:hi]))
    return float(x[i]), float("nan")


def core_by_midpoint(work: Working) -> Tuple[float, float]:
    """Core guess: halfway along the long axis, on the midline."""
    x, _, _ = _edges(work.mask)
    return float((x[0] + x[-1]) / 2), float("nan")


CORE_DETECTORS = {"thickness": core_by_thickness, "midpoint": core_by_midpoint}


def load_profile(path: str, band: float = 0.012, core: Optional[Tuple[float, float]] = None,
                 detector: str = "thickness") -> Profile:
    """Brightness profiles along reading lines for one image.

    band is the half-height of the averaging strip around each line, as a
    fraction of otolith length. core, if given, is the core position in
    original image pixels; otherwise it comes from CORE_DETECTORS[detector].
    The reading path follows the midline near the tips and bends to pass
    through the core.
    """
    work = prepare(path)
    gray, mask = work.gray, work.mask
    x, top, bottom = _edges(mask)
    length = len(x)
    mid = (top + bottom) / 2
    good = ~np.isnan(mid)
    mid = np.interp(np.arange(len(x)), np.nonzero(good)[0], mid[good])
    mid = ndimage.gaussian_filter1d(mid, 0.02 * length)

    core_x, core_y = work.to_work(*core) if core is not None else CORE_DETECTORS[detector](work)
    core_index = int(np.clip(round(core_x - x[0]), 0, length - 1))
    # Bend the path through the core, fading back to the midline toward the tips.
    shift = 0.0 if math.isnan(core_y) else core_y - mid[core_index]
    mid = mid + shift * np.exp(-0.5 * ((np.arange(length) - core_index) / (0.15 * length)) ** 2)

    half = max(1, int(round(band * length)))
    halfthick = np.nan_to_num(bottom - top) / 2
    rows = []
    for offset in OFFSETS:
        centre = mid + offset * halfthick
        row = np.full(len(x), np.nan)
        for i, c in enumerate(x):
            r0 = max(int(round(centre[i])) - half, 0)
            r1 = min(int(round(centre[i])) + half + 1, gray.shape[0])
            column, inside = gray[r0:r1, c], mask[r0:r1, c]
            if inside.any():
                row[i] = column[inside].mean()
        good = ~np.isnan(row)
        rows.append(np.interp(np.arange(len(row)), np.nonzero(good)[0], row[good]))
    values = np.array(rows)

    thickness = ndimage.gaussian_filter1d(np.nan_to_num(bottom - top), 0.03 * length)
    return Profile(values, x, mid, core_index, thickness, gray, mask)


def filtered_signal(values: np.ndarray, polarity: int, detrend: float, smooth: float) -> Tuple[np.ndarray, float]:
    """Shading-free, smoothed signal along one reading line, and its robust SD."""
    n = len(values)
    signal = polarity * values
    signal = signal - ndimage.gaussian_filter1d(signal, detrend * n)
    signal = ndimage.gaussian_filter1d(signal, max(smooth * n, 0.5))
    spread = 1.4826 * np.median(np.abs(signal - np.median(signal))) or 1.0
    return signal, spread


def find_rings(signal: np.ndarray, spread: float, min_gap: float, prominence: float) -> np.ndarray:
    n = len(signal)
    peaks, _ = find_peaks(signal, prominence=prominence * spread, distance=max(1, int(min_gap * n)))
    return peaks


def split_sides(peaks: np.ndarray, n: int, core: int, edge_trim: float, core_trim: float):
    """Peaks left and right of the core, ignoring the tips and the core zone."""
    trim, ctrim = int(edge_trim * n), int(core_trim * n)
    left = peaks[(peaks >= trim) & (peaks < core - ctrim)]
    right = peaks[(peaks > core + ctrim) & (peaks < n - trim)]
    return left, right


def ring_peaks(profile: Profile, p: Params, row: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Ring positions left and right of the core on one reading line (default: the midline)."""
    values = profile.values[len(profile.values) // 2 if row is None else row]
    signal, spread = filtered_signal(values, p.polarity, p.detrend, p.smooth)
    peaks = find_rings(signal, spread, p.min_gap, p.prominence)
    return split_sides(peaks, len(values), profile.core_index, p.edge_trim, p.core_trim)


def ring_count(profile: Profile, p: Params) -> float:
    """Rings counted tip to tip (both sides of the core), averaged over the reading lines."""
    return float(np.mean([sum(len(side) for side in ring_peaks(profile, p, row))
                          for row in range(len(profile.values))]))


def estimate_age(profile: Profile, p: Params) -> float:
    """Age from the tip-to-tip ring count via the linear calibration in p.

    The default calibration (intercept 0, slope 0.5) is the textbook reading:
    each annulus is crossed once on each side of the core.
    """
    return p.intercept + p.slope * ring_count(profile, p)


def plot_profile(profile: Profile, p: Params, path: str, title: Optional[str] = None) -> None:
    """Rotated otolith with midline and detected rings, above the profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left, right = ring_peaks(profile, p)
    rings = np.concatenate([left, right])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={"height_ratios": [3, 2]})
    top.imshow(profile.gray, cmap="gray")
    top.plot(profile.x, profile.midline, color="#2a78d6", lw=1)
    top.plot(profile.x[rings], profile.midline[rings], "o", ms=5, color="#eb6834")
    top.axvline(profile.x[profile.core_index], color="#1baf7a", lw=1)
    rows = np.nonzero(profile.mask.any(axis=1))[0]
    top.set_xlim(profile.x[0] - 10, profile.x[-1] + 10)
    top.set_ylim(rows.max() + 10, rows.min() - 10)
    top.axis("off")
    top.set_title(title or "", fontsize=10)
    mid_values = profile.values[len(profile.values) // 2]
    bottom.plot(profile.x, mid_values, color="#52514e", lw=1)
    bottom.plot(profile.x[rings], mid_values[rings], "o", ms=5, color="#eb6834")
    bottom.axvline(profile.x[profile.core_index], color="#1baf7a", lw=1)
    bottom.set_xlim(profile.x[0] - 10, profile.x[-1] + 10)
    bottom.set_ylabel("brightness")
    bottom.set_xlabel("position along long axis (core in green, rings in orange)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _predict_one(job):
    path, params, plot_path = job
    try:
        profile = load_profile(path, detector=params.core_detector)
    except Exception as e:  # unreadable image or no otolith found
        return path, None, None, repr(e)
    if plot_path:
        plot_profile(profile, params, plot_path, os.path.basename(path))
    return path, estimate_age(profile, params), ring_count(profile, params), None


def predict(paths, params: Params, workers: int = 0, plot_dir: Optional[str] = None, n_plots: int = 0):
    """Yield (path, predicted_age, ring_count, error) for each image, in parallel."""
    from multiprocessing import Pool
    jobs = []
    for i, path in enumerate(paths):
        plot_path = None
        if plot_dir and i < n_plots:
            plot_path = os.path.join(plot_dir, os.path.splitext(os.path.basename(path))[0] + "_rings.png")
        jobs.append((path, params, plot_path))
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        yield from map(_predict_one, jobs)
    else:
        with Pool(workers) as pool:
            yield from pool.imap(_predict_one, jobs, chunksize=4)


def main(argv=None) -> int:
    import argparse
    import csv
    import sys

    from . import thuenen

    parser = argparse.ArgumentParser(description="Count annuli in otolith images and write predicted ages.")
    parser.add_argument("--images", required=True, help="folder of otolith images (searched recursively)")
    parser.add_argument("--dataset", choices=thuenen.DATASETS,
                        help="only predict this Thünen dataset's labelled images")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all",
                        help="with --dataset: which split to predict (default all)")
    parser.add_argument("--config", help="JSON written by ageingfish.tune; default: uncalibrated settings")
    parser.add_argument("--out", required=True, help="CSV to write: image,predicted_age,ring_count")
    parser.add_argument("--plots", help="folder for diagnostic plots of the first --n-plots images")
    parser.add_argument("--n-plots", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0, help="processes to use (default: all CPUs)")
    args = parser.parse_args(argv)

    params = Params.load(args.config) if args.config else Params()
    if args.dataset:
        records = thuenen.load_dataset(args.dataset, args.images, require_images=True)
        if args.split != "all":
            records = [r for r in records if thuenen.split_of(r.image) == args.split]
        paths = [r.path for r in records]
    else:
        found = thuenen.find_images(args.images)
        paths = [found[name] for name in sorted(found)]
    if not paths:
        print("error: no images found", file=sys.stderr)
        return 1

    failures = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "predicted_age", "ring_count"])
        for path, age, count, error in predict(paths, params, args.workers, args.plots, args.n_plots):
            if error:
                failures += 1
                print("skipped {}: {}".format(path, error), file=sys.stderr)
                continue
            writer.writerow([os.path.basename(path), "{:.3f}".format(age), "{:.2f}".format(count)])
    print("wrote {} predictions to {}{}".format(len(paths) - failures, args.out,
                                               " ({} skipped)".format(failures) if failures else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
