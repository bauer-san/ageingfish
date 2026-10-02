# Thünen otolith label files

Small label files copied from the Thünen Institute's
[`arjaycc/ai_otolith`](https://github.com/arjaycc/ai_otolith) repository
(commit `461b20f`, MIT licence, see `LICENSE-ai_otolith.md`). They accompany
Sigurðardóttir et al. (2024), *Fish age reading using deep learning methods
for object-detection and segmentation*, ICES J. Mar. Sci. 81(4):687,
doi:10.1093/icesjms/fsae020.

| File | Contents |
|---|---|
| `north_species_map.json` | North Sea image filename → species (`cod`, `saithe`, `schellfisch` = haddock, `wittling` = whiting). 660 images. The age is in the filename (`..._age_<N>.png`). |
| `baltic_age_map.json` | Baltic Sea image filename → age read by Thünen readers (ages 1–5). 1,155 images. Baltic filenames carry no age. |

The images themselves are **not** in this repository. Download them from
Zenodo and unzip anywhere:

- North Sea: <https://doi.org/10.5281/zenodo.8341092> (`datasets_north.zip`, ~1.3 GB)
- Baltic Sea: <https://doi.org/10.5281/zenodo.8341149>

Ring annotations (VIA polyline JSON files) are optional and live in the
`ai_otolith` repo under `datasets_north/` and `datasets_baltic/`; clone it and
pass that folder to `load_ring_annotations`.
