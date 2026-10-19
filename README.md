# multimedia-retrieval

Content-based 3D shape retrieval system for the Utrecht University course Multimedia Retrieval (INFOMR).
Given a 3D mesh, it finds and shows the most similar shapes in a database of 2,483 shapes in 69 classes.

## Setup

Requires Python 3.10 or newer (tested with Python 3.12.3).

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Tested with open3d 0.20.0, numpy 2.0.2, pandas 2.2.3, scipy 1.16.1, matplotlib 3.11.2 and scikit-learn 1.7.2.
Run every script from the repository root.

## What is included

- `ShapeDatabase_INFOMR/` - the original shape database, one folder per class
- `out/resampled/` and `out/normalised/` - the database after resampling (step 2) and after full normalisation (step 3)
- `out/features.csv` - the feature database (one row per shape) and `out/features_{log,standard,minmax}.csv` - the normalised feature tables
- `out/` also holds all statistics, plots and snapshots used in the report

Everything below can be run directly on these files. Re-running the pipeline from scratch is only needed after changing the code.

## Querying

```
python query.py Gun/D00204                    # 5 best matches of a database shape
python query.py Gun/D00204 -k 10              # 10 best matches
python query.py Gun/D00204 -t 0.35            # all shapes closer than distance 0.35
python query.py path/to/mesh.obj              # a mesh that is not in the database
python knn.py Gun/D00204 -k 5                 # same, using the kd-tree (step 5)
python knn.py Gun/D00204 -r 0.4               # range search: all shapes within distance 0.4
```

Shapes are named `Class/file` without the extension. Results are printed, saved as a snapshot in `out/query_examples/` and shown in an interactive viewer (add `-nogui` to skip the viewer).
Ten shapes are held out of the database to act as unseen queries; they are listed in `out/holdout.csv`.

## Running the steps

| Step | Command | Output |
|---|---|---|
| 1 viewer | `python main.py [mesh.obj]` | interactive viewer (mouse: rotate, zoom, pan; `W` toggles the wireframe) |
| | `python show_viewer_modes.py [Class/shape] [Class/shape]` | `out/viewer_modes.png` (display modes), `out/viewer_manipulation.png` (rotate, zoom, pan), `out/viewer_variety.png` (smallest, largest and out-of-scale shapes) |
| 2 statistics, resampling, normalisation | `python preprocessing.py` | `out/shape_stats*.csv`, `out/resampled/`, `out/normalised/`, histograms |
| | `python preprocessing.py normalise` | only redo the normalisation and its check |
| | `python preprocessing.py check` | only redo the normalisation check: statistics before and after, `out/normalisation_check*.csv`, `out/histograms_normalised.png` |
| 2 cleaning | `python cleaning_check.py` | vertices merged and faces removed by each cleaning step, `out/cleaning_check.csv` |
| 2 inspection | `python show_average_outliers.py [-nogui]`, `python show_resampled.py [Class/shape ...] [-nogui]` | viewer windows; `out/average_and_outliers.png`, `out/resampling_examples.png` (original vs resampled, wireframe) |
| 3 normalisation check | `python show_normalised.py [Class/shape ...]`, `python show_normalisation_steps.py [Class/shape ...]` | before/after viewer windows; `out/normalisation_steps.png` (each step with a fixed camera) |
| 3 features | `python features.py` | `out/features.csv`, `out/skipped_shapes.csv` |
| | `python features.py Gun/D00204` | prints the features of one shape |
| 3 checks | `python validate_features.py`, `python bin_choice.py`, `python show_descriptors.py [view]` | descriptor checks on simple shapes, bin choice, `out/descriptor_examples.png` |
| | `python show_class_histograms.py [Class ...]` | `out/class_histograms.png` (every histogram descriptor of every shape of a few classes), within/between-class distances in `out/class_histogram_spread.csv` |
| | `python d1_surface.py` | D1 from the vertices vs from random surface points, and retrieval with each, `out/d1_surface*.csv` |
| 3 curvature | `python show_curvature.py [Class/shape ...]`, `python curvature_choice.py` | `out/curvature_examples.png` (meshes coloured by curvature); validation on primitives and the radius / robustness comparison, `out/curvature_choice.csv` |
| | `python curvature_sign.py` | retrieval with the signed, the unsigned and no curvature histogram, `out/curvature_sign.csv` |
| 3 volume | `python volume_check.py` | voxel resolution convergence, timing and number of ray directions, `out/volume_check.csv` |
| | `python volume_leak.py [-redo]` | leakage of every shape, voxels vs tetrahedra vs convex hull, and retrieval for each leak limit, `out/volume_leak*.csv`, `out/volume_leak.png` (the scan of all shapes takes about an hour) |
| 4 feature normalisation | `python normalise_features.py` | `out/features_{log,standard,minmax}.csv`, `out/holdout.csv` |
| | `python normalise_features.py Gun/D00204 [log\|standard\|minmax]` | prints the normalised features of one shape |
| 4 choosing the distance | `python compare_distances.py`, `python threshold_choice.py` | `out/distance_comparison.*`, `out/threshold_choice.png` |
| 5 kd-tree check | `python knn.py -check` | agreement with brute force and timings |
| 5 t-SNE | `python tsne.py [-redo] [-nogui] [-brush Class/shape]` | `out/tsne.png`; hover a point for its name and class, click it to open the shape; `-brush` saves the plot with that shape hovered as `out/tsne_brush.png` |
| 6 evaluation | `python evaluate.py [-redo]` | `out/evaluation_*.csv`, `out/evaluation_classes.png` |
| | `python evaluation_details.py` | prints the other numbers quoted in the report: perfect and empty top-10 shares, duplicates, flagged shapes, what a threshold of 0.35 returns, and which classes are confused with each other |

To rebuild everything from the original database, run in this order:
`preprocessing.py`, `features.py`, `normalise_features.py`, then any of the step 4-6 scripts.
`preprocessing.py` and `features.py` process all 2,483 shapes and take a while.