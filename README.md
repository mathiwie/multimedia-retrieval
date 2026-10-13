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
| 2 statistics, resampling, normalisation | `python preprocessing.py` | `out/shape_stats*.csv`, `out/resampled/`, `out/normalised/`, histograms |
| | `python preprocessing.py normalise` | only redo the normalisation and its check |
| 2 inspection | `python show_average_outliers.py [-nogui]`, `python show_resampled.py [Class/shape ...] [-nogui]` | viewer windows; `out/average_and_outliers.png`, `out/resampling_examples.png` (original vs resampled, wireframe) |
| 3 normalisation check | `python show_normalised.py [Class/shape ...]`, `python show_normalisation_steps.py [Class/shape ...]` | before/after viewer windows; `out/normalisation_steps.png` (each step with a fixed camera) |
| 3 features | `python features.py` | `out/features.csv`, `out/skipped_shapes.csv` |
| | `python features.py Gun/D00204` | prints the features of one shape |
| 3 checks | `python validate_features.py`, `python bin_choice.py`, `python show_descriptors.py [view]` | descriptor checks on simple shapes, bin choice, `out/descriptor_examples.png` |
| 3 curvature | `python show_curvature.py [Class/shape ...]`, `python curvature_choice.py` | `out/curvature_examples.png` (meshes coloured by curvature); validation on primitives and the radius / robustness comparison, `out/curvature_choice.csv` |
| 3 volume | `python volume_check.py` | voxel resolution convergence, timing and multi-ray inside test, `out/volume_check.csv` |
| 4 feature normalisation | `python normalise_features.py` | `out/features_{log,standard,minmax}.csv`, `out/holdout.csv` |
| | `python normalise_features.py Gun/D00204 [log\|standard\|minmax]` | prints the normalised features of one shape |
| 4 choosing the distance | `python compare_distances.py`, `python threshold_choice.py` | `out/distance_comparison.*`, `out/threshold_choice.png` |
| 5 kd-tree check | `python knn.py -check` | agreement with brute force and timings |
| 5 t-SNE | `python tsne.py [-redo] [-nogui]` | `out/tsne.png`; hover a point for its name and class, click it to open the shape |
| 6 evaluation | `python evaluate.py [-redo]` | `out/evaluation_*.csv`, `out/evaluation_classes.png` |
| | `python evaluation_details.py` | prints the other numbers quoted in the report: perfect and empty top-10 shares, duplicates, flagged shapes, what a threshold of 0.35 returns, and which classes are confused with each other |

To rebuild everything from the original database, run in this order:
`preprocessing.py`, `features.py`, `normalise_features.py`, then any of the step 4-6 scripts.
`preprocessing.py` and `features.py` process all 2,483 shapes and take a while.