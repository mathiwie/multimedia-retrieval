"""Presets for the live demo: every step of the assignment with fixed inputs.

    python demo.py          menu: a number runs that preset, Enter runs the next one, q quits
    python demo.py 4        run one preset and exit
    python demo.py list     list the presets

query.py, knn.py and tsne.py save a snapshot in out/ every time they run. After every preset the
demo puts these files back as they were (and removes new ones), so a demo never changes the
delivered results.
"""
import os
import subprocess
import sys

import pandas as pd

repo_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(repo_dir, "out")
watched_dirs = [os.path.join(out_dir, "query_examples")]
watched_files = [os.path.join(out_dir, name) for name in ("query_shape.obj", "tsne.png", "tsne_brush.png", "tsne.csv")]

view_keys = "drag: rotate | scroll: zoom | Ctrl+drag: pan | W: edges | S: flat/smooth | R: reset | Q: close"


def show_features(name):
    data = pd.read_csv(os.path.join(out_dir, "features.csv"))
    match = data[data["path"].str.replace("\\", "/", regex=False) == f"out/normalised/{name}.obj"]
    if len(match) == 0:
        print(f"{name} is not in out/features.csv")
        return
    row = match.iloc[0]
    print(f"\n{name}: the 247 numbers that the distance function compares")
    print("  global descriptors")
    for column in ("area", "compactness", "rectangularity", "diameter", "convexity", "eccentricity"):
        print(f"    {column:15s} {row[column]:10.3f}")
    print(f"    (volume {row['volume']:.4f} from the voxel grid, leakage {row['leakage']:.2f}, "
          f"hull fallback {'yes' if row['hull_fallback'] else 'no'})")
    print("  histograms (each sums to 1, bar height = share of the samples in that bin)")
    for block, meaning in (("A3", "angle between 3 surface points"), ("D1", "barycenter to surface point"),
                           ("D2", "distance between 2 points"), ("D3", "sqrt of the area of 3 points"),
                           ("D4", "cube root of the volume of 4 points"), ("CURV", "curvature: concave | flat | convex")):
        columns = [c for c in data.columns if c.startswith(block + "_") and c.split("_")[-1].isdigit()]
        print(f"    {block:4s} {len(columns)} bins  {sparkline(row[columns].to_numpy(dtype=float))}  {meaning}")


def sparkline(values):
    bars = "▁▂▃▄▅▆▇█"
    top = values.max() if values.max() > 0 else 1.0
    return "".join(" " if v <= 0 else bars[min(len(bars) - 1, int(v / top * len(bars)))] for v in values)


def show_evaluation():
    shapes = pd.read_csv(os.path.join(out_dir, "evaluation_shapes.csv"))
    classes = pd.read_csv(os.path.join(out_dir, "evaluation_classes.csv"))
    n = len(shapes)
    print(f"\n{n} queries (every database shape once), {len(classes)} classes")
    for column, label in (("precision@10", "precision at 10"), ("r_precision", "R precision")):
        print(f"  {label:16s} {shapes[column].mean():.3f} over all queries, "
              f"{classes[column].mean():.3f} averaged over the classes")
    print(f"  random retrieval {((shapes['class_size'] - 1) / (n - 1)).mean():.3f}")
    print(f"  all 10 results correct for {(shapes['precision@10'] == 1).mean():.1%} of the queries, "
          f"none correct for {(shapes['precision@10'] == 0).mean():.1%}")
    ordered = classes.sort_values("r_precision", ascending=False)
    for label, part in (("best classes", ordered.head(5)), ("worst classes", ordered.tail(5))):
        print(f"  {label:25s} size  R prec.  prec@10")
        for _, row in part.iterrows():
            print(f"    {row['class']:23s} {int(row['size']):4d}  {row['r_precision']:7.3f}  {row['precision@10']:7.3f}")


# key, step, title, what to run (script and arguments, or a function), talking points
presets = [
    ("1", "Step 1", "viewer: Car/D00236 as stored in the database",
     ["main.py", os.path.join("ShapeDatabase_INFOMR", "Car", "D00236.obj")],
     [view_keys, "the shape as stored: 26,704 vertices and 45,234 faces (Figure 2)"]),
    ("2", "Step 2-3", "normalisation: Car/D00236 resampled (left) and normalised (right)",
     ["show_normalised.py", "Car/D00236"],
     ["right: barycenter at the origin, major axis on x, flipped, scaled into the unit cube",
      "x red, y green, z blue, the grey box is the unit cube; " + view_keys]),
    ("3", "Step 3", "features of Gun/D00204 (the average shape)",
     lambda: show_features("Gun/D00204"),
     ["6 global descriptors + 5 histograms of 40 bins + the curvature histogram of 41 bins"]),
    ("4", "Step 4", "query: Bicycle/D00974, held out of the database (a good result)",
     ["query.py", "Bicycle/D00974"],
     ["orange = query, matches left to right, distances in the window title and the terminal",
      "#1 Bicycle/m1475 at 0.03 is the same model from the other benchmark",
      "#3 and #4 are City street layouts: also thin, spread out, low convexity"]),
    ("5", "Step 4", "query: Vase/m541 (a poor result)",
     ["query.py", "Vase/m541"],
     ["only #4 is a vase; the others are round objects of a similar size (blimps, drums)",
      "its volume falls back to the convex hull, so compactness, rectangularity and convexity are less reliable"]),
    ("6", "Step 4", "query: a mesh file that is not in the database",
     ["query.py", os.path.join("ShapeDatabase_INFOMR", "RoundTable", "m935.obj")],
     ["the original file of a held-out shape: cleaned, resampled, normalised and described now (about 10 s)",
      "the matches share its flat top on a thin support"]),
    ("7", "Step 5", "kd-tree, K nearest neighbours: Car/D00260 (held out), K = 5",
     ["knn.py", "Car/D00260", "-k", "5"],
     ["K is set by the user (-k); exactly the results and distances of the brute-force search",
      "cars and military vehicles: shapes with the same proportions"]),
    ("8", "Step 5", "kd-tree, range search: Humanoid/m198, r = 0.24",
     ["knn.py", "Humanoid/m198", "-r", "0.24"],
     ["every shape within distance r (the same as -t in query.py): 14 shapes, 7 of them humanoids",
      "r = 0.3 returns 60 and r = 0.4 returns 245 shapes, which is why a fixed K is the default"]),
    ("9", "Step 5", "t-SNE map of the 2,473 database shapes",
     ["tsne.py"],
     ["colour and marker = class; hover: name and class, black rings mark the whole class",
      "click a point to open the shape; HumanHead far left, Sword top right, 19 of the 20 fish together"]),
    ("0", "Step 6", "evaluation: precision at 10 and R precision",
     show_evaluation,
     ["every database shape is used once as the query (K = 10 and K = class size - 1)"]),
]
by_key = {p[0]: i for i, p in enumerate(presets)}


def snapshot():
    saved = {}
    for path in watched_paths():
        try:
            with open(path, "rb") as f:
                saved[path] = f.read()
        except OSError as e:
            print(f"    (could not back up {os.path.relpath(path, repo_dir)}: {e})")
    return saved


def watched_paths():
    paths = [p for p in watched_files if os.path.isfile(p)]
    for folder in watched_dirs:
        if os.path.isdir(folder):
            paths += [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return paths


def restore(saved):
    changed = 0
    for path in watched_paths():
        if path not in saved:
            os.remove(path)
            changed += 1
    for path, data in saved.items():
        if os.path.isfile(path):
            with open(path, "rb") as f:
                if f.read() == data:
                    continue
        with open(path, "wb") as f:
            f.write(data)
        changed += 1
    return changed


def run(index):
    key, step, title, action, notes = presets[index]
    print(f"\n=== {key}. {step}: {title}")
    for line in notes:
        print(f"    {line}")
    saved = snapshot()
    try:
        if callable(action):
            action()
        else:
            subprocess.run([sys.executable] + action, cwd=repo_dir)
    except KeyboardInterrupt:
        print("\n    (stopped)")
    finally:
        changed = restore(saved)
        if changed:
            print(f"    (put back {changed} file{'s' if changed > 1 else ''} in out/)")


def print_menu(next_index):
    print("\nMultimedia retrieval demo")
    for i, (key, step, title, _, _) in enumerate(presets):
        print(f"  {'>' if i == next_index else ' '} {key}  {step:8s} {title}")
    print("    a number runs that preset | Enter runs the one marked > | m: this menu | q: quit")


def read_key():
    if not sys.stdin.isatty():
        return (sys.stdin.readline() or "q").strip()[:1] or "\r"
    if os.name == "nt":
        import msvcrt
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            msvcrt.getwch()
            return ""
        return key
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def interactive():
    following = 0
    print_menu(following)
    while True:
        key = read_key()
        if key in ("q", "Q", "\x03", "\x1b"):
            break
        if key in ("m", "M", "?"):
            print_menu(following)
            continue
        if key in ("\r", "\n", " "):
            if following >= len(presets):
                print("    that was the last preset: a number runs one again, q quits")
                continue
            index = following
        elif key in by_key:
            index = by_key[key]
        else:
            continue
        run(index)
        following = index + 1
        if following < len(presets):
            key, step, title, _, _ = presets[following]
            print(f"\n--- next (Enter): {key}. {step}: {title}   | number | m: menu | q: quit")
        else:
            print("\n--- end of the demo   | number | m: menu | q: quit")


def main(argv):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if not argv:
        try:
            interactive()
        except KeyboardInterrupt:
            pass
    elif argv[0] in ("list", "-l", "--list"):
        print_menu(None)
    elif argv[0] in by_key:
        run(by_key[argv[0]])
    else:
        sys.exit(f"unknown preset {argv[0]!r}; use one of {', '.join(by_key)} or 'list'")


if __name__ == "__main__":
    main(sys.argv[1:])
