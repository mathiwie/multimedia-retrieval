"""Demo window: every step of the assignment one click away, plus a panel to run any query.

    python demo.py          opens the demo window
    python demo.py 4        runs preset 4 in the terminal, without the window
    python demo.py list     lists the presets

The buttons run the normal scripts (main.py, query.py, knn.py, tsne.py, ...) and show what they
print in the window. query.py, knn.py and tsne.py save a snapshot in out/ every time they run;
after every run the demo puts these files back as they were and removes new ones, so a demo never
changes the delivered results.
"""
import os
import queue
import re
import subprocess
import sys
import threading

import pandas as pd

repo_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(repo_dir, "out")
demo_meshes_dir = os.path.join(repo_dir, "demo_meshes")
watched_dirs = [os.path.join(out_dir, "query_examples")]
watched_files = [os.path.join(out_dir, name) for name in ("query_shape.obj", "tsne.png", "tsne_brush.png", "tsne.csv")]
noise = ("SetViewPoint() failed",)

view_keys = "drag: rotate | scroll: zoom | Ctrl+drag: pan | W: edges | S: flat/smooth | R: reset | Q: close"


def shape_key(path):
    parts = re.split(r"[\\/]", path)
    return f"{parts[-2]}/{os.path.splitext(parts[-1])[0]}"


def show_features(name, out=print):
    data = pd.read_csv(os.path.join(out_dir, "features.csv"))
    match = data[data["path"].map(shape_key) == name]
    if len(match) == 0:
        out(f"{name} is not in out/features.csv")
        return
    row = match.iloc[0]
    out(f"{name}: the 247 numbers that the distance function compares")
    out("  global descriptors")
    for column in ("area", "compactness", "rectangularity", "diameter", "convexity", "eccentricity"):
        out(f"    {column:15s} {row[column]:10.3f}")
    out(f"    (volume {row['volume']:.4f} from the voxel grid, leakage {row['leakage']:.2f}, "
        f"hull fallback {'yes' if row['hull_fallback'] else 'no'})")
    out("  histograms (each sums to 1, bar height = share of the samples in that bin)")
    for block, meaning in (("A3", "angle between 3 surface points"), ("D1", "barycenter to surface point"),
                           ("D2", "distance between 2 points"), ("D3", "sqrt of the area of 3 points"),
                           ("D4", "cube root of the volume of 4 points"), ("CURV", "curvature: concave | flat | convex")):
        columns = [c for c in data.columns if c.startswith(block + "_") and c.split("_")[-1].isdigit()]
        out(f"    {block:4s} {len(columns)} bins  {sparkline(row[columns].to_numpy(dtype=float))}  {meaning}")


def sparkline(values):
    bars = "▁▂▃▄▅▆▇█"
    top = values.max() if values.max() > 0 else 1.0
    return "".join(" " if v <= 0 else bars[min(len(bars) - 1, int(v / top * len(bars)))] for v in values)


def show_evaluation(out=print):
    shapes = pd.read_csv(os.path.join(out_dir, "evaluation_shapes.csv"))
    classes = pd.read_csv(os.path.join(out_dir, "evaluation_classes.csv"))
    n = len(shapes)
    out(f"{n} queries (every database shape once), {len(classes)} classes")
    for column, label in (("precision@10", "precision at 10"), ("r_precision", "R precision")):
        out(f"  {label:16s} {shapes[column].mean():.3f} over all queries, "
            f"{classes[column].mean():.3f} averaged over the classes")
    out(f"  random retrieval {((shapes['class_size'] - 1) / (n - 1)).mean():.3f}")
    out(f"  all 10 results correct for {(shapes['precision@10'] == 1).mean():.1%} of the queries, "
        f"none correct for {(shapes['precision@10'] == 0).mean():.1%}")
    ordered = classes.sort_values("r_precision", ascending=False)
    for label, part in (("best classes", ordered.head(5)), ("worst classes", ordered.tail(5))):
        out(f"  {label:25s} size  R prec.  prec@10")
        for _, row in part.iterrows():
            out(f"    {row['class']:23s} {int(row['size']):4d}  {row['r_precision']:7.3f}  {row['precision@10']:7.3f}")


# key, step, title, what to run (script and arguments, or a function), talking points
presets = [
    ("1", "Step 1", "Viewer: Car/D00236 as stored",
     ["main.py", os.path.join("ShapeDatabase_INFOMR", "Car", "D00236.obj")],
     [view_keys, "the shape as stored: 26,709 vertices and 45,234 faces (Figure 2)"]),
    ("2", "Step 2-3", "Normalisation: Car/D00236 before and after",
     ["show_normalised.py", "Car/D00236"],
     ["left: the resampled mesh; right: barycenter at the origin, major axis on x, flipped, scaled into the unit cube",
      "x red, y green, z blue, the grey box is the unit cube"]),
    ("3", "Step 3", "Features of Gun/D00204",
     lambda out: show_features("Gun/D00204", out),
     ["6 global descriptors + 5 histograms of 40 bins + the curvature histogram of 41 bins"]),
    ("4", "Step 4", "Query: Bicycle/D00974 (held out, good)",
     ["query.py", "Bicycle/D00974"],
     ["orange = query, matches left to right, distances in the window title and below",
      "#1 Bicycle/m1475 at 0.03 is the same model from the other benchmark",
      "#3 and #4 are City street layouts: also thin, spread out, low convexity"]),
    ("5", "Step 4", "Query: Vase/m541 (poor)",
     ["query.py", "Vase/m541"],
     ["only #4 is a vase; the others are round objects of a similar size (blimps, drums)",
      "its volume falls back to the convex hull, so compactness, rectangularity and convexity are less reliable"]),
    ("6", "Step 4", "Query: a mesh file not in the database",
     ["query.py", os.path.join("demo_meshes", "table.obj")],
     ["a table made of 5 boxes, rotated, scaled 37x and moved: cleaned, resampled, normalised and described now",
      "4 of the 5 matches are tables; other files can be picked in the Query panel"]),
    ("7", "Step 5", "kd-tree, K = 5: Car/D00260 (held out)",
     ["knn.py", "Car/D00260", "-k", "5"],
     ["K is set by the user; one exact tree query for the l2 combination of the blocks, ~10x faster than brute force",
      "the same 5 shapes as the brute-force search of step 4, with #2 and #3 swapped",
      "cars and military vehicles: shapes with the same proportions"]),
    ("8", "Step 5", "kd-tree, range r = 0.28: Humanoid/m198",
     ["knn.py", "Humanoid/m198", "-r", "0.28"],
     ["every shape within distance r: 16 shapes, 8 of them humanoids",
      "r = 0.35 returns 85 and r = 0.45 returns 299 shapes, which is why a fixed K is the default"]),
    ("9", "Step 5", "t-SNE map of the database",
     ["tsne.py"],
     ["colour and marker = class; hover: name and class, black rings mark the whole class",
      "click a point to open the shape; HumanHead far left, Sword top right, 19 of the 20 fish together"]),
    ("0", "Step 6", "Evaluation: precision and R precision",
     show_evaluation,
     ["every database shape is used once as the query (K = 10 and K = class size - 1)"]),
]
by_key = {p[0]: i for i, p in enumerate(presets)}


def query_command(target, kdtree, k=None, distance=None):
    script = "knn.py" if kdtree else "query.py"
    if distance is None:
        return [script, target, "-k", str(k)]
    return [script, target, "-r" if kdtree else "-t", f"{distance:g}"]


def watched_paths():
    paths = [p for p in watched_files if os.path.isfile(p)]
    for folder in watched_dirs:
        if os.path.isdir(folder):
            paths += [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return paths


def snapshot():
    saved = {}
    for path in watched_paths():
        with open(path, "rb") as f:
            saved[path] = f.read()
    return saved


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


def kill_tree(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        process.terminate()


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import font, ttk
        self.tk = tk
        self.root = root
        self.process = None
        self.saved = None
        self.stopping = False
        self.lines = queue.Queue()
        self.next_index = 0
        self.actions = []
        self.file_path = None

        root.title("Multimedia retrieval demo")
        root.geometry("1250x760")
        root.minsize(1000, 600)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.configure("Step.TButton", anchor="w", padding=(8, 4))
        style.configure("Big.TButton", padding=(8, 6))

        table = pd.read_csv(os.path.join(out_dir, "features.csv"), usecols=["path", "class"])
        held_out = set(pd.read_csv(os.path.join(out_dir, "holdout.csv"))["path"].map(shape_key))
        self.shapes = {}
        for path, cls in zip(table["path"], table["class"]):
            key = shape_key(path)
            self.shapes.setdefault(cls, []).append(key.split("/")[1] + (" (held out)" if key in held_out else ""))

        left = ttk.Frame(root, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(root, padding=(0, 10, 10, 10))
        right.pack(side="left", fill="both", expand=True)

        steps = ttk.LabelFrame(left, text="Demo steps", padding=8)
        steps.pack(fill="x")
        for i, (_, step, title, _, _) in enumerate(presets):
            button = ttk.Button(steps, text=f"{step:9s} {title}", style="Step.TButton", width=46,
                                command=lambda i=i: self.run_preset(i))
            button.pack(fill="x", pady=1)
            self.actions.append(button)
        self.next_button = ttk.Button(steps, text="Next step  ▶", style="Big.TButton", command=self.run_next)
        self.next_button.pack(fill="x", pady=(8, 2))
        self.actions.append(self.next_button)
        self.next_label = ttk.Label(steps, foreground="#555")
        self.next_label.pack(fill="x")

        panel = ttk.LabelFrame(left, text="Query", padding=8)
        panel.pack(fill="x", pady=(10, 0))
        self.source = tk.StringVar(value="database")
        ttk.Radiobutton(panel, text="Database shape", variable=self.source, value="database").grid(
            row=0, column=0, columnspan=2, sticky="w")
        self.class_box = ttk.Combobox(panel, values=sorted(self.shapes), state="readonly", width=22)
        self.class_box.grid(row=1, column=0, sticky="we", padx=(18, 4), pady=2)
        self.shape_box = ttk.Combobox(panel, state="readonly", width=20)
        self.shape_box.grid(row=1, column=1, sticky="we", pady=2)
        self.class_box.bind("<<ComboboxSelected>>", lambda e: self.class_changed())
        self.shape_box.bind("<<ComboboxSelected>>", lambda e: self.source.set("database"))
        self.class_box.set("Bicycle")
        self.class_changed("D00974 (held out)")

        ttk.Radiobutton(panel, text="Mesh file (.obj, .ply, .off, .stl)", variable=self.source, value="file").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        choose = ttk.Button(panel, text="Choose a file...", command=self.choose_file)
        choose.grid(row=3, column=0, sticky="w", padx=(18, 4), pady=2)
        self.actions.append(choose)
        self.file_label = ttk.Label(panel, text="no file chosen", foreground="#555", width=24)
        self.file_label.grid(row=3, column=1, sticky="w")

        ttk.Separator(panel).grid(row=4, column=0, columnspan=2, sticky="we", pady=8)
        self.search = tk.StringVar(value="brute")
        ttk.Label(panel, text="Search").grid(row=5, column=0, sticky="w")
        modes = ttk.Frame(panel)
        modes.grid(row=5, column=1, sticky="w")
        ttk.Radiobutton(modes, text="brute force", variable=self.search, value="brute").pack(side="left")
        ttk.Radiobutton(modes, text="kd-tree", variable=self.search, value="kdtree").pack(side="left", padx=(8, 0))
        self.limit = tk.StringVar(value="k")
        ttk.Radiobutton(panel, text="K best matches", variable=self.limit, value="k").grid(row=6, column=0, sticky="w")
        self.k = tk.StringVar(value="5")
        ttk.Spinbox(panel, from_=1, to=50, textvariable=self.k, width=6).grid(row=6, column=1, sticky="w", pady=2)
        ttk.Radiobutton(panel, text="All within distance", variable=self.limit, value="distance").grid(
            row=7, column=0, sticky="w")
        self.distance = tk.StringVar(value="0.25")
        ttk.Spinbox(panel, from_=0.05, to=1.0, increment=0.01, textvariable=self.distance, width=6).grid(
            row=7, column=1, sticky="w", pady=2)
        ttk.Label(panel, text="0.2 to 0.25 keeps the result window readable", foreground="#555").grid(
            row=8, column=0, columnspan=2, sticky="w")
        run = ttk.Button(panel, text="Run query", style="Big.TButton", command=self.run_query)
        run.grid(row=9, column=0, columnspan=2, sticky="we", pady=(8, 0))
        self.actions.append(run)
        panel.columnconfigure(1, weight=1)

        bar = ttk.Frame(right)
        bar.pack(side="bottom", fill="x", pady=(6, 0))
        self.status = ttk.Label(bar, text="ready")
        self.status.pack(side="left")
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="right")
        self.stop_button = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="right", padx=(0, 6))

        fixed = font.nametofont("TkFixedFont").copy()
        fixed.configure(size=10)
        bold = fixed.copy()
        bold.configure(weight="bold", size=11)
        self.text = tk.Text(right, wrap="word", font=fixed, state="disabled", padx=10, pady=8, relief="flat",
                            background="#fbfbfb")
        scroll = ttk.Scrollbar(right, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.tag_configure("heading", font=bold, spacing1=10, foreground="#1d4f91")
        self.text.tag_configure("note", foreground="#666")
        self.text.tag_configure("error", foreground="#b00020")

        self.write("Click a step on the left, or run your own query.\n", "heading")
        self.write("Viewer windows open separately; close them (Q) to come back here.\n", "note")
        self.write(f"{view_keys}\n", "note")
        self.update_next()
        root.after(100, self.poll)

    def class_changed(self, shape=None):
        names = self.shapes[self.class_box.get()]
        self.shape_box.configure(values=names)
        self.shape_box.set(shape if shape in names else names[0])
        self.source.set("database")

    def choose_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            parent=self.root, title="Choose a mesh to query",
            initialdir=demo_meshes_dir if os.path.isdir(demo_meshes_dir) else repo_dir,
            filetypes=[("3D meshes", "*.obj *.ply *.off *.stl"), ("All files", "*.*")])
        if path:
            self.file_path = os.path.normpath(path)
            self.file_label.configure(text=os.path.basename(path))
            self.source.set("file")

    def write(self, text, tag=None):
        self.text.configure(state="normal")
        self.text.insert("end", text, tag)
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def update_next(self):
        if self.next_index < len(presets):
            _, step, title, _, _ = presets[self.next_index]
            self.next_label.configure(text=f"next: {step}  {title}")
        else:
            self.next_label.configure(text="that was the last step")

    def run_preset(self, index):
        _, step, title, action, notes = presets[index]
        if self.start(f"{step}: {title}", notes, action):
            self.next_index = index + 1
            self.update_next()

    def run_next(self):
        if self.next_index < len(presets):
            self.run_preset(self.next_index)

    def run_query(self):
        from tkinter import messagebox
        notes = []
        if self.source.get() == "file":
            if not self.file_path:
                messagebox.showinfo("Query", "Choose a mesh file first.", parent=self.root)
                return
            target, label = self.file_path, os.path.basename(self.file_path)
            notes.append("a new mesh: cleaned, resampled, normalised and described first (a few seconds)")
        else:
            target = label = f"{self.class_box.get()}/{self.shape_box.get().split(' ')[0]}"
        kdtree = self.search.get() == "kdtree"
        if kdtree:
            notes.append("the kd-tree combines the blocks as an l2 distance (step 5), so its distances are a "
                         "little larger than those of the brute-force search")
        try:
            if self.limit.get() == "k":
                k, distance = int(self.k.get()), None
                if k < 1:
                    raise ValueError
            else:
                k, distance = None, float(self.distance.get())
                if distance <= 0:
                    raise ValueError
                notes.append("every shape within the distance is shown in the viewer, so large distances get crowded")
        except ValueError:
            messagebox.showerror("Query", "K must be a whole number of at least 1 and the distance a positive number.",
                                 parent=self.root)
            return
        limit = f"K = {k}" if distance is None else f"distance < {distance:g}"
        self.start(f"Query: {label}  ({'kd-tree' if kdtree else 'brute force'}, {limit})", notes,
                   query_command(target, kdtree, k, distance))

    def start(self, heading, notes, action):
        if self.process is not None:
            return False
        self.write(f"\n{heading}\n", "heading")
        for line in notes:
            self.write(f"  {line}\n", "note")
        self.saved = snapshot()
        if callable(action):
            try:
                action(lambda line="": self.write(line + "\n"))
            except Exception as e:
                self.write(f"  error: {e}\n", "error")
            self.finish(None)
            return True
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u"] + action, cwd=repo_dir, env=env, text=True, encoding="utf-8",
                errors="replace", stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as e:
            self.write(f"  could not start: {e}\n", "error")
            self.finish(None)
            return False
        threading.Thread(target=self.pump, args=(self.process,), daemon=True).start()
        self.set_busy(True, heading)
        return True

    def pump(self, process):
        for line in process.stdout:
            self.lines.put(line)
        process.wait()
        self.lines.put(None)

    def poll(self):
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    self.finish(self.process.returncode if self.process else None)
                elif not any(n in line for n in noise):
                    self.write(line)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def finish(self, code):
        if self.saved is not None:
            changed = restore(self.saved)
            self.saved = None
            if changed:
                self.write(f"  (put back {changed} file{'s' if changed > 1 else ''} in out/)\n", "note")
        if self.stopping:
            self.write("  (stopped)\n", "note")
        elif code not in (None, 0):
            self.write(f"  (ended with exit code {code})\n", "error")
        self.process = None
        self.stopping = False
        self.set_busy(False)

    def set_busy(self, busy, heading=""):
        for button in self.actions:
            button.state(["disabled"] if busy else ["!disabled"])
        self.stop_button.state(["!disabled"] if busy else ["disabled"])
        self.status.configure(text=f"running: {heading}  (close its window to continue)" if busy else "ready")

    def stop(self):
        if self.process is not None:
            self.stopping = True
            kill_tree(self.process)

    def close(self):
        if self.process is not None:
            process = self.process
            self.stop()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            self.finish(process.returncode)
        self.root.destroy()


def run_in_terminal(index):
    key, step, title, action, notes = presets[index]
    print(f"=== {key}. {step}: {title}")
    for line in notes:
        print(f"    {line}")
    saved = snapshot()
    try:
        if callable(action):
            action(print)
        else:
            subprocess.run([sys.executable] + action, cwd=repo_dir)
    except KeyboardInterrupt:
        print("    (stopped)")
    finally:
        changed = restore(saved)
        if changed:
            print(f"    (put back {changed} file{'s' if changed > 1 else ''} in out/)")


def main(argv):
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if argv and argv[0] in ("list", "-l", "--list"):
        for key, step, title, _, _ in presets:
            print(f"  {key}  {step:9s} {title}")
    elif argv and argv[0] in by_key:
        run_in_terminal(by_key[argv[0]])
    elif argv:
        sys.exit(f"unknown preset {argv[0]!r}; use one of {', '.join(by_key)} or 'list', or no argument for the window")
    else:
        if os.name == "nt":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass
        import tkinter as tk
        root = tk.Tk()
        App(root)
        root.mainloop()


if __name__ == "__main__":
    main(sys.argv[1:])
