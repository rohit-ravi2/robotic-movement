"""The platform never depends on demonstration data: no module under movement_discovery imports or reads motion-capture assets
(LAFAN, AMASS, BVH, the mocap_baseline script, human_planar). 'mocap' alone is MuJoCo's kinematic-body term and is allowed."""
import os, re

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"(lafan1?|amass|mocap_baseline|human_planar|\.bvh\b|bvh_|motion_capture|motioncapture)", re.I)


def _py_files():
    for root, _, files in os.walk(PKG):
        if "tests" in root.split(os.sep) or "__pycache__" in root: continue
        for f in files:
            if f.endswith(".py"): yield os.path.join(root, f)


def test_no_mocap_imports_or_paths():
    hits = []
    for p in _py_files():
        for i, line in enumerate(open(p), 1):
            if PAT.search(line): hits.append(f"{os.path.relpath(p, PKG)}:{i}: {line.strip()[:100]}")
    assert not hits, "demonstration-data reference(s):\n" + "\n".join(hits)


def test_no_trajectory_files_loaded():
    """No np.load / pickle.load of a *motion* file outside the archive's own side files and the stored throw prior (a body-derived reference, not a recording)."""
    bad = []
    for p in _py_files():
        for i, line in enumerate(open(p), 1):
            if re.search(r"(np\.load|pickle\.load|loadtxt)\(", line) and not re.search(r"(archive|refs/|traces/|prior|\.npz|_meta)", line): bad.append(f"{os.path.relpath(p, PKG)}:{i}: {line.strip()[:100]}")
    assert not bad, "\n".join(bad)
