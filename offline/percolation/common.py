"""Settings and helpers shared by the offline percolation scripts.

Protocol: the one-cluster study of Brill & Carlson, arXiv:2606.20347 (Sec. 5, App. B, Tables 1, 4, 5).
"""
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Data (paper Table 4, one-cluster column).
N_POINTS = 200_000
DIM = 100
DATA_SEEDS = {"graph": 0, "embed": 10_000, "value": 20_000}

# The authors' generator, aribrill/percolation-synthetic-data, MIT license.
GENERATOR_REPO = "aribrill/percolation-synthetic-data"
GENERATOR_COMMIT = "3dd47a808f02d161a34615570814f107ab1c1627"
GENERATOR_FILE = "percolation_dataset.py"
GENERATOR_SHA256 = "a9e2edc7f6b14d371d65a5b6ce4c4d5b20b8fffd56bb0adfc4a0926e44d3900a"

# Network (paper Table 5, one-cluster column).
D_MODEL, N_BLOCKS, EXPANSION = 256, 3, 4
BATCH, EPOCHS, LR, WEIGHT_DECAY, ETA_MIN = 1024, 500, 1e-4, 0.01, 1e-6
SPLIT = (160_000, 20_000, 20_000)  # 80/10/10
TRAIN_SEED = 42

# Probes (paper Sec. 5). Depths 0 and 5 have a constant target under the 150-point filter.
ANC_DEPTHS = list(range(0, 201, 5))
PROBE_DEPTHS = list(range(10, 201, 5))
MIN_LATENT_POINTS = 150
PROBE_SEED = 42
PROBE_VAL_FRAC = 0.3
RIDGE_ALPHA = 1.0
MIN_BASELINE_MSE = 1e-4
SITES = ["input_layer", "hidden0", "block0", "hidden1", "block1", "hidden2", "block2"]
HIDDEN_SITES = ["hidden0", "hidden1", "hidden2"]
RESIDUAL_SITES = ["input_layer", "block0", "block1", "block2"]


def load_generator(path):
    """Import the authors' generator from its checkout after checking the file hash."""
    path = Path(path).resolve()
    src = path / GENERATOR_FILE
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    if digest != GENERATOR_SHA256:
        raise SystemExit(f"{src} has sha256 {digest}, expected {GENERATOR_SHA256} (commit {GENERATOR_COMMIT[:7]})")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path))
    import percolation_dataset  # noqa: E402

    return percolation_dataset


def pick_device(float64=False):
    """cuda, then mps, then cpu. MPS has no float64, so float64 work falls back to the CPU there."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and not float64:
        return torch.device("mps")
    return torch.device("cpu")


def gpu_peak_mb(dev):
    import torch
    if dev.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2**20)
    if dev.type == "mps":
        return round(torch.mps.driver_allocated_memory() / 2**20)
    return 0


def array_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def peak_rss_mb():
    """Peak resident memory of this process (macOS reports bytes, Linux kilobytes)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def pipeline_revision():
    """Git HEAD of the repository plus a hash of every pipeline file (the files may be uncommitted)."""
    head, dirty = None, None  # None: not run from a git checkout (the file hashes identify the code)
    try:
        r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True)
        if r.returncode == 0:
            head = r.stdout.strip()
            dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", "."],
                                   capture_output=True, text=True).stdout.strip() != ""
    except OSError:
        pass
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(HERE.glob("*.py"))}
    return {"git_head": head, "uncommitted_changes": dirty, "files_sha256_16": files}


def environment():
    import scipy
    info = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "machine": platform.machine(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        dev = pick_device()
        info["device"] = dev.type
        if dev.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=_json_default) + "\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def r2_mse(y, pred):
    y = np.asarray(y, np.float64)
    pred = np.asarray(pred, np.float64)
    mse = float(np.mean((y - pred) ** 2))
    return {"r2": 1.0 - mse / float(np.var(y)), "mse": mse}


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks = {}

    def mark(self, name):
        self.marks[name] = round(time.perf_counter() - self.t0, 2)
        return self.marks[name]
