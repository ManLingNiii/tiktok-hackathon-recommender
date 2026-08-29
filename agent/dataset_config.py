"""Approved KuaiRand dataset locations; raw data remains external/read-only."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_1K = os.environ.get("AGENT_1K_DATA_DIR", os.path.join(ROOT, "kuairand-starter-kit", "KuaiRand-1K", "data"))
DATASETS = {"pure": os.path.join(ROOT, "kuairand-starter-kit", "KuaiRand-Pure", "data"), "1k": LOCAL_1K, "kuairand_1k": LOCAL_1K}
def dataset_name(): return os.environ.get("AGENT_DATASET", "pure").lower()
def data_dir():
    name = dataset_name()
    if name not in DATASETS: raise ValueError(f"dataset is not allowlisted: {name}")
    path = DATASETS[name]
    if not os.path.isdir(path): raise FileNotFoundError(path)
    return path
def runs_dir(): return os.path.join(ROOT, "runs", dataset_name())
def outputs_dir(): return os.path.join(ROOT, "outputs", dataset_name())
