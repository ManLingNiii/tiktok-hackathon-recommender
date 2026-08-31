"""Task-0 train-only feature governance for the KuaiRand agent."""
import csv
import json
import os
from pathlib import Path

try:
    from dataset_config import data_dir, dataset_name, runs_dir
except ImportError:
    from agent.dataset_config import data_dir, dataset_name, runs_dir


CATALOG_VERSION = "pure-feature-catalog-v1"
RAW_FIELDS = (
    "date", "user_id", "video_id", "author_id", "tab", "duration_ms", "hourmin"
)
EXCLUDED_FEEDBACK = (
    "long_view", "is_click", "is_like", "is_follow", "is_comment",
    "is_forward", "play_time_ms",
)


def _suffix():
    return "_1k" if dataset_name() in {"1k", "kuairand_1k"} else "_pure"


def _train_log(path):
    return os.path.join(path, f"log_standard_4_08_to_4_21{_suffix()}.csv")


def _video_table(path):
    return os.path.join(path, f"video_features_basic{_suffix()}.csv")


def _inside_project(path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.commonpath([root, os.path.abspath(path)]) == root


def _feature(name, source, dtype, missing_policy, unique_policy,
             inference_available=True, family_scope=("composition_fm",)):
    return {
        "name": name,
        "source": source,
        "dtype": dtype,
        "missing_ratio": 0.0,
        "missing_policy": missing_policy,
        "unique_count": None,
        "unique_count_policy": unique_policy,
        "inference_available": inference_available,
        "family_scope": list(family_scope),
        "leakage_safe": True,
    }


def build_catalog(root=None):
    """Build a catalog by reading only the approved training log."""
    path = os.path.abspath(root or data_dir())
    if not _inside_project(path):
        raise ValueError("feature catalog data path must remain inside the project")
    log_path = _train_log(path)
    video_path = _video_table(path)
    if not os.path.isfile(log_path) or not os.path.isfile(video_path):
        raise FileNotFoundError(log_path)

    authors = {}
    with open(video_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            authors[row["video_id"]] = row.get("author_id", "UNK") or "UNK"

    counts = {field: 0 for field in RAW_FIELDS}
    unique = {field: set() for field in RAW_FIELDS}
    train_rows = 0
    with open(log_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        for row in reader:
            train_rows += 1
            values = {
                "date": row.get("date", ""),
                "user_id": row.get("user_id", ""),
                "video_id": row.get("video_id", ""),
                "author_id": authors.get(row.get("video_id", ""), "UNK"),
                "tab": row.get("tab", ""),
                "duration_ms": row.get("duration_ms", ""),
                "hourmin": row.get("hourmin", ""),
            }
            for field, value in values.items():
                value = str(value).strip()
                if not value:
                    counts[field] += 1
                else:
                    unique[field].add(value)

    features = [
        _feature("duration_ms", "train log raw duration_ms", "float32",
                 "reject missing rows", "exact unique count in train"),
        _feature("tab", "train log raw tab", "categorical",
                 "map unseen values to UNK", "train vocabulary plus UNK"),
        _feature("hourmin", "train log raw hourmin", "int32",
                 "fill missing with 0", "train vocabulary plus UNK"),
        _feature("user_history_count", "prior train long_view history per user", "float32",
                 "fill unseen users with 0", "integer count; no vocabulary"),
        _feature("video_exposure_count", "train exposure count per video", "float32",
                 "fill unseen videos with 0", "integer count; no vocabulary"),
        _feature("family_prediction_std", "five frozen family predictions", "float32",
                 "fail on non-finite prediction", "continuous; not categorical"),
        _feature("family_prediction_range", "five frozen family predictions", "float32",
                 "fail on non-finite prediction", "continuous; not categorical"),
        _feature("family_prediction_mean_abs_disagreement", "five frozen family predictions", "float32",
                 "fail on non-finite prediction", "continuous; not categorical"),
    ]
    raw_map = {x["name"]: x for x in features if x["name"] in RAW_FIELDS}
    for field, item in raw_map.items():
        item["missing_ratio"] = counts[field] / max(train_rows, 1)
        item["unique_count"] = len(unique[field])

    return {
        "catalog_version": CATALOG_VERSION,
        "dataset": dataset_name(),
        "source_split": "train_only",
        "source_files": [os.path.relpath(log_path, os.path.dirname(os.path.dirname(log_path))),
                         os.path.relpath(video_path, os.path.dirname(os.path.dirname(video_path)))],
        "train_rows": train_rows,
        "raw_fields": headers,
        "normalized_fields": list(RAW_FIELDS),
        "excluded_feedback": list(EXCLUDED_FEEDBACK),
        "selected_features": [],
        "features": features,
        "approved_interactions": [
            {"name": "p_bpr_duration", "source": "frozen bpr prediction x duration_ms",
             "inference_available": True, "leakage_safe": True, "family_scope": ["composition_fm"]}
        ],
        "leakage_audit": {
            "label_columns_used_as_current_row_features": False,
            "future_history_used": False,
            "validation_statistics_fit_from_train_only": True,
            "test_or_hidden_test_loaded": False,
            "zero_variance_policy": "replace zero std with 1.0 and keep finite output",
        },
    }


def validate_catalog(catalog):
    if catalog.get("catalog_version") != CATALOG_VERSION:
        raise ValueError("unsupported feature catalog version")
    if catalog.get("source_split") != "train_only":
        raise ValueError("feature catalog must be train-only")
    if catalog.get("selected_features") != []:
        raise ValueError("Task 0 catalog must not preselect Task 1 features")
    audit = catalog.get("leakage_audit", {})
    if any(audit.get(key) is not expected for key, expected in (
        ("label_columns_used_as_current_row_features", False),
        ("future_history_used", False),
        ("validation_statistics_fit_from_train_only", True),
        ("test_or_hidden_test_loaded", False),
    )):
        raise ValueError("feature catalog leakage audit failed")
    for feature in catalog.get("features", []):
        if not feature.get("leakage_safe") or not feature.get("inference_available"):
            raise ValueError(f"feature is not safe and inference-available: {feature.get('name')}")
        if not 0.0 <= float(feature.get("missing_ratio", 1.0)) <= 1.0:
            raise ValueError(f"invalid missing ratio: {feature.get('name')}")
    return True


def ensure_feature_catalog(project_root=None):
    project_root = os.path.abspath(project_root or os.path.join(os.path.dirname(__file__), ".."))
    output = Path(runs_dir()) / "feature_catalog.json"
    if output.exists():
        catalog = json.loads(output.read_text(encoding="utf-8"))
        validate_catalog(catalog)
        return catalog
    catalog = build_catalog(data_dir())
    validate_catalog(catalog)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


if __name__ == "__main__":
    result = ensure_feature_catalog()
    print(json.dumps({"catalog": str(Path(runs_dir()) / "feature_catalog.json"),
                      "catalog_version": result["catalog_version"],
                      "train_rows": result["train_rows"],
                      "selected_features": result["selected_features"],
                      "validated": True}, ensure_ascii=False, indent=2))
