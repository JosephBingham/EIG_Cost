#!/usr/bin/env python
"""Build a 5-class subset of MIMIC-IV for AFA benchmarking.

Reads the existing AFA-Benchmark MIMIC-IV dataset, drops the ESR feature
(which is 100% missing for all patients), filters patients to 5 selected
diseases, remaps labels to 0-4, and writes a new dataset directory with
identical file layout that can be registered as a separate dataset in
AFA-Benchmark.

Compatibility
-------------
Requires only Python 3.8+ and numpy. Uses no 3.10+ syntax (no `X | Y`
unions, no `list[int]` generics), no 3.12 syntax (no PEP 695 type
aliases, no `typing.override`).

Why this exists
---------------
The original MIMIC-IV dataset has 21 highly imbalanced classes (60 vs
13614). With most baselines trained without a budget, they collapse to
majority-class prediction and the comparison becomes meaningless. This
script produces a smaller, less imbalanced 5-class subset focused on
five methodologically distinct acute conditions:

  - diabetic_emergency (endocrine)
  - myocardial_infarction (cardiac)
  - pancreatitis (GI)
  - sepsis (infectious)
  - stroke (neurologic)

Each has its own discriminating expensive test (HbA1c+pH; troponin;
lipase; lactate+blood-culture; clinical/imaging respectively), so the
AFA acquisition decision is informative.

The single all-NaN feature (ESR) is dropped because strict-intersection
on the original 55 features yields an empty cohort. After dropping ESR
all remaining 54 features are 100% observed.

Class imbalance is preserved (no downsampling). Expected ratio in train
is min/max = 4590/13614 = 0.337.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("cohort")


# Default selection -- 5 methodologically distinct acute conditions.
DEFAULT_SELECTED_CLASS_NAMES = (
    "diabetic_emergency",
    "myocardial_infarction",
    "pancreatitis",
    "sepsis",
    "stroke",
)

# Features with NaN rate above this threshold are dropped wholesale.
# Set to 1.0 to drop only fully-empty columns (currently just ESR).
NAN_DROP_THRESHOLD = 0.999


# ---------------------------------------------------------------------------
# Step 0: load inputs
# ---------------------------------------------------------------------------
def load_input(in_dir):
    """Load all input arrays and JSON metadata.

    Verifies expected file presence and basic shape consistency before
    returning. Raises FileNotFoundError or ValueError on problems.
    """
    required = [
        "X_train.npy", "X_val.npy", "X_test.npy",
        "y_train.npy", "y_val.npy", "y_test.npy",
        "costs.npy", "groups.json", "metadata.json",
    ]
    missing = [f for f in required if not (in_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files in {}: {}".format(in_dir, missing)
        )

    X = {}
    y = {}
    for s in ("train", "val", "test"):
        X[s] = np.load(str(in_dir / ("X_" + s + ".npy")))
        y[s] = np.load(str(in_dir / ("y_" + s + ".npy")))
    costs = np.load(str(in_dir / "costs.npy"))

    with open(str(in_dir / "groups.json")) as f:
        groups = json.load(f)
    with open(str(in_dir / "metadata.json")) as f:
        meta = json.load(f)

    # Sanity checks on shape consistency
    n_features = X["train"].shape[1]
    for s in ("train", "val", "test"):
        if X[s].shape[1] != n_features:
            raise ValueError(
                "X_{} has {} features but X_train has {}".format(
                    s, X[s].shape[1], n_features
                )
            )
        if X[s].shape[0] != y[s].shape[0]:
            raise ValueError(
                "X_{} has {} rows but y_{} has {}".format(
                    s, X[s].shape[0], s, y[s].shape[0]
                )
            )

    if costs.shape != (n_features,):
        raise ValueError(
            "costs.npy has shape {} but expected ({},)".format(
                costs.shape, n_features
            )
        )

    if len(groups["feature_to_group"]) != n_features:
        raise ValueError(
            "groups.json feature_to_group has {} entries but X has {} features"
            .format(len(groups["feature_to_group"]), n_features)
        )

    # Verify costs are consistent with group_costs
    g2c = groups["group_costs"]
    for i, gid in enumerate(groups["feature_to_group"]):
        if not np.isclose(costs[i], g2c[gid], atol=1e-3):
            raise ValueError(
                "Cost mismatch at feature {}: costs.npy says {} but groups.json"
                " says group {} has cost {}".format(i, costs[i], gid, g2c[gid])
            )

    return {"X": X, "y": y, "costs": costs, "groups": groups, "meta": meta}


# ---------------------------------------------------------------------------
# Step 1: identify features to drop based on NaN rate
# ---------------------------------------------------------------------------
def identify_features_to_drop(X_train, threshold, feature_names):
    """Return sorted list of feature indices with NaN rate > threshold."""
    nan_rate = np.isnan(X_train).mean(axis=0)
    drop_idx = sorted([int(i) for i, r in enumerate(nan_rate) if r > threshold])
    if drop_idx:
        log.info("Features to drop (NaN rate > %.3f):", threshold)
        for i in drop_idx:
            log.info(
                "  feature %d (%s): NaN rate %.4f",
                i, feature_names[i], nan_rate[i],
            )
    else:
        log.info("No features exceed NaN threshold %.3f", threshold)
    return drop_idx


# ---------------------------------------------------------------------------
# Step 2: drop features and renumber groups
# ---------------------------------------------------------------------------
def drop_features_and_groups(data, drop_feature_idx):
    """Drop the specified feature columns and any groups that become empty.

    Returns a new data dict with feature dimension reduced and groups
    metadata renumbered. Original data is not modified.
    """
    n_features_old = data["X"]["train"].shape[1]
    keep_feature_mask = np.ones(n_features_old, dtype=bool)
    keep_feature_mask[drop_feature_idx] = False

    # Drop features from X and costs
    new_X = {}
    for s in ("train", "val", "test"):
        new_X[s] = data["X"][s][:, keep_feature_mask]
    new_costs = data["costs"][keep_feature_mask]

    # Determine which groups become empty
    old_feature_to_group = np.asarray(data["groups"]["feature_to_group"])
    kept_feature_to_group_old = old_feature_to_group[keep_feature_mask]
    remaining_old_group_ids = set(int(g) for g in kept_feature_to_group_old)
    n_groups_old = len(data["groups"]["group_names"])
    dropped_old_group_ids = sorted(
        g for g in range(n_groups_old) if g not in remaining_old_group_ids
    )
    if dropped_old_group_ids:
        dropped_names = [
            data["groups"]["group_names"][g] for g in dropped_old_group_ids
        ]
        log.info(
            "Dropping %d empty groups: %s",
            len(dropped_old_group_ids), dropped_names,
        )

    # Build old_group_id -> new_group_id mapping
    new_group_id_of = {}
    next_new = 0
    for old in range(n_groups_old):
        if old in remaining_old_group_ids:
            new_group_id_of[old] = next_new
            next_new += 1

    # Remap feature_to_group
    new_feature_to_group = [
        new_group_id_of[int(g)] for g in kept_feature_to_group_old
    ]
    new_group_names = [
        data["groups"]["group_names"][g]
        for g in range(n_groups_old)
        if g in remaining_old_group_ids
    ]
    new_group_costs = [
        data["groups"]["group_costs"][g]
        for g in range(n_groups_old)
        if g in remaining_old_group_ids
    ]
    new_group_cpt_codes = [
        data["groups"]["group_cpt_codes"][g]
        for g in range(n_groups_old)
        if g in remaining_old_group_ids
    ]

    # Update feature_names if present in metadata
    old_feature_names = data["meta"].get("feature_names", [])
    if old_feature_names:
        new_feature_names = [
            old_feature_names[i]
            for i in range(n_features_old)
            if keep_feature_mask[i]
        ]
    else:
        new_feature_names = []

    log.info(
        "After feature drop: %d -> %d features, %d -> %d groups",
        n_features_old, int(keep_feature_mask.sum()),
        n_groups_old, len(new_group_names),
    )

    new_meta = dict(data["meta"])
    new_meta["feature_names"] = new_feature_names

    return {
        "X": new_X,
        "y": data["y"],
        "costs": new_costs,
        "groups": {
            "feature_to_group": new_feature_to_group,
            "group_names": new_group_names,
            "group_costs": new_group_costs,
            "group_cpt_codes": new_group_cpt_codes,
        },
        "meta": new_meta,
    }


# ---------------------------------------------------------------------------
# Step 3: filter to 5 classes and remap labels
# ---------------------------------------------------------------------------
def filter_classes(data, selected_class_names):
    """Filter rows to those with label in selected_class_names; remap to 0..k-1."""
    all_class_names = data["meta"]["class_names"]
    name_to_old_id = {n: i for i, n in enumerate(all_class_names)}

    missing = [n for n in selected_class_names if n not in name_to_old_id]
    if missing:
        raise ValueError(
            "Selected class names not found in metadata: {}. Available: {}"
            .format(missing, all_class_names)
        )

    # Sort selected by their old label so the new mapping is stable
    selected_with_old_id = sorted(
        [(name_to_old_id[n], n) for n in selected_class_names],
        key=lambda x: x[0],
    )
    old_ids = [oid for oid, _ in selected_with_old_id]
    new_class_names = [name for _, name in selected_with_old_id]
    old_to_new = {oid: new_i for new_i, (oid, _) in enumerate(selected_with_old_id)}

    log.info("Class remapping (old -> new):")
    for new_i, (oid, name) in enumerate(selected_with_old_id):
        log.info("  %d -> %d  %s", oid, new_i, name)

    new_X = {}
    new_y = {}
    for s in ("train", "val", "test"):
        y_old = data["y"][s]
        keep_mask = np.isin(y_old, old_ids)
        new_X[s] = data["X"][s][keep_mask]
        # Remap labels
        y_kept = y_old[keep_mask]
        y_new = np.empty(int(keep_mask.sum()), dtype=np.int32)
        for oid, nid in old_to_new.items():
            y_new[y_kept == oid] = nid
        new_y[s] = y_new
        log.info(
            "  %s: kept %d / %d rows", s, int(keep_mask.sum()), len(y_old)
        )

    new_meta = dict(data["meta"])
    new_meta["class_names"] = new_class_names
    new_meta["n_classes"] = len(new_class_names)
    return {
        "X": new_X,
        "y": new_y,
        "costs": data["costs"],
        "groups": data["groups"],
        "meta": new_meta,
    }


# ---------------------------------------------------------------------------
# Step 4: verify and save
# ---------------------------------------------------------------------------
def verify(data):
    """Sanity checks on the final dataset. Raises AssertionError on problems."""
    n_features = data["X"]["train"].shape[1]
    n_classes = data["meta"]["n_classes"]
    n_groups = len(data["groups"]["group_names"])
    log.info("Final verification:")
    log.info("  features: %d", n_features)
    log.info("  groups:   %d", n_groups)
    log.info("  classes:  %d", n_classes)

    for s in ("train", "val", "test"):
        X = data["X"][s]
        y = data["y"][s]
        n_nan = int(np.isnan(X).sum())
        if n_nan > 0:
            raise AssertionError(
                "X_{} still has {} NaN values -- feature drop incomplete"
                .format(s, n_nan)
            )
        if y.min() < 0 or y.max() >= n_classes:
            raise AssertionError(
                "y_{} has labels outside [0, {}]: min={}, max={}".format(
                    s, n_classes - 1, y.min(), y.max()
                )
            )
        if X.shape[0] != y.shape[0]:
            raise AssertionError(
                "X_{} and y_{} length mismatch: {} vs {}".format(
                    s, s, X.shape[0], y.shape[0]
                )
            )
        if X.shape[1] != n_features:
            raise AssertionError(
                "X_{} has {} features but expected {}".format(
                    s, X.shape[1], n_features
                )
            )

    if data["costs"].shape != (n_features,):
        raise AssertionError(
            "costs has shape {} but expected ({},)".format(
                data["costs"].shape, n_features
            )
        )
    if len(data["groups"]["feature_to_group"]) != n_features:
        raise AssertionError(
            "feature_to_group length does not match feature count"
        )
    max_gid = max(data["groups"]["feature_to_group"])
    if max_gid >= n_groups:
        raise AssertionError(
            "feature_to_group has group id {} but only {} groups".format(
                max_gid, n_groups
            )
        )
    if bool(np.isnan(data["costs"]).any()):
        raise AssertionError("costs.npy contains NaN")

    log.info("  OK: no NaN in X arrays")
    log.info("  OK: all labels in valid range")
    log.info("  OK: shape consistency confirmed")
    log.info("  OK: group structure consistent")


def save_output(data, out_dir):
    """Write all arrays and JSON to out_dir. Creates the directory if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in ("train", "val", "test"):
        np.save(
            str(out_dir / ("X_" + s + ".npy")),
            data["X"][s].astype(np.float32),
        )
        np.save(
            str(out_dir / ("y_" + s + ".npy")),
            data["y"][s].astype(np.int32),
        )
    np.save(str(out_dir / "costs.npy"), data["costs"].astype(np.float32))

    with open(str(out_dir / "groups.json"), "w") as f:
        json.dump(data["groups"], f, indent=2)

    new_meta = dict(data["meta"])
    new_meta["name"] = data["meta"].get("name", "") + " (5-class subset)"
    new_meta["n_features"] = int(data["X"]["train"].shape[1])
    new_meta["n_groups"] = len(data["groups"]["group_names"])
    new_meta["n_classes"] = data["meta"]["n_classes"]
    new_meta["train_size"] = int(data["X"]["train"].shape[0])
    new_meta["val_size"] = int(data["X"]["val"].shape[0])
    new_meta["test_size"] = int(data["X"]["test"].shape[0])
    new_meta["total_cost"] = float(sum(data["groups"]["group_costs"]))
    new_meta["derived_from"] = "mimic_iv (21-class)"
    new_meta["selection_rationale"] = (
        "Five methodologically distinct acute conditions; "
        "imbalance preserved; ESR feature dropped (100% NaN)."
    )

    with open(str(out_dir / "metadata.json"), "w") as f:
        json.dump(new_meta, f, indent=2)

    log.info("Wrote output to %s", out_dir)
    for f in sorted(out_dir.iterdir()):
        log.info("  %s (%d bytes)", f.name, f.stat().st_size)


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
def print_summary(data):
    """Print final per-class counts for sanity-checking against expectations."""
    print("\n" + "=" * 70)
    print("FINAL COHORT SUMMARY")
    print("=" * 70)
    n_classes = data["meta"]["n_classes"]
    for s in ("train", "val", "test"):
        y = data["y"][s]
        print("\n--- {} (N={:,}) ---".format(s, len(y)))
        counts = np.bincount(y, minlength=n_classes)
        for cid in range(n_classes):
            n = int(counts[cid])
            pct = 100.0 * n / len(y) if len(y) else 0.0
            name = data["meta"]["class_names"][cid]
            print("  {}: {:25s} {:6,d} ({:5.2f}%)".format(cid, name, n, pct))
        if counts.min() > 0:
            print("  balance min/max = {:.3f}".format(
                float(counts.min()) / float(counts.max())
            ))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir", type=Path,
        default=Path(
            "/bigdata/MIMIC-IV/MIMIC-IV/AFA-Benchmark/extra/data/mimic_iv"
        ),
        help="Directory containing original X_*.npy, y_*.npy, etc.",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "/bigdata/MIMIC-IV/MIMIC-IV/AFA-Benchmark/extra/data/mimic_iv_5class"
        ),
        help="Directory to write new 5-class arrays.",
    )
    p.add_argument(
        "--classes", nargs="+",
        default=list(DEFAULT_SELECTED_CLASS_NAMES),
        help="Class names to keep (must match metadata.json class_names).",
    )
    p.add_argument(
        "--nan-threshold", type=float, default=NAN_DROP_THRESHOLD,
        help="Drop features whose NaN rate in TRAIN exceeds this.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Allow overwriting existing output directory.",
    )
    args = p.parse_args()

    if (args.output_dir.exists()
            and any(args.output_dir.iterdir())
            and not args.force):
        log.error(
            "Output directory %s is not empty. Pass --force to overwrite.",
            args.output_dir,
        )
        return 2

    log.info("Loading input from %s", args.input_dir)
    data = load_input(args.input_dir)
    log.info(
        "Loaded: train=%d, val=%d, test=%d, features=%d, groups=%d, classes=%d",
        data["X"]["train"].shape[0],
        data["X"]["val"].shape[0],
        data["X"]["test"].shape[0],
        data["X"]["train"].shape[1],
        len(data["groups"]["group_names"]),
        len(data["meta"]["class_names"]),
    )

    n_feat = data["X"]["train"].shape[1]
    feat_names = data["meta"].get(
        "feature_names", ["f{}".format(i) for i in range(n_feat)]
    )
    drop_idx = identify_features_to_drop(
        data["X"]["train"], args.nan_threshold, feat_names
    )
    data = drop_features_and_groups(data, drop_idx)

    log.info("Selecting classes: %s", args.classes)
    data = filter_classes(data, args.classes)

    verify(data)
    save_output(data, args.output_dir)
    print_summary(data)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
