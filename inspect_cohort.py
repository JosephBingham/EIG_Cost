#!/usr/bin/env python
"""Inspect and verify a built 5-class cohort.

Standalone sanity-check for the output of build_cohort.py. Verifies:
  - All expected files present, correct shapes and dtypes
  - No NaN values in any X array
  - Labels in valid range [0, n_classes - 1]
  - Group structure internally consistent
  - costs.npy values match groups.json group_costs
  - Per-class counts in each split

Exits 0 if everything checks out, 1 if any assertion fails.

Compatibility
-------------
Python 3.8+. No 3.10/3.12 syntax.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def fail(msg, errors):
    print("  FAIL: " + msg)
    errors.append(msg)


def ok(msg):
    print("  OK: " + msg)


def inspect(d):
    errors = []

    required = [
        "X_train.npy", "X_val.npy", "X_test.npy",
        "y_train.npy", "y_val.npy", "y_test.npy",
        "costs.npy", "groups.json", "metadata.json",
    ]
    print("\n[1/6] File presence in " + str(d))
    missing = [f for f in required if not (d / f).exists()]
    if missing:
        fail("missing files: " + str(missing), errors)
        return 1
    ok("all {} expected files present".format(len(required)))

    print("\n[2/6] Loading arrays")
    X = {}
    y = {}
    for s in ("train", "val", "test"):
        X[s] = np.load(str(d / ("X_" + s + ".npy")))
        y[s] = np.load(str(d / ("y_" + s + ".npy")))
    costs = np.load(str(d / "costs.npy"))
    with open(str(d / "groups.json")) as f:
        groups = json.load(f)
    with open(str(d / "metadata.json")) as f:
        meta = json.load(f)
    ok("all arrays load without error")

    n_features = X["train"].shape[1]
    n_classes = meta.get("n_classes")
    n_groups = len(groups["group_names"])

    print("\n[3/6] Shape and dtype")
    for s in ("train", "val", "test"):
        if X[s].dtype != np.float32:
            fail(
                "X_{} dtype is {}, expected float32".format(s, X[s].dtype),
                errors,
            )
        if y[s].dtype != np.int32:
            fail(
                "y_{} dtype is {}, expected int32".format(s, y[s].dtype),
                errors,
            )
        if X[s].shape[1] != n_features:
            fail(
                "X_{} has {} features but X_train has {}".format(
                    s, X[s].shape[1], n_features
                ),
                errors,
            )
        if X[s].shape[0] != y[s].shape[0]:
            fail(
                "X_{} has {} rows but y_{} has {}".format(
                    s, X[s].shape[0], s, y[s].shape[0]
                ),
                errors,
            )
    if costs.shape != (n_features,):
        fail(
            "costs.npy shape {} != ({},)".format(costs.shape, n_features),
            errors,
        )
    if not errors:
        ok("shapes consistent: features={}, classes={}, groups={}".format(
            n_features, n_classes, n_groups
        ))

    print("\n[4/6] Value sanity")
    nan_errors = []
    for s in ("train", "val", "test"):
        n_nan = int(np.isnan(X[s]).sum())
        if n_nan > 0:
            msg = "X_{} has {} NaN values".format(s, n_nan)
            fail(msg, errors)
            nan_errors.append(msg)
        if y[s].min() < 0 or y[s].max() >= n_classes:
            msg = "y_{} labels [{}, {}] outside [0, {}]".format(
                s, y[s].min(), y[s].max(), n_classes - 1
            )
            fail(msg, errors)
            nan_errors.append(msg)
    if bool(np.isnan(costs).any()):
        fail("costs.npy contains NaN", errors)
        nan_errors.append("costs NaN")
    if bool((costs < 0).any()):
        fail("costs.npy contains negative values", errors)
        nan_errors.append("negative cost")
    if not nan_errors:
        ok("no NaN in X, all y in valid range, costs non-negative")

    print("\n[5/6] Group structure")
    group_errors = []
    if len(groups["feature_to_group"]) != n_features:
        msg = "feature_to_group length {} != {}".format(
            len(groups["feature_to_group"]), n_features
        )
        fail(msg, errors)
        group_errors.append(msg)
    if len(groups["group_costs"]) != n_groups:
        msg = "group_costs length {} != {}".format(
            len(groups["group_costs"]), n_groups
        )
        fail(msg, errors)
        group_errors.append(msg)
    if len(groups["group_cpt_codes"]) != n_groups:
        msg = "group_cpt_codes length {} != {}".format(
            len(groups["group_cpt_codes"]), n_groups
        )
        fail(msg, errors)
        group_errors.append(msg)
    max_gid = max(groups["feature_to_group"])
    min_gid = min(groups["feature_to_group"])
    if max_gid >= n_groups or min_gid < 0:
        msg = "feature_to_group group ids out of range: [{}, {}], n_groups={}".format(
            min_gid, max_gid, n_groups
        )
        fail(msg, errors)
        group_errors.append(msg)
    used = set(groups["feature_to_group"])
    if len(used) != n_groups:
        unused = [i for i in range(n_groups) if i not in used]
        msg = "some declared groups have no features: " + str(unused)
        fail(msg, errors)
        group_errors.append(msg)
    cost_mismatches = []
    for i, gid in enumerate(groups["feature_to_group"]):
        if not np.isclose(costs[i], groups["group_costs"][gid], atol=1e-3):
            cost_mismatches.append(
                "feature {}: costs.npy={} but group {} cost={}".format(
                    i, costs[i], gid, groups["group_costs"][gid]
                )
            )
    if cost_mismatches:
        msg = "costs.npy disagrees with group_costs for {} features".format(
            len(cost_mismatches)
        )
        fail(msg, errors)
        group_errors.append(msg)
    if not group_errors:
        ok("group structure consistent (all {} groups used, costs match)".format(
            n_groups
        ))

    print("\n[6/6] Per-class counts and balance")
    print("  class names: " + str(meta["class_names"]))
    for s in ("train", "val", "test"):
        ys = y[s]
        print("  --- {} (N={:,}) ---".format(s, len(ys)))
        counts = np.bincount(ys, minlength=n_classes)
        for c in range(n_classes):
            n = int(counts[c])
            pct = 100.0 * n / len(ys) if len(ys) else 0.0
            print("    {}: {:25s} {:6,d} ({:5.2f}%)".format(
                c, meta["class_names"][c], n, pct
            ))
        if counts.min() > 0:
            balance = float(counts.min()) / float(counts.max())
            print("    balance min/max = {:.3f}".format(balance))
        else:
            empty = [int(i) for i in np.where(counts == 0)[0].tolist()]
            fail("y_{} has empty class(es): {}".format(s, empty), errors)

    print("\n" + "=" * 70)
    if errors:
        print("FAILED: {} issue(s) found".format(len(errors)))
        for e in errors:
            print("  - " + e)
        return 1
    print("ALL CHECKS PASSED")
    print("=" * 70)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort-dir", type=Path,
        default=Path(
            "/bigdata/MIMIC-IV/MIMIC-IV/AFA-Benchmark/extra/data/mimic_iv_5class"
        ),
        help="Directory containing the built cohort",
    )
    args = p.parse_args()
    if not args.cohort_dir.is_dir():
        print("ERROR: " + str(args.cohort_dir) + " is not a directory")
        return 2
    return inspect(args.cohort_dir)


if __name__ == "__main__":
    sys.exit(main())
