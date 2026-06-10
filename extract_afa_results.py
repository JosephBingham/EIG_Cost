#!/usr/bin/env python3
"""
Extract and analyze AFA-Benchmark evaluation results for the EIG-cost MIMIC-IV paper.

Walks `eval_results/eval_split-test/initializer-cold/` and discovers every
(method, eval_budget, eval_seed) combination. Parses the per-step CSVs that
`afabench.eval.eval_afa_method` writes, reconstructs trajectories, and produces
the tables and figures needed for the MLHC submission.

Usage
-----
    python extract_afa_results.py \\
        --root  /bigdata/MIMIC-IV/MIMIC-IV/AFA-Benchmark/extra/output/eval_results/eval_split-test/initializer-cold \\
        --groups /bigdata/MIMIC-IV/MIMIC-IV/AFA-Benchmark/extra/data/mimic_iv/groups.json \\
        --out    /bigdata/MIMIC-IV/MIMIC-IV/AFA-Benchmark/extra/output/analysis \\
        [--class-names class_names.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception as e:  # noqa: BLE001
    HAS_MPL = False
    print(f"[warn] matplotlib unavailable, skipping figures: {e}", file=sys.stderr)

try:
    from sklearn.metrics import f1_score, balanced_accuracy_score
    HAS_SKLEARN = True
except Exception as e:  # noqa: BLE001
    HAS_SKLEARN = False
    print(f"[warn] sklearn unavailable, macro-F1 will be skipped: {e}", file=sys.stderr)


logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("extract_afa")


# Method-family grouping. Keys are *directory* names under initializer-cold/.
METHOD_FAMILY: dict[str, tuple[str, str]] = {
    "eig_cost":          ("Information-theoretic (ours)", "EIG-Cost (ours)"),
    "dime":              ("Information-theoretic",        "DIME"),
    "gadgil2023":        ("Information-theoretic",        "DIME (Gadgil 2023)"),
    "eddi":              ("Information-theoretic",        "EDDI"),
    "ma2018":            ("Information-theoretic",        "EDDI (Ma 2018)"),
    "cae":               ("Information-theoretic",        "CAE"),
    "odin_mb":           ("Information-theoretic",        "ODIN (model-based)"),
    "odin_model_based":  ("Information-theoretic",        "ODIN (model-based)"),
    "aaco":              ("RL / order-learning",          "AACO"),
    "aaco_nn":           ("RL / order-learning",          "AACO-NN"),
    "jafa":              ("RL / order-learning",          "JAFA"),
    "ol":                ("RL / order-learning",          "OL"),
    "permutation":       ("Baselines",                    "Permutation"),
    "odin_mf":           ("Baselines",                    "ODIN (model-free)"),
    "odin_model_free":   ("Baselines",                    "ODIN (model-free)"),
    "random":            ("Baselines",                    "Random"),
}

FAMILY_ORDER = [
    "Information-theoretic (ours)",
    "Information-theoretic",
    "RL / order-learning",
    "Baselines",
    "Unknown",
]


@dataclass
class RunMeta:
    method: str
    display_name: str
    family: str
    eval_budget: int
    eval_seed: int
    csv_path: Path


_BUDGET_RE = re.compile(r"eval_hard_budget-(\d+|null)")
_SEED_RE   = re.compile(r"eval_seed-(\d+)")


def discover_runs(root: Path) -> list[RunMeta]:
    """Walk the tree and return one RunMeta per CSV file found."""
    runs: list[RunMeta] = []
    if not root.exists():
        raise FileNotFoundError(f"root does not exist: {root}")

    for csv_path in sorted(root.rglob("*.csv")):
        rel = csv_path.relative_to(root)
        parts = rel.parts
        if len(parts) < 2:
            log.warning("Skipping unexpected file at top level: %s", csv_path)
            continue
        method_dir = parts[0]
        path_str = str(csv_path)

        b_match = _BUDGET_RE.search(path_str)
        s_match = _SEED_RE.search(path_str)
        if b_match is None or s_match is None:
            log.warning("Could not parse budget/seed from %s; skipping", csv_path)
            continue

        budget_raw = b_match.group(1)
        eval_budget = -1 if budget_raw == "null" else int(budget_raw)
        eval_seed = int(s_match.group(1))

        family, display = METHOD_FAMILY.get(method_dir, ("Unknown", method_dir))
        runs.append(RunMeta(
            method=method_dir, display_name=display, family=family,
            eval_budget=eval_budget, eval_seed=eval_seed, csv_path=csv_path,
        ))
    return runs


def _parse_prev_selections(s: object) -> list[int]:
    if isinstance(s, list):
        return s
    if pd.isna(s):
        return []
    try:
        v = ast.literal_eval(s)
        return list(v) if isinstance(v, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def load_steps(meta: RunMeta) -> pd.DataFrame:
    df = pd.read_csv(meta.csv_path)
    df["prev_selections_performed"] = df["prev_selections_performed"].map(_parse_prev_selections)
    df["step_in_trajectory"] = df["prev_selections_performed"].str.len()
    df["method"] = meta.method
    df["display_name"] = meta.display_name
    df["family"] = meta.family
    df["eval_budget"] = meta.eval_budget
    df["eval_seed"] = meta.eval_seed
    # New trajectory starts whenever step_in_trajectory == 0
    df["traj_id_local"] = (df["step_in_trajectory"] == 0).cumsum() - 1
    return df


def trajectories_from_steps(step_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce per-step rows to one row per trajectory (terminal row per traj)."""
    keys = ["method", "eval_budget", "eval_seed", "traj_id_local"]
    last = step_df.groupby(keys, sort=False).tail(1).copy()
    last = last.rename(columns={
        "external_predicted_class": "external_pred",
        "builtin_predicted_class":  "builtin_pred",
        "accumulated_cost":         "final_cost",
        "step_in_trajectory":       "n_acquisitions",
    })
    last["action_sequence"] = last["prev_selections_performed"]
    cols = [
        "method", "display_name", "family",
        "eval_budget", "eval_seed", "traj_id_local",
        "true_class", "external_pred", "builtin_pred",
        "final_cost", "n_acquisitions", "forced_stop", "action_sequence",
    ]
    return last[cols].reset_index(drop=True)


def summarize(traj_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, eval_budget), aggregated across seeds if any."""
    g = traj_df.groupby(["family", "display_name", "method", "eval_budget"])

    def _agg(group: pd.DataFrame) -> pd.Series:
        y_true = group["true_class"].to_numpy()
        y_ext = group["external_pred"].to_numpy()
        y_blt = group["builtin_pred"].to_numpy()
        ext_valid = pd.notna(y_ext).all() and len(y_ext) > 0
        blt_valid = pd.notna(y_blt).all() and len(y_blt) > 0
        out = {
            "n_trajectories": len(group),
            "n_seeds":        group["eval_seed"].nunique(),
            "acc_external":   (y_ext == y_true).mean() if ext_valid else np.nan,
            "acc_builtin":    (y_blt == y_true).mean() if blt_valid else np.nan,
            "mean_cost":      group["final_cost"].mean(),
            "median_cost":    group["final_cost"].median(),
            "p25_cost":       group["final_cost"].quantile(0.25),
            "p75_cost":       group["final_cost"].quantile(0.75),
            "mean_n_acquisitions":   group["n_acquisitions"].mean(),
            "median_n_acquisitions": group["n_acquisitions"].median(),
            "voluntary_stop_rate":   (~group["forced_stop"]).mean(),
        }
        if HAS_SKLEARN and ext_valid:
            out["macro_f1_external"] = f1_score(y_true, y_ext, average="macro", zero_division=0)
            out["balanced_acc_external"] = balanced_accuracy_score(y_true, y_ext)
        return pd.Series(out)

    # Compat: include_groups added in pandas 2.1; drop the group cols manually for older versions.
    import inspect
    if "include_groups" in inspect.signature(g.apply).parameters:
        summary = g.apply(_agg, include_groups=False).reset_index()
    else:
        summary = g.apply(_agg).reset_index()
    summary["family_rank"] = summary["family"].map(
        {f: i for i, f in enumerate(FAMILY_ORDER)}
    ).fillna(len(FAMILY_ORDER))
    summary = summary.sort_values(
        ["family_rank", "display_name", "eval_budget"]
    ).drop(columns=["family_rank"]).reset_index(drop=True)
    return summary


def per_class_breakdown(
    traj_df: pd.DataFrame, n_classes: int, class_names: list[str]
) -> pd.DataFrame:
    rows = []
    grp = traj_df.groupby(["method", "display_name", "family", "eval_budget"])
    for (m, disp, fam, b), g in grp:
        for c in range(n_classes):
            mask = g["true_class"] == c
            n = int(mask.sum())
            if n == 0:
                rows.append({
                    "method": m, "display_name": disp, "family": fam,
                    "eval_budget": b, "class_id": c,
                    "class_name": class_names[c],
                    "n_patients": 0,
                    "accuracy_external": np.nan,
                    "mean_cost": np.nan,
                })
                continue
            y_ext = g.loc[mask, "external_pred"].to_numpy()
            acc = float((y_ext == c).mean()) if pd.notna(y_ext).all() else np.nan
            rows.append({
                "method": m, "display_name": disp, "family": fam,
                "eval_budget": b, "class_id": c,
                "class_name": class_names[c],
                "n_patients": n,
                "accuracy_external": acc,
                "mean_cost": float(g.loc[mask, "final_cost"].mean()),
            })
    return pd.DataFrame(rows)


def action_frequency_by_group(
    traj_df: pd.DataFrame,
    feature_to_group: list[int],
    group_names: list[str],
) -> pd.DataFrame:
    rows = []
    grp = traj_df.groupby(["method", "display_name", "family", "eval_budget"])
    for (m, disp, fam, b), g in grp:
        group_counter: Counter[int] = Counter()
        for seq in g["action_sequence"]:
            for feat_idx in seq:
                if 0 <= feat_idx < len(feature_to_group):
                    group_counter[feature_to_group[feat_idx]] += 1
        total = sum(group_counter.values())
        n_traj = len(g)
        for gid, gname in enumerate(group_names):
            cnt = group_counter.get(gid, 0)
            rows.append({
                "method": m, "display_name": disp, "family": fam,
                "eval_budget": b, "group_id": gid, "group_name": gname,
                "count": cnt,
                "fraction_of_acquisitions": cnt / total if total > 0 else 0.0,
                "acquisitions_per_patient": cnt / n_traj if n_traj > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def anytime_curves(
    step_df: pd.DataFrame, cost_grid: np.ndarray, step_grid: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-(method, budget) accuracy as a function of step or cumulative cost."""
    df = step_df.dropna(subset=["external_predicted_class"]).copy()
    df["correct"] = df["external_predicted_class"] == df["true_class"]
    keys = ["method", "display_name", "family", "eval_budget"]

    rows_step = []
    for k, g in df.groupby(keys, sort=False):
        for s in step_grid:
            sub = g[g["step_in_trajectory"] == s]
            if len(sub) == 0:
                continue
            rows_step.append({
                **dict(zip(keys, k)),
                "n_acquisitions": int(s),
                "accuracy": float(sub["correct"].mean()),
                "n_patients_at_step": int(len(sub)),
            })
    curve_step = pd.DataFrame(rows_step)

    rows_cost = []
    for k, g in df.groupby(keys, sort=False):
        traj_groups = g.groupby(["eval_seed", "traj_id_local"], sort=False)
        traj_arrays = []
        for _, t in traj_groups:
            t = t.sort_values("step_in_trajectory")
            traj_arrays.append((
                t["accumulated_cost"].to_numpy(),
                t["correct"].to_numpy(),
            ))
        for c in cost_grid:
            correct_count = 0
            n = 0
            for cost_arr, corr_arr in traj_arrays:
                idx = np.searchsorted(cost_arr, c, side="right") - 1
                if idx < 0:
                    continue
                n += 1
                correct_count += int(corr_arr[idx])
            if n == 0:
                continue
            rows_cost.append({
                **dict(zip(keys, k)),
                "cost_threshold": float(c),
                "accuracy": correct_count / n,
                "n_patients_at_cost": n,
            })
    curve_cost = pd.DataFrame(rows_cost)
    return curve_step, curve_cost


def render_main_table_md(summary: pd.DataFrame) -> str:
    cols = [
        "family", "display_name", "eval_budget",
        "n_trajectories", "acc_external", "macro_f1_external",
        "mean_cost", "median_cost", "mean_n_acquisitions",
        "voluntary_stop_rate",
    ]
    cols = [c for c in cols if c in summary.columns]
    out = summary[cols].copy()
    for c in ["acc_external", "macro_f1_external", "voluntary_stop_rate"]:
        if c in out.columns:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    for c in ["mean_cost", "median_cost"]:
        if c in out.columns:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"${v:.2f}")
    for c in ["mean_n_acquisitions"]:
        if c in out.columns:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    # Hand-rolled markdown (no `tabulate` dependency).
    headers = list(out.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, r in out.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in headers) + " |")
    return "\n".join(lines)


def render_main_table_tex(summary: pd.DataFrame) -> str:
    rows = [
        r"\begin{tabular}{l l r r r r r r}",
        r"\toprule",
        r"Family & Method & Budget & Acc. & Macro-F1 & Mean cost & Mean \#tests & Vol. stop \\",
        r"\midrule",
    ]
    current_family = None
    for _, r in summary.iterrows():
        fam = r["family"]
        fam_cell = fam if fam != current_family else ""
        current_family = fam
        acc = f"{r['acc_external']:.3f}" if pd.notna(r.get("acc_external")) else "--"
        f1 = (f"{r['macro_f1_external']:.3f}"
              if "macro_f1_external" in r and pd.notna(r["macro_f1_external"]) else "--")
        cost = f"\\${r['mean_cost']:.2f}" if pd.notna(r["mean_cost"]) else "--"
        nq = f"{r['mean_n_acquisitions']:.2f}" if pd.notna(r["mean_n_acquisitions"]) else "--"
        vs = f"{r['voluntary_stop_rate']:.2f}" if pd.notna(r["voluntary_stop_rate"]) else "--"
        rows.append(
            f"{fam_cell} & {r['display_name']} & {r['eval_budget']} & "
            f"{acc} & {f1} & {cost} & {nq} & {vs} \\\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def _highlight(name: str) -> bool:
    return "ours" in name.lower() or name.lower().startswith("eig-cost")


def plot_acc_vs_budget(summary: pd.DataFrame, out_path: Path) -> None:
    if not HAS_MPL: return
    fig, ax = plt.subplots(figsize=(7, 5))
    for disp, g in summary.groupby("display_name", sort=False):
        g = g.sort_values("eval_budget")
        lw = 2.6 if _highlight(disp) else 1.4
        ax.plot(g["eval_budget"], g["acc_external"], marker="o",
                linewidth=lw, label=disp)
    ax.set_xlabel("Budget (max # tests)")
    ax.set_ylabel("Top-1 accuracy (external classifier)")
    ax.set_title("Accuracy vs. acquisition budget")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_acc_vs_cost(curve_cost: pd.DataFrame, out_path: Path) -> None:
    """Headline figure: accuracy as a function of cumulative $ spent."""
    if not HAS_MPL or len(curve_cost) == 0: return
    fig, ax = plt.subplots(figsize=(7, 5))
    for disp, g in curve_cost.groupby("display_name", sort=False):
        best_b = g["eval_budget"].max()
        gb = g[g["eval_budget"] == best_b].sort_values("cost_threshold")
        if len(gb) == 0: continue
        lw = 2.6 if _highlight(disp) else 1.4
        ax.plot(gb["cost_threshold"], gb["accuracy"],
                linewidth=lw, label=f"{disp} (B={best_b})")
    ax.set_xlabel("Cumulative cost ($ USD)")
    ax.set_ylabel("Top-1 accuracy (external classifier)")
    ax.set_title("Accuracy vs. cumulative test cost")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_per_class_heatmap(per_class: pd.DataFrame, target_budget: int, out_path: Path) -> None:
    if not HAS_MPL: return
    sub = per_class[per_class["eval_budget"] == target_budget].copy()
    if len(sub) == 0:
        target_budget = per_class["eval_budget"].max()
        sub = per_class[per_class["eval_budget"] == target_budget].copy()
        if len(sub) == 0: return
    pivot = sub.pivot_table(index="class_name", columns="display_name",
                            values="accuracy_external", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.0 * len(pivot.columns) + 2,
                                    0.4 * len(pivot.index) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Per-class accuracy (budget = {target_budget})")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Accuracy")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_test_group_usage(action_freq: pd.DataFrame, target_budget: int, out_path: Path) -> None:
    if not HAS_MPL: return
    sub = action_freq[action_freq["eval_budget"] == target_budget].copy()
    if len(sub) == 0:
        target_budget = action_freq["eval_budget"].max()
        sub = action_freq[action_freq["eval_budget"] == target_budget].copy()
        if len(sub) == 0: return
    pivot = sub.pivot_table(index="group_name", columns="display_name",
                            values="acquisitions_per_patient", aggfunc="mean").fillna(0.0)
    fig, ax = plt.subplots(figsize=(1.0 * len(pivot.columns) + 2,
                                    0.4 * len(pivot.index) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Mean acquisitions per patient, by test group (budget = {target_budget})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def sanity_checks(
    runs: list[RunMeta],
    step_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    n_classes: int,
) -> list[str]:
    out: list[str] = []
    methods = sorted({r.method for r in runs})
    out.append(f"Discovered methods: {methods}")

    unknown = [r.method for r in runs if r.family == "Unknown"]
    if unknown:
        out.append("UNKNOWN METHOD DIRS (add them to METHOD_FAMILY): "
                   f"{sorted(set(unknown))}")

    cov = (traj_df.groupby(["method", "eval_budget"]).size()
           .unstack(fill_value=0).sort_index())
    out.append("Trajectory counts per (method, budget):\n" + cov.to_string())

    null_external = (traj_df.assign(_null=traj_df["external_pred"].isna())
                     .groupby("method")["_null"].mean())
    bad = null_external[null_external > 0]
    if len(bad) > 0:
        out.append("METHODS WITH MISSING EXTERNAL PREDICTIONS (fraction): "
                   f"{bad.to_dict()}")

    has_builtin = (traj_df.assign(_has=traj_df["builtin_pred"].notna())
                   .groupby("method")["_has"].mean())
    out.append(f"Builtin-classifier coverage per method: {has_builtin.to_dict()}")

    obs_max = int(traj_df["true_class"].max())
    obs_min = int(traj_df["true_class"].min())
    out.append(f"Observed true_class range: [{obs_min}, {obs_max}] "
               f"(configured n_classes={n_classes})")
    if obs_max >= n_classes:
        out.append(f"WARNING: observed class id {obs_max} >= n_classes {n_classes}")

    class_counts = traj_df.groupby(["method", "eval_budget", "true_class"]).size()
    rare = class_counts[class_counts < 5]
    if len(rare) > 0:
        out.append(f"VERY RARE CLASSES (<5 patients) in {len(rare)} cells: "
                   "per-class accuracy will be noisy")

    forced = traj_df.groupby("method")["forced_stop"].mean()
    extreme = forced[(forced < 0.05) | (forced > 0.95)]
    if len(extreme) > 0:
        out.append("NOTE: methods with extreme forced-stop rates "
                   f"(<5% or >95%): {extreme.to_dict()}")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--class-names", type=Path, default=None)
    parser.add_argument("--n-classes", type=int, default=21)
    parser.add_argument("--target-budget", type=int, default=10)
    parser.add_argument("--cost-grid-n", type=int, default=40)
    parser.add_argument("--step-grid-max", type=int, default=30)
    parser.add_argument("--no-figures", action="store_true",
                        help="skip matplotlib figure generation (use if env is broken)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    (args.out / "tables").mkdir(exist_ok=True)

    with args.groups.open() as f:
        groups_data = json.load(f)
    feature_to_group: list[int] = groups_data["feature_to_group"]
    group_names: list[str] = groups_data["group_names"]
    log.info("Loaded %d features in %d groups", len(feature_to_group), len(group_names))

    if args.class_names:
        with args.class_names.open() as f:
            class_names: list[str] = json.load(f)
    else:
        class_names = [f"class_{i}" for i in range(args.n_classes)]

    runs = discover_runs(args.root)
    if not runs:
        log.error("No CSV files found under %s", args.root)
        return 1
    log.info("Discovered %d run files across %d methods",
             len(runs), len({r.method for r in runs}))

    step_frames = []
    for r in runs:
        try:
            step_frames.append(load_steps(r))
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to load %s: %s", r.csv_path, e)
    step_df = pd.concat(step_frames, ignore_index=True)
    log.info("Loaded %d total step rows", len(step_df))

    traj_df = trajectories_from_steps(step_df)
    log.info("Reconstructed %d trajectories", len(traj_df))

    try:
        traj_df.to_parquet(args.out / "per_trajectory.parquet", index=False)
        step_df.drop(columns=["prev_selections_performed"]).to_parquet(
            args.out / "per_step.parquet", index=False
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Parquet write failed (%s); falling back to CSV", e)
        traj_df.to_csv(args.out / "per_trajectory.csv", index=False)

    summary = summarize(traj_df)
    summary.to_csv(args.out / "summary_by_method_budget.csv", index=False)
    (args.out / "tables" / "main_results.md").write_text(render_main_table_md(summary))
    (args.out / "tables" / "main_results.tex").write_text(render_main_table_tex(summary))
    log.info("Wrote summary table (%d rows)", len(summary))

    per_class = per_class_breakdown(traj_df, args.n_classes, class_names)
    per_class.to_csv(args.out / "per_class_accuracy.csv", index=False)

    action_freq = action_frequency_by_group(traj_df, feature_to_group, group_names)
    action_freq.to_csv(args.out / "action_frequency.csv", index=False)

    max_cost = float(step_df["accumulated_cost"].max() or 0.0)
    cost_grid = np.linspace(0, max_cost, args.cost_grid_n) if max_cost > 0 else np.array([0.0])
    step_grid = np.arange(0, args.step_grid_max + 1)
    curve_step, curve_cost = anytime_curves(step_df, cost_grid, step_grid)
    curve_step.to_csv(args.out / "curve_accuracy_vs_step.csv", index=False)
    curve_cost.to_csv(args.out / "curve_accuracy_vs_cost.csv", index=False)

    if HAS_MPL and not args.no_figures:
        try:
            plot_acc_vs_budget(summary, args.out / "figures" / "accuracy_vs_budget.pdf")
            plot_acc_vs_cost(curve_cost, args.out / "figures" / "accuracy_vs_cost.pdf")
            plot_per_class_heatmap(per_class, args.target_budget,
                                   args.out / "figures" / "per_class_heatmap.pdf")
            plot_test_group_usage(action_freq, args.target_budget,
                                  args.out / "figures" / "test_group_usage.pdf")
            log.info("Wrote figures to %s", args.out / "figures")
        except Exception as e:  # noqa: BLE001
            log.warning("Figure generation failed (%s); CSV outputs are still complete.", e)
    elif args.no_figures:
        log.info("Skipping figures (--no-figures set).")

    warns = sanity_checks(runs, step_df, traj_df, args.n_classes)
    (args.out / "warnings.txt").write_text("\n\n".join(warns))
    log.info("Wrote warnings.txt with %d notes", len(warns))

    log.info("Done. Outputs in %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
