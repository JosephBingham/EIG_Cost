#!/usr/bin/env python3
"""
Diagnose why a baseline AFA method (DIME by default) collapses to trivial
behavior on MIMIC-IV — i.e., why it voluntarily stops at step 0 or 1 on
nearly every patient and reports ~7.8% accuracy by predicting class 18.

For each of N test patients, this script:
  1. Resets to an empty observation (only the initialized features observable).
  2. Runs the method's policy step-by-step, printing at each step:
     - which features are currently observed,
     - the full softmax distribution over actions (acquire each of K features, or stop=0),
     - the action chosen and the mass it received,
     - the external classifier's class prediction + top-3 probabilities,
     - (if has_builtin) the method's builtin classifier's prediction.
  3. Stops when the policy picks 0 OR when budget is exhausted.
  4. Summarises across the 10 patients: how often is action 0 chosen on step 0?
     How does external-classifier output change as features are acquired?
     Is the policy's action distribution flat, peaked on 0, or peaked on a feature?

Usage
-----
    python diagnose_dime.py \\
        --method-name gadgil2023 \\
        --method-checkpoint /bigdata/.../method.pt \\
        --external-classifier /bigdata/.../external_classifier.pt \\
        --dataset mimic_iv \\
        --n-patients 10 \\
        --max-steps 30 \\
        --device cuda:0

This script does NOT modify any model weights. It only runs inference.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("diag")


# ---------------------------------------------------------------------------
# Imports from afabench. Adjust if your method lives under a different path.
# ---------------------------------------------------------------------------
def import_method_class(method_name: str):
    """Find and import the AFAMethod subclass for `method_name`.

    Tries common locations under afabench. Adjust the candidate list if your
    repo organises methods differently.
    """
    candidates = [
        f"afabench.methods.{method_name}.afa_method",
        f"afabench.methods.{method_name}",
        f"afabench.{method_name}.afa_method",
        f"afabench.{method_name}",
    ]
    for modpath in candidates:
        try:
            mod = importlib.import_module(modpath)
        except ImportError:
            continue
        # Find the AFAMethod subclass inside the module
        from afabench.common.custom_types import AFAMethod
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (isinstance(attr, type)
                    and issubclass(attr, AFAMethod)
                    and attr is not AFAMethod):
                log.info("Loaded method %s.%s", modpath, attr_name)
                return attr
    raise ImportError(
        f"Could not find an AFAMethod subclass for '{method_name}' under "
        f"afabench. Tried: {candidates}. Pass --method-module-path to override."
    )


def import_dataset(dataset_name: str):
    """Import the dataset class. Adjust if MimicIVDataset lives elsewhere."""
    from afabench.common.datasets.datasets import MimicIVDataset
    if dataset_name.lower() in ("mimic_iv", "mimiciv", "mimic-iv"):
        return MimicIVDataset
    raise ValueError(
        f"Unknown dataset {dataset_name}. Add it to import_dataset() in this script."
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def fmt_prob_row(probs: np.ndarray, top_k: int = 5, name_fn=None) -> str:
    """Return e.g. '  18: 0.812 | 12: 0.094 | 8: 0.041 | 0: 0.020 | 9: 0.011'."""
    order = np.argsort(-probs)[:top_k]
    parts = []
    for i in order:
        label = name_fn(int(i)) if name_fn else str(int(i))
        parts.append(f"{label}: {float(probs[i]):.3f}")
    return "  " + " | ".join(parts)


def fmt_observed(mask: np.ndarray, feature_names: list[str] | None) -> str:
    obs_idx = np.where(mask)[0]
    if len(obs_idx) == 0:
        return "(none)"
    if feature_names is not None:
        return ", ".join(feature_names[i] for i in obs_idx)
    return ", ".join(str(int(i)) for i in obs_idx)


# ---------------------------------------------------------------------------
# Step-by-step rollout with verbose printing
# ---------------------------------------------------------------------------
@dataclass
class StepRecord:
    step: int
    chosen_action: int               # 0 = stop, else 1-indexed feature
    chosen_action_prob: float | None # mass the policy placed on its choice
    policy_top5: list[tuple[int, float]] | None  # (action_id, prob)
    external_top3: list[tuple[int, float]]
    builtin_top3: list[tuple[int, float]] | None
    accumulated_cost: float


@dataclass
class PatientTrace:
    patient_idx: int
    true_class: int
    steps: list[StepRecord]
    final_external_pred: int
    final_builtin_pred: int | None
    final_cost: float
    forced_stop: bool


def policy_distribution_via_score_sweep(
    method,
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
    selection_mask: torch.Tensor,
    label: torch.Tensor,
    feature_shape: torch.Size,
    n_features: int,
    device: torch.device,
) -> np.ndarray | None:
    """Try to extract a per-action distribution from the method's policy.

    Many AFAMethod implementations expose a `score()` or `policy()` method that
    returns logits over actions. If so, we use it. Otherwise we return None
    and only report the chosen action.
    """
    for attr in ("score", "policy", "action_distribution", "_action_logits"):
        fn = getattr(method, attr, None)
        if fn is None or not callable(fn):
            continue
        try:
            with torch.no_grad():
                out = fn(
                    masked_features=masked_features,
                    feature_mask=feature_mask,
                    selection_mask=selection_mask,
                    label=label,
                    feature_shape=feature_shape,
                )
        except TypeError:
            try:
                with torch.no_grad():
                    out = fn(masked_features, feature_mask)
            except Exception:  # noqa: BLE001
                continue
        except Exception:  # noqa: BLE001
            continue
        # Coerce to numpy 1D over (n_features+1) actions
        if isinstance(out, torch.Tensor):
            arr = out.detach().cpu().numpy().squeeze()
        else:
            arr = np.asarray(out).squeeze()
        if arr.ndim == 1 and len(arr) in (n_features, n_features + 1):
            # Apply softmax for stability
            arr = arr - arr.max()
            probs = np.exp(arr) / np.exp(arr).sum()
            return probs
    return None


def trace_one_patient(
    method,
    features: torch.Tensor,
    label: torch.Tensor,
    afa_initialize_fn,
    afa_unmask_fn,
    external_predict_fn,
    n_selection_choices: int,
    feature_shape: torch.Size,
    feature_costs: list[float],
    feature_names: list[str] | None,
    class_names: list[str] | None,
    max_steps: int,
    budget: float,
    patient_idx: int,
    device: torch.device,
) -> PatientTrace:
    """Run one patient through `max_steps` and record everything."""
    n_features = feature_shape[0]

    # Initialise observation
    feature_mask = afa_initialize_fn(
        features, label, feature_shape=feature_shape
    ).to(device)
    masked_features = features.clone()
    masked_features[~feature_mask] = 0.0

    selection_mask = torch.zeros(
        (1, n_selection_choices), device=device, dtype=torch.bool
    )

    true_class = int(label.argmax(-1).item())
    print("\n" + "=" * 78)
    print(f"PATIENT idx={patient_idx} true_class={true_class}"
          + (f" ({class_names[true_class]})" if class_names else ""))
    print("=" * 78)

    steps: list[StepRecord] = []
    accumulated_cost = 0.0
    forced_stop = False
    has_builtin = bool(getattr(method, "has_builtin_classifier", False))

    for step in range(max_steps):
        print(f"\n--- step {step} ---")
        print(f"observed features: "
              f"{fmt_observed(feature_mask[0].cpu().numpy().astype(bool), feature_names)}")
        print(f"accumulated cost: ${accumulated_cost:.2f}")

        # External classifier output BEFORE this step's acquisition
        with torch.no_grad():
            ext_logits = external_predict_fn(
                masked_features=masked_features,
                feature_mask=feature_mask,
                label=label,
                feature_shape=feature_shape,
            )
        ext_probs = torch.softmax(ext_logits, dim=-1).detach().cpu().numpy().squeeze()
        ext_top3 = [(int(i), float(ext_probs[i]))
                    for i in np.argsort(-ext_probs)[:3]]
        print("external classifier (top 3):")
        print(fmt_prob_row(ext_probs, top_k=3,
                           name_fn=(lambda i: class_names[i]) if class_names else None))

        # Builtin classifier output
        builtin_top3 = None
        if has_builtin:
            with torch.no_grad():
                blt_logits = method.predict(
                    masked_features=masked_features.squeeze(0),
                    feature_mask=feature_mask.squeeze(0),
                    label=label,
                    feature_shape=feature_shape,
                )
            blt_probs = blt_logits.detach().cpu().numpy().squeeze()
            builtin_top3 = [(int(i), float(blt_probs[i]))
                            for i in np.argsort(-blt_probs)[:3]]
            print("builtin classifier (top 3):")
            print(fmt_prob_row(blt_probs, top_k=3,
                               name_fn=(lambda i: class_names[i]) if class_names else None))

        # Try to get the policy's full action distribution
        action_probs = policy_distribution_via_score_sweep(
            method=method,
            masked_features=masked_features,
            feature_mask=feature_mask,
            selection_mask=selection_mask,
            label=label,
            feature_shape=feature_shape,
            n_features=n_features,
            device=device,
        )

        # Always call act() to get the chosen action
        with torch.no_grad():
            action_tensor = method.act(
                masked_features=masked_features,
                feature_mask=feature_mask,
                selection_mask=selection_mask,
                label=label,
                feature_shape=feature_shape,
            )
        chosen_action = int(action_tensor.squeeze().item())

        # Report policy info
        policy_top5 = None
        chosen_prob = None
        if action_probs is not None:
            # Action index convention: try both (0..n_features) and (1..n_features+1)
            if len(action_probs) == n_features + 1:
                # 0 = stop, 1..n = acquire feature (chosen_action is in this space)
                idx_for_chosen = chosen_action
            else:
                # length == n_features: assume these are over features only;
                # chosen_action 0 = stop has no entry; otherwise feature idx
                idx_for_chosen = chosen_action - 1 if chosen_action > 0 else None
            if idx_for_chosen is not None and 0 <= idx_for_chosen < len(action_probs):
                chosen_prob = float(action_probs[idx_for_chosen])
            top_idx = np.argsort(-action_probs)[:5]
            policy_top5 = [(int(i), float(action_probs[i])) for i in top_idx]
            print("policy distribution (top 5 actions):")
            for i, p in policy_top5:
                label_str = (f"STOP" if (len(action_probs) == n_features + 1 and i == 0)
                             else (feature_names[i - 1] if (feature_names and i > 0 and i <= len(feature_names))
                                   else f"feature_{i - 1}" if len(action_probs) == n_features + 1
                                   else feature_names[i] if feature_names else f"feature_{i}"))
                print(f"  {i:4d} ({label_str}): {p:.4f}")
        else:
            print("policy distribution: (method does not expose .score()/.policy() — only .act() was called)")

        print(f"CHOSEN ACTION: {chosen_action}"
              + (f"  (mass={chosen_prob:.4f})" if chosen_prob is not None else "")
              + ("  [STOP]" if chosen_action == 0
                 else f"  [acquire feature {chosen_action - 1}"
                      + (f" = {feature_names[chosen_action - 1]}" if feature_names else "")
                      + "]"))

        steps.append(StepRecord(
            step=step,
            chosen_action=chosen_action,
            chosen_action_prob=chosen_prob,
            policy_top5=policy_top5,
            external_top3=ext_top3,
            builtin_top3=builtin_top3,
            accumulated_cost=accumulated_cost,
        ))

        if chosen_action == 0:
            print(">> stopping (voluntary)")
            break

        # Check budget BEFORE applying the action
        feat_idx = chosen_action - 1
        step_cost = feature_costs[feat_idx] if feat_idx < len(feature_costs) else 1.0
        if accumulated_cost + step_cost > budget:
            print(f">> stopping (budget would be exceeded: "
                  f"${accumulated_cost:.2f} + ${step_cost:.2f} > ${budget:.2f})")
            forced_stop = True
            break

        # Apply acquisition
        afa_selection = action_tensor - 1
        new_mask = afa_unmask_fn(
            masked_features=masked_features,
            feature_mask=feature_mask,
            features=features,
            afa_selection=afa_selection,
            selection_mask=selection_mask,
            label=label,
            feature_shape=feature_shape,
        )
        feature_mask = new_mask
        masked_features = features.clone()
        masked_features[~feature_mask] = 0.0
        selection_mask[0, feat_idx] = True
        accumulated_cost += step_cost

    # Final predictions
    with torch.no_grad():
        final_ext_logits = external_predict_fn(
            masked_features=masked_features,
            feature_mask=feature_mask,
            label=label,
            feature_shape=feature_shape,
        )
    final_ext_pred = int(final_ext_logits.argmax(-1).item())

    final_builtin_pred = None
    if has_builtin:
        with torch.no_grad():
            final_blt = method.predict(
                masked_features=masked_features.squeeze(0),
                feature_mask=feature_mask.squeeze(0),
                label=label,
                feature_shape=feature_shape,
            )
        final_builtin_pred = int(final_blt.argmax(-1).item())

    print(f"\nFINAL: external_pred={final_ext_pred}"
          + (f" ({class_names[final_ext_pred]})" if class_names else "")
          + f" | true={true_class}"
          + (f" ({class_names[true_class]})" if class_names else "")
          + f" | cost=${accumulated_cost:.2f} | steps={len(steps)}"
          + (" | FORCED" if forced_stop else ""))

    return PatientTrace(
        patient_idx=patient_idx,
        true_class=true_class,
        steps=steps,
        final_external_pred=final_ext_pred,
        final_builtin_pred=final_builtin_pred,
        final_cost=accumulated_cost,
        forced_stop=forced_stop,
    )


# ---------------------------------------------------------------------------
# Across-patient summary
# ---------------------------------------------------------------------------
def summarise(traces: list[PatientTrace], class_names: list[str] | None) -> None:
    print("\n\n" + "#" * 78)
    print("# SUMMARY ACROSS PATIENTS")
    print("#" * 78)
    n = len(traces)

    stops_at_0 = sum(1 for t in traces if len(t.steps) == 1 and t.steps[0].chosen_action == 0)
    stops_at_1 = sum(1 for t in traces if len(t.steps) >= 2 and t.steps[1].chosen_action == 0)
    print(f"\nPatients: {n}")
    print(f"Voluntarily stopped at step 0 (no acquisition): {stops_at_0}/{n}")
    print(f"Voluntarily stopped at step 1 (after 1 acquisition): {stops_at_1}/{n}")
    print(f"Mean steps: {np.mean([len(t.steps) for t in traces]):.2f}")
    print(f"Mean acquisitions (non-stop actions): "
          f"{np.mean([sum(1 for s in t.steps if s.chosen_action != 0) for t in traces]):.2f}")

    # Did external classifier change its top-1 between step 0 and the end?
    changed = sum(1 for t in traces
                  if t.steps and t.steps[0].external_top3[0][0] != t.final_external_pred)
    print(f"External classifier top-1 changed during episode: {changed}/{n}")

    # What does external classifier predict on the FIRST step (i.e., with only
    # initialised features observable, before any acquisition)?
    initial_preds = [t.steps[0].external_top3[0][0] for t in traces if t.steps]
    from collections import Counter
    init_dist = Counter(initial_preds)
    print(f"External classifier predictions at step 0 (over {len(initial_preds)} patients):")
    for c, k in init_dist.most_common():
        name = f" ({class_names[c]})" if class_names else ""
        print(f"  class {c}{name}: {k}")

    # Final accuracy on this small sample
    correct = sum(1 for t in traces if t.final_external_pred == t.true_class)
    print(f"\nFinal external-classifier accuracy on these {n} patients: {correct}/{n}")

    # Policy-mass-on-STOP at step 0
    masses_on_stop = []
    for t in traces:
        if t.steps and t.steps[0].policy_top5 is not None:
            for aid, p in t.steps[0].policy_top5:
                if aid == 0:
                    masses_on_stop.append(p)
                    break
    if masses_on_stop:
        print(f"Policy mass on STOP at step 0: "
              f"mean={np.mean(masses_on_stop):.3f}, "
              f"min={np.min(masses_on_stop):.3f}, "
              f"max={np.max(masses_on_stop):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method-name", default="gadgil2023",
                   help="Method directory name under afabench.methods (e.g. gadgil2023, ma2018, jafa)")
    p.add_argument("--method-checkpoint", type=Path, required=True,
                   help="Path to the directory containing method.pt for the method")
    p.add_argument("--external-classifier", type=Path, required=True,
                   help="Path to the external classifier checkpoint used at eval time")
    p.add_argument("--dataset", default="mimic_iv")
    p.add_argument("--data-root", type=Path,
                   default=Path("/bigdata/MIMIC-IV/MIMIC-IV/AFA-Benchmark/extra/data/mimic_iv"),
                   help="Directory containing the dataset's numpy arrays + groups.json")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-patients", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=30,
                   help="Cap on rollout steps per patient (also used as the cost budget surrogate)")
    p.add_argument("--budget", type=float, default=30.0,
                   help="Cost budget in dollars (set to a large value to never force-stop)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--class-names", type=Path, default=None,
                   help="Optional JSON list of 21 condition names")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Class names
    class_names: list[str] | None = None
    if args.class_names:
        with args.class_names.open() as f:
            class_names = json.load(f)

    # Group / feature metadata for human-readable printing
    groups_path = args.data_root / "groups.json"
    feature_names: list[str] | None = None
    feature_costs: list[float] = []
    if groups_path.exists():
        with groups_path.open() as f:
            g = json.load(f)
        feature_to_group: list[int] = g["feature_to_group"]
        group_names: list[str] = g["group_names"]
        group_costs: list[float] = g["group_costs"]
        feature_names = [f"{group_names[gid]}[{i}]"
                         for i, gid in enumerate(feature_to_group)]
        feature_costs = [group_costs[gid] for gid in feature_to_group]
        log.info("Loaded %d features mapped to %d groups",
                 len(feature_names), len(group_names))

    # Dataset
    DatasetCls = import_dataset(args.dataset)
    dataset = DatasetCls(root=args.data_root, split=args.split)
    log.info("Loaded %s split with %d samples", args.split, len(dataset))

    # Method
    MethodCls = import_method_class(args.method_name)
    method = MethodCls.load(args.method_checkpoint, device=str(device))
    log.info("Loaded method %s from %s", args.method_name, args.method_checkpoint)

    # External classifier predict_fn — adjust to your repo's API.
    # Most AFA-Benchmark setups have a helper to build this; we try a couple.
    try:
        from afabench.common.classifiers import load_external_classifier_predict_fn
        external_predict_fn = load_external_classifier_predict_fn(
            args.external_classifier, device=device
        )
    except ImportError:
        log.warning("Could not import load_external_classifier_predict_fn; "
                    "falling back to a generic torch.load. You may need to adapt this.")
        clf = torch.load(args.external_classifier, map_location=device)
        clf.eval()
        def external_predict_fn(masked_features, feature_mask, label=None, feature_shape=None):
            with torch.no_grad():
                return clf(masked_features, feature_mask)

    # AFA helpers — adjust paths if your repo organises these differently.
    from afabench.common.initialize import cold_initialize_fn as afa_initialize_fn
    from afabench.common.unmask import standard_unmask_fn as afa_unmask_fn

    feature_shape = dataset.feature_shape
    n_selection_choices = feature_shape[0]

    # Sample N patients with diverse true classes for an informative diagnostic
    indices = []
    seen_classes = set()
    perm = torch.randperm(len(dataset)).tolist()
    for i in perm:
        _, y = dataset[i]
        c = int(y.argmax(-1).item())
        # Prefer one patient per class, then fall back to filling out the sample
        if c not in seen_classes:
            indices.append(i)
            seen_classes.add(c)
        if len(indices) >= args.n_patients:
            break
    if len(indices) < args.n_patients:
        remaining = [i for i in perm if i not in indices]
        indices.extend(remaining[: args.n_patients - len(indices)])
    indices = indices[: args.n_patients]
    log.info("Sampled %d patients covering true classes %s",
             len(indices),
             sorted({int(dataset[i][1].argmax(-1).item()) for i in indices}))

    traces: list[PatientTrace] = []
    for patient_idx in indices:
        features, label = dataset[patient_idx]
        features = features.unsqueeze(0).to(device)  # (1, n_features)
        label = label.unsqueeze(0).to(device)
        trace = trace_one_patient(
            method=method,
            features=features,
            label=label,
            afa_initialize_fn=afa_initialize_fn,
            afa_unmask_fn=afa_unmask_fn,
            external_predict_fn=external_predict_fn,
            n_selection_choices=n_selection_choices,
            feature_shape=feature_shape,
            feature_costs=feature_costs if feature_costs else [1.0] * n_selection_choices,
            feature_names=feature_names,
            class_names=class_names,
            max_steps=args.max_steps,
            budget=args.budget,
            patient_idx=patient_idx,
            device=device,
        )
        traces.append(trace)

    summarise(traces, class_names)

    return 0


if __name__ == "__main__":
    sys.exit(main())
