#!/usr/bin/env python
"""Apply all EIG-Cost pipeline integration changes to the AFA-Benchmark repo.

Run from the repo root:
    python3 install_eig_cost_integration.py [--repo /path/to/repo] [--dry-run]

Changes applied:
  1. scripts/train/eig_cost.py           — new pipeline-compatible training script
  2. extra/conf/scripts/train/eig_cost/  — Hydra config directory
  3. extra/workflow/conf/method_options.yaml  — register eig_cost
  4. extra/workflow/conf/methods.yaml         — add to methods list + main set
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO_DEFAULT = Path("/bigdata/MIMIC-IV/MIMIC-IV/AFA-Benchmark")

# ── File contents ─────────────────────────────────────────────────────────────

TRAIN_SCRIPT = '''\
"""Train EIG-Cost AFA method and save as an AFA-Benchmark bundle.

Pipeline-integrated version: uses Hydra for config, reads dataset from
a bundle path, saves weights and method.pt inside the save_path directory.

Design note
-----------
EIG-Cost training is budget-agnostic — the MaskableClassifier and CVAE are
trained once and the cost penalty (cost_lambda) is applied at inference time.
The pipeline generates one training job per hard_budget value, so the same
weights are trained multiple times when multiple budgets are evaluated. This
is wasteful but correct and keeps the pipeline structure uniform. The
hard_budget argument is accepted but not used during training.
"""

import logging
import os
import sys
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/eig_cost",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    import tensorflow as tf
    from afabench.common.bundle import load_bundle
    from afabench.eig_cost.models import build_maskable_classifier, build_tabular_mae
    from afabench.eig_cost.afa_method import EIGCostAFAMethod

    if cfg.seed is not None:
        np.random.seed(int(cfg.seed))
        tf.random.set_seed(int(cfg.seed))

    log.info("Loading training data from %s", cfg.train_dataset_bundle_path)
    train_ds, _ = load_bundle(Path(cfg.train_dataset_bundle_path))
    X_train, y_train = train_ds.get_all_data()
    X_np = X_train.numpy().astype(np.float32)
    y_np = y_train.argmax(dim=-1).numpy().astype(np.int32)
    n_features = int(X_np.shape[1])
    n_classes = int(y_train.shape[1])
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

    if hasattr(train_ds, "get_feature_acquisition_costs"):
        costs = train_ds.get_feature_acquisition_costs().tolist()
    else:
        costs = [1.0] * n_features

    log.info("Dataset: %s samples, %d features, %d classes",
             X_np.shape[0], n_features, n_classes)

    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    cls_path = str(save_path / "classifier.weights.h5")
    mae_path = str(save_path / "mae.weights.h5")

    batch_size = int(cfg.get("batch_size", 256))
    classifier_epochs = int(cfg.get("classifier_epochs", 50))
    mae_epochs = int(cfg.get("mae_epochs", 30))
    n_mc_samples = int(cfg.get("n_mc_samples", 300))
    cost_lambda = float(cfg.get("cost_lambda", 0.005))

    log.info("Training MaskableClassifier (%d epochs)...", classifier_epochs)
    classifier = build_maskable_classifier(n_features, n_classes)
    optimizer = tf.keras.optimizers.Adam(5e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    class_counts = np.bincount(y_np, minlength=n_classes).astype(np.float32)
    class_weights = len(y_np) / (n_classes * class_counts + 1e-6)

    cls_ds = (
        tf.data.Dataset.from_tensor_slices((X_np, y_np))
        .shuffle(10000).batch(batch_size).prefetch(2)
    )
    for epoch in range(classifier_epochs):
        for features, labels in cls_ds:
            bs = tf.shape(features)[0]
            frac = tf.random.uniform([bs, 1], 0.1, 1.0)
            mask = tf.cast(tf.random.uniform([bs, n_features]) < frac, tf.float32)
            use_full = tf.cast(tf.random.uniform([bs, 1]) < 0.3, tf.float32)
            mask = mask * (1 - use_full) + tf.ones([bs, n_features]) * use_full
            with tf.GradientTape() as tape:
                logits = classifier(features, mask, training=True)
                w = tf.gather(tf.constant(class_weights, dtype=tf.float32), labels)
                loss = loss_fn(labels, logits, sample_weight=w)
            grads = tape.gradient(loss, classifier.trainable_variables)
            optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
        if (epoch + 1) % 10 == 0:
            log.info("  Classifier epoch %d/%d", epoch + 1, classifier_epochs)
    classifier.save_weights(cls_path)
    log.info("Classifier weights saved to %s", cls_path)

    log.info("Training CVAE (%d epochs)...", mae_epochs)
    mae = build_tabular_mae(n_features)
    mae_opt = tf.keras.optimizers.Adam(5e-4)
    mae_ds = (
        tf.data.Dataset.from_tensor_slices(X_np)
        .shuffle(10000).batch(batch_size).prefetch(2)
    )
    for epoch in range(mae_epochs):
        kl_w = min(1.0, (epoch + 1) / 10.0) * 0.05
        for batch in mae_ds:
            bs = tf.shape(batch)[0]
            frac = tf.random.uniform([bs, 1], 0.0, 0.95)
            mask = tf.cast(tf.random.uniform([bs, n_features]) < frac, tf.float32)
            with tf.GradientTape() as tape:
                fm, zm, zlv = mae(batch, mask, training=True)
                unmasked = 1.0 - mask
                rl = tf.reduce_sum((batch - fm) ** 2 * unmasked) / (tf.reduce_sum(unmasked) + 1e-6)
                kl = -0.5 * tf.reduce_mean(1 + zlv - zm ** 2 - tf.exp(zlv))
                loss = rl + kl_w * kl
            grads = tape.gradient(loss, mae.trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            mae_opt.apply_gradients(zip(grads, mae.trainable_variables))
        if (epoch + 1) % 10 == 0:
            log.info("  CVAE epoch %d/%d", epoch + 1, mae_epochs)
    mae.save_weights(mae_path)
    log.info("CVAE weights saved to %s", mae_path)

    method = EIGCostAFAMethod(
        classifier_weights_path=cls_path,
        mae_weights_path=mae_path,
        n_features=n_features,
        n_classes=n_classes,
        n_mc_samples=n_mc_samples,
        cost_lambda=cost_lambda,
        feature_costs=costs,
    )
    method.save(save_path)
    log.info("EIG-Cost bundle saved to %s", save_path)


if __name__ == "__main__":
    main()
'''

CONF_CONFIG = '''\
hydra:
  searchpath:
    - file://extra/conf
    - file://extra/conf/global

defaults:
  - hydra: custom
  - /components/initializers@initializer: ???
  - /components/unmaskers@unmasker: ???
  - _self_
  - optional experiment@_global_: ???
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog
  - override hydra/launcher: custom_slurm

train_dataset_bundle_path: ???
val_dataset_bundle_path: ???
classifier_bundle_path: ???
save_path: ???

device: null
seed: null
hard_budget: null
soft_budget_param: null
use_wandb: false
smoke_test: false

batch_size: 256
classifier_epochs: 50
mae_epochs: 30
n_mc_samples: 300
cost_lambda: 0.005
'''

EXPERIMENT_MIMIC_IV = '''\
batch_size: 256
classifier_epochs: 50
mae_epochs: 30
n_mc_samples: 300
cost_lambda: 0.005
'''

METHOD_OPTIONS_ENTRY = '''\
  eig_cost:
    train_script_name: "eig_cost"
    hard_budget_ignored_datasets: [imagenette, mnist, fashion_mnist]
    soft_budget_ignored_datasets: [imagenette, mnist, fashion_mnist]
'''


def apply_method_options(path: Path, dry_run: bool) -> None:
    text = path.read_text()
    if "eig_cost:" in text:
        print("[skip]    method_options.yaml (eig_cost already present)")
        return
    new_text = text.rstrip() + "\n" + METHOD_OPTIONS_ENTRY
    if dry_run:
        print("[would add] method_options.yaml: eig_cost entry")
    else:
        path.write_text(new_text)
        print("[updated] method_options.yaml")


def apply_methods_yaml(path: Path, dry_run: bool) -> None:
    text = path.read_text()
    changed = False

    if "- eig_cost" not in text:
        # Add to the methods list after aaco_nn
        text = text.replace(
            "  - aaco_nn\n",
            "  - aaco_nn\n  - eig_cost\n",
        )
        changed = True
        if dry_run:
            print("[would add] methods.yaml: eig_cost to methods list")

    if "main:" in text and "eig_cost" not in text.split("main:")[1].split("\n\n")[0]:
        # Add to the main method set
        text = text.replace(
            "    - aaco\n",
            "    - aaco\n    - eig_cost\n",
            1  # Only replace first occurrence (inside main:)
        )
        changed = True
        if dry_run:
            print("[would add] methods.yaml: eig_cost to main method set")

    if not dry_run and changed:
        path.write_text(text)
        print("[updated] methods.yaml")
    elif not changed:
        print("[skip]    methods.yaml (eig_cost already present)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    repo = args.repo
    dry_run = args.dry_run

    if not repo.is_dir():
        print("ERROR: repo not found at " + str(repo))
        return 1

    tag = "[would create]" if dry_run else "[created]   "

    # 1. Training script
    train_script = repo / "scripts" / "train" / "eig_cost.py"
    if train_script.exists():
        print("[skip]    " + str(train_script) + " (already exists)")
    else:
        if not dry_run:
            train_script.write_text(TRAIN_SCRIPT)
        print(tag + " " + str(train_script))

    # 2. Hydra config directory + config.yaml
    conf_dir = repo / "extra" / "conf" / "scripts" / "train" / "eig_cost"
    exp_dir = conf_dir / "experiment"
    if not dry_run:
        conf_dir.mkdir(parents=True, exist_ok=True)
        exp_dir.mkdir(parents=True, exist_ok=True)

    config_yaml = conf_dir / "config.yaml"
    if config_yaml.exists():
        print("[skip]    " + str(config_yaml) + " (already exists)")
    else:
        if not dry_run:
            config_yaml.write_text(CONF_CONFIG)
        print(tag + " " + str(config_yaml))

    # 3. Per-dataset experiment configs
    for dataset_name in ("mimic_iv", "mimic_iv_5class"):
        exp_yaml = exp_dir / (dataset_name + ".yaml")
        if exp_yaml.exists():
            print("[skip]    " + str(exp_yaml) + " (already exists)")
        else:
            if not dry_run:
                exp_yaml.write_text(EXPERIMENT_MIMIC_IV)
            print(tag + " " + str(exp_yaml))

    # 4. method_options.yaml
    mo_path = repo / "extra" / "workflow" / "conf" / "method_options.yaml"
    apply_method_options(mo_path, dry_run)

    # 5. methods.yaml
    methods_path = repo / "extra" / "workflow" / "conf" / "methods.yaml"
    apply_methods_yaml(methods_path, dry_run)

    if dry_run:
        print("\nDry run complete. Re-run without --dry-run to apply.")
    else:
        print("\nAll changes applied.")
        print("\nNext steps:")
        print("  1. Verify with dry-run: uv run snakemake ... --dry-run | grep eig_cost")
        print("  2. Launch: nohup ./run.sh > pipeline.log 2>&1 &")

    return 0


if __name__ == "__main__":
    sys.exit(main())
