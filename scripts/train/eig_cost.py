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
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force TensorFlow to use CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
original_cwd = os.getcwd()
from afabench.common.bundle import save_bundle
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
    # TF imported here to keep startup fast and allow Hydra to configure first
    import tensorflow as tf
    from afabench.common.bundle import load_bundle
    from afabench.eig_cost.models import build_maskable_classifier, build_tabular_mae
    from afabench.eig_cost.afa_method import EIGCostAFAMethod

    # ── Reproducibility ───────────────────────────────────────────────────────
    if cfg.seed is not None:
        np.random.seed(int(cfg.seed))
        tf.random.set_seed(int(cfg.seed))

    # ── Load dataset from bundle ──────────────────────────────────────────────
    train_bundle_path = Path(os.path.abspath(cfg.train_dataset_bundle_path))
    save_path = Path(os.path.abspath(cfg.save_path))
    log.info("Loading training data from %s", cfg.train_dataset_bundle_path)
    train_ds, _ = load_bundle(train_bundle_path)
    X_train, y_train = train_ds.get_all_data()
    X_np = X_train.numpy().astype(np.float32)
    y_np = y_train.argmax(dim=-1).numpy().astype(np.int32)
    n_features = int(X_np.shape[1])
    n_classes = int(y_train.shape[1])

    # Replace NaN with 0 (some datasets have sparse features)
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

    if hasattr(train_ds, "get_feature_acquisition_costs"):
        costs = train_ds.get_feature_acquisition_costs().tolist()
    else:
        costs = [1.0] * n_features

    log.info(
        "Dataset: %s samples, %d features, %d classes",
        X_np.shape[0], n_features, n_classes,
    )

    # ── Prepare save paths (weights live inside the bundle directory) ─────────
    #save_path = Path(cfg.save_path)
    #save_path.mkdir(parents=True, exist_ok=True)
    save_path = Path(os.path.abspath(cfg.save_path))
    save_path.mkdir(parents=True, exist_ok=True)
    data_path = save_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    cls_path = str(data_path / "classifier.weights.h5")
    mae_path = str(data_path / "mae.weights.h5")

    # Read hyperparameters from config (with sensible defaults)
    batch_size = int(cfg.get("batch_size", 256))
    classifier_epochs = int(cfg.get("classifier_epochs", 50))
    mae_epochs = int(cfg.get("mae_epochs", 30))
    n_mc_samples = int(cfg.get("n_mc_samples", 300))
    cost_lambda = float(cfg.get("cost_lambda", 0.005))

    # ── Train mask-aware classifier ───────────────────────────────────────────
    log.info("Training MaskableClassifier (%d epochs, batch %d)...",
             classifier_epochs, batch_size)

    classifier = build_maskable_classifier(n_features, n_classes)
    optimizer = tf.keras.optimizers.Adam(5e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Class-balanced sample weights
    class_counts = np.bincount(y_np, minlength=n_classes).astype(np.float32)
    class_weights = len(y_np) / (n_classes * class_counts + 1e-6)

    cls_ds = (
        tf.data.Dataset.from_tensor_slices((X_np, y_np))
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(2)
    )

    for epoch in range(classifier_epochs):
        for features, labels in cls_ds:
            bs = tf.shape(features)[0]
            # Random masking fraction per sample
            frac = tf.random.uniform([bs, 1], 0.1, 1.0)
            mask = tf.cast(
                tf.random.uniform([bs, n_features]) < frac, tf.float32
            )
            # 30% of samples see the full feature set (unmasked)
            use_full = tf.cast(tf.random.uniform([bs, 1]) < 0.3, tf.float32)
            mask = mask * (1 - use_full) + tf.ones([bs, n_features]) * use_full
            with tf.GradientTape() as tape:
                logits = classifier(features, mask, training=True)
                w = tf.gather(
                    tf.constant(class_weights, dtype=tf.float32), labels
                )
                loss = loss_fn(labels, logits, sample_weight=w)
            grads = tape.gradient(loss, classifier.trainable_variables)
            optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
        if (epoch + 1) % 10 == 0:
            log.info("  Classifier epoch %d/%d", epoch + 1, classifier_epochs)

    classifier.save_weights(cls_path)
    log.info("Classifier weights saved to %s", cls_path)

    # ── Train CVAE (tabular MAE) ──────────────────────────────────────────────
    log.info("Training CVAE (%d epochs, batch %d)...", mae_epochs, batch_size)

    mae = build_tabular_mae(n_features)
    mae_opt = tf.keras.optimizers.Adam(5e-4)
    mae_ds = (
        tf.data.Dataset.from_tensor_slices(X_np)
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(2)
    )

    for epoch in range(mae_epochs):
        # KL weight anneals from 0 → 0.05 over first 10 epochs
        kl_w = min(1.0, (epoch + 1) / 10.0) * 0.05
        for batch in mae_ds:
            bs = tf.shape(batch)[0]
            frac = tf.random.uniform([bs, 1], 0.0, 0.95)
            mask = tf.cast(
                tf.random.uniform([bs, n_features]) < frac, tf.float32
            )
            with tf.GradientTape() as tape:
                fm, zm, zlv = mae(batch, mask, training=True)
                unmasked = 1.0 - mask
                recon_loss = tf.reduce_sum(
                    (batch - fm) ** 2 * unmasked
                ) / (tf.reduce_sum(unmasked) + 1e-6)
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + zlv - zm ** 2 - tf.exp(zlv)
                )
                loss = recon_loss + kl_w * kl_loss
            grads = tape.gradient(loss, mae.trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            mae_opt.apply_gradients(zip(grads, mae.trainable_variables))
        if (epoch + 1) % 10 == 0:
            log.info("  CVAE epoch %d/%d", epoch + 1, mae_epochs)

    mae.save_weights(mae_path)
    log.info("CVAE weights saved to %s", mae_path)

    # ── Build and save the method bundle ─────────────────────────────────────
    method = EIGCostAFAMethod(
        classifier_weights_path=cls_path,
        mae_weights_path=mae_path,
        n_features=n_features,
        n_classes=n_classes,
        n_mc_samples=n_mc_samples,
        cost_lambda=cost_lambda,
        feature_costs=costs,
    )
    
    save_bundle(obj=method, path=save_path, metadata={"method": "eig_cost", "class_name": "EIGCostAFAMethod"})
    log.info("EIG-Cost bundle saved to %s", save_path)


if __name__ == "__main__":
    main()
