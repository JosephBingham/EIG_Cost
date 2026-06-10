"""Train EIG-Cost method and save as AFA-Benchmark bundle."""
import sys, os
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from pathlib import Path
from afabench.common.bundle import load_bundle, save_bundle
from afabench.eig_cost.models import build_maskable_classifier, build_tabular_mae
from afabench.eig_cost.afa_method import EIGCostAFAMethod

print("Loading training data...")
train_ds, _ = load_bundle(Path("extra/output/datasets/mimic_iv/0/train.bundle"))
X_train, y_train = train_ds.get_all_data()
X_np = X_train.numpy()
y_np = y_train.argmax(dim=-1).numpy()
n_features = X_np.shape[1]
n_classes = int(y_train.shape[1])

if hasattr(train_ds, "get_feature_acquisition_costs"):
    costs = train_ds.get_feature_acquisition_costs().tolist()
else:
    costs = [1.0] * n_features

print(f"Data: {X_np.shape}, classes: {n_classes}, features: {n_features}")

print("Training mask-aware classifier (50 epochs)...")
classifier = build_maskable_classifier(n_features, n_classes)
optimizer = tf.keras.optimizers.Adam(5e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
class_counts = np.bincount(y_np, minlength=n_classes)
class_weights = len(y_np) / (n_classes * class_counts + 1e-6)

dataset = tf.data.Dataset.from_tensor_slices(
    (X_np.astype(np.float32), y_np.astype(np.int32))
).shuffle(10000).batch(256).prefetch(2)

for epoch in range(50):
    for features, labels in dataset:
        bs = tf.shape(features)[0]
        frac = tf.random.uniform([bs, 1], 0.1, 1.0)
        mask = tf.cast(tf.random.uniform([bs, n_features]) < frac, tf.float32)
        full = tf.ones([bs, n_features])
        use_full = tf.cast(tf.random.uniform([bs, 1]) < 0.3, tf.float32)
        mask = mask * (1 - use_full) + full * use_full
        with tf.GradientTape() as tape:
            logits = classifier(features, mask, training=True)
            w = tf.gather(tf.constant(class_weights, dtype=tf.float32), labels)
            loss = loss_fn(labels, logits, sample_weight=w)
        grads = tape.gradient(loss, classifier.trainable_variables)
        optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
    if (epoch + 1) % 10 == 0:
        print(f"  Classifier epoch {epoch+1}/50")

print("Training CVAE (30 epochs)...")
mae = build_tabular_mae(n_features)
mae_opt = tf.keras.optimizers.Adam(5e-4)
mae_ds = tf.data.Dataset.from_tensor_slices(
    X_np.astype(np.float32)
).shuffle(10000).batch(256).prefetch(2)

for epoch in range(30):
    kl_w = min(1.0, (epoch + 1) / 10.0) * 0.05
    for batch in mae_ds:
        bs = tf.shape(batch)[0]
        frac = tf.random.uniform([bs, 1], 0.0, 0.95)
        mask = tf.cast(tf.random.uniform([bs, n_features]) < frac, tf.float32)
        with tf.GradientTape() as tape:
            fm, zm, zlv = mae(batch, mask, training=True)
            um = 1.0 - mask
            rl = tf.reduce_sum((batch - fm)**2 * um) / (tf.reduce_sum(um) + 1e-6)
            kl = -0.5 * tf.reduce_mean(1 + zlv - zm**2 - tf.exp(zlv))
            loss = rl + kl_w * kl
        grads = tape.gradient(loss, mae.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        mae_opt.apply_gradients(zip(grads, mae.trainable_variables))
    if (epoch + 1) % 10 == 0:
        print(f"  MAE epoch {epoch+1}/30")

save_dir = Path("extra/output/trained_methods/eig_cost/mimic_iv")
save_dir.mkdir(parents=True, exist_ok=True)

CLS_FILE = "classifier.weights.h5"
MAE_FILE = "mae.weights.h5"
cls_path = str(save_dir / CLS_FILE)
mae_path = str(save_dir / MAE_FILE)

classifier.save_weights(cls_path)
mae.save_weights(mae_path)

method = EIGCostAFAMethod(
    classifier_weights_path=cls_path,
    mae_weights_path=mae_path,
    n_features=n_features,
    n_classes=n_classes,
    n_mc_samples=300,
    cost_lambda=0.005,
    feature_costs=costs,
)

bundle_path = save_dir / "method.bundle"
save_bundle(obj=method, path=bundle_path, metadata={"method": "eig_cost"})
print(f"Saved method bundle to {bundle_path}")
print("Done!")
