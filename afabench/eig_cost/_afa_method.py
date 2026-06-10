"""EIG-Cost active feature acquisition method."""
import logging
from pathlib import Path
from typing_extensions import override

import numpy as np
import torch
from torch import Tensor

from afabench.common.custom_types import (
    AFAAction,
    AFAMethod,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)

log = logging.getLogger(__name__)


class EIGCostAFAMethod(AFAMethod):
    """Cost-penalized Expected Information Gain method."""

    def __init__(
        self,
        classifier_weights_path: str = "",
        mae_weights_path: str = "",
        n_features: int = 55,
        n_classes: int = 21,
        n_mc_samples: int = 50,
        cost_lambda: float = 0.005,
        feature_costs: list[float] | None = None,
        device: torch.device | None = None,
    ):
        self.classifier_weights_path = classifier_weights_path
        self.mae_weights_path = mae_weights_path
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_mc_samples = n_mc_samples
        self.cost_lambda = cost_lambda
        self.feature_costs = feature_costs or [1.0] * n_features
        self._device = device or torch.device("cpu")
        self._classifier = None
        self._mae = None

    def _ensure_models_loaded(self):
        if self._classifier is not None:
            return
        import tensorflow as tf
        from afabench.eig_cost.models import (
            build_maskable_classifier,
            build_tabular_mae,
        )
        self._classifier = build_maskable_classifier(
            self.n_features, self.n_classes
        )
        dummy_f = tf.zeros((1, self.n_features))
        dummy_m = tf.ones((1, self.n_features))
        self._classifier(dummy_f, dummy_m)
        self._classifier.load_weights(self.classifier_weights_path)

        self._mae = build_tabular_mae(self.n_features)
        self._mae(dummy_f, dummy_m)
        self._mae.load_weights(self.mae_weights_path)

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True
    
    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Tensor:
        self._ensure_models_loaded()
        import tensorflow as tf

        if masked_features.dim() == 1:
            features_np = masked_features.cpu().numpy().astype(np.float32)
            mask_np = feature_mask.cpu().float().numpy().astype(np.float32)
            batched = False
        else:
            features_np = masked_features[0].cpu().numpy().astype(np.float32)
            mask_np = feature_mask[0].cpu().float().numpy().astype(np.float32)
            batched = True

        f = tf.constant(features_np[np.newaxis], dtype=tf.float32)
        m = tf.constant(mask_np[np.newaxis], dtype=tf.float32)
        logits = self._classifier(f, m, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        result = torch.tensor(probs, dtype=torch.float32, device=self._device)
        if batched:
            result = result.unsqueeze(0)

        return torch.tensor(probs, dtype=torch.float32, device=self._device).unsqueeze(0)

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        self._ensure_models_loaded()
        import tensorflow as tf

        if masked_features.dim() == 1:
            features_np = masked_features.cpu().numpy().astype(np.float32)
            mask_np = feature_mask.cpu().float().numpy().astype(np.float32)
        else:
            features_np = masked_features[0].cpu().numpy().astype(np.float32)
            mask_np = feature_mask[0].cpu().float().numpy().astype(np.float32)

        unacquired = np.where(mask_np == 0)[0]

        if len(unacquired) == 0:
            return torch.tensor([[0]], device=self._device)

        f = tf.constant(features_np[np.newaxis], dtype=tf.float32)
        m = tf.constant(mask_np[np.newaxis], dtype=tf.float32)
        logits = self._classifier(f, m, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        current_entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        if current_entropy < 0.1:
            return torch.tensor([[0]], device=self._device)

        vals = tf.constant(features_np[np.newaxis], dtype=tf.float32)
        masked_batch = tf.repeat(vals * m, self.n_mc_samples, axis=0)
        mask_batch = tf.repeat(m, self.n_mc_samples, axis=0)

        z_mean, z_log_var = self._mae.encode(masked_batch, mask_batch)
        eps = tf.random.normal(tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        feat_mean = self._mae.decode(z, mask_batch)
        completions = feat_mean + tf.random.normal(tf.shape(feat_mean)) * 0.1
        vb = tf.repeat(vals, self.n_mc_samples, axis=0)
        completions = (vb * mask_batch + completions * (1.0 - mask_batch)).numpy()

        best_score = -np.inf
        best_feature_idx = int(unacquired[0])

        for feat_idx in unacquired:
            batch_f = np.tile(features_np, (self.n_mc_samples, 1))
            batch_m = np.tile(mask_np, (self.n_mc_samples, 1))
            batch_f[:, feat_idx] = completions[:, feat_idx]
            batch_m[:, feat_idx] = 1.0

            logits_new = self._classifier(
                tf.constant(batch_f, dtype=tf.float32),
                tf.constant(batch_m, dtype=tf.float32),
                training=False,
            )
            probs_new = tf.nn.softmax(logits_new, axis=-1).numpy()
            cond_ent = -np.sum(
                probs_new * np.log(probs_new + 1e-10), axis=-1
            ).mean()

            eig = current_entropy - cond_ent
            cost = self.feature_costs[feat_idx] if feat_idx < len(self.feature_costs) else 1.0
            score = eig - self.cost_lambda * cost

            if score > best_score:
                best_score = score
                best_feature_idx = int(feat_idx)

        action = best_feature_idx + 1
        return torch.tensor([[action]], device=self._device)

    def save(self, path: Path, **kwargs) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "classifier_weights_path": self.classifier_weights_path,
                "mae_weights_path": self.mae_weights_path,
                "n_features": self.n_features,
                "n_classes": self.n_classes,
                "n_mc_samples": self.n_mc_samples,
                "cost_lambda": self.cost_lambda,
                "feature_costs": self.feature_costs,
            },
            path / "method.pt",
        )

    @classmethod
    def load(cls, path: Path, **kwargs) -> "EIGCostAFAMethod":
        data = torch.load(path / "method.pt")
        device = kwargs.get("device", None)
        return cls(
            classifier_weights_path=data["classifier_weights_path"],
            mae_weights_path=data["mae_weights_path"],
            n_features=data["n_features"],
            n_classes=data["n_classes"],
            n_mc_samples=data["n_mc_samples"],
            cost_lambda=data["cost_lambda"],
            feature_costs=data["feature_costs"],
            device=torch.device(device) if device else None,
        )
