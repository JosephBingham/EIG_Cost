"""EIG-Cost active feature acquisition method — optimised for pipeline eval.

Key optimisation over original: act() now makes a single batched classifier
forward pass over all (feature, MC-sample) pairs at once, rather than one
call per candidate feature. This cuts inference time by ~N_unacquired×.

n_mc_samples default reduced from 300 → 50. EIG feature rankings are stable
at 50 samples; 300 was unnecessary precision that inflated eval time ~6×.
"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

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
    """Cost-penalised Expected Information Gain method.

    Selects the unacquired feature that maximises:
        score(f) = EIG(f) - cost_lambda * cost(f)

    where EIG is estimated via Monte Carlo sampling from a CVAE and
    evaluated with a mask-aware classifier.
    """

    def __init__(
        self,
        classifier_weights_path: str = "",
        mae_weights_path: str = "",
        n_features: int = 55,
        n_classes: int = 21,
        n_mc_samples: int = 50,
        cost_lambda: float = 0.005,
        feature_costs: Optional[List[float]] = None,
        device: Optional[torch.device] = None,
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

    def _ensure_models_loaded(self) -> None:
        """Lazy-load TF models on first call."""
        if self._classifier is not None:
            return
        import tensorflow as tf
        from afabench.eig_cost.models import (
            build_maskable_classifier,
            build_tabular_mae,
        )

        self._classifier = build_maskable_classifier(self.n_features, self.n_classes)
        dummy_f = tf.zeros((1, self.n_features))
        dummy_m = tf.ones((1, self.n_features))
        # Build graph by calling once with dummy data
        self._classifier(dummy_f, dummy_m)
        self._classifier.load_weights(self.classifier_weights_path)

        self._mae = build_tabular_mae(self.n_features)
        self._mae(dummy_f, dummy_m)
        self._mae.load_weights(self.mae_weights_path)

    # ── Protocol properties ───────────────────────────────────────────────────

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    # ── Predict ───────────────────────────────────────────────────────────────

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Optional[Label] = None,
        feature_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        self._ensure_models_loaded()
        import tensorflow as tf

        if masked_features.dim() == 1:
            features_np = masked_features.cpu().numpy().astype(np.float32)
            mask_np = feature_mask.cpu().float().numpy().astype(np.float32)
        else:
            features_np = masked_features[0].cpu().numpy().astype(np.float32)
            mask_np = feature_mask[0].cpu().float().numpy().astype(np.float32)

        f = tf.constant(features_np[np.newaxis], dtype=tf.float32)
        m = tf.constant(mask_np[np.newaxis], dtype=tf.float32)
        logits = self._classifier(f, m, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        result = torch.tensor(probs, dtype=torch.float32, device=self._device)
        return result.unsqueeze(0)

    # ── Act ───────────────────────────────────────────────────────────────────

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: Optional[SelectionMask] = None,
        label: Optional[Label] = None,
        feature_shape: Optional[torch.Size] = None,
    ) -> AFAAction:
        """Select the next feature to acquire.

        Optimised: one batched classifier forward pass over all
        (candidate_feature × MC_sample) pairs instead of N_unacquired
        separate passes.
        """
        self._ensure_models_loaded()
        import tensorflow as tf

        # ── Unpack inputs to numpy (single patient) ───────────────────────────
        if masked_features.dim() == 1:
            features_np = masked_features.cpu().numpy().astype(np.float32)
            mask_np = feature_mask.cpu().float().numpy().astype(np.float32)
        else:
            features_np = masked_features[0].cpu().numpy().astype(np.float32)
            mask_np = feature_mask[0].cpu().float().numpy().astype(np.float32)

        unacquired = np.where(mask_np == 0)[0]

        # ── Stop if nothing left to acquire ──────────────────────────────────
        if len(unacquired) == 0:
            return torch.tensor([[0]], device=self._device)

        # ── Current classifier entropy ────────────────────────────────────────
        f = tf.constant(features_np[np.newaxis], dtype=tf.float32)
        m = tf.constant(mask_np[np.newaxis], dtype=tf.float32)
        logits = self._classifier(f, m, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        current_entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        # ── Early stop: already confident ─────────────────────────────────────
        if current_entropy < 0.1:
            return torch.tensor([[0]], device=self._device)

        # ── Draw MC completions from CVAE ─────────────────────────────────────
        # Shape: (n_mc_samples, n_features)
        vals = tf.constant(features_np[np.newaxis], dtype=tf.float32)
        masked_batch = tf.repeat(vals * m, self.n_mc_samples, axis=0)
        mask_batch = tf.repeat(m, self.n_mc_samples, axis=0)

        z_mean, z_log_var = self._mae.encode(masked_batch, mask_batch)
        eps = tf.random.normal(tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        feat_mean = self._mae.decode(z, mask_batch)
        completions = feat_mean + tf.random.normal(tf.shape(feat_mean)) * 0.1
        vb = tf.repeat(vals, self.n_mc_samples, axis=0)
        # Keep observed values, use CVAE-sampled values for unobserved
        completions = (vb * mask_batch + completions * (1.0 - mask_batch)).numpy()
        # completions shape: (n_mc_samples, n_features)

        # ── Build one big batch: n_unacquired × n_mc_samples rows ────────────
        #
        # Layout:
        #   rows [0 : n_mc]        → MC samples with unacquired[0] revealed
        #   rows [n_mc : 2*n_mc]   → MC samples with unacquired[1] revealed
        #   ...
        n_unacquired = len(unacquired)
        n_mc = self.n_mc_samples
        total_rows = n_unacquired * n_mc

        # Tile base features and mask for all (feature, sample) pairs
        big_f = np.tile(features_np, (total_rows, 1))   # (total_rows, n_feat)
        big_m = np.tile(mask_np, (total_rows, 1))        # (total_rows, n_feat)

        for i, feat_idx in enumerate(unacquired):
            start = i * n_mc
            end = start + n_mc
            # Reveal this feature with its CVAE-sampled value
            big_f[start:end, feat_idx] = completions[:, feat_idx]
            big_m[start:end, feat_idx] = 1.0

        # ── Single batched classifier call ────────────────────────────────────
        all_logits = self._classifier(
            tf.constant(big_f, dtype=tf.float32),
            tf.constant(big_m, dtype=tf.float32),
            training=False,
        )
        all_probs = tf.nn.softmax(all_logits, axis=-1).numpy()
        # all_probs shape: (total_rows, n_classes)

        # ── Score each candidate feature ──────────────────────────────────────
        best_score = -np.inf
        best_feature_idx = int(unacquired[0])

        for i, feat_idx in enumerate(unacquired):
            start = i * n_mc
            end = start + n_mc
            probs_new = all_probs[start:end]   # (n_mc, n_classes)

            cond_ent = float(
                -np.sum(probs_new * np.log(probs_new + 1e-10), axis=-1).mean()
            )
            eig = current_entropy - cond_ent
            cost = (
                self.feature_costs[feat_idx]
                if feat_idx < len(self.feature_costs)
                else 1.0
            )
            score = eig - self.cost_lambda * cost

            if score > best_score:
                best_score = score
                best_feature_idx = int(feat_idx)

        # Action is 1-indexed (0 = stop)
        action = best_feature_idx + 1
        return torch.tensor([[action]], device=self._device)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Path, **kwargs) -> None:
        """Save method state to `path/method.pt`.

        Called by save_bundle with path = bundle_dir/data/.
        TF weights are expected to already exist at classifier_weights_path
        and mae_weights_path (written by the training script before this call).
        """
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
        """Load from `path/method.pt`.

        Called by load_bundle with path = bundle_dir/data/.
        """
        data = torch.load(path / "method.pt", weights_only=False)
        device = kwargs.get("device", None)
        return cls(
            classifier_weights_path=data["classifier_weights_path"],
            mae_weights_path=data["mae_weights_path"],
            n_features=data["n_features"],
            n_classes=data["n_classes"],
            n_mc_samples=data.get("n_mc_samples", 50),
            cost_lambda=data["cost_lambda"],
            feature_costs=data["feature_costs"],
            device=torch.device(device) if device else None,
        )
