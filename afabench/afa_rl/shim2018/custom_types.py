from abc import ABC, abstractmethod
from typing_extensions import override

from jaxtyping import Float
from torch import Tensor, nn

from afabench.common.custom_types import FeatureMask, Logits, MaskedFeatures
from typing_extensions import TypeAlias

Embedding: TypeAlias = Float[Tensor, "*batch embedding_size"]


class Embedder(nn.Module, ABC):
    """An Embedder converts feature values and feature indices (1 if a feature is observed, 0 if not) to an embedding."""

    @abstractmethod
    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding: ...

    @override
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding:
        return super().__call__(masked_features, feature_mask)


class EmbeddingClassifier(nn.Module, ABC):
    @abstractmethod
    @override
    def forward(self, embedding: Embedding) -> Logits: ...

    @override
    def __call__(self, embedding: Embedding) -> Logits:
        return super().__call__(embedding)
