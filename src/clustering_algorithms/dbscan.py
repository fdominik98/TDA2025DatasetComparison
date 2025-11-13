from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from clustering_algorithms.base import ClusteringAlgorithm


class DBSCANClustering(ClusteringAlgorithm):
    """Density-based clustering using a precomputed distance matrix."""

    def __init__(self, eps: float, min_samples: int = 5) -> None:
        super().__init__(name="dbscan")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        self.eps = eps
        self.min_samples = min_samples
        self._model: DBSCAN | None = None

    def _cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        self._model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed")
        return self._model.fit_predict(distance_matrix)

    def _build_metadata(
        self,
        distance_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float] | None:
        metadata = super()._build_metadata(distance_matrix, labels)
        core_samples = int(self._model.core_sample_indices_.shape[0]) if self._model is not None else None
        extra = {
            "eps": self.eps,
            "min_samples": self.min_samples,
        }
        if core_samples is not None:
            extra["core_samples"] = core_samples
        if metadata:
            metadata.update(extra)
            return metadata
        return extra

