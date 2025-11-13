from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering

from clustering_algorithms.base import ClusteringAlgorithm


class SingleLinkageKClusters(ClusteringAlgorithm):
    """Single linkage agglomerative clustering with a fixed number of clusters."""

    def __init__(self, n_clusters: int) -> None:
        super().__init__(name="single_linkage_k_clusters")
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        self.n_clusters = n_clusters

    def _cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage="single",
            metric="precomputed",
        )
        return model.fit_predict(distance_matrix)


class SingleLinkageThreshold(ClusteringAlgorithm):
    """Single linkage agglomerative clustering with a fixed distance threshold."""

    def __init__(self, threshold: float) -> None:
        super().__init__(name="single_linkage_threshold")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold

    def _cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        model = AgglomerativeClustering(
            n_clusters=None,
            linkage="single",
            metric="precomputed",
            distance_threshold=self.threshold,
        )
        return model.fit_predict(distance_matrix)


class SingleLinkageAlphaScaled(ClusteringAlgorithm):
    """Single linkage agglomerative clustering with alpha-scaled max distance threshold."""

    def __init__(self, alpha: float) -> None:
        super().__init__(name="single_linkage_alpha_scaled")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in the interval (0, 1]")
        self.alpha = alpha
        self._resolved_threshold: float | None = None

    def _cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        condensed = squareform(distance_matrix, checks=False)
        max_distance = condensed.max()
        self._resolved_threshold = self.alpha * float(max_distance)
        model = AgglomerativeClustering(
            n_clusters=None,
            linkage="single",
            metric="precomputed",
            distance_threshold=self._resolved_threshold,
        )
        return model.fit_predict(distance_matrix)

    def _build_metadata(
        self,
        distance_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float] | None:
        metadata = super()._build_metadata(distance_matrix, labels)
        resolved_threshold = self._resolved_threshold
        extra = {"threshold": round(resolved_threshold, 3)} if resolved_threshold is not None else {}
        if metadata:
            metadata.update(extra)
            return metadata
        return extra or None

