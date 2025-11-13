from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from clustering_algorithms.base import ClusteringAlgorithm


class KMeansClustering(ClusteringAlgorithm):
    """K-means clustering treating each distance matrix row as a feature vector."""

    def __init__(self, n_clusters: int, random_state: int | None = None, n_init: int = 10) -> None:
        super().__init__(name="kmeans")
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self._model: KMeans | None = None

    def _cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        self._model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        return self._model.fit_predict(distance_matrix)

    def _build_metadata(
        self,
        distance_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float] | None:
        metadata = super()._build_metadata(distance_matrix, labels)
        if self._model is None:
            return metadata
        extra = {"inertia": round(float(self._model.inertia_), 3)}
        if metadata:
            metadata.update(extra)
            return metadata
        return extra

