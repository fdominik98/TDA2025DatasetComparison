from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class ClusteringResult:
    """Encapsulates the output of a clustering algorithm."""

    labels: np.ndarray
    scores: dict[str, float]
    metadata: dict[str, Any] | None = None


class ClusteringAlgorithm(ABC):
    """Abstract base class for clustering algorithms operating on distance matrices."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    def run(
        self,
        distance_matrix: np.ndarray,
        true_labels: np.ndarray | None = None,
    ) -> ClusteringResult:
        """Execute the clustering algorithm and return labels and quality scores."""
        labels = self._cluster(distance_matrix)
        scores = self._compute_scores(labels, true_labels)
        metadata = self._build_metadata(distance_matrix, labels)
        return ClusteringResult(labels=labels, scores=scores, metadata=metadata)

    @abstractmethod
    def _cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Return cluster labels for each sample."""

    def _compute_scores(
        self,
        labels: np.ndarray,
        true_labels: np.ndarray | None,
    ) -> dict[str, float]:
        """Compute clustering quality metrics relative to the ground truth."""
        if true_labels is None:
            return {}
        return {
            "adjusted_rand_index": adjusted_rand_score(true_labels, labels),
            "normalized_mutual_information": normalized_mutual_info_score(true_labels, labels),
        }

    def _build_metadata(
        self,
        distance_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any] | None:
        """Default metadata includes the number of detected clusters (excluding noise labels)."""
        labels_array = np.asarray(labels)
        try:
            non_noise = labels_array[labels_array >= 0]
            n_clusters = int(np.unique(non_noise).size)
        except TypeError:
            n_clusters = int(np.unique(labels_array).size)
        return {"n_clusters": n_clusters}

