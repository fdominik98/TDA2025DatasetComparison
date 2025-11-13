"""Clustering utilities built on modular distance-matrix algorithms."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from collections.abc import Sequence

from config import RANDOM_STATE
from datasets.base_dataset import BaseDataset

from clustering_algorithms.base import ClusteringAlgorithm, ClusteringResult
from clustering_algorithms.single_linkage import SingleLinkageKClusters, SingleLinkageThreshold, SingleLinkageAlphaScaled
from clustering_algorithms.kmeans import KMeansClustering
from clustering_algorithms.dbscan import DBSCANClustering
from clustering_algorithms.tda_persistent import TDAPersistentComponents
from datasets.mnist_dataset import MNISTDataset
from datasets.concentric_triangles_dataset import ConcentricTrianglesDataset
from datasets.mnist_pca_dataset import MNISTPcaDataset
from datasets.mnist_umap_dataset import MNISTUmapDataset
from datasets.cifar_dataset import CIFARDataset
from datasets.cifar_pca_dataset import CIFARPcaDataset
from datasets.cifar_umap_dataset import CIFARUmapDataset


class Clustering:
    """Coordinate clustering experiments on datasets backed by distance matrices."""

    def __init__(self, dataset: BaseDataset, embedding_metric: str, clustering_metric: str) -> None:
        self.dataset = dataset
        self.embedding_metric = embedding_metric
        self.clustering_metric = clustering_metric
        self.random_state = RANDOM_STATE

        self.distance_matrix = dataset.get_distance_matrix(distance_metric=clustering_metric)
        self.true_labels = np.asarray(dataset.labels[: dataset.number_of_samples])

        (
            self.embedding,
            self.embedding_axis_labels,
            self.embedding_source_description,
        ) = self._resolve_embedding()
        self._distance_matrix_modified = False
        self.algorithms: list[ClusteringAlgorithm] = []
        self.results: dict[str, ClusteringResult] = {}

    def _resolve_embedding(self) -> tuple[np.ndarray, tuple[str, str], str]:
        """Best-effort retrieval of 2D coordinates for visualisation."""
        default_axis_labels = ("Embedding Dimension 1", "Embedding Dimension 2")
        metric_label = self.embedding_metric if self.embedding_metric else "raw"
        default_description = f"{metric_label} embedding"

        if self._is_mnist_dataset() and hasattr(self.dataset, "get_visualization_embedding"):
            try:
                embedding = np.asarray(self.dataset.get_visualization_embedding())
                embedding = self._ensure_two_dimensions(embedding[: self.dataset.number_of_samples])
                return embedding, ("UMAP 1", "UMAP 2"), "UMAP (visualisation)"
            except Exception:  # pragma: no cover - conservative fallback
                pass

        if hasattr(self.dataset, "get_embedding"):
            try:
                embedding = np.asarray(self.dataset.get_embedding(self.embedding_metric))
                embedding = self._ensure_two_dimensions(embedding[: self.dataset.number_of_samples])
                return embedding, default_axis_labels, default_description
            except Exception:  # pragma: no cover - conservative fallback
                pass
        if hasattr(self.dataset, "X"):
            embedding = self._ensure_two_dimensions(np.asarray(self.dataset.X[: self.dataset.number_of_samples]))
            return embedding, default_axis_labels, "Raw features"
        embedding = self._ensure_two_dimensions(np.asarray(self.dataset.get_distance_features()))
        return embedding, default_axis_labels, "Distance features"

    @staticmethod
    def _ensure_two_dimensions(data: np.ndarray) -> np.ndarray:
        """Guarantee at least two dimensions for plotting."""
        array = np.asarray(data)
        if array.ndim == 1:
            array = array[:, np.newaxis]
        if array.shape[1] < 2:
            pad = 2 - array.shape[1]
            array = np.pad(array, ((0, 0), (0, pad)), mode="constant")
        return array

    def set_distance_matrix(self, distance_matrix: np.ndarray) -> None:
        """Replace the cached distance matrix."""
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError(f"Distance matrix must be square, got shape {distance_matrix.shape}")
        if distance_matrix.shape[0] != self.distance_matrix.shape[0]:
            raise ValueError(
                f"Distance matrix size ({distance_matrix.shape[0]}) must match number of samples "
                f"({self.distance_matrix.shape[0]})",
            )
        self.distance_matrix = distance_matrix
        self._distance_matrix_modified = True
        self.results.clear()

    def get_distance_matrix(self) -> np.ndarray:
        """Return the current distance matrix."""
        return self.distance_matrix

    def run_algorithm(self, algorithm: ClusteringAlgorithm) -> ClusteringResult:
        """Execute an algorithm and cache its result."""
        result = algorithm.run(self.distance_matrix, self.true_labels)
        self.results[algorithm.name] = result
        return result

    def single_linkage_k_clusters(self, k: int) -> ClusteringResult:
        return self.run_algorithm(SingleLinkageKClusters(k))

    def single_linkage_distance_threshold(self, threshold: float) -> ClusteringResult:
        return self.run_algorithm(SingleLinkageThreshold(threshold))

    def single_linkage_scale_alpha(self, alpha: float) -> ClusteringResult:
        return self.run_algorithm(SingleLinkageAlphaScaled(alpha))

    def kmeans_clustering(self, k: int) -> ClusteringResult:
        return self.run_algorithm(KMeansClustering(k, random_state=self.random_state))

    def dbscan_clustering(self, eps: float, min_samples: int = 5) -> ClusteringResult:
        return self.run_algorithm(DBSCANClustering(eps=eps, min_samples=min_samples))

    def tda_persistent_components(self) -> ClusteringResult:
        return self.run_algorithm(TDAPersistentComponents())


    def configure_algorithms(self, algorithms: Sequence[ClusteringAlgorithm]) -> None:
        """Configure the suite of algorithms to be executed during comparisons."""
        self.algorithms = list(algorithms)

    def add_algorithm(self, algorithm: ClusteringAlgorithm) -> None:
        """Append a single algorithm to the configured suite."""
        self.algorithms.append(algorithm)

    def clear_algorithms(self) -> None:
        """Remove all configured algorithms."""
        self.algorithms.clear()

    def _is_mnist_dataset(self) -> bool:
        """Heuristic check for MNIST dataset variants to ensure consistent visualisation."""
        module_name = getattr(self.dataset, "__module__", "")
        class_name = self.dataset.__class__.__name__
        combined = f"{module_name}.{class_name}".lower()
        return "mnist" in combined

    def _default_algorithms(self) -> list[ClusteringAlgorithm]:
        """Fallback algorithm configuration mirroring previous defaults."""
        k = int(np.unique(self.true_labels).size)
        return [
            SingleLinkageKClusters(k),
            SingleLinkageAlphaScaled(0.5),
            KMeansClustering(k, random_state=self.random_state),
            DBSCANClustering(eps=0.5, min_samples=5),
            TDAPersistentComponents(),
        ]

    def compare_all_methods(
        self,
        algorithms: Sequence[ClusteringAlgorithm] | None = None,
        figsize: tuple[int, int] = (15, 10),
    ) -> dict[str, ClusteringResult]:
        """Run a configurable suite of algorithms and visualise results."""
        if algorithms is not None:
            algorithms_to_run = list(algorithms)
        elif self.algorithms:
            algorithms_to_run = list(self.algorithms)
        else:
            algorithms_to_run = self._default_algorithms()

        self.algorithms = list(algorithms_to_run)

        results: dict[str, ClusteringResult] = {}
        for algorithm in algorithms_to_run:
            result = self.run_algorithm(algorithm)
            results[algorithm.name] = result

        n_methods = len(results) + 1
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols
        fig = plt.figure(figsize=figsize)

        ax = plt.subplot(n_rows, n_cols, 1)
        ax.scatter(self.embedding[:, 0], self.embedding[:, 1], c=self.true_labels, s=10, cmap="tab10")
        title = f"Ground Truth | Clusters: {len(np.unique(self.true_labels))}"
        ax.set_title(title)
        ax.set_xlabel(self.embedding_axis_labels[0])
        ax.set_ylabel(self.embedding_axis_labels[1])

        for idx, (name, result) in enumerate(results.items(), start=2):
            ax = plt.subplot(n_rows, n_cols, idx)
            ax.scatter(self.embedding[:, 0], self.embedding[:, 1], c=result.labels, s=10, cmap="tab10")
            ari = result.scores.get("adjusted_rand_index")
            nmi = result.scores.get("normalized_mutual_information")
            n_clusters = result.metadata.get("n_clusters") if result.metadata else None
            extras: list[str] = []
            if result.metadata:
                extras = [
                    f"{key}: {value}"
                    for key, value in result.metadata.items()
                    if key not in {"n_clusters", "barcode", "stable_rank"}
                ]
            lines = [
                name.replace("_", " ").title(),
                f"Clusters: {n_clusters if n_clusters is not None else '?'}",
            ]
            if ari is not None:
                lines.append(f"ARI: {ari:.3f}")
            if nmi is not None:
                lines.append(f"NMI: {nmi:.3f}")
            lines.extend(extras)
            ax.set_title(" | ".join(lines), fontsize=9)
            ax.set_xlabel(self.embedding_axis_labels[0])
            ax.set_ylabel(self.embedding_axis_labels[1])

        plt.suptitle(
            f"Clustering Comparison - {self.dataset.name} "
            f"(Embedding: {self.embedding_metric}, Clustering: {self.clustering_metric}, "
            f"Viz: {self.embedding_source_description})",
            fontsize=14,
            y=0.995,
        )
        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 80)
        print(
            f"CLUSTERING RESULTS SUMMARY - Embedding: {self.embedding_metric.upper()}, "
            f"Clustering: {self.clustering_metric.upper()}",
        )
        print("=" * 80)
        print(f"{'Method':<35} {'ARI':>8} {'NMI':>8} {'#Clusters':>10}")
        print("-" * 80)
        for name, result in results.items():
            ari = result.scores.get("adjusted_rand_index", float("nan"))
            nmi = result.scores.get("normalized_mutual_information", float("nan"))
            n_clusters = result.metadata.get("n_clusters") if result.metadata else float("nan")
            print(f"{name:<35} {ari:>8.3f} {nmi:>8.3f} {n_clusters:>10}")
        print("=" * 80)

        tda_result = results.get("single_linkage_threshold_by_tda_persistent_components")
        if tda_result and tda_result.metadata and "barcode" in tda_result.metadata:
            print("\nPlotting H₀ persistence barcode for TDA method...")
            barcode = np.asarray(tda_result.metadata["barcode"])
            stable_rank = tda_result.metadata.get("stable_rank") if tda_result.metadata else None
            TDAPersistentComponents.plot_persistence_summary(
                barcode,
                dataset_name=self.dataset.name,
                clustering_metric=self.clustering_metric,
                stable_rank=stable_rank,
            )

        return results


if __name__ == "__main__":

    print("Testing Clustering on Concentric Triangles Dataset")
    print("=" * 80)

    triangles = ConcentricTrianglesDataset(number_of_samples=1000, n_triangles=5)
    clustering = Clustering(triangles, embedding_metric="euclidean", clustering_metric="euclidean")
    clustering.configure_algorithms(
        [
            SingleLinkageKClusters(5),
            SingleLinkageAlphaScaled(0.05),
            KMeansClustering(5, random_state=clustering.random_state),
            DBSCANClustering(eps=0.15, min_samples=20),
            TDAPersistentComponents(),
        ],
    )
    clustering.compare_all_methods()

    print("\n\nTesting Clustering on MNIST Dataset")
    print("=" * 80)

    mnist = MNISTDataset(number_of_samples=1000)
    clustering_mnist = Clustering(mnist, embedding_metric="euclidean", clustering_metric="euclidean")
    clustering_mnist.configure_algorithms(
        [
            SingleLinkageKClusters(10),
            SingleLinkageAlphaScaled(0.05),
            KMeansClustering(10, random_state=clustering_mnist.random_state),
            DBSCANClustering(eps=5, min_samples=20),
            TDAPersistentComponents(),
        ],
    )
    clustering_mnist.compare_all_methods()
    
    
    print("Testing Clustering on MNIST UMAP Dataset")
    print("=" * 80)

    mnist_umap = MNISTUmapDataset(number_of_samples=1000, n_components=2)
    clustering_mnist_umap = Clustering(mnist_umap, embedding_metric="euclidean", clustering_metric="euclidean")
    clustering_mnist_umap.configure_algorithms(
        [
            SingleLinkageKClusters(10),
            SingleLinkageAlphaScaled(0.05),
            KMeansClustering(10, random_state=clustering_mnist_umap.random_state),
            DBSCANClustering(eps=2, min_samples=20),
            TDAPersistentComponents(),
        ],
    )
    clustering_mnist_umap.compare_all_methods()
    
    
    print("Testing Clustering on MNIST PCA Dataset")
    print("=" * 80)

    mnist_pca = MNISTPcaDataset(number_of_samples=1000, n_components=2)
    clustering_mnist_pca = Clustering(mnist_pca, embedding_metric="euclidean", clustering_metric="euclidean")
    clustering_mnist_pca.configure_algorithms(
        [
            SingleLinkageKClusters(10),
            SingleLinkageAlphaScaled(0.02),
            KMeansClustering(10, random_state=clustering_mnist_pca.random_state),
            DBSCANClustering(eps=2, min_samples=20),
            TDAPersistentComponents(),
        ],
    )
    clustering_mnist_pca.compare_all_methods()
    
    
    print("Testing Clustering on CIFAR Dataset")
    print("=" * 80)
    
    cifar = CIFARDataset(number_of_samples=1000)
    clustering_cifar = Clustering(cifar, embedding_metric="euclidean", clustering_metric="euclidean")
    clustering_cifar.configure_algorithms(
        [
            SingleLinkageKClusters(10),
            SingleLinkageAlphaScaled(0.01),
            KMeansClustering(10, random_state=clustering_cifar.random_state),
            DBSCANClustering(eps=1, min_samples=20),
            TDAPersistentComponents(),
        ],
    )
    clustering_cifar.compare_all_methods()
    
    print("Testing Clustering on CIFAR UMAP Dataset")
    print("=" * 80)
    
    cifar_umap = CIFARUmapDataset(number_of_samples=1000, n_components=2)
    clustering_cifar_umap = Clustering(cifar_umap, embedding_metric="euclidean", clustering_metric="euclidean")
    clustering_cifar_umap.configure_algorithms(
        [
            SingleLinkageKClusters(10),
            SingleLinkageAlphaScaled(0.02),
            KMeansClustering(10, random_state=clustering_cifar_umap.random_state),
            DBSCANClustering(eps=0.1, min_samples=20),
            TDAPersistentComponents(),
        ]
    )
    clustering_cifar_umap.compare_all_methods()
    
    
    print("Testing Clustering on CIFAR PCA Dataset")
    print("=" * 80)
    
    cifar_pca = CIFARPcaDataset(number_of_samples=1000, n_components=2)
    clustering_cifar_pca = Clustering(cifar_pca, embedding_metric="euclidean", clustering_metric="euclidean")
    clustering_cifar_pca.configure_algorithms(
        [
            SingleLinkageKClusters(10),
            SingleLinkageAlphaScaled(0.05),
            KMeansClustering(10, random_state=clustering_cifar_pca.random_state),
            DBSCANClustering(eps=0.1, min_samples=20),
            TDAPersistentComponents(),
        ]
    )
    clustering_cifar_pca.compare_all_methods()
    
    