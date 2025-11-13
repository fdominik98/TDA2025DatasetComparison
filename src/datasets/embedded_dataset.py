from __future__ import annotations

if __name__ == "__main__" and (__package__ is None or __package__ == "") and __spec__ is None:
    import runpy
    import sys
    from pathlib import Path

    _package_dir = Path(__file__).resolve().parent
    _parent_dir = _package_dir.parent
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
    runpy.run_module(f"{_package_dir.name}.{Path(__file__).stem}", run_name="__main__")
    sys.exit()

import matplotlib.pyplot as plt
from .base_dataset import BaseDataset  # type: ignore[import]
from embedding import Embedding  # type: ignore[import]


class EmbeddedDataset(BaseDataset):
    """
    Base class for datasets that use UMAP embedding for distance computations
    and can plot embeddings. Delegates to `Embedding`.
    """

    def __init__(
        self,
        number_of_samples=100,
        n_components=None,
        n_neighbors=None,
        min_dist=None,
        embedding_metric='euclidean',
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.embedding_metric = embedding_metric
        super().__init__(number_of_samples=number_of_samples)
        # Initialize embedding after data is loaded
        self._embedding = Embedding(
            self.X,
            number_of_samples=self.number_of_samples,
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            use_embedding=True,
        )

    def get_embedding(self, metric=None):
        metric_to_use = metric if metric is not None else self.embedding_metric
        return self._embedding.get_embedding(metric_to_use)

    def get_distance_features(self):
        # Use the current embedding as the distance features
        return self.get_embedding(self.embedding_metric)

    def plot_embedding(self, metric=None, figsize=(8, 6), s=10, cmap='viridis', label='Label'):
        metric_to_use = metric if metric is not None else self.embedding_metric
        embedded = self.get_embedding(metric_to_use)
        plt.figure(figsize=figsize)
        plt.scatter(embedded[:, 0], embedded[:, 1], c=self.y[:self.number_of_samples], s=s, cmap=cmap)
        plt.colorbar(label=label)
        plt.title(f"{self.__class__.__name__} UMAP Embedding ({metric_to_use} metric)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.show()


