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
import numpy as np
from sklearn.datasets import fetch_openml  # type: ignore[import]
from sklearn.decomposition import PCA  # type: ignore[import]
from .base_dataset import BaseDataset  # type: ignore[import]

__all__ = ["MNISTPcaDataset"]


class MNISTPcaDataset(BaseDataset):  # type: ignore[misc, valid-type]
    """
    MNIST dataset projected through PCA for downstream distance-based analyses.

    Provides access to PCA embeddings, distance features, and quick plotting helpers.
    """

    def __init__(
        self,
        number_of_samples: int = 1000,
        n_components: int = 2,
        whiten: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self._pca_embedding: np.ndarray | None = None
        super().__init__(number_of_samples=number_of_samples)
        self._fit_pca()
        self.name = f"MNIST PCA Dataset (n_components={n_components})"

    def _load_data(self) -> None:
        """Load raw MNIST digits."""
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        self.X, self.y = mnist.data, mnist.target.astype(int)

    def _fit_pca(self) -> None:
        """Compute PCA embedding for the chosen number of samples."""
        pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        subset = self.X[: self.number_of_samples]
        self._pca_embedding = pca.fit_transform(subset)
        self._pca_model = pca

    def get_embedding(self) -> np.ndarray:
        """Return the cached PCA embedding."""
        if self._pca_embedding is None:
            self._fit_pca()
        return np.asarray(self._pca_embedding)

    def get_distance_features(self) -> np.ndarray:
        """Use PCA coordinates when computing distance matrices."""
        return self.get_embedding()

    def plot_embedding(
        self,
        figsize: tuple[int, int] = (8, 6),
        s: int = 10,
        cmap: str = "viridis",
        title: str | None = None,
    ) -> None:
        """
        Visualise the dataset using a dedicated 2D UMAP embedding (independent of PCA distance features).
        """
        embedding = self.get_visualization_embedding()
        if embedding.shape[1] < 2:
            raise ValueError("UMAP visualization embedding must have at least two components to plot.")
        plt.figure(figsize=figsize)
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=self.y[: self.number_of_samples],
            s=s,
            cmap=cmap,
        )
        plt.colorbar(label="Label")
        final_title = title or "MNIST UMAP Visualization"
        plt.title(final_title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plt.show()

    def plot_image(self, index: int) -> None:
        """Display a single MNIST digit."""
        print(f"Label: {self.y[index]}")
        plt.imshow(self.X[index].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.show()

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Return PCA explained variance ratios."""
        return np.asarray(self._pca_model.explained_variance_ratio_)


if __name__ == "__main__":
    dataset = MNISTPcaDataset(number_of_samples=2000, n_components=2)
    dataset.plot_image(0)
    dataset.plot_embedding()
    matrix = dataset.get_distance_matrix(distance_metric="euclidean")
    print(matrix.shape)

