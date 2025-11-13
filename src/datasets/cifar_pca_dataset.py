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

__all__ = ["CIFARPcaDataset"]


class CIFARPcaDataset(BaseDataset):  # type: ignore[misc, valid-type]
    """
    CIFAR-10 dataset projected through PCA for downstream analyses.
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
        self.image_shape = (32, 32, 3)
        self._pca_embedding: np.ndarray | None = None
        self.class_names: np.ndarray | None = None
        super().__init__(number_of_samples=number_of_samples)
        self._fit_pca()
        self.name = f"CIFAR-10 PCA Dataset (n_components={n_components})"

    def _load_data(self) -> None:
        """Load raw CIFAR-10 samples."""
        cifar = fetch_openml("CIFAR_10_small", version=1, as_frame=False)
        data = np.asarray(cifar.data, dtype=np.float32)
        targets = np.asarray(cifar.target)

        try:
            labels = targets.astype(int)
            class_names = np.unique(labels)
        except ValueError:
            class_names = np.unique(targets)
            label_lookup = {label: idx for idx, label in enumerate(class_names)}
            labels = np.asarray([label_lookup[label] for label in targets], dtype=int)

        self.X = data
        self.y = labels
        self.class_names = class_names

    def _fit_pca(self) -> None:
        """Compute PCA embedding using the selected number of samples."""
        pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        subset = self.X[: self.number_of_samples]
        self._pca_embedding = pca.fit_transform(subset)
        self._pca_model = pca

    def get_embedding(self) -> np.ndarray:
        """Return PCA embedding features."""
        if self._pca_embedding is None:
            self._fit_pca()
        return np.asarray(self._pca_embedding)

    def get_distance_features(self) -> np.ndarray:
        """Use PCA embedding for distance-based computations."""
        return self.get_embedding()

    def plot_embedding(
        self,
        figsize: tuple[int, int] = (8, 6),
        s: int = 10,
        cmap: str = "viridis",
        title: str | None = None,
    ) -> None:
        """Visualize CIFAR-10 samples in a 2D UMAP projection for inspection."""
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
        final_title = title or "CIFAR-10 UMAP Visualization"
        plt.title(final_title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plt.show()

    def plot_image(self, index: int) -> None:
        """Display a single CIFAR-10 image."""
        image = self.X[index].reshape(3, 32, 32).transpose(1, 2, 0)
        image = np.clip(image, 0, 255).astype(np.uint8)
        label = self.y[index]
        title = f"Label: {label}"
        if self.class_names is not None and 0 <= label < len(self.class_names):
            title = f"Label: {label} ({self.class_names[label]})"
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.show()

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Return PCA explained variance ratios."""
        return np.asarray(self._pca_model.explained_variance_ratio_)


if __name__ == "__main__":
    dataset = CIFARPcaDataset(number_of_samples=2000, n_components=2)
    dataset.plot_image(0)
    dataset.plot_embedding()
    matrix = dataset.get_distance_matrix(distance_metric="euclidean")
    print(matrix.shape)

