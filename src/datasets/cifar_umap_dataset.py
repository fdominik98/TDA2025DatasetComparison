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

from .embedded_dataset import EmbeddedDataset  # type: ignore[import]

__all__ = ["CIFARUmapDataset"]


class CIFARUmapDataset(EmbeddedDataset):  # type: ignore[misc, valid-type]
    """
    CIFAR-10 dataset with UMAP embedding support.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.class_names: np.ndarray | None = None
        self.image_shape = (32, 32, 3)
        super().__init__(*args, **kwargs)
        self.name = f"CIFAR-10 UMAP Dataset (n_components={self.n_components})"

    def _load_data(self) -> None:
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

    def plot_embedding(
        self,
        metric: str | None = None,
        figsize: tuple[int, int] = (8, 6),
        s: int = 10,
        cmap: str = "viridis",
        label: str = "Label",
        title: str | None = None,
    ) -> None:
        """
        Visualize the CIFAR-10 samples using a dedicated 2D UMAP projection.
        """
        embedding = self.get_visualization_embedding()
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=self.y[: self.number_of_samples],
            s=s,
            cmap=cmap,
        )
        plt.colorbar(scatter, label=label)
        final_title = title or "CIFAR-10 UMAP Visualization"
        plt.title(final_title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dataset = CIFARUmapDataset(number_of_samples=1000, n_components=2)
    dataset.plot_image(0)
    dataset.plot_embedding(metric="euclidean")
    dm = dataset.get_distance_matrix(distance_metric="euclidean")
    print(dm.shape)

