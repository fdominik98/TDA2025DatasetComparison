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
from sklearn.datasets import fetch_openml  # type: ignore[import]
from .embedded_dataset import EmbeddedDataset  # type: ignore[import]


class MNISTUmapDataset(EmbeddedDataset):
    """
    MNIST dataset with UMAP embedding support and embedding plot.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = f"MNIST UMAP Dataset (n_components={self.n_components})"

    def _load_data(self):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        self.X, self.y = mnist.data, mnist.target.astype(int)

    def plot_image(self, index):
        # Defer to raw display logic similar to MNISTDataset (user can implement if needed)
        print(f"Label: {self.y[index]}")
        plt.imshow(self.X[index].reshape(28, 28), cmap='gray')
        plt.show()

    def plot_embedding(
        self,
        metric=None,
        figsize=(8, 6),
        s=10,
        cmap='viridis',
        label='Label',
        title=None,
    ):
        """
        Visualize MNIST samples via a fixed 2D UMAP projection, independent of the distance embedding.

        The `metric` argument is accepted for API compatibility but ignored.
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
        final_title = title or "MNIST UMAP Visualization"
        plt.title(final_title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dataset = MNISTUmapDataset(number_of_samples=1000, n_components=2)
    dataset.plot_image(0)
    dataset.plot_embedding(metric='euclidean')
    dm = dataset.get_distance_matrix(distance_metric='euclidean')
    print(dm.shape)

