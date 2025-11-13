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

from sklearn.datasets import fetch_openml  # type: ignore[import]
import matplotlib.pyplot as plt
from .base_dataset import BaseDataset  # type: ignore[import]


class MNISTDataset(BaseDataset):
    """
    MNIST dataset (raw). No embedding logic here.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "MNIST Raw Dataset"
    
    def _load_data(self):
        """Load MNIST data from sklearn."""
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        self.X, self.y = mnist.data, mnist.target.astype(int)
    
    def plot_image(self, index):
        """Display a single MNIST digit."""
        print(f"Label: {self.y[index]}")
        plt.imshow(self.X[index].reshape(28, 28), cmap='gray')
        plt.show()

    def plot_embedding(
        self,
        figsize=(8, 6),
        s=10,
        cmap='viridis',
        title=None,
    ):
        """
        Visualize the dataset using a 2D UMAP projection for consistency across MNIST variants.
        """
        embedding = self.get_visualization_embedding()
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
    
# Example usage
if __name__ == "__main__":
    dataset = MNISTDataset(number_of_samples=1000)
    dataset.print_first_10_labels()
    dataset.print_first_10_images_and_labels()
    dataset.plot_image(0)