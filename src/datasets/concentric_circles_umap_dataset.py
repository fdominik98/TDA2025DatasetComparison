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

from .embedded_dataset import EmbeddedDataset  # type: ignore[import]
from .concentric_circles_dataset import ConcentricCirclesDataset  # type: ignore[import]


class ConcentricCirclesUmapDataset(EmbeddedDataset, ConcentricCirclesDataset):
    """
    UMAP-embedded variant of the synthetic ConcentricCircles dataset with embedding plot.
    """

    def __init__(
        self,
        number_of_samples=1000,
        n_circles=10,
        points_per_circle=200,
        radii=None,
        noise=0.05,
        random_state=None,
        n_components=None,
        n_neighbors=None,
        min_dist=None,
        embedding_metric='euclidean',
    ):
        ConcentricCirclesDataset.__init__(
            self,
            number_of_samples=number_of_samples,
            n_circles=n_circles,
            points_per_circle=points_per_circle,
            radii=radii,
            noise=noise,
            random_state=random_state,
        )
        EmbeddedDataset.__init__(
            self,
            number_of_samples=number_of_samples,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            embedding_metric=embedding_metric,
        )
        self.name = f"Concentric Circles UMAP Dataset (n_components={n_components})"

if __name__ == "__main__":
    dataset = ConcentricCirclesUmapDataset(number_of_samples=1000, n_components=2)
    dataset.plot_embedding(metric='euclidean')
    dm = dataset.get_distance_matrix(distance_metric='euclidean')
    print(dm.shape)


