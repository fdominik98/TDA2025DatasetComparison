"""
Abstract base class for datasets with raw data only (no embedding).
"""
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

from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, Optional

from sklearn.metrics import pairwise_distances  # type: ignore[import]
from umap import UMAP  # type: ignore[import]

from config import (  # type: ignore[import]
    RANDOM_STATE,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
)


class BaseDataset(ABC):
    """
    Abstract base class for datasets with raw data and distance matrices (no embedding).
    Subclasses must implement `_load_data` and `plot_image`.
    """
    
    def __init__(self, number_of_samples=100):
        """
        Initialize the dataset and load raw data.
        """
        self.number_of_samples = number_of_samples
        self._visualization_embedding_cache: Dict[Hashable, Any] = {}
        self.X: Any = None
        self.y: Any = None
        
        # Load the actual data (implemented by subclasses)
        self._load_data()
        # Shuffle the data
        # self.X, self.y = shuffle(self.X, self.y, random_state=RANDOM_STATE)
        self.name = f"{self.__class__.__name__} Dataset"
    
    @abstractmethod
    def _load_data(self):
        """
        Load the dataset and set self.X (features) and self.y (labels).
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def plot_image(self, index):
        """
        Visualize a single data point at the given index.
        Must be implemented by subclasses.
        """
        pass
    
    @property
    def shape(self):
        """Return the shape of the feature matrix."""
        return self.X.shape
    
    @property
    def labels(self):
        """Return the labels array."""
        return self.y
    
    @property
    def data(self):
        """Return features and labels as a tuple."""
        return self.X, self.y
    
    def print_first_10_labels(self):
        """Print the first 10 labels."""
        print(self.y[:10])
    
    def print_first_10_images(self):
        """Print the first 10 feature vectors."""
        print(self.X[:10])
    
    def print_first_10_images_and_labels(self):
        """Print the first 10 feature vectors and their labels."""
        print(self.X[:10], self.y[:10])
    
    def get_distance_features(self):
        """
        Return the feature matrix used for distance computations.
        Raw datasets return `self.X`. Embedded datasets can override this.
        """
        return self.X[:self.number_of_samples]
    
    def get_distance_matrix(self, distance_metric='euclidean'):
        """
        Compute pairwise distances on the dataset's distance features.
        Raw datasets use X; embedded datasets may override the distance features.
        """
        features = self.get_distance_features()
        return pairwise_distances(features, metric=distance_metric)

    def get_visualization_embedding(
        self,
        *,
        features=None,
        n_components: int = 2,
        n_neighbors: Optional[int] = None,
        min_dist: Optional[float] = None,
        metric: str = 'euclidean',
        random_state: Optional[int] = None,
        cache_key: Optional[Hashable] = None,
    ):
        """
        Compute (and optionally cache) a UMAP embedding intended solely for visualization.
        Subclasses can override or extend, but by default this operates on the raw samples.
        """
        if features is None:
            features = self.X[: self.number_of_samples]
        if n_neighbors is None:
            n_neighbors = UMAP_N_NEIGHBORS
        if min_dist is None:
            min_dist = UMAP_MIN_DIST
        if random_state is None:
            random_state = RANDOM_STATE

        effective_cache_key: Optional[Hashable]
        if cache_key is not None:
            effective_cache_key = cache_key
        else:
            effective_cache_key = (metric, n_components, n_neighbors, min_dist, random_state, self.number_of_samples)

        if effective_cache_key is not None and effective_cache_key in self._visualization_embedding_cache:
            return self._visualization_embedding_cache[effective_cache_key]

        umap_model = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
        )
        embedding = umap_model.fit_transform(features)

        if effective_cache_key is not None:
            self._visualization_embedding_cache[effective_cache_key] = embedding

        return embedding
    
