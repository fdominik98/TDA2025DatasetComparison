from umap import UMAP
from sklearn.metrics import pairwise_distances
from config import RANDOM_STATE, UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_N_COMPONENTS


class Embedding:
    """
    Manages UMAP embeddings and distance matrices for a dataset's feature matrix.

    Caches per-metric embeddings and distance matrices to avoid recomputation.
    """

    def __init__(
        self,
        X,
        number_of_samples=100,
        n_components=None,
        n_neighbors=None,
        min_dist=None,
        use_embedding=True,
    ):
        self.X = X
        self.number_of_samples = number_of_samples
        self.n_components = n_components if n_components is not None else UMAP_N_COMPONENTS
        self.n_neighbors = n_neighbors if n_neighbors is not None else UMAP_N_NEIGHBORS
        self.min_dist = min_dist if min_dist is not None else UMAP_MIN_DIST
        self.use_embedding = use_embedding

        # Embedding caches
        self._X_embedded_euclidean = None
        self._X_embedded_manhattan = None
        self._X_embedded_cosine = None

        # Distance matrices cache: [embedding_metric][distance_metric] -> matrix
        self._distance_matrices = {}

    def update_data(self, X):
        """
        Update the underlying data matrix and clear all caches.
        """
        self.X = X
        self._X_embedded_euclidean = None
        self._X_embedded_manhattan = None
        self._X_embedded_cosine = None
        self._distance_matrices = {}

    def get_embedding(self, metric='euclidean'):
        """
        Get or compute UMAP embedding for the specified metric.
        Caches results per metric.
        """
        if metric == 'euclidean':
            if self._X_embedded_euclidean is None:
                print(f"Computing UMAP embedding with {metric} metric...")
                umap_model = UMAP(
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    n_components=self.n_components,
                    metric=metric,
                    random_state=RANDOM_STATE,
                )
                self._X_embedded_euclidean = umap_model.fit_transform(
                    self.X[: self.number_of_samples]
                )
            return self._X_embedded_euclidean
        elif metric == 'manhattan':
            if self._X_embedded_manhattan is None:
                print(f"Computing UMAP embedding with {metric} metric...")
                umap_model = UMAP(
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    n_components=self.n_components,
                    metric=metric,
                    random_state=RANDOM_STATE,
                )
                self._X_embedded_manhattan = umap_model.fit_transform(
                    self.X[: self.number_of_samples]
                )
            return self._X_embedded_manhattan
        elif metric == 'cosine':
            if self._X_embedded_cosine is None:
                print(f"Computing UMAP embedding with {metric} metric...")
                umap_model = UMAP(
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    n_components=self.n_components,
                    metric=metric,
                    random_state=RANDOM_STATE,
                )
                self._X_embedded_cosine = umap_model.fit_transform(
                    self.X[: self.number_of_samples]
                )
            return self._X_embedded_cosine
        else:
            raise ValueError("metric must be euclidean, manhattan, or cosine")

    def get_distance_matrix(
        self, embedding_metric='euclidean', distance_metric=None, use_embedding=None
    ):
        """
        Get or compute a distance matrix for the specified metrics.
        If use_embedding is False, computes distances on raw data.
        """
        if use_embedding is None:
            use_embedding = self.use_embedding

        if distance_metric is None:
            distance_metric = embedding_metric

        valid_metrics = ['euclidean', 'manhattan', 'cosine']
        if use_embedding and embedding_metric not in valid_metrics:
            raise ValueError(f"embedding_metric must be one of {valid_metrics}")
        if distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {valid_metrics}")

        if not use_embedding:
            cache_key = 'raw'
            if cache_key not in self._distance_matrices:
                self._distance_matrices[cache_key] = {}
            if distance_metric not in self._distance_matrices[cache_key]:
                print(
                    f"Computing distance matrix on raw data (distance: {distance_metric})..."
                )
                self._distance_matrices[cache_key][distance_metric] = pairwise_distances(
                    self.X[: self.number_of_samples], metric=distance_metric
                )
            return self._distance_matrices[cache_key][distance_metric]

        if embedding_metric not in self._distance_matrices:
            self._distance_matrices[embedding_metric] = {}

        if distance_metric not in self._distance_matrices[embedding_metric]:
            print(
                f"Computing distance matrix (embedding: {embedding_metric}, distance: {distance_metric})..."
            )
            embedding = self.get_embedding(embedding_metric)
            self._distance_matrices[embedding_metric][distance_metric] = pairwise_distances(
                embedding, metric=distance_metric
            )

        return self._distance_matrices[embedding_metric][distance_metric]

    def get_raw_distance_matrix(self, distance_metric='euclidean'):
        return self.get_distance_matrix(distance_metric=distance_metric, use_embedding=False)

    # Convenience properties (optional)
    @property
    def X_embedded_euclidean(self):
        return self.get_embedding('euclidean')

    @property
    def X_embedded_manhattan(self):
        return self.get_embedding('manhattan')

    @property
    def X_embedded_cosine(self):
        return self.get_embedding('cosine')

    @property
    def euclidean_distance_matrix(self):
        return self.get_distance_matrix(embedding_metric='euclidean', distance_metric='euclidean')

    @property
    def manhattan_distance_matrix(self):
        return self.get_distance_matrix(embedding_metric='manhattan', distance_metric='manhattan')

    @property
    def cosine_distance_matrix(self):
        return self.get_distance_matrix(embedding_metric='cosine', distance_metric='cosine')


