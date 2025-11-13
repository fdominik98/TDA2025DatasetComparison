from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence

from stablerank.distances import Distance
from stablerank.barcodes import BC
from stablerank.rtorf import Pcnif

from clustering_algorithms.base import ClusteringAlgorithm
from clustering_algorithms.single_linkage import SingleLinkageThreshold


class TDAPersistentComponents(ClusteringAlgorithm):
    """Clustering via persistent connected components derived from H₀ barcodes."""

    def __init__(self) -> None:
        super().__init__(name="single_linkage_threshold_by_tda_persistent_components")
        self._barcode: BC | None = None
        self._stable_rank: Pcnif | None = None
        self._resolved_threshold: float | None = None

    def _cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        self._barcode = self._compute_barcode(distance_matrix)
        self._stable_rank = self._compute_stable_rank(self._barcode)
        threshold = self._best_finite_bar_distance(self._barcode.bars)
        self._resolved_threshold = threshold
        single_linkage = SingleLinkageThreshold(threshold)
        return single_linkage._cluster(distance_matrix)

    def _compute_barcode(self, distance_matrix: np.ndarray) -> BC:
        sym_matrix = np.asarray(distance_matrix, dtype=float)
        if sym_matrix.ndim != 2 or sym_matrix.shape[0] != sym_matrix.shape[1]:
            raise ValueError("Distance matrix must be square.")
        sym_matrix = (sym_matrix + sym_matrix.T) / 2.0
        np.fill_diagonal(sym_matrix, 0.0)
        distance = Distance(sym_matrix)
        barcode = distance.get_bc(maxdim=0, reduced=False)["H0"]
        return barcode

    def _best_finite_bar_distance(self, barcode_array: np.ndarray) -> float:
        finite_bars = barcode_array[barcode_array[:, 1] < np.inf]
        if finite_bars.size == 0:
            raise ValueError("Unable to compute threshold: no finite bars found in barcode.")
        return float(np.median(finite_bars[:, 1]))

    def _compute_stable_rank(self, barcode: BC) -> Pcnif:
        return barcode.stable_rank()

    def _build_metadata(
        self,
        distance_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, object] | None:
        metadata = super()._build_metadata(distance_matrix, labels)
        extra: dict[str, object] = {}
        if self._resolved_threshold is not None:
            extra["threshold"] = round(self._resolved_threshold, 3)
        if self._barcode is not None:
            extra["barcode"] = self._barcode.bars.tolist()
        if self._stable_rank is not None:
            extra["stable_rank"] = self._stable_rank.content.tolist()
        if metadata:
            metadata.update(extra)
            return metadata
        return extra or None

    # ------------------------------------------------------------------ #
    # Visualisation utilities.
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_persistence_summary(
        barcode: np.ndarray,
        dataset_name : str,
        clustering_metric : str,
        stable_rank: Sequence[Sequence[float]] | None = None,
        figsize: tuple[int, int] = (18, 6),
    ) -> None:
        """Plot barcode, persistence diagram, and stable-rank curve for H₀."""
        barcode_array = np.asarray(barcode, dtype=float)
        if barcode_array.ndim != 2 or barcode_array.shape[1] != 2:
            raise ValueError("Barcode must be a 2D array with birth/death columns.")

        fig, (ax_barcode, ax_diagram, ax_stable_rank) = plt.subplots(1, 3, figsize=figsize)

        fig.suptitle(f"TDA Persistence Summary - Dataset: {dataset_name} | Clustering: {clustering_metric}")

        if barcode_array.size == 0:
            ax_barcode.text(0.5, 0.5, "Empty barcode", ha="center", va="center")
            ax_diagram.text(0.5, 0.5, "Empty persistence diagram", ha="center", va="center")
            stable_rank_fn = TDAPersistentComponents._stable_rank_from_metadata_or_barcode(
                barcode_array,
                stable_rank,
            )
            stable_rank_fn.plot(ax=ax_stable_rank, color="purple", linewidth=1.5)
        else:
            finite_bars = barcode_array[np.isfinite(barcode_array[:, 1])]
            infinite_bars = barcode_array[~np.isfinite(barcode_array[:, 1])]

            for i, (birth, death) in enumerate(finite_bars):
                ax_barcode.plot([birth, death], [i, i], "r-", linewidth=1.5)

            max_death_candidate = finite_bars[:, 1].max() if len(finite_bars) > 0 else 1.0
            max_birth_candidate = barcode_array[:, 0].max() if barcode_array.size else 1.0
            max_extent = max(max_death_candidate, max_birth_candidate, 1.0)

            for i, (birth, _death) in enumerate(infinite_bars):
                idx = len(finite_bars) + i
                ax_barcode.plot([birth, max_extent * 1.2], [idx, idx], "b-", linewidth=1.5)
                ax_barcode.plot(max_extent * 1.2, idx, "b>", markersize=8)

            if len(finite_bars) > 0 and len(infinite_bars) > 0:
                ax_barcode.plot([], [], "r-", linewidth=1.5, label="Finite bars")
                ax_barcode.plot([], [], "b-", linewidth=1.5, label="Infinite bars")
                ax_barcode.legend()

            ax_barcode.set_xlabel("Distance")
            ax_barcode.set_ylabel("Bar Index")
            ax_barcode.set_title(f"H₀ Persistence Barcode")
            ax_barcode.grid(True, alpha=0.3)

            finite_points = finite_bars if len(finite_bars) > 0 else np.empty((0, 2))
            if finite_points.size > 0:
                ax_diagram.scatter(finite_points[:, 0], finite_points[:, 1], color="red", label="Finite bars")
            if len(infinite_bars) > 0:
                ax_diagram.scatter(
                    infinite_bars[:, 0],
                    np.full(len(infinite_bars), max_extent * 1.1),
                    color="blue",
                    marker="^",
                    label="Infinite bars",
                )

            diag_limit = max_extent * 1.1
            ax_diagram.plot([0, diag_limit], [0, diag_limit], color="black", linewidth=1.0, linestyle="--")
            ax_diagram.set_xlim(0, diag_limit)
            ax_diagram.set_ylim(0, diag_limit)
            ax_diagram.set_xlabel("Birth")
            ax_diagram.set_ylabel("Death / Proxy")
            ax_diagram.set_title("Persistence Diagram (H₀)")
            if ax_diagram.legend_ is None and (finite_points.size > 0 or len(infinite_bars) > 0):
                ax_diagram.legend()
            ax_diagram.grid(True, alpha=0.3)

            stable_rank_fn = TDAPersistentComponents._stable_rank_from_metadata_or_barcode(barcode_array, stable_rank)
            stable_rank_fn.plot(ax=ax_stable_rank, color="purple", linewidth=1.5)

        ax_stable_rank.set_xlabel("Persistence Threshold")
        ax_stable_rank.set_ylabel("PCF / Stable Rank")
        ax_stable_rank.set_title("Stable Rank Diagram")
        ax_stable_rank.set_xlim(left=0.0)
        ax_stable_rank.set_ylim(bottom=0.0)
        ax_stable_rank.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _stable_rank_from_metadata_or_barcode(
        barcode: np.ndarray,
        stable_rank: Sequence[Sequence[float]] | None,
    ) -> Pcnif:
        if stable_rank is not None:
            sr_array = np.asarray(stable_rank, dtype=float)
            if sr_array.ndim == 2 and sr_array.shape[0] == 2 and sr_array.size:
                return Pcnif(sr_array)

        barcode_array = np.asarray(barcode, dtype=float)
        if barcode_array.size == 0:
            return Pcnif(np.array([[0.0], [0.0]], dtype=float))

        bc = BC(np.asarray(barcode_array, dtype="double"))
        return bc.stable_rank()

