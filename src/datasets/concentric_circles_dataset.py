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

import numpy as np
import pandas as pd  # type: ignore[import]
import matplotlib.pyplot as plt
from config import RANDOM_STATE  # type: ignore[import]
from .base_dataset import BaseDataset  # type: ignore[import]


class ConcentricCirclesDataset(BaseDataset):
    
    def __init__(self, number_of_samples=100, n_circles=10, points_per_circle=200, radii=None, noise=0.05, random_state=None):
        """
        Initialize the Concentric Circles dataset.
        
        Parameters:
        - number_of_samples: Number of samples to use for embeddings
        - n_circles: Number of concentric circles to generate
        - points_per_circle: Number of points per circle
        - radii: List/array of radii for circles
        - noise: Standard deviation of Gaussian noise
        - random_state: Random state for reproducibility (defaults to RANDOM_STATE from config)
        """
        # Store generation parameters
        self.n_circles = n_circles
        self.points_per_circle = points_per_circle
        self.radii = radii
        self.noise = noise
        self.random_state = random_state if random_state is not None else RANDOM_STATE
        
        # Call parent constructor which will call _load_data
        super().__init__(number_of_samples)
        self.name = f"Concentric Circles Dataset (n_circles={n_circles})"
    
    def _load_data(self):
        """Generate fresh concentric circles data."""
        self.df = self.make_concentric_circles(
            n_circles=self.n_circles,
            points_per_circle=self.points_per_circle,
            radii=self.radii,
            noise=self.noise,
            random_state=self.random_state
        )
        
        # Extract X (features) and y (labels)
        self.X = self.df[['x', 'y']].values
        self.y = self.df['label'].values
    
    def make_concentric_circles(self, n_circles=3, points_per_circle=200, radii=None, noise=0.05, random_state=None):
        """
        Generate 2D points lying on concentric circles with optional Gaussian noise.
        Returns a pandas DataFrame with columns: x, y, label, radius, angle.
        
        Parameters:
        - n_circles: number of concentric circles
        - points_per_circle: number of points per circle (int or list of ints for per-circle counts)
        - radii: list/array of radii. If None, radii will be [1,2,...,n_circles]
        - noise: standard deviation of Gaussian noise added to (x,y)
        - random_state: int or None for reproducibility
        """
        rng = np.random.default_rng(random_state)
        
        # allow points_per_circle to be either int or list
        if isinstance(points_per_circle, int):
            counts = [points_per_circle] * n_circles
        else:
            counts = list(points_per_circle)
            if len(counts) != n_circles:
                raise ValueError("points_per_circle list must match n_circles length")
        
        if radii is None:
            radii = np.arange(1, n_circles+1)
        else:
            radii = np.array(radii)
            if radii.shape[0] != n_circles:
                raise ValueError("radii length must equal n_circles")
        
        rows = []
        for i, (r, cnt) in enumerate(zip(radii, counts)):
            # uniform angles around circle
            angles = rng.uniform(0, 2*np.pi, size=cnt)
            x = r * np.cos(angles) + rng.normal(0, noise, size=cnt)
            y = r * np.sin(angles) + rng.normal(0, noise, size=cnt)
            label = np.full(cnt, i, dtype=int)
            radius_col = np.full(cnt, r, dtype=float)
            for xi, yi, lab, ang, rad in zip(x, y, label, angles, radius_col):
                rows.append((xi, yi, int(lab), float(rad), float(ang)))
        
        df = pd.DataFrame(rows, columns=["x", "y", "label", "radius", "angle"])
        
        return df
    
    def display_dataframe(self, df):
        print(df.head(12).to_string(index=False))
        
    def plot_concentric(self, figsize=(6,6), marker='o', s=20, alpha=0.8):
        """
        Scatter-plot the concentric circles dataset. Each label plotted with a separate point series.
        """
        labels = sorted(self.df['label'].unique())
        plt.figure(figsize=figsize)
        for lab in labels:
            sub = self.df[self.df['label'] == lab]
            plt.scatter(sub['x'], sub['y'], label=f"label={lab}, r={sub['radius'].iloc[0]:.2f}", marker=marker, s=s, alpha=alpha)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Concentric circles (synthetic)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_image(self, index):
        """
        Plot a single point from the dataset, highlighting it within the context of all circles.
        Similar to MNIST's plot_image method.
        """
        print(f"Label: {self.y[index]}")
        print(f"Coordinates: x={self.X[index][0]:.4f}, y={self.X[index][1]:.4f}")
        
        # Plot all circles with low opacity
        labels = sorted(self.df['label'].unique())
        plt.figure(figsize=(6,6))
        for lab in labels:
            sub = self.df[self.df['label'] == lab]
            plt.scatter(sub['x'], sub['y'], label=f"label={lab}", marker='o', s=20, alpha=0.3)
        
        # Highlight the specific point
        plt.scatter(self.X[index][0], self.X[index][1], 
                   color='red', s=200, marker='*', 
                   edgecolors='black', linewidths=2,
                   label=f'Selected point (index={index})', zorder=5)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Point at index {index} (Label: {self.y[index]})')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Raw dataset: no embedding plot here


# Example usage
if __name__ == "__main__":
    dataset = ConcentricCirclesDataset(number_of_samples=1000)
    dataset.print_first_10_labels()
    dataset.print_first_10_images_and_labels()
    dataset.plot_image(0)