"""
Global configuration for the TDA project.
"""
import numpy as np
import random

# Random state for reproducibility
RANDOM_STATE = 42

# Set random seeds
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# UMAP parameters
UMAP_N_NEIGHBORS = 50
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2

