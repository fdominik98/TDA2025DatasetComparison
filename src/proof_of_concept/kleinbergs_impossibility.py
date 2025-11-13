"""
THIS CLASS IS NOT USED IN THE PROJECT. IT IS HERE FOR REFERENCE ONLY.
Kleinberg's Impossibility Theorem Testing

Kleinberg's impossibility theorem:
- Scale-invariance: Scaling all distances shouldn't change clustering
- Consistency: Shrinking intra-cluster distances and expanding inter-cluster 
               distances shouldn't change clustering
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from clustering import Clustering
from clustering_algorithms import ClusteringResult
from datasets.concentric_triangles_dataset import ConcentricTrianglesDataset
from datasets.mnist_dataset import MNISTDataset
from distance_transformations import apply_consistency_transformation, apply_consistency_transformation_for_one_pair


class KleinbergsImpossibility:
    """
    Test clustering algorithms for scale-invariance and consistency properties
    from Kleinberg's impossibility theorem.
    """
    
    def __init__(self, clustering: Clustering):
        """
        Initialize with a Clustering object.
        """
        self.clustering = clustering
        self.original_distance_matrix = clustering.get_distance_matrix().copy()
        self.results: dict = {}
    
    def test_scale_invariance(self, clustering_method, method_params, scale_factors=None):
        """
        Test scale-invariance: clustering should remain the same when all distances
        are multiplied by a constant factor.
        """
        if scale_factors is None:
            scale_factors = [0.5, 2.0, 5.0, 10.0]
        
        # Reset to original distance matrix
        self.clustering.set_distance_matrix(self.original_distance_matrix.copy())
        
        # Get original clustering
        print(f"\nTesting Scale-Invariance for {clustering_method}")
        print("=" * 80)
        print(f"Running original clustering with params: {method_params}")
        original_labels = self._run_clustering(clustering_method, method_params)
        
        results = {'scale_factors': [], 'ari_scores': [], 'same_clustering': []}
        
        # Test each scale factor
        for scale in scale_factors:
            print(f"\nTesting scale factor: {scale}")
            
            # Scale the distance matrix
            scaled_dist = self.original_distance_matrix * scale
            self.clustering.set_distance_matrix(scaled_dist)
            
            # Adjust parameters that depend on distances
            scaled_params = self._scale_parameters(method_params, scale, clustering_method)
            print(f"  Scaled parameters: {scaled_params}")
            
            # Run clustering on scaled distances
            scaled_labels = self._run_clustering(clustering_method, scaled_params)
            
            # Compare clusterings
            ari = adjusted_rand_score(original_labels, scaled_labels)
            is_same = np.array_equal(original_labels, scaled_labels)
            
            results['scale_factors'].append(scale)
            results['ari_scores'].append(ari)
            results['same_clustering'].append(is_same)
            
            print(f"  ARI: {ari:.4f} | Same clustering: {is_same}")
        
        # Summary
        print("\n" + "=" * 80)
        print("SCALE-INVARIANCE TEST SUMMARY")
        print("=" * 80)
        print(f"{'Scale Factor':<15} {'ARI':>10} {'Same Clustering':>18}")
        print("-" * 45)
        for i, scale in enumerate(results['scale_factors']):
            print(f"{scale:<15.2f} {results['ari_scores'][i]:>10.4f} {str(results['same_clustering'][i]):>18}")
        
        avg_ari = np.mean(results['ari_scores'])
        all_same = all(results['same_clustering'])
        print("=" * 45)
        print(f"Average ARI: {avg_ari:.4f}")
        print(f"Scale-invariant: {all_same}")
        
        self.results['scale_invariance'] = results
        
        # Reset to original distance matrix
        self.clustering.set_distance_matrix(self.original_distance_matrix.copy())
        
        return results
    
    def test_consistency(self, clustering_method, method_params, 
                        shrink_factors=None, expand_factors=None):
        """
        Test consistency: clustering should remain the same when intra-cluster distances
        are shrunk and inter-cluster distances are expanded.
        Tests all combinations of shrink and expand factors.
        """
        if shrink_factors is None:
            shrink_factors = [0.5, 0.25]
        if expand_factors is None:
            expand_factors = [2.0, 4.0]
        
        # Ensure they are lists
        if not isinstance(shrink_factors, list):
            shrink_factors = [shrink_factors]
        if not isinstance(expand_factors, list):
            expand_factors = [expand_factors]
        
        # Reset to original distance matrix
        self.clustering.set_distance_matrix(self.original_distance_matrix.copy())
        
        # Get original clustering
        print(f"\nTesting Consistency for {clustering_method}")
        print("=" * 80)
        print(f"Running original clustering with params: {method_params}")
        original_labels = self._run_clustering(clustering_method, method_params)
        
        # Test all combinations
        results = {
            'combinations': [],
            'ari_scores': [],
            'same_clustering': [],
            'n_clusters': [],
            'original_labels': original_labels,
            'n_clusters_original': len(np.unique(original_labels[original_labels >= 0]))
        }
        
        print(f"\nTesting {len(shrink_factors)} x {len(expand_factors)} = {len(shrink_factors) * len(expand_factors)} combinations")
        print(f"Shrink factors: {shrink_factors}")
        print(f"Expand factors: {expand_factors}")
        
        for shrink in shrink_factors:
            for expand in expand_factors:
                print(f"\n  Testing shrink={shrink}, expand={expand}...")
                
                # Create consistency-modified distance matrix
                modified_dist = apply_consistency_transformation(
                    self.original_distance_matrix.copy(),
                    original_labels,
                    shrink,
                    expand
                )
                
                self.clustering.set_distance_matrix(modified_dist)
                
                # Run clustering on modified distances
                modified_labels = self._run_clustering(clustering_method, method_params)
                
                # Compare clusterings
                ari = adjusted_rand_score(original_labels, modified_labels)
                is_same = np.array_equal(original_labels, modified_labels)
                n_clusters = len(np.unique(modified_labels[modified_labels >= 0]))
                
                results['combinations'].append((shrink, expand))
                results['ari_scores'].append(ari)
                results['same_clustering'].append(is_same)
                results['n_clusters'].append(n_clusters)
                
                print(f"    ARI: {ari:.4f} | Same clustering: {is_same} | Clusters: {n_clusters}")
        
        # Summary
        print("\n" + "=" * 80)
        print("CONSISTENCY TEST SUMMARY")
        print("=" * 80)
        print(f"{'Shrink':>8} {'Expand':>8} {'ARI':>10} {'Same Clustering':>18} {'#Clusters':>12}")
        print("-" * 60)
        for i, (shrink, expand) in enumerate(results['combinations']):
            print(f"{shrink:>8.2f} {expand:>8.2f} {results['ari_scores'][i]:>10.4f} "
                  f"{str(results['same_clustering'][i]):>18} {results['n_clusters'][i]:>12}")
        
        avg_ari = np.mean(results['ari_scores'])
        all_same = all(results['same_clustering'])
        print("-" * 60)
        print(f"Average ARI: {avg_ari:.4f}")
        print(f"All clusterings consistent: {all_same}")
        print(f"Original clusters: {results['n_clusters_original']}")
        
        self.results['consistency'] = results
        
        # Reset to original distance matrix
        self.clustering.set_distance_matrix(self.original_distance_matrix.copy())
        
        return results
    
    def test_both_properties(self, clustering_method, method_params,
                            scale_factors=None, shrink_factors=None, expand_factors=None):
        """
        Test both scale-invariance and consistency.
        """
        print("=" * 80)
        print("KLEINBERG'S IMPOSSIBILITY THEOREM TESTING")
        print(f"Method: {clustering_method}")
        print(f"Parameters: {method_params}")
        print("=" * 80)
        
        # Test scale-invariance
        scale_results = self.test_scale_invariance(
            clustering_method, method_params, scale_factors
        )
        
        # Test consistency
        consistency_results = self.test_consistency(
            clustering_method, method_params, shrink_factors, expand_factors
        )
        
        # Combined summary
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        
        scale_invariant = all(scale_results['same_clustering'])
        consistent = all(consistency_results['same_clustering'])
        
        print(f"Scale-invariant: {scale_invariant}")
        print(f"  - Average ARI across scales: {np.mean(scale_results['ari_scores']):.4f}")
        print(f"  - All clusterings identical: {scale_invariant}")
        
        print(f"\nConsistent: {consistent}")
        print(f"  - Average ARI across consistency tests: {np.mean(consistency_results['ari_scores']):.4f}")
        print(f"  - All clusterings identical: {consistent}")
        print(f"  - Tested {len(consistency_results['combinations'])} combinations")
        
        print("\n" + "=" * 80)
        print("KLEINBERG'S THEOREM IMPLICATIONS:")
        print("-" * 80)
        if scale_invariant and consistent:
            print("WARNING: Algorithm satisfies both scale-invariance and consistency!")
            print("  According to Kleinberg's theorem, it cannot have richness.")
        elif scale_invariant and not consistent:
            print("OK: Algorithm is scale-invariant but not consistent.")
            print("  This is expected for many clustering algorithms.")
        elif not scale_invariant and consistent:
            print("OK: Algorithm is consistent but not scale-invariant.")
            print("  This is expected for distance-threshold methods.")
        else:
            print("OK: Algorithm is neither scale-invariant nor consistent.")
            print("  This is common for practical clustering algorithms.")
        print("=" * 80)
        
        combined_results = {
            'scale_invariance': scale_results,
            'consistency': consistency_results,
            'scale_invariant': scale_invariant,
            'consistent': consistent
        }
        
        return combined_results
    
    def _run_clustering(self, method_name, params):
        """Run the specified clustering method with given parameters."""
        method = getattr(self.clustering, method_name)
        
        # Handle methods that return tuple (labels, barcode)
        result = method(**params)
        if isinstance(result, ClusteringResult):
            return result.labels
        if isinstance(result, tuple):
            return result[0]
        return result
    
    def _scale_parameters(self, params, scale_factor, method_name):
        """
        Scale parameters that depend on distances.
        
        For threshold-based methods, scale the threshold.
        For k-based methods, keep k the same.
        """
        scaled_params = params.copy()
        
        # Distance threshold parameters need scaling
        if 'r' in scaled_params:
            scaled_params['r'] = scaled_params['r'] * scale_factor
        
        if 'eps' in scaled_params:
            scaled_params['eps'] = scaled_params['eps'] * scale_factor
        
        if 'threshold' in scaled_params and scaled_params['threshold'] is not None:
            scaled_params['threshold'] = scaled_params['threshold'] * scale_factor
        
        # Alpha and k don't need scaling (they're scale-invariant)
        # alpha is already a fraction of max distance
        # k is just number of clusters
        
        return scaled_params

    
    def get_results(self):
        """Return all stored test results."""
        return self.results


# Example usage
if __name__ == "__main__":
    from datasets.concentric_circles_dataset import ConcentricCirclesDataset
    
    print("=" * 80)
    print("KLEINBERG'S IMPOSSIBILITY THEOREM - DEMONSTRATION")
    print("=" * 80)
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = ConcentricCirclesDataset(
        number_of_samples=1000,
        n_circles=3,
        points_per_circle=100
    )
    dataset = MNISTDataset(
        number_of_samples=1000
    )
    dataset = ConcentricTrianglesDataset(
        number_of_samples=2000,
        n_triangles=2,
        points_per_triangle=1000,
        noise=0.1,
    )
    
    # Create clustering object
    print("Creating clustering object...")
    clustering = Clustering(
        dataset,
        embedding_metric='manhattan',
        clustering_metric='manhattan'
    )
    
    # Create Kleinberg test object
    kb_test = KleinbergsImpossibility(clustering)
    
    # Test 1: Single linkage with k clusters (should be scale-invariant)
    print("\n" + "=" * 80)
    print("TEST 1: Single Linkage with k=3")
    print("=" * 80)
    results1 = kb_test.test_both_properties(
        clustering_method='single_linkage_k_clusters',
        method_params={'k': 2},
        scale_factors=[0.5, 2.0, 10.0],
        shrink_factors=[0.5, 0.25, 0.1],
        expand_factors=[2.0, 4.0]
    )
    
    # Test 2: Single linkage with distance threshold (NOT scale-invariant)
    print("\n\n" + "=" * 80)
    print("TEST 2: Single Linkage with Distance Threshold r=2.0")
    print("=" * 80)
    results2 = kb_test.test_both_properties(
        clustering_method='single_linkage_distance_threshold',
        method_params={'r': 0.01},
        scale_factors=[0.01, 0.5, 2.0, 10.0, 100.0, 1000.0],
        shrink_factors=[0.5, 0.25],
        expand_factors=[2.0, 3.0]
    )
    
    # Test 3: Single linkage with alpha (should be scale-invariant)
    print("\n\n" + "=" * 80)
    print("TEST 3: Single Linkage with Alpha=0.3")
    print("=" * 80)
    results3 = kb_test.test_both_properties(
        clustering_method='single_linkage_scale_alpha',
        method_params={'alpha': 0.2},
        scale_factors=[0.5, 2.0, 10.0],
        shrink_factors=[0.5, 0.2],
        expand_factors=[2.0, 5.0]
    )
    
    # Test 4: K-means clustering (should not be consistent)
    print("\n" + "=" * 80)
    print("TEST 4: KMeans with k=2")
    print("=" * 80)
    results4 = kb_test.test_both_properties(
        clustering_method='kmeans_clustering',
        method_params={'k': 2},
        scale_factors=[0.5, 2.0, 10.0],
        shrink_factors=[0.5, 0.25, 0.01, 0.001, 1, 0.95],
        expand_factors=[2.0, 3.0, 100.0, 1, 1.05]
    )

