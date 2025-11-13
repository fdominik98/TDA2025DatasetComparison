def apply_consistency_transformation(dist_matrix, labels, 
                                        shrink_factor, expand_factor):
    """
    Apply consistency transformation to distance matrix:
    - Shrink distances within same cluster
    - Expand distances between different clusters
    """
    modified_dist = dist_matrix.copy()
    n = len(labels)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if both points are in the same cluster (and not noise)
            if labels[i] >= 0 and labels[j] >= 0:
                if labels[i] == labels[j]:
                    # Same cluster: shrink distance
                    modified_dist[i, j] *= shrink_factor
                    modified_dist[j, i] *= shrink_factor
                else:
                    # Different clusters: expand distance
                    modified_dist[i, j] *= expand_factor
                    modified_dist[j, i] *= expand_factor
    
    return modified_dist



def apply_consistency_transformation_for_one_pair(dist_matrix, labels, 
                                        shrink_factor, expand_factor):
    """
    Apply consistency transformation to distance matrix:
    - Shrink distances within same cluster
    - Expand distances between different clusters
    """
    modified_dist = dist_matrix.copy()
    n = len(labels)
    
    shrinked_one = False
    expanded_one = False
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if both points are in the same cluster (and not noise)
            if labels[i] >= 0 and labels[j] >= 0:
                if labels[i] == labels[j] and not shrinked_one:
                    # Same cluster: shrink distance
                    modified_dist[i, j] *= shrink_factor
                    modified_dist[j, i] *= shrink_factor
                    shrinked_one = True
                elif labels[i] != labels[j] and not expanded_one:
                    # Different clusters: expand distance
                    modified_dist[i, j] *= expand_factor
                    modified_dist[j, i] *= expand_factor
                    expanded_one = True
    return modified_dist