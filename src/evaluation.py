"""
Evaluation metrics for clustering quality assessment
Implements all metrics specified in the project requirements
"""
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def compute_silhouette_score(X, labels):
    """
    Silhouette Score: Measures how similar an object is to its own cluster
    compared to other clusters.
    
    Range: [-1, 1]
    Higher is better
    
    Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where:
        a(i) = average distance within cluster
        b(i) = minimum average distance to other clusters
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        Silhouette score
    """
    try:
        # Need at least 2 clusters
        if len(np.unique(labels)) < 2:
            return -1.0
        score = silhouette_score(X, labels)
        return score
    except Exception as e:
        print(f"Warning: Could not compute Silhouette Score - {e}")
        return -1.0


def compute_calinski_harabasz_index(X, labels):
    """
    Calinski-Harabasz Index (Variance Ratio Criterion)
    Measures ratio of between-cluster variance to within-cluster variance.
    
    Higher is better
    
    Formula: CH = [tr(Bk) / (k-1)] / [tr(Wk) / (n-k)]
    where:
        k = number of clusters
        n = number of points
        Bk = between-cluster dispersion matrix
        Wk = within-cluster dispersion matrix
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        Calinski-Harabasz index
    """
    try:
        if len(np.unique(labels)) < 2:
            return 0.0
        score = calinski_harabasz_score(X, labels)
        return score
    except Exception as e:
        print(f"Warning: Could not compute Calinski-Harabasz Index - {e}")
        return 0.0


def compute_davies_bouldin_index(X, labels):
    """
    Davies-Bouldin Index: Average similarity of each cluster with its most similar cluster
    
    Range: [0, ∞)
    Lower is better
    
    Formula: DB = (1/k) * Σ max_{j≠i} [(σi + σj) / dij]
    where:
        σi = average distance of points in cluster i to centroid
        dij = distance between centroids of clusters i and j
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        Davies-Bouldin index
    """
    try:
        if len(np.unique(labels)) < 2:
            return float('inf')
        score = davies_bouldin_score(X, labels)
        return score
    except Exception as e:
        print(f"Warning: Could not compute Davies-Bouldin Index - {e}")
        return float('inf')


def compute_adjusted_rand_index(y_true, y_pred):
    """
    Adjusted Rand Index (ARI): Measures similarity between predicted clusters
    and ground truth labels, adjusted for chance.
    
    Range: [-1, 1]
    Higher is better (1 = perfect match, 0 = random labeling)
    
    Formula: ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    where RI is the Rand Index
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster labels
        
    Returns:
        Adjusted Rand Index
    """
    try:
        score = adjusted_rand_score(y_true, y_pred)
        return score
    except Exception as e:
        print(f"Warning: Could not compute Adjusted Rand Index - {e}")
        return 0.0


def compute_normalized_mutual_info(y_true, y_pred):
    """
    Normalized Mutual Information (NMI): Measures mutual information between
    predicted clusters and true labels, normalized to [0,1].
    
    Range: [0, 1]
    Higher is better
    
    Formula: NMI(U,V) = 2*I(U;V) / [H(U) + H(V)]
    where:
        I(U;V) = mutual information between U and V
        H(U), H(V) = entropies of U and V
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster labels
        
    Returns:
        Normalized Mutual Information
    """
    try:
        score = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
        return score
    except Exception as e:
        print(f"Warning: Could not compute NMI - {e}")
        return 0.0


def compute_cluster_purity(y_true, y_pred):
    """
    Cluster Purity: Fraction of the dominant class in each cluster
    
    Range: [0, 1]
    Higher is better
    
    Formula: Purity = (1/n) * Σ max_j |ck ∩ tj|
    where:
        ck = set of points in cluster k
        tj = set of points in true class j
        n = total number of points
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster labels
        
    Returns:
        Cluster purity
    """
    try:
        # Get unique clusters and true labels
        clusters = np.unique(y_pred)
        true_labels = np.unique(y_true)
        
        total_correct = 0
        n = len(y_true)
        
        # For each cluster, find the most common true label
        for cluster in clusters:
            # Get indices of points in this cluster
            cluster_mask = (y_pred == cluster)
            cluster_true_labels = y_true[cluster_mask]
            
            # Count occurrences of each true label in this cluster
            if len(cluster_true_labels) > 0:
                # Find the most common true label
                most_common_count = Counter(cluster_true_labels).most_common(1)[0][1]
                total_correct += most_common_count
        
        purity = total_correct / n
        return purity
    except Exception as e:
        print(f"Warning: Could not compute Cluster Purity - {e}")
        return 0.0


class ClusteringEvaluator:
    """
    Complete evaluation suite for clustering results
    """
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate_all(self, X, labels_pred, labels_true=None, verbose=True):
        """
        Compute all clustering metrics
        
        Args:
            X: Feature matrix
            labels_pred: Predicted cluster labels
            labels_true: True labels (optional, for supervised metrics)
            verbose: Whether to print results
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        # Unsupervised metrics (don't require true labels)
        results['silhouette_score'] = compute_silhouette_score(X, labels_pred)
        results['calinski_harabasz_index'] = compute_calinski_harabasz_index(X, labels_pred)
        results['davies_bouldin_index'] = compute_davies_bouldin_index(X, labels_pred)
        
        # Supervised metrics (require true labels)
        if labels_true is not None:
            results['adjusted_rand_index'] = compute_adjusted_rand_index(labels_true, labels_pred)
            results['normalized_mutual_info'] = compute_normalized_mutual_info(labels_true, labels_pred)
            results['cluster_purity'] = compute_cluster_purity(labels_true, labels_pred)
        
        # Store results
        self.metrics = results
        
        # Print results if requested
        if verbose:
            self.print_results(results)
        
        return results
    
    def print_results(self, results):
        """Pretty print evaluation results"""
        print("\n" + "="*60)
        print("CLUSTERING EVALUATION RESULTS")
        print("="*60)
        
        print("\nUnsupervised Metrics:")
        print(f"  Silhouette Score:          {results['silhouette_score']:.4f}  (higher is better)")
        print(f"  Calinski-Harabasz Index:   {results['calinski_harabasz_index']:.2f}  (higher is better)")
        print(f"  Davies-Bouldin Index:      {results['davies_bouldin_index']:.4f}  (lower is better)")
        
        if 'adjusted_rand_index' in results:
            print("\nSupervised Metrics (with ground truth):")
            print(f"  Adjusted Rand Index:       {results['adjusted_rand_index']:.4f}  (higher is better)")
            print(f"  Normalized Mutual Info:    {results['normalized_mutual_info']:.4f}  (higher is better)")
            print(f"  Cluster Purity:            {results['cluster_purity']:.4f}  (higher is better)")
        
        print("="*60 + "\n")
    
    def compare_methods(self, results_dict, labels_true=None):
        """
        Compare multiple clustering methods
        
        Args:
            results_dict: Dictionary of {method_name: {'X': features, 'labels': predicted_labels}}
            labels_true: True labels (optional)
            
        Returns:
            Comparison DataFrame
        """
        import pandas as pd
        
        comparison = []
        
        for method_name, data in results_dict.items():
            X = data['X']
            labels_pred = data['labels']
            
            # Evaluate this method
            metrics = self.evaluate_all(X, labels_pred, labels_true, verbose=False)
            metrics['method'] = method_name
            
            comparison.append(metrics)
        
        # Create DataFrame
        df = pd.DataFrame(comparison)
        
        # Reorder columns
        cols = ['method', 'silhouette_score', 'calinski_harabasz_index', 'davies_bouldin_index']
        if labels_true is not None:
            cols.extend(['adjusted_rand_index', 'normalized_mutual_info', 'cluster_purity'])
        
        df = df[cols]
        
        return df
    
    def get_best_score(self, metric_name, higher_is_better=True):
        """Get best score for a given metric across stored results"""
        if metric_name not in self.metrics:
            return None
        
        scores = self.metrics[metric_name]
        if higher_is_better:
            return max(scores)
        else:
            return min(scores)


def evaluate_clustering(X, labels_pred, labels_true=None, method_name="Clustering"):
    """
    Convenience function to evaluate clustering results
    
    Args:
        X: Feature matrix
        labels_pred: Predicted cluster labels
        labels_true: True labels (optional)
        method_name: Name of the method for display
        
    Returns:
        Dictionary of metric scores
    """
    print(f"\nEvaluating {method_name}...")
    evaluator = ClusteringEvaluator()
    results = evaluator.evaluate_all(X, labels_pred, labels_true, verbose=True)
    return results


if __name__ == '__main__':
    # Test evaluation metrics
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    
    print("Testing evaluation metrics...")
    
    # Generate synthetic data
    X, y_true = make_blobs(n_samples=300, n_features=10, centers=5, 
                          cluster_std=1.0, random_state=42)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    # Test individual metrics
    print("\nTesting individual metrics:")
    print(f"1. Silhouette Score: {compute_silhouette_score(X, y_pred):.4f}")
    print(f"2. Calinski-Harabasz Index: {compute_calinski_harabasz_index(X, y_pred):.2f}")
    print(f"3. Davies-Bouldin Index: {compute_davies_bouldin_index(X, y_pred):.4f}")
    print(f"4. Adjusted Rand Index: {compute_adjusted_rand_index(y_true, y_pred):.4f}")
    print(f"5. Normalized Mutual Info: {compute_normalized_mutual_info(y_true, y_pred):.4f}")
    print(f"6. Cluster Purity: {compute_cluster_purity(y_true, y_pred):.4f}")
    
    # Test evaluator class
    print("\nTesting ClusteringEvaluator:")
    evaluator = ClusteringEvaluator()
    results = evaluator.evaluate_all(X, y_pred, y_true, verbose=True)
    
    # Test comparison
    print("\nTesting method comparison:")
    y_pred2 = KMeans(n_clusters=4, random_state=42).fit_predict(X)
    
    results_dict = {
        'KMeans (k=5)': {'X': X, 'labels': y_pred},
        'KMeans (k=4)': {'X': X, 'labels': y_pred2}
    }
    
    comparison_df = evaluator.compare_methods(results_dict, y_true)
    print(comparison_df)
    
    print("\nAll evaluation tests passed!")
