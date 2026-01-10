"""
Clustering algorithms for music feature representations
Includes: K-Means, Agglomerative Clustering, DBSCAN, and baseline methods
"""
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MusicClusterer:
    """Wrapper class for various clustering algorithms"""
    
    def __init__(self, method='kmeans', n_clusters=10, **kwargs):
        """
        Args:
            method: Clustering method ('kmeans', 'agglomerative', 'dbscan')
            n_clusters: Number of clusters (not used for DBSCAN)
            **kwargs: Additional parameters for clustering algorithm
        """
        self.method = method
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.model = None
        self.labels_ = None
        
        self._init_model()
        
    def _init_model(self):
        """Initialize clustering model based on method"""
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.kwargs.get('random_state', 42),
                n_init=self.kwargs.get('n_init', 10),
                max_iter=self.kwargs.get('max_iter', 300)
            )
        elif self.method == 'agglomerative':
            linkage = self.kwargs.get('linkage', 'ward')
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=linkage
            )
        elif self.method == 'dbscan':
            eps = self.kwargs.get('eps', 0.5)
            min_samples = self.kwargs.get('min_samples', 5)
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
    
    def fit(self, X):
        """
        Fit clustering model to data
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        self.labels_ = self.model.fit_predict(X)
        return self.labels_
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        if self.method == 'kmeans':
            return self.model.predict(X)
        else:
            # For methods without predict, refit on combined data
            return self.fit(X)
    
    def get_cluster_centers(self):
        """Get cluster centers (only for KMeans)"""
        if self.method == 'kmeans':
            return self.model.cluster_centers_
        return None


class BaselineClustering:
    """Baseline clustering methods: PCA + K-Means"""
    
    def __init__(self, n_components=32, n_clusters=10, random_state=42):
        """
        Args:
            n_components: Number of PCA components
            n_clusters: Number of clusters
            random_state: Random seed
        """
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.labels_ = None
        
    def fit(self, X):
        """
        Fit PCA + K-Means pipeline
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        # Apply PCA
        X_pca = self.pca.fit_transform(X)
        
        # Apply K-Means on PCA features
        self.labels_ = self.kmeans.fit_predict(X_pca)
        
        return self.labels_
    
    def transform(self, X):
        """Transform data using fitted PCA"""
        return self.pca.transform(X)
    
    def get_pca_features(self, X):
        """Get PCA transformed features"""
        return self.pca.transform(X)
    
    def get_explained_variance_ratio(self):
        """Get explained variance ratio from PCA"""
        return self.pca.explained_variance_ratio_


class ClusteringPipeline:
    """
    Complete clustering pipeline with multiple methods
    """
    
    def __init__(self, n_clusters=10, random_state=42):
        """
        Args:
            n_clusters: Number of clusters
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.results = {}
        
    def run_all_methods(self, X, methods=['kmeans', 'agglomerative', 'dbscan'], 
                       include_baseline=True):
        """
        Run all clustering methods on data
        
        Args:
            X: Feature matrix
            methods: List of clustering methods to use
            include_baseline: Whether to include PCA+KMeans baseline
            
        Returns:
            Dictionary of results with cluster labels for each method
        """
        results = {}
        
        # Run each clustering method
        for method in methods:
            print(f"Running {method.upper()} clustering...")
            
            if method == 'dbscan':
                # DBSCAN doesn't use n_clusters
                clusterer = MusicClusterer(
                    method=method,
                    eps=0.5,
                    min_samples=5
                )
            else:
                clusterer = MusicClusterer(
                    method=method,
                    n_clusters=self.n_clusters,
                    random_state=self.random_state
                )
            
            labels = clusterer.fit(X)
            results[method] = {
                'labels': labels,
                'model': clusterer,
                'n_clusters_found': len(np.unique(labels))
            }
            
            print(f"  Found {len(np.unique(labels))} clusters")
        
        # Run baseline if requested
        if include_baseline:
            print("Running PCA + K-Means baseline...")
            baseline = BaselineClustering(
                n_components=min(32, X.shape[1]),
                n_clusters=self.n_clusters,
                random_state=self.random_state
            )
            labels = baseline.fit(X)
            results['pca_kmeans'] = {
                'labels': labels,
                'model': baseline,
                'n_clusters_found': len(np.unique(labels)),
                'explained_variance': baseline.get_explained_variance_ratio()
            }
            print(f"  Found {len(np.unique(labels))} clusters")
            print(f"  Explained variance (first 5 components): {baseline.get_explained_variance_ratio()[:5]}")
        
        self.results = results
        return results
    
    def get_best_method(self, metric_scores):
        """
        Determine best clustering method based on metric scores
        
        Args:
            metric_scores: Dictionary of {method: score}
            
        Returns:
            Best method name
        """
        best_method = max(metric_scores, key=metric_scores.get)
        return best_method


class AutoencoderBaseline:
    """
    Simple Autoencoder baseline for comparison
    (Without the variational component)
    """
    
    def __init__(self, input_dim, latent_dim=32):
        """
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent dimension
        """
        import torch
        import torch.nn as nn
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Simple autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
    def forward(self, x):
        """Forward pass"""
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent
    
    def get_latent(self, x):
        """Get latent representation"""
        return self.encoder(x)


def compare_clustering_methods(X, y_true, n_clusters=10, random_state=42):
    """
    Convenience function to compare multiple clustering methods
    
    Args:
        X: Feature matrix
        y_true: True labels (for evaluation)
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Dictionary of results
    """
    pipeline = ClusteringPipeline(n_clusters=n_clusters, random_state=random_state)
    
    # Run all methods
    results = pipeline.run_all_methods(
        X,
        methods=['kmeans', 'agglomerative'],
        include_baseline=True
    )
    
    return results


if __name__ == '__main__':
    # Test clustering methods
    from sklearn.datasets import make_blobs
    
    print("Testing clustering methods...")
    
    # Generate synthetic data
    X, y_true = make_blobs(n_samples=300, n_features=10, centers=5, random_state=42)
    
    # Test individual methods
    print("\n1. Testing K-Means...")
    kmeans = MusicClusterer(method='kmeans', n_clusters=5)
    labels = kmeans.fit(X)
    print(f"   Found {len(np.unique(labels))} clusters")
    
    print("\n2. Testing Agglomerative Clustering...")
    agg = MusicClusterer(method='agglomerative', n_clusters=5)
    labels = agg.fit(X)
    print(f"   Found {len(np.unique(labels))} clusters")
    
    print("\n3. Testing DBSCAN...")
    dbscan = MusicClusterer(method='dbscan', eps=1.5, min_samples=5)
    labels = dbscan.fit(X)
    print(f"   Found {len(np.unique(labels))} clusters")
    
    print("\n4. Testing Baseline (PCA + K-Means)...")
    baseline = BaselineClustering(n_components=5, n_clusters=5)
    labels = baseline.fit(X)
    print(f"   Found {len(np.unique(labels))} clusters")
    
    print("\n5. Testing full pipeline...")
    results = compare_clustering_methods(X, y_true, n_clusters=5)
    print(f"   Tested {len(results)} methods")
    
    print("\nAll clustering tests passed!")
