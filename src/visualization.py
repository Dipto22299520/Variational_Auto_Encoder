"""
Visualization tools for clustering results and latent space analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import umap
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_latent_space_tsne(latent_vectors, labels=None, perplexity=30, 
                            title="t-SNE Visualization of Latent Space",
                            save_path=None, label_names=None):
    """
    Visualize latent space using t-SNE
    
    Args:
        latent_vectors: Latent representations (n_samples, latent_dim)
        labels: Cluster or true labels (optional)
        perplexity: t-SNE perplexity parameter
        title: Plot title
        save_path: Path to save figure (optional)
        label_names: Names for labels (optional)
    """
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                max_iter=1000)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(12, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names[label] if label_names is not None else f"Cluster {label}"
            plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                       c=[colors[i]], label=label_name, alpha=0.6, s=50)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  fontsize=10, frameon=True)
    else:
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=50)
    
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    
    plt.show()
    
    return latent_2d


def plot_latent_space_umap(latent_vectors, labels=None, n_neighbors=15,
                           title="UMAP Visualization of Latent Space",
                           save_path=None, label_names=None):
    """
    Visualize latent space using UMAP
    
    Args:
        latent_vectors: Latent representations
        labels: Cluster or true labels (optional)
        n_neighbors: UMAP n_neighbors parameter
        title: Plot title
        save_path: Path to save figure
        label_names: Names for labels
    """
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                       random_state=42, verbose=False)
    latent_2d = reducer.fit_transform(latent_vectors)
    
    plt.figure(figsize=(12, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names[label] if label_names is not None else f"Cluster {label}"
            plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                       c=[colors[i]], label=label_name, alpha=0.6, s=50)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize=10, frameon=True)
    else:
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=50)
    
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved UMAP plot to {save_path}")
    
    plt.show()
    
    return latent_2d


def plot_cluster_distribution(labels_pred, labels_true=None, 
                              title="Cluster Distribution",
                              save_path=None, label_names=None):
    """
    Plot distribution of samples across clusters
    
    Args:
        labels_pred: Predicted cluster labels
        labels_true: True labels (optional, for stacked bar plot)
        title: Plot title
        save_path: Path to save figure
        label_names: Names for true labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Cluster sizes
    unique_clusters, counts = np.unique(labels_pred, return_counts=True)
    axes[0].bar(unique_clusters, counts, color='steelblue', alpha=0.7)
    axes[0].set_xlabel("Cluster ID", fontsize=12)
    axes[0].set_ylabel("Number of Samples", fontsize=12)
    axes[0].set_title("Cluster Sizes", fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (cluster, count) in enumerate(zip(unique_clusters, counts)):
        axes[0].text(cluster, count + max(counts)*0.01, str(count), 
                    ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Confusion matrix if true labels available
    if labels_true is not None:
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(labels_true, labels_pred)
        
        # Create heatmap
        im = axes[1].imshow(cm, cmap='Blues', aspect='auto')
        axes[1].set_xlabel("Predicted Cluster", fontsize=12)
        axes[1].set_ylabel("True Genre", fontsize=12)
        axes[1].set_title("Cluster-Genre Confusion Matrix", fontsize=13, fontweight='bold')
        
        # Set ticks
        axes[1].set_xticks(np.arange(len(unique_clusters)))
        axes[1].set_yticks(np.arange(len(np.unique(labels_true))))
        
        if label_names is not None:
            axes[1].set_yticklabels(label_names)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1])
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = axes[1].text(j, i, cm[i, j],
                                  ha="center", va="center", 
                                  color="white" if cm[i, j] > cm.max()/2 else "black",
                                  fontsize=8)
    else:
        # Just show cluster size pie chart
        axes[1].pie(counts, labels=[f"Cluster {c}" for c in unique_clusters],
                   autopct='%1.1f%%', startangle=90)
        axes[1].set_title("Cluster Proportions", fontsize=13, fontweight='bold')
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cluster distribution plot to {save_path}")
    
    plt.show()


def plot_reconstruction_comparison(original, reconstructed, n_samples=5,
                                   title="Reconstruction Examples",
                                   save_path=None):
    """
    Plot comparison of original and reconstructed features
    
    Args:
        original: Original features
        reconstructed: Reconstructed features
        n_samples: Number of samples to plot
        title: Plot title
        save_path: Path to save figure
    """
    n_samples = min(n_samples, len(original))
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original
        axes[i, 0].plot(original[i], 'b-', linewidth=1, alpha=0.7)
        axes[i, 0].set_title(f"Sample {i+1}: Original", fontsize=11)
        axes[i, 0].set_ylabel("Feature Value", fontsize=10)
        axes[i, 0].grid(alpha=0.3)
        
        # Reconstructed
        axes[i, 1].plot(reconstructed[i], 'r-', linewidth=1, alpha=0.7)
        axes[i, 1].set_title(f"Sample {i+1}: Reconstructed", fontsize=11)
        axes[i, 1].set_ylabel("Feature Value", fontsize=10)
        axes[i, 1].grid(alpha=0.3)
        
        if i == n_samples - 1:
            axes[i, 0].set_xlabel("Feature Index", fontsize=10)
            axes[i, 1].set_xlabel("Feature Index", fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reconstruction comparison to {save_path}")
    
    plt.show()


def plot_training_loss(losses, title="VAE Training Loss", save_path=None):
    """
    Plot training loss over epochs
    
    Args:
        losses: List of loss values
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training loss plot to {save_path}")
    
    plt.show()


def plot_metric_comparison(comparison_df, metric_name, higher_is_better=True,
                           title=None, save_path=None):
    """
    Plot comparison of a metric across different methods
    
    Args:
        comparison_df: DataFrame with method comparison results
        metric_name: Name of metric to plot
        higher_is_better: Whether higher values are better
        title: Plot title
        save_path: Path to save figure
    """
    if title is None:
        title = f"{metric_name.replace('_', ' ').title()} Comparison"
    
    plt.figure(figsize=(12, 6))
    
    methods = comparison_df['method'].values
    scores = comparison_df[metric_name].values
    
    # Create color based on performance
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(methods)))
    if not higher_is_better:
        colors = colors[::-1]
    
    bars = plt.bar(range(len(methods)), scores, color=colors, alpha=0.7)
    
    # Highlight best method
    best_idx = np.argmax(scores) if higher_is_better else np.argmin(scores)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(3)
    
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (score, bar) in enumerate(zip(scores, bars)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metric comparison plot to {save_path}")
    
    plt.show()


def plot_latent_dimensions(latent_vectors, labels=None, max_dims=10,
                           title="Latent Dimension Analysis", save_path=None):
    """
    Visualize individual latent dimensions
    
    Args:
        latent_vectors: Latent representations
        labels: Labels for coloring (optional)
        max_dims: Maximum number of dimensions to plot
        title: Plot title
        save_path: Path to save figure
    """
    n_dims = min(max_dims, latent_vectors.shape[1])
    
    fig, axes = plt.subplots(2, (n_dims + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(n_dims):
        if labels is not None:
            for label in np.unique(labels):
                mask = labels == label
                axes[i].hist(latent_vectors[mask, i], bins=30, alpha=0.5, 
                           label=f"Class {label}")
        else:
            axes[i].hist(latent_vectors[:, i], bins=30, alpha=0.7, color='steelblue')
        
        axes[i].set_xlabel(f"Latent Dim {i+1}", fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)
        axes[i].grid(alpha=0.3)
        
        if i == 0 and labels is not None:
            axes[i].legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved latent dimension analysis to {save_path}")
    
    plt.show()


def create_comprehensive_report(latent_vectors, labels_pred, labels_true=None,
                               original=None, reconstructed=None,
                               comparison_df=None, label_names=None,
                               save_dir=None):
    """
    Create comprehensive visualization report
    
    Args:
        latent_vectors: Latent representations
        labels_pred: Predicted cluster labels
        labels_true: True labels (optional)
        original: Original features (optional)
        reconstructed: Reconstructed features (optional)
        comparison_df: Method comparison DataFrame (optional)
        label_names: Names for labels
        save_dir: Directory to save figures
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")
    print("="*60 + "\n")
    
    # 1. Latent space visualization (t-SNE)
    print("1. Creating t-SNE visualization...")
    tsne_path = os.path.join(save_dir, "latent_tsne.png") if save_dir else None
    plot_latent_space_tsne(latent_vectors, labels_pred, 
                          title="t-SNE: Predicted Clusters",
                          save_path=tsne_path, label_names=label_names)
    
    if labels_true is not None:
        tsne_true_path = os.path.join(save_dir, "latent_tsne_true.png") if save_dir else None
        plot_latent_space_tsne(latent_vectors, labels_true,
                              title="t-SNE: True Genres",
                              save_path=tsne_true_path, label_names=label_names)
    
    # 2. Latent space visualization (UMAP)
    print("\n2. Creating UMAP visualization...")
    umap_path = os.path.join(save_dir, "latent_umap.png") if save_dir else None
    plot_latent_space_umap(latent_vectors, labels_pred,
                          title="UMAP: Predicted Clusters",
                          save_path=umap_path, label_names=label_names)
    
    # 3. Cluster distribution
    print("\n3. Creating cluster distribution plots...")
    dist_path = os.path.join(save_dir, "cluster_distribution.png") if save_dir else None
    plot_cluster_distribution(labels_pred, labels_true,
                             title="Cluster Analysis",
                             save_path=dist_path, label_names=label_names)
    
    # 4. Reconstruction examples
    if original is not None and reconstructed is not None:
        print("\n4. Creating reconstruction examples...")
        recon_path = os.path.join(save_dir, "reconstructions.png") if save_dir else None
        plot_reconstruction_comparison(original, reconstructed, n_samples=5,
                                     save_path=recon_path)
    
    # 5. Latent dimension analysis
    print("\n5. Creating latent dimension analysis...")
    latent_path = os.path.join(save_dir, "latent_dimensions.png") if save_dir else None
    plot_latent_dimensions(latent_vectors, labels_true,
                          save_path=latent_path)
    
    # 6. Method comparison
    if comparison_df is not None:
        print("\n6. Creating method comparison plots...")
        for metric in ['silhouette_score', 'calinski_harabasz_index', 'davies_bouldin_index']:
            if metric in comparison_df.columns:
                comp_path = os.path.join(save_dir, f"comparison_{metric}.png") if save_dir else None
                higher_better = metric != 'davies_bouldin_index'
                plot_metric_comparison(comparison_df, metric, 
                                     higher_is_better=higher_better,
                                     save_path=comp_path)
    
    print("\n" + "="*60)
    print("VISUALIZATION REPORT COMPLETE")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Test visualization functions
    from sklearn.datasets import make_blobs
    
    print("Testing visualization functions...")
    
    # Generate synthetic data
    X, y = make_blobs(n_samples=300, n_features=50, centers=5, random_state=42)
    latent = X[:, :10]  # Simulate latent space
    
    # Test t-SNE
    print("\n1. Testing t-SNE visualization...")
    plot_latent_space_tsne(latent, y, title="Test t-SNE")
    
    # Test UMAP
    print("\n2. Testing UMAP visualization...")
    plot_latent_space_umap(latent, y, title="Test UMAP")
    
    # Test cluster distribution
    print("\n3. Testing cluster distribution...")
    plot_cluster_distribution(y, y, title="Test Distribution")
    
    print("\nAll visualization tests passed!")
