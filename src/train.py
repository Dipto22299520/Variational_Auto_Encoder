"""
Main training script for VAE-based music clustering
Supports Easy, Medium, and Hard tasks
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import load_gtzan_data
from vae import SimpleVAE, ConvVAE, BetaVAE, ConditionalVAE, VAETrainer
from clustering import MusicClusterer, BaselineClustering, ClusteringPipeline
from evaluation import ClusteringEvaluator
from visualization import create_comprehensive_report, plot_training_loss


def train_easy_task(data_loader, device='cpu', save_dir='results'):
    """
    Easy Task: Basic VAE + K-Means clustering
    
    - Implement basic VAE for feature extraction
    - Perform clustering using K-Means
    - Visualize with t-SNE/UMAP
    - Compare with PCA + K-Means baseline
    """
    print("\n" + "="*70)
    print("EASY TASK: Basic VAE + K-Means Clustering")
    print("="*70 + "\n")
    
    # Load data
    X, y_true = data_loader.get_full_dataset(scale=True)
    input_dim = data_loader.get_num_features()
    n_clusters = data_loader.get_num_classes()
    
    # Create dataloaders
    train_loader = data_loader.get_dataloader(X, batch_size=32, shuffle=True)
    test_loader = data_loader.get_dataloader(X, batch_size=32, shuffle=False)
    
    # 1. Train Basic VAE
    print("1. Training Simple VAE...")
    vae = SimpleVAE(input_dim=input_dim, hidden_dims=[256, 128], latent_dim=32)
    trainer = VAETrainer(vae, device=device, learning_rate=1e-3)
    trainer.train(train_loader, num_epochs=100, beta=1.0, verbose=True)
    
    # Plot training loss
    plot_training_loss(trainer.train_losses, 
                      title="Simple VAE Training Loss",
                      save_path=os.path.join(save_dir, 'easy_training_loss.png'))
    
    # 2. Extract latent representations
    print("\n2. Extracting latent representations...")
    latent_vectors, _ = trainer.get_latent_representations(test_loader)
    
    # 3. Perform K-Means clustering on latent features
    print("\n3. Performing K-Means clustering...")
    kmeans = MusicClusterer(method='kmeans', n_clusters=n_clusters, random_state=42)
    labels_pred = kmeans.fit(latent_vectors)
    
    # 4. Baseline: PCA + K-Means
    print("\n4. Running PCA + K-Means baseline...")
    baseline = BaselineClustering(n_components=32, n_clusters=n_clusters)
    labels_baseline = baseline.fit(X)
    
    # 5. Evaluation
    print("\n5. Evaluating clustering quality...")
    evaluator = ClusteringEvaluator()
    
    # VAE + K-Means results
    print("\n--- VAE + K-Means Results ---")
    vae_results = evaluator.evaluate_all(latent_vectors, labels_pred, y_true, verbose=True)
    
    # Baseline results
    print("\n--- PCA + K-Means Baseline Results ---")
    baseline_results = evaluator.evaluate_all(X, labels_baseline, y_true, verbose=True)
    
    # Create comparison DataFrame
    comparison_data = []
    for metric in ['silhouette_score', 'calinski_harabasz_index', 'davies_bouldin_index',
                   'adjusted_rand_index', 'normalized_mutual_info', 'cluster_purity']:
        comparison_data.append({
            'metric': metric,
            'VAE + K-Means': vae_results.get(metric, np.nan),
            'PCA + K-Means': baseline_results.get(metric, np.nan)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n--- Comparison Table ---")
    print(comparison_df.to_string(index=False))
    
    # Save results
    comparison_df.to_csv(os.path.join(save_dir, 'clustering_metrics', 'easy_task_comparison.csv'), index=False)
    
    # 6. Visualization
    print("\n6. Generating visualizations...")
    label_names = data_loader.label_names
    
    # Get reconstructions
    vae.eval()
    with torch.no_grad():
        sample_data = torch.FloatTensor(X[:5]).to(device)
        recon_data, _, _ = vae(sample_data)
        recon_data = recon_data.cpu().numpy()
    
    create_comprehensive_report(
        latent_vectors=latent_vectors,
        labels_pred=labels_pred,
        labels_true=y_true,
        original=X[:5],
        reconstructed=recon_data,
        label_names=label_names,
        save_dir=os.path.join(save_dir, 'latent_visualization', 'easy_task')
    )
    
    # Save model
    torch.save(vae.state_dict(), os.path.join(save_dir, 'easy_vae_model.pth'))
    print(f"\nModel saved to {os.path.join(save_dir, 'easy_vae_model.pth')}")
    
    print("\n" + "="*70)
    print("EASY TASK COMPLETE!")
    print("="*70 + "\n")
    
    return vae, latent_vectors, labels_pred, vae_results


def train_medium_task(data_loader, device='cpu', save_dir='results'):
    """
    Medium Task: Convolutional VAE + Multiple clustering algorithms
    
    - Enhance VAE with convolutional architecture
    - Experiment with K-Means, Agglomerative, DBSCAN
    - Comprehensive evaluation with multiple metrics
    """
    print("\n" + "="*70)
    print("MEDIUM TASK: Convolutional VAE + Multiple Clustering Algorithms")
    print("="*70 + "\n")
    
    # Load data
    X, y_true = data_loader.get_full_dataset(scale=True)
    input_dim = data_loader.get_num_features()
    n_clusters = data_loader.get_num_classes()
    
    # Create dataloaders
    train_loader = data_loader.get_dataloader(X, batch_size=32, shuffle=True)
    test_loader = data_loader.get_dataloader(X, batch_size=32, shuffle=False)
    
    # 1. Train Convolutional VAE
    print("1. Training Convolutional VAE...")
    conv_vae = ConvVAE(input_dim=input_dim, latent_dim=64, channels=[32, 64, 128])
    trainer = VAETrainer(conv_vae, device=device, learning_rate=1e-3)
    trainer.train(train_loader, num_epochs=100, beta=1.0, verbose=True)
    
    # Plot training loss
    plot_training_loss(trainer.train_losses,
                      title="Convolutional VAE Training Loss",
                      save_path=os.path.join(save_dir, 'medium_training_loss.png'))
    
    # 2. Extract latent representations
    print("\n2. Extracting latent representations...")
    latent_vectors, _ = trainer.get_latent_representations(test_loader)
    
    # 3. Run multiple clustering algorithms
    print("\n3. Running multiple clustering algorithms...")
    pipeline = ClusteringPipeline(n_clusters=n_clusters, random_state=42)
    clustering_results = pipeline.run_all_methods(
        latent_vectors,
        methods=['kmeans', 'agglomerative'],
        include_baseline=True
    )
    
    # 4. Evaluate all methods
    print("\n4. Evaluating all clustering methods...")
    evaluator = ClusteringEvaluator()
    
    all_results = []
    for method_name, result in clustering_results.items():
        print(f"\n--- {method_name.upper()} Results ---")
        labels_pred = result['labels']
        metrics = evaluator.evaluate_all(latent_vectors, labels_pred, y_true, verbose=True)
        metrics['method'] = method_name
        all_results.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results)
    cols = ['method', 'silhouette_score', 'calinski_harabasz_index', 'davies_bouldin_index',
            'adjusted_rand_index', 'normalized_mutual_info', 'cluster_purity']
    comparison_df = comparison_df[cols]
    
    print("\n--- Method Comparison Table ---")
    print(comparison_df.to_string(index=False))
    
    # Save results
    comparison_df.to_csv(os.path.join(save_dir, 'clustering_metrics', 'medium_task_comparison.csv'), index=False)
    
    # 5. Visualization
    print("\n5. Generating visualizations...")
    label_names = data_loader.label_names
    
    # Use best method for visualization (highest Silhouette Score)
    best_method = comparison_df.loc[comparison_df['silhouette_score'].idxmax(), 'method']
    best_labels = clustering_results[best_method]['labels']
    
    print(f"\nBest method: {best_method.upper()}")
    
    # Get reconstructions
    conv_vae.eval()
    with torch.no_grad():
        sample_data = torch.FloatTensor(X[:5]).to(device)
        recon_data, _, _ = conv_vae(sample_data)
        recon_data = recon_data.cpu().numpy()
    
    create_comprehensive_report(
        latent_vectors=latent_vectors,
        labels_pred=best_labels,
        labels_true=y_true,
        original=X[:5],
        reconstructed=recon_data,
        comparison_df=comparison_df,
        label_names=label_names,
        save_dir=os.path.join(save_dir, 'latent_visualization', 'medium_task')
    )
    
    # Save model
    torch.save(conv_vae.state_dict(), os.path.join(save_dir, 'medium_convvae_model.pth'))
    print(f"\nModel saved to {os.path.join(save_dir, 'medium_convvae_model.pth')}")
    
    print("\n" + "="*70)
    print("MEDIUM TASK COMPLETE!")
    print("="*70 + "\n")
    
    return conv_vae, latent_vectors, best_labels, comparison_df


def train_hard_task(data_loader, device='cpu', save_dir='results'):
    """
    Hard Task: Beta-VAE for disentangled representations
    
    - Implement Beta-VAE with beta > 1
    - Multi-modal clustering with comprehensive evaluation
    - All evaluation metrics including NMI, ARI, Purity
    """
    print("\n" + "="*70)
    print("HARD TASK: Beta-VAE for Disentangled Representations")
    print("="*70 + "\n")
    
    # Load data
    X, y_true = data_loader.get_full_dataset(scale=True)
    input_dim = data_loader.get_num_features()
    n_clusters = data_loader.get_num_classes()
    
    # Create dataloaders
    train_loader = data_loader.get_dataloader(X, y_true, batch_size=32, shuffle=True)
    test_loader = data_loader.get_dataloader(X, y_true, batch_size=32, shuffle=False)
    
    # 1. Train Beta-VAE with different beta values
    print("1. Training Beta-VAE with beta=4.0...")
    beta_vae = BetaVAE(input_dim=input_dim, hidden_dims=[256, 128], latent_dim=64, beta=4.0)
    trainer = VAETrainer(beta_vae, device=device, learning_rate=1e-3)
    trainer.train(train_loader, num_epochs=100, beta=4.0, verbose=True)
    
    # Plot training loss
    plot_training_loss(trainer.train_losses,
                      title="Beta-VAE Training Loss (β=4.0)",
                      save_path=os.path.join(save_dir, 'hard_training_loss.png'))
    
    # 2. Extract latent representations
    print("\n2. Extracting latent representations...")
    latent_vectors, _ = trainer.get_latent_representations(test_loader)
    
    # 3. Run comprehensive clustering pipeline
    print("\n3. Running comprehensive clustering pipeline...")
    pipeline = ClusteringPipeline(n_clusters=n_clusters, random_state=42)
    clustering_results = pipeline.run_all_methods(
        latent_vectors,
        methods=['kmeans', 'agglomerative'],
        include_baseline=True
    )
    
    # 4. Also compare with simple VAE and Conv VAE
    print("\n4. Training comparison models...")
    
    # Simple VAE
    print("   - Training Simple VAE...")
    simple_vae = SimpleVAE(input_dim=input_dim, latent_dim=64)
    simple_trainer = VAETrainer(simple_vae, device=device, learning_rate=1e-3)
    simple_trainer.train(train_loader, num_epochs=100, beta=1.0, verbose=False)
    latent_simple, _ = simple_trainer.get_latent_representations(test_loader)
    
    # Conv VAE
    print("   - Training Convolutional VAE...")
    conv_vae = ConvVAE(input_dim=input_dim, latent_dim=64)
    conv_trainer = VAETrainer(conv_vae, device=device, learning_rate=1e-3)
    conv_trainer.train(train_loader, num_epochs=100, beta=1.0, verbose=False)
    latent_conv, _ = conv_trainer.get_latent_representations(test_loader)
    
    # Cluster with all methods
    print("\n5. Clustering with all VAE variants...")
    all_results = []
    
    # Beta-VAE results
    for method_name, result in clustering_results.items():
        labels_pred = result['labels']
        evaluator = ClusteringEvaluator()
        metrics = evaluator.evaluate_all(latent_vectors, labels_pred, y_true, verbose=False)
        metrics['vae_type'] = 'Beta-VAE'
        metrics['method'] = method_name
        all_results.append(metrics)
    
    # Simple VAE results
    kmeans_simple = MusicClusterer(method='kmeans', n_clusters=n_clusters)
    labels_simple = kmeans_simple.fit(latent_simple)
    evaluator = ClusteringEvaluator()
    metrics = evaluator.evaluate_all(latent_simple, labels_simple, y_true, verbose=False)
    metrics['vae_type'] = 'Simple VAE'
    metrics['method'] = 'kmeans'
    all_results.append(metrics)
    
    # Conv VAE results
    kmeans_conv = MusicClusterer(method='kmeans', n_clusters=n_clusters)
    labels_conv = kmeans_conv.fit(latent_conv)
    evaluator = ClusteringEvaluator()
    metrics = evaluator.evaluate_all(latent_conv, labels_conv, y_true, verbose=False)
    metrics['vae_type'] = 'Conv VAE'
    metrics['method'] = 'kmeans'
    all_results.append(metrics)
    
    # PCA baseline
    n_pca_components = min(64, input_dim)  # Ensure PCA components <= features
    baseline = BaselineClustering(n_components=n_pca_components, n_clusters=n_clusters)
    labels_baseline = baseline.fit(X)
    evaluator = ClusteringEvaluator()
    metrics = evaluator.evaluate_all(X, labels_baseline, y_true, verbose=False)
    metrics['vae_type'] = 'Baseline'
    metrics['method'] = 'pca_kmeans'
    all_results.append(metrics)
    
    # Create comprehensive comparison DataFrame
    comparison_df = pd.DataFrame(all_results)
    cols = ['vae_type', 'method', 'silhouette_score', 'calinski_harabasz_index', 
            'davies_bouldin_index', 'adjusted_rand_index', 'normalized_mutual_info', 'cluster_purity']
    comparison_df = comparison_df[cols]
    
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70 + "\n")
    
    # Save results
    comparison_df.to_csv(os.path.join(save_dir, 'clustering_metrics', 'hard_task_comparison.csv'), index=False)
    
    # 6. Visualization
    print("\n6. Generating comprehensive visualizations...")
    label_names = data_loader.label_names
    
    # Use Beta-VAE + best method for visualization
    beta_results = comparison_df[comparison_df['vae_type'] == 'Beta-VAE']
    best_method = beta_results.loc[beta_results['silhouette_score'].idxmax(), 'method']
    best_labels = clustering_results[best_method]['labels']
    
    print(f"\nBest Beta-VAE method: {best_method.upper()}")
    
    # Get reconstructions
    beta_vae.eval()
    with torch.no_grad():
        sample_data = torch.FloatTensor(X[:5]).to(device)
        recon_data, _, _ = beta_vae(sample_data)
        recon_data = recon_data.cpu().numpy()
    
    create_comprehensive_report(
        latent_vectors=latent_vectors,
        labels_pred=best_labels,
        labels_true=y_true,
        original=X[:5],
        reconstructed=recon_data,
        comparison_df=comparison_df,
        label_names=label_names,
        save_dir=os.path.join(save_dir, 'latent_visualization', 'hard_task')
    )
    
    # Save model
    torch.save(beta_vae.state_dict(), os.path.join(save_dir, 'hard_betavae_model.pth'))
    print(f"\nModel saved to {os.path.join(save_dir, 'hard_betavae_model.pth')}")
    
    print("\n" + "="*70)
    print("HARD TASK COMPLETE!")
    print("="*70 + "\n")
    
    return beta_vae, latent_vectors, best_labels, comparison_df


def main():
    parser = argparse.ArgumentParser(description='VAE-based Music Clustering')
    parser.add_argument('--task', type=str, default='all', 
                       choices=['easy', 'medium', 'hard', 'all'],
                       help='Task to run (default: all)')
    parser.add_argument('--data_dir', type=str, default='archive/Data',
                       help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Create save directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'clustering_metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'latent_visualization'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'reconstructions'), exist_ok=True)
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print("\n" + "="*70)
    print("VAE-BASED MUSIC CLUSTERING PROJECT")
    print("="*70)
    print(f"\nTask: {args.task.upper()}")
    print(f"Device: {args.device.upper()}")
    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {args.save_dir}")
    print("="*70 + "\n")
    
    # Load data
    print("Loading GTZAN dataset...")
    data_loader = load_gtzan_data(data_dir=args.data_dir, use_30sec=True)
    
    # Run tasks
    if args.task in ['easy', 'all']:
        train_easy_task(data_loader, device=args.device, save_dir=args.save_dir)
    
    if args.task in ['medium', 'all']:
        train_medium_task(data_loader, device=args.device, save_dir=args.save_dir)
    
    if args.task in ['hard', 'all']:
        train_hard_task(data_loader, device=args.device, save_dir=args.save_dir)
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {args.save_dir}")
    print("Check the following directories:")
    print(f"  - Metrics: {os.path.join(args.save_dir, 'clustering_metrics')}")
    print(f"  - Visualizations: {os.path.join(args.save_dir, 'latent_visualization')}")
    print(f"  - Models: {args.save_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
