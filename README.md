# VAE for Hybrid Language Music Clustering

## Project Overview
This project implements a Variational Autoencoder (VAE) based unsupervised learning pipeline for clustering music tracks. The implementation includes:

- **Easy Task**: Basic VAE with K-Means clustering
- **Medium Task**: Convolutional VAE with multiple clustering algorithms
- **Hard Task**: Beta-VAE for disentangled representations with comprehensive evaluation

## Dataset
Using GTZAN Genre Collection with 1000 audio tracks across 10 genres:
- Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- Pre-extracted audio features (MFCCs, spectral features, chroma, etc.)

## Project Structure
```
425_project/
├── archive/                    # GTZAN dataset
│   └── Data/
│       ├── features_30_sec.csv
│       └── genres_original/
├── src/                        # Source code
│   ├── dataset.py             # Data loading and preprocessing
│   ├── vae.py                 # VAE architectures
│   ├── clustering.py          # Clustering algorithms
│   ├── evaluation.py          # Metrics computation
│   └── visualization.py       # Plotting functions
├── notebooks/                  # Jupyter notebooks
│   └── exploratory.ipynb      # Analysis and experiments
├── results/                    # Output files
│   ├── latent_visualization/
│   ├── clustering_metrics/
│   └── reconstructions/
└── requirements.txt
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
# Train basic VAE and perform clustering
python src/train.py --task easy

# Train convolutional VAE with multiple algorithms
python src/train.py --task medium

# Train Beta-VAE with comprehensive evaluation
python src/train.py --task hard
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/exploratory.ipynb
```

## Evaluation Metrics
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Cluster Purity

## Results
Results are saved in the `results/` directory:
- Clustering metrics: CSV files with quantitative results
- Latent visualizations: t-SNE/UMAP plots
- Reconstructions: Sample reconstructions from VAE

## Author
Dipto Sumit

## Course
Neural Network CSE425

## Submission Date
January 10th, 2026
