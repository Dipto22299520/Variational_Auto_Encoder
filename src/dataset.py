"""
Dataset loading and preprocessing for music clustering
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os


class MusicDataset(Dataset):
    """PyTorch Dataset for music features"""
    
    def __init__(self, features, labels=None, transform=None):
        """
        Args:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,) - optional
            transform: optional transform to apply
        """
        self.features = torch.FloatTensor(features)
        self.labels = labels
        if labels is not None:
            self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        sample = self.features[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample


class MusicDataLoader:
    """Handles loading and preprocessing of GTZAN dataset"""
    
    def __init__(self, csv_path, test_size=0.2, random_state=42):
        """
        Args:
            csv_path: Path to features CSV file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self._preprocess()
        
    def _preprocess(self):
        """Extract features and labels, handle missing values"""
        # Drop filename column
        if 'filename' in self.df.columns:
            self.df = self.df.drop('filename', axis=1)
        
        # Handle 'length' column if present
        if 'length' in self.df.columns:
            self.df = self.df.drop('length', axis=1)
            
        # Extract labels
        self.labels = self.df['label'].values
        self.label_names = np.unique(self.labels)
        
        # Encode labels as integers
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        # Extract features (all columns except label)
        self.feature_columns = [col for col in self.df.columns if col != 'label']
        self.features = self.df[self.feature_columns].values
        
        # Handle any NaN or inf values
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Loaded {len(self.features)} samples with {len(self.feature_columns)} features")
        print(f"Genres: {self.label_names}")
        print(f"Samples per genre: {len(self.features) // len(self.label_names)}")
        
    def get_train_test_split(self, scale=True):
        """
        Split data into train and test sets
        
        Args:
            scale: Whether to apply standard scaling
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, 
            self.encoded_labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.encoded_labels
        )
        
        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
        return X_train, X_test, y_train, y_test
    
    def get_full_dataset(self, scale=True):
        """
        Get full dataset (useful for unsupervised learning)
        
        Args:
            scale: Whether to apply standard scaling
            
        Returns:
            features, labels
        """
        features = self.features.copy()
        
        if scale:
            features = self.scaler.fit_transform(features)
            
        return features, self.encoded_labels
    
    def get_dataloader(self, features, labels=None, batch_size=32, shuffle=True):
        """
        Create PyTorch DataLoader
        
        Args:
            features: Feature array
            labels: Label array (optional)
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader object
        """
        dataset = MusicDataset(features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def decode_labels(self, encoded_labels):
        """Convert encoded labels back to genre names"""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_columns
    
    def get_num_features(self):
        """Return number of features"""
        return len(self.feature_columns)
    
    def get_num_classes(self):
        """Return number of classes (genres)"""
        return len(self.label_names)


def load_gtzan_data(data_dir='archive/Data', use_30sec=True):
    """
    Convenience function to load GTZAN dataset
    
    Args:
        data_dir: Directory containing the data
        use_30sec: Whether to use 30-second features (True) or 3-second features (False)
        
    Returns:
        MusicDataLoader object
    """
    if use_30sec:
        csv_path = os.path.join(data_dir, 'features_30_sec.csv')
    else:
        csv_path = os.path.join(data_dir, 'features_3_sec.csv')
        
    return MusicDataLoader(csv_path)


if __name__ == '__main__':
    # Test the data loader
    loader = load_gtzan_data()
    X_train, X_test, y_train, y_test = loader.get_train_test_split()
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Feature names: {loader.get_feature_names()[:5]}...")
    
    # Test DataLoader
    train_loader = loader.get_dataloader(X_train, y_train, batch_size=32)
    batch = next(iter(train_loader))
    print(f"\nBatch shape: {batch[0].shape}")
    print(f"Labels shape: {batch[1].shape}")
