"""
VAE Architectures for Music Clustering
Includes: Basic VAE, Convolutional VAE, and Beta-VAE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np


class BaseVAE(nn.Module):
    """Base Variational Autoencoder class"""
    
    def __init__(self, input_dim, latent_dim):
        super(BaseVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """
        VAE loss = Reconstruction loss + KL divergence
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL divergence (for Beta-VAE)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


class SimpleVAE(BaseVAE):
    """
    Simple fully-connected VAE for Easy Task
    Architecture: Encoder -> Latent Space -> Decoder
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128], latent_dim=32):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
        """
        super(SimpleVAE, self).__init__(input_dim, latent_dim)
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def get_latent(self, x):
        """Get latent representation (mean) without reparameterization"""
        mu, _ = self.encode(x)
        return mu


class ConvVAE(BaseVAE):
    """
    Convolutional VAE for Medium Task
    Treats features as 1D signal for CNN processing
    """
    
    def __init__(self, input_dim, latent_dim=64, channels=[32, 64, 128]):
        """
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            channels: List of channel dimensions for conv layers
        """
        super(ConvVAE, self).__init__(input_dim, latent_dim)
        
        self.channels = channels
        
        # Reshape input to [batch, 1, input_dim] for 1D convolution
        # Encoder
        encoder_layers = []
        in_channels = 1
        for out_channels in channels:
            encoder_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(0.2)
            ])
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened dimension after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            dummy_out = self.encoder(dummy)
            self.flattened_dim = dummy_out.view(1, -1).shape[1]
        
        # Latent space
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_dim)
        
        # Decoder
        decoder_layers = []
        for i in range(len(channels) - 1, 0, -1):
            decoder_layers.extend([
                nn.ConvTranspose1d(channels[i], channels[i-1], 
                                 kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(channels[i-1]),
                nn.Dropout(0.2)
            ])
        
        # Final layer to get back to original dimension
        decoder_layers.append(
            nn.ConvTranspose1d(channels[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Store shape for reshaping in decoder
        self.decoder_input_shape = dummy_out.shape[1:]
        
    def encode(self, x):
        """Encode input to latent parameters"""
        # Reshape to [batch, 1, features]
        x = x.unsqueeze(1)
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        """Decode from latent space"""
        h = self.fc_decode(z)
        h = h.view(h.size(0), *self.decoder_input_shape)  # Reshape for conv
        recon = self.decoder(h)
        recon = recon.squeeze(1)  # Remove channel dimension
        # Ensure output matches input dimension
        if recon.size(1) != self.input_dim:
            recon = F.interpolate(recon.unsqueeze(1), size=self.input_dim, mode='linear', align_corners=False).squeeze(1)
        return recon
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def get_latent(self, x):
        """Get latent representation"""
        mu, _ = self.encode(x)
        return mu


class BetaVAE(SimpleVAE):
    """
    Beta-VAE for Hard Task
    Uses beta > 1 for disentangled representations
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128], latent_dim=64, beta=4.0):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
            beta: Weight for KL divergence (typically 2-10 for disentanglement)
        """
        super(BetaVAE, self).__init__(input_dim, hidden_dims, latent_dim)
        self.beta = beta
        
    def compute_loss(self, recon_x, x, mu, logvar):
        """Compute Beta-VAE loss with specified beta"""
        return self.loss_function(recon_x, x, mu, logvar, beta=self.beta)


class ConditionalVAE(BaseVAE):
    """
    Conditional VAE (CVAE) for Hard Task
    Conditions on genre labels for better clustering
    """
    
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128], latent_dim=64):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes (genres)
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
        """
        super(ConditionalVAE, self).__init__(input_dim, latent_dim)
        self.num_classes = num_classes
        
        # Encoder (input + one-hot class)
        encoder_layers = []
        prev_dim = input_dim + num_classes
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder (latent + one-hot class)
        decoder_layers = []
        prev_dim = latent_dim + num_classes
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x, c):
        """
        Encode input conditioned on class
        Args:
            x: Input features
            c: Class labels (integers)
        """
        # One-hot encode class
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        # Concatenate input and class
        xc = torch.cat([x, c_onehot], dim=1)
        h = self.encoder(xc)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z, c):
        """
        Decode from latent space conditioned on class
        Args:
            z: Latent vector
            c: Class labels (integers)
        """
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        zc = torch.cat([z, c_onehot], dim=1)
        return self.decoder(zc)
    
    def forward(self, x, c):
        """Forward pass through CVAE"""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar
    
    def get_latent(self, x, c):
        """Get latent representation"""
        mu, _ = self.encode(x, c)
        return mu


class VAETrainer:
    """Trainer class for VAE models"""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3):
        """
        Args:
            model: VAE model instance
            device: Device to train on
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
        
    def train_epoch(self, dataloader, beta=1.0):
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader object
            beta: Beta parameter for VAE loss
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                data = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device) if len(batch_data) > 1 else None
            else:
                data = batch_data.to(self.device)
                labels = None
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, ConditionalVAE) and labels is not None:
                recon_batch, mu, logvar = self.model(data, labels)
            else:
                recon_batch, mu, logvar = self.model(data)
            
            # Compute loss
            loss, recon_loss, kl_loss = self.model.loss_function(
                recon_batch, data, mu, logvar, beta=beta
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon = total_recon_loss / len(dataloader.dataset)
        avg_kl = total_kl_loss / len(dataloader.dataset)
        
        return avg_loss, avg_recon, avg_kl
    
    def train(self, dataloader, num_epochs, beta=1.0, verbose=True):
        """
        Train the model
        
        Args:
            dataloader: DataLoader object
            num_epochs: Number of training epochs
            beta: Beta parameter for VAE loss
            verbose: Whether to print training progress
        """
        for epoch in range(num_epochs):
            avg_loss, avg_recon, avg_kl = self.train_epoch(dataloader, beta=beta)
            self.train_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
                      f'Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}')
    
    def get_latent_representations(self, dataloader):
        """
        Extract latent representations for all data
        
        Args:
            dataloader: DataLoader object
            
        Returns:
            latent_vectors: numpy array of shape (n_samples, latent_dim)
            labels: numpy array of shape (n_samples,) if available
        """
        self.model.eval()
        latent_vectors = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                if isinstance(batch_data, (list, tuple)):
                    data = batch_data[0].to(self.device)
                    labels = batch_data[1] if len(batch_data) > 1 else None
                else:
                    data = batch_data.to(self.device)
                    labels = None
                
                if isinstance(self.model, ConditionalVAE) and labels is not None:
                    mu = self.model.get_latent(data, labels.to(self.device))
                else:
                    mu = self.model.get_latent(data)
                
                latent_vectors.append(mu.cpu().numpy())
                
                if labels is not None:
                    all_labels.append(labels.numpy())
        
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        
        if all_labels:
            all_labels = np.concatenate(all_labels, axis=0)
            return latent_vectors, all_labels
        
        return latent_vectors, None


if __name__ == '__main__':
    # Test VAE architectures
    input_dim = 57
    batch_size = 32
    
    # Test SimpleVAE
    print("Testing SimpleVAE...")
    vae = SimpleVAE(input_dim=input_dim, latent_dim=32)
    x = torch.randn(batch_size, input_dim)
    recon_x, mu, logvar = vae(x)
    print(f"Input shape: {x.shape}, Recon shape: {recon_x.shape}, Latent shape: {mu.shape}")
    
    # Test ConvVAE
    print("\nTesting ConvVAE...")
    conv_vae = ConvVAE(input_dim=input_dim, latent_dim=64)
    recon_x, mu, logvar = conv_vae(x)
    print(f"Input shape: {x.shape}, Recon shape: {recon_x.shape}, Latent shape: {mu.shape}")
    
    # Test BetaVAE
    print("\nTesting BetaVAE...")
    beta_vae = BetaVAE(input_dim=input_dim, latent_dim=64, beta=4.0)
    recon_x, mu, logvar = beta_vae(x)
    print(f"Input shape: {x.shape}, Recon shape: {recon_x.shape}, Latent shape: {mu.shape}")
    
    # Test CVAE
    print("\nTesting ConditionalVAE...")
    cvae = ConditionalVAE(input_dim=input_dim, num_classes=10, latent_dim=64)
    labels = torch.randint(0, 10, (batch_size,))
    recon_x, mu, logvar = cvae(x, labels)
    print(f"Input shape: {x.shape}, Recon shape: {recon_x.shape}, Latent shape: {mu.shape}")
    
    print("\nAll tests passed!")
