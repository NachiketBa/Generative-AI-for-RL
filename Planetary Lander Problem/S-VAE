#This program implements the S-VAE for the Mars Lander problem
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(2)

# Hyperparameters
hidden_sizes_encoder = [324]
hidden_sizes_decoder = [324]
latent_size = 32
learning_rate = 1e-3
epochs = 2000
n_features = 1200
batch_size = 32
num_generated_samples = 1000


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        encoder_layers = []
        input_size = n_features
        for hidden_size in hidden_sizes_encoder:
            encoder_layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),   # layer norm instead of batch norm
                nn.ReLU()
            ])
            input_size = hidden_size
        encoder_layers.append(nn.Linear(hidden_sizes_encoder[-1], latent_size * 2))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        input_size = latent_size
        for hidden_size in hidden_sizes_decoder:
            decoder_layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),   # layer norm instead of batch norm
                nn.ReLU()
            ])
            input_size = hidden_size
        decoder_layers.append(nn.Linear(hidden_sizes_decoder[-1], n_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        latent_params = self.encoder(x)
        mu, logvar = torch.chunk(latent_params, 2, dim=1)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def load_csv_folder(folder_path: Path):
    # Get all CSV files in sorted order
    files = sorted(folder_path.glob("*.csv"))

    # Read each CSV into a DataFrame, flatten to 1D, then convert to tensor
    data_list = []
    for f in files:
        arr = pd.read_csv(f, header=None).values.squeeze()  # shape (1010,)
        data_list.append(torch.tensor(arr, dtype=torch.float32))

    # Stack into tensor of shape (1000, 1010)
    data = torch.stack(data_list, dim=1)  # shape (1010, 1000)
    return data


def loss_function(x_recon, x, mu, logvar):
    recon_loss = nn.MSELoss(reduction='sum')(x_recon, x)
    kl_loss = -0.5 * torch.sum(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss, kl_loss

def main():
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # Load data
    data_path = Path("C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Desktop/Journal_2025_work/wind_vae_final_mod_params")
    data = load_csv_folder(data_path)
    data = data.T.clone().detach()

    data = data[:25,:]
    print("Shape of loaded data:", data.shape)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0) + 1e-6
    data = (data - data_mean) / data_std
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # Training loop
    vae.train()
    epoch_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for traj in train_loader:
            inputs = traj.float().to(device)
            x_recon, mu, logvar = vae(inputs)
            recon_loss, kl_loss = loss_function(x_recon, inputs, mu, logvar)
            loss = recon_loss+ kl_loss
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, kl_loss: {kl_loss.item():.4f}")
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('VAE Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Generate 1000 samples
    generate_samples(vae, num_generated_samples,data_std,data_mean)


def generate_samples(vae, num_samples,data_std,data_mean):
    vae.eval()

    with torch.no_grad():
        # Generate all samples in one batch
        z_samples = torch.randn(num_samples, latent_size).to(device)
        generated_samples = vae.decoder(z_samples).cpu().numpy()

    # Undo normalization
    generated_samples = generated_samples * data_std.cpu().numpy()  + data_mean.cpu().numpy()

    out_csv_folder = Path(
        "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Desktop/Journal_2025_work/Mars_lander_VAE_noise_25")
    out_csv_folder.mkdir(parents=True, exist_ok=True)

    # Save each sample as its own CSV file
    for i, sample in enumerate(generated_samples):
        sample_path = out_csv_folder / f"sample_{i:04d}.csv"
        pd.DataFrame(sample).to_csv(sample_path, header=False, index=False)

    print(f"Generated {num_generated_samples} samples saved to {out_csv_folder}")



if __name__ == "__main__":
    main()
