# This script implements the MI-Split-VAE for the Mars Lander Problem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Device and seed
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(32)

# ---------------------------
# Load datasets
# ---------------------------
A_path = Path(
    "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Desktop/Journal_2025_work/wind_vae_sensor_noise")
B_path = Path(
    "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Desktop/Journal_2025_work/no_wind_vae_sensor_noise")


def load_csv_folder(folder_path: Path):
    files = sorted(folder_path.glob("*.csv"))
    data_list = []
    for f in files:
        arr = pd.read_csv(f, header=None).values.squeeze()
        data_list.append(torch.tensor(arr, dtype=torch.float32))
    data = torch.stack(data_list, dim=1)
    return data


A = load_csv_folder(A_path).T.clone().detach()  # shape [N, features]
B = load_csv_folder(B_path).T.clone().detach()

A = A[:25, :]
B = B[:1000, :]

# Standardization
A_mean = A.mean(dim=0)
A_std = A.std(dim=0) + 1e-6
A = (A - A_mean) / A_std

B_mean = B.mean(dim=0)
B_std = B.std(dim=0) + 1e-6
B = (B - B_mean) / B_std

# ---------------------------
# Dataset
# ---------------------------
class LabeledTensorDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.label


dataset_A = LabeledTensorDataset(A, 0)
dataset_B = LabeledTensorDataset(B, 1)

# ---------------------------
# Encoder / Decoder
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, in_dim, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        h = F.relu(self.ln1(self.fc1(x)))
        h = F.relu(self.ln2(self.fc2(h)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim, out_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, z):
        h = F.relu(self.ln1(self.fc1(z)))
        h = F.relu(self.ln2(self.fc2(h)))
        return self.fc_out(h)


# ---------------------------
# Model + optimizer
# ---------------------------
input_dim = A.size(1)
z1_dim = 32
z2_dim = 32
# z1_dim = 64
# z2_dim = 64
hidden_dim = 324

encoder1 = Encoder(input_dim, z1_dim, hidden_dim).to(device)
encoder2 = Encoder(input_dim, z2_dim, hidden_dim).to(device)
decoderA = Decoder(z1_dim + z2_dim, input_dim, hidden_dim).to(device)
decoderB = Decoder(z1_dim + z2_dim, input_dim, hidden_dim).to(device)

# Priors for KL
mu1_A = torch.zeros(z1_dim, device=device)
logvar1_A = torch.zeros(z1_dim, device=device)
mu1_B = torch.ones(z1_dim, device=device)
logvar1_B = torch.log(torch.ones(z1_dim, device=device) * 2.0)

params = list(encoder1.parameters()) + list(encoder2.parameters()) + \
         list(decoderA.parameters()) + list(decoderB.parameters())
optimizer = optim.Adam(params, lr=1e-3)

lambda1, lambda2 = 1.0, 1.0
beta = 20.0  # MI loss weight 
mi_warmup_epochs = 50  # can be 0 if you want MI from the start

# ---------------------------
# Utility functions
# ---------------------------
def reparam(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_diag(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (torch.log(var_p) - torch.log(var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)
    return kl.sum(dim=1)

# ---------------------------
# EMA-based MI estimator
# ---------------------------
ema_decay = 0.99
ema_mean = None
ema_cov = None

def update_ema(new_mean, new_cov):
    global ema_mean, ema_cov
    if ema_mean is None:
        ema_mean = new_mean
        ema_cov = new_cov
    else:
        ema_mean = ema_decay * ema_mean + (1 - ema_decay) * new_mean
        ema_cov = ema_decay * ema_cov + (1 - ema_decay) * new_cov

def mi_loss_z1(z1_A, z1_B):
    batch = torch.cat([z1_A, z1_B], dim=1)
    mean = batch.mean(dim=0)
    cov = torch.cov(batch.T)
    update_ema(mean.detach(), cov.detach())

    if ema_cov is None:
        return torch.tensor(0.0, device=z1_A.device)

    z_dim = z1_A.size(1)
    cov_joint = ema_cov + 1e-6 * torch.eye(2 * z_dim, device=z1_A.device)
    cov_A = cov_joint[:z_dim, :z_dim]
    cov_B = cov_joint[z_dim:, z_dim:]

    sign_joint, logdet_joint = torch.slogdet(cov_joint)
    sign_A, logdet_A = torch.slogdet(cov_A)
    sign_B, logdet_B = torch.slogdet(cov_B)
    mi = 0.5 * (logdet_A + logdet_B - logdet_joint)
    return mi

# ---------------------------
# Training loop
# ---------------------------
epochs = 2000
batch_size = 32
loss_history = []
mi_history = []

A_samples = dataset_A.data.to(device)
B_samples = dataset_B.data.to(device)

for epoch in range(epochs):
    idx_A = torch.randperm(len(A_samples))
    idx_B = torch.randperm(len(B_samples))

    max_len = max(len(A_samples), len(B_samples))
    x_A = A_samples.repeat((max_len // len(A_samples) + 1), 1)[:max_len]
    x_B = B_samples.repeat((max_len // len(B_samples) + 1), 1)[:max_len]

    total_loss_epoch = 0.0
    total_mi_epoch = 0.0
    total_kl1_epoch = 0.0
    total_kl2_epoch = 0.0
    num_batches = max_len // batch_size

    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size
        x_A_batch = x_A[start:end]
        x_B_batch = x_B[start:end]

        # Encode z1
        mu1_A_enc, logvar1_A_enc = encoder1(x_A_batch)
        z1_A_batch = reparam(mu1_A_enc, logvar1_A_enc)
        mu1_B_enc, logvar1_B_enc = encoder1(x_B_batch)
        z1_B_batch = reparam(mu1_B_enc, logvar1_B_enc)

        # Encode shared z2
        mu2_enc, logvar2_enc = encoder2(torch.cat([x_A_batch, x_B_batch], dim=0))
        z_shared = reparam(mu2_enc, logvar2_enc)
        z_shared_A_batch = z_shared[:len(x_A_batch)]
        z_shared_B_batch = z_shared[len(x_A_batch):]

        # Reconstruction
        x_hat_A = decoderA(torch.cat([z1_A_batch, z_shared_A_batch], dim=1))
        x_hat_B = decoderB(torch.cat([z1_B_batch, z_shared_B_batch], dim=1))
        rec_loss = F.mse_loss(x_hat_A, x_A_batch, reduction='sum') + \
                   F.mse_loss(x_hat_B, x_B_batch, reduction='sum')

        # KL
        kl1_A = kl_diag(mu1_A_enc, logvar1_A_enc, mu1_A, logvar1_A).sum()
        kl1_B = kl_diag(mu1_B_enc, logvar1_B_enc, mu1_B, logvar1_B).sum()
        kl1 = kl1_A + kl1_B
        kl2 = kl_diag(mu2_enc, logvar2_enc,
                      torch.zeros_like(mu2_enc), torch.zeros_like(logvar2_enc)).sum()

        # MI
        mi_z1 = mi_loss_z1(z1_A_batch, z1_B_batch)

        # Total loss
        if epoch >= mi_warmup_epochs:
            loss = rec_loss + lambda1 * kl1 + lambda2 * kl2 + beta * mi_z1
        else:
            loss = rec_loss + lambda1 * kl1 + lambda2 * kl2  # MI warmup

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        total_mi_epoch += mi_z1.item()
        total_kl1_epoch += kl1.item()
        total_kl2_epoch += kl2.item()

    avg_loss_epoch = total_loss_epoch / max_len
    avg_mi_epoch = total_mi_epoch / num_batches
    loss_history.append(avg_loss_epoch)
    mi_history.append(avg_mi_epoch)
    avg_kl1_epoch = total_kl1_epoch / max_len
    avg_kl2_epoch = total_kl2_epoch / max_len
    print(f"Epoch {epoch+1}/{epochs}, "
          f"Total Loss: {avg_loss_epoch:.4f}, "
          f"KL1: {avg_kl1_epoch:.4f}, KL2: {avg_kl2_epoch:.4f}, "
          f"MI: {avg_mi_epoch:.4f}")
# ---------------------------
# Plot Loss + MI
# ---------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("Total Loss")
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(mi_history)
plt.title("EMA-based MI(z1_A,z1_B)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Generate samples for A# ---------------------------
num_gen = 1000
z1_gen_A = torch.randn(num_gen, z1_dim, device=device)
z_shared_gen = torch.randn(num_gen, z2_dim, device=device)
with torch.no_grad():
    x_gen_A = decoderA(torch.cat([z1_gen_A, z_shared_gen], dim=1)).cpu().numpy()
    x_gen_A = x_gen_A * A_std.cpu().numpy() + A_mean.cpu().numpy()

out_csv_folder = Path(
    "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Desktop/Journal_2025_work/Mars_lander_2AE_noise_25")
out_csv_folder.mkdir(parents=True, exist_ok=True)

for i, sample in enumerate(x_gen_A):
    pd.DataFrame(sample).to_csv(out_csv_folder / f"sample_{i:04d}.csv",
                                header=False, index=False)

print(f"Generated {num_gen} samples saved to {out_csv_folder}")
