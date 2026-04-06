# Generative AI for RL — Mars Lander Problem

> **Using Variational Autoencoders (VAE and MI-VAE) to generate synthetic RL policy parameters for the Mars Lander problem under wind and no-wind conditions.**

---

## Overview

This sub-project is part of the broader **Generative AI for Reinforcement Learning** research initiative. The goal is to use **generative models** to synthesize new RL policy parameter trajectories from a small set of real training runs, enabling data augmentation for the Mars Lander control problem.

Two VAE architectures are implemented and compared:

| Script | Model | Purpose |
|---|---|---|
| `S_VAE_mars_lander.py` | Standard VAE (S-VAE) | Learn a latent distribution over policy parameters from wind-condition runs; generate synthetic samples |
| `MI_VAE_mars_lander.py` | Mutual Information VAE (MI-VAE) | Disentangle **condition-specific** (wind vs. no-wind) latent factors from **shared** latent factors across both domains |

The generated synthetic parameter sets can then be injected back into the RL training pipeline as data augmentation, improving generalization with very few real rollouts.

---

## Problem Context

The Mars Lander problem involves controlling a lander's **rotation** and **thrust** to reach a flat landing zone safely under Martian gravity (3.711 m/s²). RL agents learn a policy parameterized by neural network weights. This project treats those **policy parameter vectors** as the data distribution to be modeled:

- **Dataset A**: Policy parameters from agents trained **with wind** (25 samples)
- **Dataset B**: Policy parameters from agents trained **without wind** (1000 samples)

The VAE learns to model and generate new, plausible parameter vectors from each distribution.

---

## Architecture

### S-VAE (`S_VAE_mars_lander.py`)

A single-domain standard VAE trained on wind-condition policy parameters.

```
Input (1200-dim policy params)
        ↓
  Encoder: Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → [μ, log σ²]
        ↓
  Reparameterization: z = μ + ε·σ,   ε ~ N(0, I)     (z ∈ ℝ³²)
        ↓
  Decoder: Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → x̂
        ↓
Output (1200-dim reconstructed params)
```

**Loss:**
```
L = MSE(x̂, x)  +  KL[ q(z|x) || N(0, I) ]
```

---

### MI-VAE (`MI_VAE_mars_lander.py`)

A dual-encoder, dual-decoder VAE designed to disentangle **domain-specific** (z₁) and **domain-shared** (z₂) latent variables across wind (A) and no-wind (B) datasets.

```
Domain A (wind)        Domain B (no-wind)
      │                       │
   Encoder1 → z1_A         Encoder1 → z1_B     ← domain-specific latent (ℝ³²)
   Encoder2 → z2_A         Encoder2 → z2_B     ← shared latent       (ℝ³²)
      │                       │
  DecoderA([z1_A, z2_A])  DecoderB([z1_B, z2_B])
      ↓                       ↓
    x̂_A                     x̂_B
```

**Loss:**
```
L = Recon_A + Recon_B
  + λ₁ · (KL[q(z1_A) || p_A(z1)] + KL[q(z1_B) || p_B(z1)])
  + λ₂ · KL[q(z2) || N(0, I)]
  + β  · MI(z1_A, z1_B)          ← maximized after warmup (epoch ≥ 50)
```

The **MI term** encourages the domain-specific encoders to capture *different* information for each domain, pushing the latent spaces apart and improving disentanglement. The EMA-based mutual information estimator is computed using the log-determinant of the joint covariance matrix.

---

## Key Hyperparameters

| Parameter | S-VAE | MI-VAE | Description |
|---|---|---|---|
| `input_dim` / `n_features` | 1200 | auto-detected | Policy parameter vector size |
| `latent_size` / `z_dim` | 32 | 32 (z1) + 32 (z2) | Latent space dimensionality |
| `hidden_dim` | 324 | 324 | Hidden layer width |
| `epochs` | 2000 | 2000 | Training epochs |
| `batch_size` | 32 | 32 | Mini-batch size |
| `learning_rate` | 1e-3 | 1e-3 | Adam optimizer LR |
| `β` | — | 20.0 | MI loss weight |
| `mi_warmup_epochs` | — | 50 | Epochs before MI term activates |
| `ema_decay` | — | 0.99 | EMA decay for MI estimator |
| `num_generated_samples` | 1000 | 1000 | Synthetic samples to generate |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/NachiketBa/Generative-AI-for-RL.git
cd "Generative-AI-for-RL/Mars Lander Problem"

# Install dependencies
pip install torch pandas numpy matplotlib
```

**Requirements:**
- Python ≥ 3.9
- PyTorch ≥ 2.0
- pandas, numpy, matplotlib

GPU is supported automatically — if CUDA is available, it will be used. Otherwise, training runs on CPU.

---

## Data Format

Each dataset is a **folder of CSV files**, one CSV per policy parameter sample.

```
wind_vae_final_mod_params/
├── sample_0000.csv     # shape: (1200, 1) — one policy parameter vector
├── sample_0001.csv
└── ...

no_wind_vae_final_mod_params/
├── sample_0000.csv
└── ...
```

Each CSV contains a single column of 1200 float values (the flattened policy network parameters). Before training, data is **standardized per-feature** (zero mean, unit variance) and de-standardized before saving generated samples.

> ⚠️ **Update the data paths** in each script to point to your local dataset folders before running.

---

## Usage

### Running the S-VAE

```bash
python S_VAE_mars_lander.py
```

**What happens:**
1. Loads CSV files from `wind_vae_final_mod_params/` (uses first 25 samples)
2. Standardizes the data
3. Trains a VAE for 2000 epochs
4. Plots training loss curve
5. Generates 1000 synthetic parameter samples
6. Saves them to `Mars_lander_VAE_noise_25/` as `sample_0000.csv … sample_0999.csv`

---

### Running the MI-VAE

```bash
python MI_VAE_mars_lander.py
```

**What happens:**
1. Loads wind data (25 samples, Dataset A) and no-wind data (1000 samples, Dataset B)
2. Standardizes each domain independently
3. Trains the MI-VAE for 2000 epochs with a 50-epoch MI warmup
4. Plots training loss and EMA-based MI curves side-by-side
5. Generates 1000 synthetic samples for the wind domain using the disentangled latent space
6. Saves them to `Mars_lander_2AE_noise_25/` as `sample_0000.csv … sample_0999.csv`

---

## Output

Both scripts produce a folder of generated CSV files that can be fed back into the RL training pipeline:

```
Mars_lander_VAE_noise_25/      ← S-VAE output
├── sample_0000.csv
├── ...
└── sample_0999.csv

Mars_lander_2AE_noise_25/      ← MI-VAE output
├── sample_0000.csv
├── ...
└── sample_0999.csv
```

Each generated CSV has the same format as the input (1200 × 1), de-normalized back to the original parameter scale.

---

## Training Monitoring

Both scripts print per-epoch diagnostics to the console:

**S-VAE:**
```
Epoch [1/2000], Loss: 142.3821, kl_loss: 0.0023
Epoch [2/2000], Loss: 138.9102, kl_loss: 0.0041
...
```

**MI-VAE:**
```
Epoch 1/2000, Total Loss: 9.4521, KL1: 0.0031, KL2: 0.0012, MI: 0.0000
...
Epoch 51/2000, Total Loss: 8.1234, KL1: 0.0045, KL2: 0.0018, MI: 0.4321
...
```

At the end of training, both scripts display **matplotlib plots**:
- **S-VAE**: Single loss curve (total loss vs. epoch)
- **MI-VAE**: Two-panel plot — total loss + EMA-based MI(z1_A, z1_B) vs. epoch

---

## Design Decisions

**Why LayerNorm instead of BatchNorm?**
The datasets are very small (especially Dataset A with only 25 samples), making batch statistics unstable. LayerNorm normalizes per-sample rather than per-batch, giving stable gradients regardless of batch size.

**Why an EMA-based MI estimator?**
Computing mutual information from a single mini-batch is noisy. The exponential moving average (decay = 0.99) smooths the covariance estimate across batches, giving a more stable MI signal for the loss.

**Why separate priors for z1_A and z1_B?**
The MI-VAE uses `N(0, I)` as the prior for z1_A (wind) and `N(1, 2I)` as the prior for z1_B (no-wind). This prior shift encourages the domain-specific encoders to occupy distinct regions of latent space from the start of training.

**Why MI warmup for 50 epochs?**
The MI estimator needs time to warm up via EMA before it produces reliable gradients. Training without MI for the first 50 epochs lets reconstruction and KL losses stabilize first.

---

## File Structure

```
Mars Lander Problem/
├── S_VAE_mars_lander.py        # Standard VAE for single-domain generation
├── MI_VAE_mars_lander.py       # Mutual Information VAE for cross-domain disentanglement
└── README.md
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{nachiket2025genairl,
  author       = {Nachiket Ba},
  title        = {Generative AI for Reinforcement Learning — Mars Lander Problem},
  year         = {2025},
  howpublished = {\url{https://github.com/NachiketBa/Generative-AI-for-RL}},
}
```

---

## License

This project is licensed under the MIT License.
