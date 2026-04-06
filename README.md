# Generative AI for RL: Mars Lander Problem

> Variational Autoencoders (S-VAE and MI-VAE) trained to generate synthetic RL policy parameters for the Mars Lander problem, with and without wind.

---

## What this is

Real RL training runs for the Mars Lander are expensive to collect, especially under wind conditions where only 25 samples were available. This project fits two VAE models to those small datasets and uses them to generate 1000 new synthetic policy parameter vectors per run. Those vectors can then be fed back into the RL training loop as data augmentation.

Two models are implemented:

| Script | Model | What it does |
|---|---|---|
| `S_VAE_mars_lander.py` | Standard VAE (S-VAE) | Fits a single latent distribution to wind-condition policy parameters and samples from it |
| `MI_VAE_mars_lander.py` | Mutual Information VAE (MI-VAE) | Trains on both wind and no-wind data simultaneously, pushing the two domains into separate latent regions via a mutual information penalty |

---

## Background

The Mars Lander problem asks an agent to control a lander's rotation and thrust to reach a flat landing zone under Martian gravity (3.711 m/s²). The RL agent is a neural network, and its weights form a ~1200-dimensional parameter vector. Rather than modeling the lander physics directly, this project models **the distribution of trained policy weights** — wind vs. no-wind — and generates new ones.

- **Dataset A**: 25 policy parameter vectors from agents trained with wind
- **Dataset B**: 1000 policy parameter vectors from agents trained without wind

The imbalance between the two domains is intentional; the MI-VAE is specifically designed to handle the case where one domain has far fewer samples.

---

## Model Architectures

### S-VAE (`S_VAE_mars_lander.py`)

A standard single-domain VAE trained only on wind-condition parameters.

```
Input (1200-dim policy params)
        |
  Encoder: Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU -> [mu, log_var]
        |
  Reparameterize: z = mu + eps * sigma,   eps ~ N(0, I)     (z in R^32)
        |
  Decoder: Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU -> x_hat
        |
Output (1200-dim reconstructed params)
```

**Loss function:**
```
L = MSE(x_hat, x)  +  KL[ q(z|x) || N(0, I) ]
```

---

### MI-VAE (`MI_VAE_mars_lander.py`)

A dual-encoder, dual-decoder VAE that separates what is unique to each domain (z1) from what both domains share (z2).

```
Domain A (wind)              Domain B (no-wind)
       |                            |
  Encoder1 -> z1_A           Encoder1 -> z1_B     <- domain-specific (R^32)
  Encoder2 -> z2_A           Encoder2 -> z2_B     <- shared           (R^32)
       |                            |
  DecoderA([z1_A, z2_A])    DecoderB([z1_B, z2_B])
       |                            |
     x_hat_A                     x_hat_B
```

**Loss function:**
```
L = Recon_A + Recon_B
  + lambda1 * (KL[q(z1_A) || p_A(z1)] + KL[q(z1_B) || p_B(z1)])
  + lambda2 * KL[q(z2) || N(0, I)]
  + beta    * MI(z1_A, z1_B)          <- added after epoch 50
```

The MI term (weighted by beta = 20.0) penalizes any information shared between z1_A and z1_B. In practice this forces the domain-specific encoders to find features that genuinely differ between the wind and no-wind conditions. The MI estimate uses an EMA-smoothed covariance matrix rather than a per-batch estimate, which keeps the signal stable given the small dataset size.

---

## Hyperparameters

| Parameter | S-VAE | MI-VAE | Notes |
|---|---|---|---|
| `input_dim` / `n_features` | 1200 | auto-detected | Flattened policy network weights |
| `latent_size` / `z_dim` | 32 | 32 per head (z1 + z2) | Latent dimensionality |
| `hidden_dim` | 324 | 324 | Width of each hidden layer |
| `epochs` | 2000 | 2000 | Full training passes |
| `batch_size` | 32 | 32 | |
| `learning_rate` | 1e-3 | 1e-3 | Adam |
| `beta` | n/a | 20.0 | Weight on the MI loss term |
| `mi_warmup_epochs` | n/a | 50 | MI term is off for the first 50 epochs |
| `ema_decay` | n/a | 0.99 | Smoothing factor for the MI covariance estimate |
| `num_generated_samples` | 1000 | 1000 | Samples written to disk after training |

---

## Installation

```bash
git clone https://github.com/NachiketBa/Generative-AI-for-RL.git
cd "Generative-AI-for-RL/Mars Lander Problem"

pip install torch pandas numpy matplotlib
```

Tested on Python 3.9+ and PyTorch 2.0+. Both scripts detect CUDA automatically and fall back to CPU if no GPU is found.

---

## Data Format

Each dataset is a folder of CSV files, one file per training run.

```
wind_vae_final_mod_params/
    sample_0000.csv     # single column, 1200 rows: one policy parameter vector
    sample_0001.csv
    ...

no_wind_vae_final_mod_params/
    sample_0000.csv
    ...
```

Before training, each feature is normalized to zero mean and unit variance. Generated samples are de-normalized back to the original scale before saving.

> **Before running either script, update the hardcoded folder paths** near the top of each file to point to your local data directories.

---

## Running the scripts

### S-VAE

```bash
python S_VAE_mars_lander.py
```

This loads the first 25 samples from the wind folder, trains a VAE for 2000 epochs, shows a loss plot, then writes 1000 generated parameter vectors to:

```
Mars_lander_VAE_noise_25/
    sample_0000.csv  ...  sample_0999.csv
```

### MI-VAE

```bash
python MI_VAE_mars_lander.py
```

This loads 25 wind samples (Dataset A) and 1000 no-wind samples (Dataset B), trains the dual-encoder model for 2000 epochs, shows a two-panel plot of total loss and MI over training, then writes 1000 generated wind-domain parameter vectors to:

```
Mars_lander_2AE_noise_25/
    sample_0000.csv  ...  sample_0999.csv
```

---

## Console output

**S-VAE** prints one line per epoch:
```
Epoch [1/2000], Loss: 142.3821, kl_loss: 0.0023
Epoch [2/2000], Loss: 138.9102, kl_loss: 0.0041
```

**MI-VAE** prints reconstruction loss, both KL terms, and the current MI estimate:
```
Epoch 1/2000,  Total Loss: 9.4521, KL1: 0.0031, KL2: 0.0012, MI: 0.0000
...
Epoch 51/2000, Total Loss: 8.1234, KL1: 0.0045, KL2: 0.0018, MI: 0.4321
```

MI reads 0.0000 for the first 50 epochs while the warmup runs. Once it activates at epoch 51, you should see it climb as the domain-specific encoders diverge.

---

## Design notes

**LayerNorm instead of BatchNorm.** With only 25 samples in Dataset A, batch statistics are too noisy to normalize reliably. LayerNorm operates per sample, so batch size does not affect its behavior.

**EMA for the MI estimate.** A covariance matrix computed from a single mini-batch of 32 samples is too noisy to use as a loss signal directly. The EMA with decay 0.99 accumulates statistics across batches and gives the optimizer a much smoother gradient.

**Separate priors for z1_A and z1_B.** The MI-VAE sets the prior for z1_A to N(0, I) and for z1_B to N(1, 2I). Starting the two domain-specific encoders from different prior regions makes it easier for the MI penalty to keep them separated throughout training.

**MI warmup.** The EMA covariance estimate is unreliable early in training before enough batches have been seen. Running 50 epochs of pure reconstruction and KL loss first gives the estimate time to stabilize before the MI term turns on.

---

## File structure

```
Mars Lander Problem/
    S_VAE_mars_lander.py        # single-domain VAE
    MI_VAE_mars_lander.py       # dual-domain MI-VAE
    README.md
```

---

## Citation

```bibtex
@misc{nachiket2025genairl,
  author       = {Nachiket Ba},
  title        = {Generative AI for Reinforcement Learning: Mars Lander Problem},
  year         = {2025},
  howpublished = {\url{https://github.com/NachiketBa/Generative-AI-for-RL}},
}
```

---

## License

MIT License.
