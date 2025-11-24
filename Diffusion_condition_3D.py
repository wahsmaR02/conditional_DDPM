import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ================================================================
# Helper: Extract a time-dependent scalar and reshape for N-D volume
# ================================================================
def extract(v, t, x_shape):
    """
    v: tensor[T]
    t: tensor[B] of timesteps
    x_shape: shape of target volume (B, C, D, H, W)
    Returns: (B, 1, 1, ..., 1) broadcastable over volume
    """
    out = v.gather(dim=0, index=t).float().to(t.device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# ================================================================
# Training-time diffusion (Forward process q(x_t | x_0))
# ================================================================
class GaussianDiffusionTrainer_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        # -------------------------------------------------
        # float32 buffers for MPS, CUDA, CPU
        # -------------------------------------------------
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float32)

        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar",
                             torch.sqrt(1.0 - alphas_bar))

    def forward(self, x_0):
        """
        x_0: [B, 2, D, H, W]
             channel 0 = clean CT (target)
             channel 1 = CBCT condition (never noised)
        """
        B = x_0.shape[0]
        device = x_0.device

        # random timesteps per batch element
        t = torch.randint(self.T, (B,), device=device)

        # --- split channels (3D-compatible) ---
        ct   = x_0[:, 0:1, ...]      # [B,1,D,H,W]
        cbct = x_0[:, 1:2, ...]      # [B,1,D,H,W]

        noise = torch.randn_like(ct)

        # forward diffusion: q(x_t) = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps
        x_t = (
            extract(self.sqrt_alphas_bar, t, ct.shape) * ct +
            extract(self.sqrt_one_minus_alphas_bar, t, ct.shape) * noise
        )

        # keep CBCT unchanged
        x_t = torch.cat((x_t, cbct), dim=1)

        # model predicts noise on CT channel
        noise_pred = self.model(x_t, t)

        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        return loss


# ================================================================
# Sampling-time reverse process (p(x_{t-1} | x_t))
# ================================================================
class GaussianDiffusionSampler_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float32)
        self.register_buffer("betas", betas)

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, (1, 0), value=1.0)[:-1]

        # precompute diffusion coefficients
        self.register_buffer("coeff1", torch.sqrt(1.0 / alphas))
        self.register_buffer("coeff2", (1 - alphas) /
                             torch.sqrt(1 - alphas_bar))
        self.register_buffer("posterior_var",
                             betas * (1 - alphas_bar_prev) /
                             (1 - alphas_bar))

    # -------------------------
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        ct = x_t[:, 0:1, ...]
        return (
            extract(self.coeff1, t, ct.shape) * ct -
            extract(self.coeff2, t, ct.shape) * eps
        )

    # -------------------------
    def p_mean_variance(self, x_t, t):
        ct = x_t[:, 0:1, ...]
        eps = self.model(x_t, t)

        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps)
        var = extract(self.posterior_var, t, ct.shape)
        return xt_prev_mean, var

    # -------------------------
    def forward(self, x_T):
        """
        x_T: random noise volume + CBCT (unnoised)
        returns: reconstructed synthetic CT
        """
        x_t = x_T
        cbct = x_T[:, 1:2, ...]   # CBCT always stays same

        for time_step in reversed(range(self.T)):
            B = x_t.shape[0]
            t = torch.full((B,), time_step, device=x_t.device, dtype=torch.long)

            mean, var = self.p_mean_variance(x_t, t)

            if time_step > 0:
                noise = torch.randn_like(mean)
                ct = mean + torch.sqrt(var) * noise
            else:
                ct = mean   # final step: no noise

            x_t = torch.cat((ct, cbct), dim=1)

        return torch.clamp(x_t, -1.0, 1.0)