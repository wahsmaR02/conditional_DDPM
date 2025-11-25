import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract(v, t, x_shape):
    """
    Extract the timestep value v[t] for each batch element and reshape it
    so it can broadcast over a tensor of shape x_shape.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer_cond(nn.Module):
    """
    Conditional diffusion trainer for 3D volumes:
    Adds noise to CT image and trains the model to predict the noise.

    Expects x_0 of shape [B, 2, D, H, W]:
      - channel 0: clean CT
      - channel 1: CBCT (conditioning)
    """
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        # float32 on purpose for MPS and stability
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        x_0: [B, 2, D, H, W]
        returns: scalar loss
        """
        #print("\n[Diffusion] x_0:", x_0.shape)   # <-- PRINT HERE

        B = x_0.shape[0]
        device = x_0.device

        # sample random timesteps for each item in batch
        t = torch.randint(self.T, size=(B,), device=device)

        # split CT and CBCT, keep all 3D spatial dims
        ct   = x_0[:, 0].unsqueeze(1)  # [B, 1, D, H, W]
        cbct = x_0[:, 1].unsqueeze(1)  # [B, 1, D, H, W]

        #print("[Diffusion] CT:", ct.shape)
       # print("[Diffusion] CBCT:", cbct.shape)

        # sample Gaussian noise
        noise = torch.randn_like(ct)

        # forward diffusion (q(x_t | x_0))
        x_t = (extract(self.sqrt_alphas_bar, t, ct.shape) * ct +
               extract(self.sqrt_one_minus_alphas_bar, t, ct.shape) * noise)

      #  print("[Diffusion] x_t:", x_t.shape)

        # concatenate noisy CT with conditioning CBCT
        x_t = torch.cat((x_t, cbct), dim=1)  # [B, 2, D, H, W]

        # model predicts epsilon for CT channel (epsilon is the noise that was added to the CT image during the forward diffusion process.)
        eps_pred = self.model(x_t, t)        # [B, 1, D, H, W]
        #print("[Diffusion] eps_pred:", eps_pred.shape)

        loss = F.mse_loss(eps_pred, noise, reduction='sum')
        return loss, noise.numel()


class GaussianDiffusionSampler_cond(nn.Module):
    """
    Conditional sampler for 3D diffusion model.
    Denoises the CT image while keeping the CBCT fixed.

    Takes an initial noisy pair [B, 2, D, H, W] (noisy CT + CBCT),
    and iteratively denoises the CT channel while keeping CBCT fixed.
    """
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        # float32 on purpose for MPS and stability
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer(
            'coeff2',
            self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar)
        )
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar)
        )

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        """
        Compute mean of p(x_{t-1} | x_t, eps_pred).
        --> Computes the models estimate of the next denoised CT

        x_t: [B, 2, D, H, W]
        eps: [B, 1, D, H, W] (predicted noise for CT)
        """
        ct = x_t[:, 0].unsqueeze(1)  # [B, 1, D, H, W]
        assert ct.shape == eps.shape

        return (extract(self.coeff1, t, ct.shape) * ct -
                extract(self.coeff2, t, ct.shape) * eps)

    def p_mean_variance(self, x_t, t):
        """
        Compute mean and variance of p(x_{t-1} | x_t).
        -->Computes the mean and varince of the reverse diffusion process.
        """
        ct = x_t[:, 0].unsqueeze(1)  # [B, 1, D, H, W]

        # posterior variance schedule
        var_schedule = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var_schedule, t, ct.shape)

        # model predicts epsilon given current x_t and timestep t
        eps = self.model(x_t, t)  # [B, 1, D, H, W]

        # compute mean of x_{t-1}
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Performs the entire reverse diffusion process:
        Iteratively denoises the CT from t = T → 0 while keeping CBCT fixed.
        Returns the final reconstructed CT.

        x_T: initial noisy pair [B, 2, D, H, W]
             (channel 0 = noisy CT, channel 1 = CBCT)

        returns: x_0 ≈ [B, 2, D, H, W] with denoised CT in channel 0.
        """
        
        x_t = x_T
        B = x_T.shape[0]
        device = x_T.device

        # split for shape convenience
        ct   = x_t[:, 0].unsqueeze(1)  # [B, 1, D, H, W]
        cbct = x_t[:, 1].unsqueeze(1)  # [B, 1, D, H, W]

        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([B], dtype=torch.long, device=device) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t)
            noise = torch.randn_like(ct) if time_step > 0 else 0
            ct = mean + torch.sqrt(var) * noise
            x_t = torch.cat((ct, cbct), dim=1)  # keep CBCT fixed
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        x_0 = x_t
        return torch.clip(x_0, -1, 1)

