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
      - Channel 1+: Conditioning (CBCT, Coords, etc.)
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

    def forward(self, x_0, mask=None): # ADD mask=None argument
        """
        x_0: [B, 5, D, H, W] (CT + CBCT + 3 Coords)
        returns: scalar loss
        """
        #print("\n[Diffusion] x_0:", x_0.shape)   # <-- PRINT HERE

        B = x_0.shape[0]
        device = x_0.device

        # sample random timesteps for each item in batch
        t = torch.randint(self.T, size=(B,), device=device)

        # split CT and CBCT, keep all 3D spatial dims
        # Channel 0 is the target (CT)
        # Channels 1:End are the condition (CBCT + Coords)
        ct   = x_0[:, 0:1]  # [B, 1, D, H, W]
        cond = x_0[:, 1:]   # [B, 4, D, H, W]

        #print("[Diffusion] CT:", ct.shape)
       # print("[Diffusion] CBCT:", cbct.shape)

        # sample Gaussian noise
        noise = torch.randn_like(ct)

        # forward diffusion (q(x_t | x_0))
        x_t = (extract(self.sqrt_alphas_bar, t, ct.shape) * ct +
               extract(self.sqrt_one_minus_alphas_bar, t, ct.shape) * noise)

      #  print("[Diffusion] x_t:", x_t.shape)

        # concatenate noisy CT with conditioning (CBCT+Coords)
        x_t = torch.cat((x_t, cond), dim=1)  # [B, 5, D, H, W]

        # model predicts epsilon for CT channel (epsilon is the noise that was added to the CT image during the forward diffusion process.)
        eps_pred = self.model(x_t, t)
        #print("[Diffusion] eps_pred:", eps_pred.shape)

        #loss = F.mse_loss(eps_pred, noise, reduction='sum')
        #return loss, noise.numel()

# --- MODIFIED: Masked Loss Calculation ---
        if mask is not None:
            # Ensure mask is correct shape [B, 1, D, H, W] and on device
            mask_t = mask.float().to(device)
            if mask_t.ndim == 4:
                mask_t = mask_t.unsqueeze(1)  # [B,1,D,H,W]

            # Compute Squared Error per pixel
            squared_error = (eps_pred - noise) ** 2
            
            # Zero out background loss
            loss = (squared_error * mask_t).sum()
            
            # Count only valid pixels to avoid overly small loss
            numel = mask_t.sum().clamp(min=1.0)
        else:
            loss = F.mse_loss(eps_pred, noise, reduction="sum")
            numel = noise.numel()
        # --- END MODIFIED ---

        return loss, numel

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
        
        # We only denoise channel 0 (CT)
        ct = x_t[:, 0:1]
        assert ct.shape == eps.shape

        return (extract(self.coeff1, t, ct.shape) * ct -
                extract(self.coeff2, t, ct.shape) * eps)

    def p_mean_variance(self, x_t, t):
        """
        Compute mean and variance of p(x_{t-1} | x_t).
        -->Computes the mean and varince of the reverse diffusion process.
        """
        
        # Extract CT for variance calculation
        ct = x_t[:, 0:1]

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

        # 1. Separate Condition (CBCT + Coords) once.
        # These channels are NEVER added noise to, so we keep them aside.
        cond = x_t[:, 1:] 
        
        # 2. Extract starting noisy CT
        ct = x_t[:, 0:1]

        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([B], dtype=torch.long, device=device) * time_step
            
            # We pass the FULL x_t (ct + cond) to the model
            mean, var = self.p_mean_variance(x_t=x_t, t=t)
            
            noise = torch.randn_like(ct) if time_step > 0 else 0
            ct = mean + torch.sqrt(var) * noise
            
            # Re-concatenate with the fixed condition
            x_t = torch.cat((ct, cond), dim=1)
            
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        x_0 = x_t
        #return torch.clip(x_0, -1, 1)
        return x_0
