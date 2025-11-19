import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)   # ensure float32
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        # >>> float32 on purpose for MPS <<<
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        #ct = x_0[:,0,:,:].unsqueeze(1)
        #cbct = x_0[:,1,:,:].unsqueeze(1)

        ct_center = x_0[:,1,:,:].unsqueeze(1)     # CT(z)
        cond = torch.cat([
            x_0[:,0:1,:,:],    # CT(z-1)
            x_0[:,2:3,:,:],    # CT(z+1)
            x_0[:,3:,:,:]      # CBCT stack (3 channels)
        ], dim=1)              # total cond channels = 5

        noise = torch.randn_like(ct_center)
        x_t_center = (extract(self.sqrt_alphas_bar, t, ct_center.shape) * ct_center +
               extract(self.sqrt_one_minus_alphas_bar, t, ct_center.shape) * noise)
        
        x_t = torch.cat((x_t_center, cond), dim=1)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='sum')
        return loss


class GaussianDiffusionSampler_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        # >>> float32 on purpose for MPS <<<
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        #ct = x_t[:,0,:,:].unsqueeze(1)
        #cbct = x_t[:,1,:,:].unsqueeze(1)
        #assert ct.shape == eps.shape

        ct_center = x_t[:,0,:,:].unsqueeze(1)
        assert ct_center.shape == eps.shape

        return (
            extract(self.coeff1, t, ct_center.shape) * ct_center -
            extract(self.coeff2, t, ct_center.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        #ct = x_t[:,0,:,:].unsqueeze(1)
        #cbct = x_t[:,1,:,:].unsqueeze(1)
        ct_center = x_t[:,0,:,:].unsqueeze(1)

        var_schedule = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var_schedule, t, ct_center.shape)
        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T):
        x_t = x_T
        ct_center = x_t[:,0,:,:].unsqueeze(1)
        cond = x_t[:,1,:,:].unsqueeze(1)

        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t)
            noise = torch.randn_like(ct_center) if time_step > 0 else 0
            ct_center = mean + torch.sqrt(var) * noise
            x_t = torch.cat((ct_center, cond), 1)
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

