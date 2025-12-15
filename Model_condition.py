
# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech

"""
3D conditional UNet for diffusion-based CT reconstruction.

Builds all components: time embedding, ResBlocks, attention,
down/upsampling, and assembles them into a full UNet that
takes (noisy CT + CBCT) and predicts CT noise (epsilon).
"""

import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    """
    Swish activation function.
    """ 
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    Used to inject the current diffusion step t into the UNet.
    Sinusoidal + MLP (Multilayer Perceptron) time embedding, (1D -> [B, tdim]).
    """
    def __init__(self, T, d_model, dim):
        """
        T: number of diffusion steps
        d_model: dimension of the embedding
        dim: dimension of the output embedding
        """
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        # t: [B]
        emb = self.timembedding(t)
        return emb  # [B, dim]


class DownSample(nn.Module):
    """
    3D downsampling: keep depth D, downsample H,W by 2.
    """
    def __init__(self, in_ch): #in_ch: input channel
        super().__init__()
        self.main = nn.Conv3d(in_ch, in_ch, 3, stride=(1, 2, 2), padding=1) #[B, in_ch, D, H, W] -> [B, in_ch, D, H/2, W/2]
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias) 

    def forward(self, x, temb):
        # x: [B, C, D, H, W]
        x = self.main(x) 
        return x


class UpSample(nn.Module):
    """
    3D upsampling: keep depth D, upsample H,W by 2.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv3d(in_ch, in_ch, 3, stride=1, padding=1) 
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        # x: [B, C, D, H, W]
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear") #trilinear neighbor interpolation
        x = self.main(x) #[B, C, D, 2*H, 2*W] -> [B, C, D, 2*H, 2*W]
        return x


class AttnBlock(nn.Module):
    """
    3D self-attention over all voxels (D*H*W).
    Add global context to the UNet features by attending across all voxels.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch) #normalize input
        self.proj_q = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)  # [B, C, D, H, W]
        k = self.proj_k(h)
        v = self.proj_v(h)

        # Flatten spatial dims: N = D*H*W
        N = D * H * W #reshape the tensor so that each voxel becomes one token for attention
        q = q.view(B, C, N).permute(0, 2, 1)   # [B, N, C]
        k = k.view(B, C, N)                    # [B, C, N]
        v = v.view(B, C, N).permute(0, 2, 1)   # [B, N, C]

        # Compute attention weights
        w = torch.bmm(q, k) * (C ** -0.5)      # [B, N, N]
        w = F.softmax(w, dim=-1)

        # Apply attention weights to values
        h = torch.bmm(w, v)                    # [B, N, C]
        h = h.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        h = self.proj(h)

        return x + h #residual connection


class ResBlock(nn.Module):
    """
    3D residual block with time embedding and optional attention.
    """
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        """
        in_ch: input channel
        out_ch: output channel
        tdim: time embedding dimension
        dropout: dropout rate
        attn: whether to use attention
        """
        super().__init__()
        self.block1 = nn.Sequential( #first block: normalize input, apply swish activation, and apply 3D convolution
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv3d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential( #second block: normalize input, apply swish activation, apply dropout, and apply 3D convolution
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch: #if input channel is not equal to output channel, apply 1x1 convolution to match the dimensions
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        # x: [B, C, D, H, W], temb: [B, tdim]
        h = self.block1(x)
        # Add time embedding (broadcast over D,H,W)
        h += self.temb_proj(temb)[:, :, None, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    """
    Builds the UNet architecture for the conditional diffusion model.
    3D conditional UNet:
      input x: [B, 2, D, H, W]
        - channel 0: noisy CT
        - channel 1: CBCT (conditioning)
      output: [B, 1, D, H, W] (epsilon for CT channel)
    """
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        # 2 channels: CT + CBCT
        self.head = nn.Conv3d(2, ch, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(
                        in_ch=now_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn),
                    )
                )
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList(
            [
                ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
                ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
            ]
        )

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(
                        in_ch=chs.pop() + now_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn),
                    )
                )
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv3d(now_ch, 1, 3, stride=1, padding=1),
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        """
        x: [B, 2, D, H, W]
        t: [B] (long)
        returns: [B, 1, D, H, W] (predicted noise for CT)
        """
      #  print("[UNet] Input x:", x.shape)   # [B, 2, D, H, W]


        temb = self.time_embedding(t)  # [B, tdim]

        h = self.head(x)  # [B, ch, D, H, W]
        #print("[UNet] Head:", h.shape)
        hs = [h]
        for layer in self.downblocks:
            if isinstance(layer, ResBlock):
                h = layer(h, temb)
            else:
                h = layer(h, temb)  # DownSample ignores temb
            hs.append(h)

        for layer in self.middleblocks:
            h = layer(h, temb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, temb)
            else:
                h = layer(h, temb)  # UpSample ignores temb

        h = self.tail(h)
        assert len(hs) == 0
        #print("[UNet] Output:", h.shape)
        return h


