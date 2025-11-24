# ------------------------------------------------------------
# 3D UNet for Conditional DDPM
# Converted from original 2D repo by Junbo Peng
# ------------------------------------------------------------
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


# ---------------------------
# Swish
# ---------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# ---------------------------
# Time Embedding (unchanged)
# ---------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        super().__init__()
        assert d_model % 2 == 0

        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, t):
        return self.timembedding(t)


# ---------------------------
# Downsample 3D
# ---------------------------
class DownSample3D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv3d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, t_emb):
        return self.main(x)


# ---------------------------
# Upsample 3D
# ---------------------------
class UpSample3D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv3d(in_ch, in_ch, 3, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, t_emb):
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return self.main(x)


# ---------------------------
# 3D Attention Block
# ---------------------------
class AttnBlock3D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)

        self.q = nn.Conv3d(in_ch, in_ch, 1)
        self.k = nn.Conv3d(in_ch, in_ch, 1)
        self.v = nn.Conv3d(in_ch, in_ch, 1)
        self.proj_out = nn.Conv3d(in_ch, in_ch, 1)

        self.initialize()

    def initialize(self):
        for m in [self.q, self.k, self.v, self.proj_out]:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.group_norm(x)

        q = self.q(h).reshape(B, C, -1)            # [B, C, DHW]
        k = self.k(h).reshape(B, C, -1)            # [B, C, DHW]
        v = self.v(h).reshape(B, C, -1).permute(0, 2, 1)  # [B, DHW, C]

        w = torch.bmm(q.permute(0, 2, 1), k) * (C ** -0.5)  # [B, DHW, DHW]
        w = F.softmax(w, dim=-1)

        out = torch.bmm(w, v)                      # [B, DHW, C]
        out = out.permute(0, 2, 1).reshape(B, C, D, H, W)

        return x + self.proj_out(out)


# ---------------------------
# ResBlock 3D
# ---------------------------
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv3d(in_ch, out_ch, 3, padding=1)
        )

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
        )

        self.shortcut = (
            nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

        self.attn = AttnBlock3D(out_ch) if attn else nn.Identity()
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, t_emb):
        h = self.block1(x)
        h += self.temb_proj(t_emb)[:, :, None, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        return self.attn(h)


# ---------------------------
# Full UNet 3D
# ---------------------------
class UNet3D(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()

        assert all(i < len(ch_mult) for i in attn)
        tdim = ch * 4

        # Time embedding
        self.time_embed = TimeEmbedding(T, ch, tdim)

        # Input stem
        self.head = nn.Conv3d(2, ch, 3, padding=1)

        # ----------------- Down path ---------------------
        self.down = nn.ModuleList()
        chs = [ch]
        curr_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock3D(curr_ch, out_ch, tdim, dropout, attn=(i in attn)))
                curr_ch = out_ch
                chs.append(curr_ch)

            if i != len(ch_mult) - 1:
                self.down.append(DownSample3D(curr_ch))
                chs.append(curr_ch)

        # ----------------- Middle ------------------------
        self.mid = nn.ModuleList([
            ResBlock3D(curr_ch, curr_ch, tdim, dropout, attn=True),
            ResBlock3D(curr_ch, curr_ch, tdim, dropout, attn=False),
        ])

        # ----------------- Up path -----------------------
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(ResBlock3D(chs.pop() + curr_ch, out_ch, tdim, dropout, attn=(i in attn)))
                curr_ch = out_ch
            if i != 0:
                self.up.append(UpSample3D(curr_ch))

        # Output
        self.tail = nn.Sequential(
            nn.GroupNorm(32, curr_ch),
            Swish(),
            nn.Conv3d(curr_ch, 1, 3, padding=1)
        )

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)

    def forward(self, x, t):
        temb = self.time_embed(t)

        h = self.head(x)
        skip = [h]

        # ---- Down ----
        for layer in self.down:
            h = layer(h, temb)
            skip.append(h)

        # ---- Middle ----
        for layer in self.mid:
            h = layer(h, temb)

        # ---- Up ----
        for layer in self.up:
            if isinstance(layer, ResBlock3D):
                h = torch.cat([h, skip.pop()], dim=1)
            h = layer(h, temb)

        return self.tail(h)