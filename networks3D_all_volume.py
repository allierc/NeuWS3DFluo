# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.fft import fft2, fftshift
from prettytable import PrettyTable

import numpy as np
import torchvision.transforms
from utils import compute_zernike_basis, fft_2xPad_Conv2D
import tifffile
import matplotlib.pyplot as plt

from BPM3Dfluo_V1 import *


class sine_act(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.sin(x)
        return out
class G_Renderer(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=32, num_layers=2, out_dim=1):
        super().__init__()

        in_dim = in_dim + 11 #z_embedding

        act_fn = nn.ReLU()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                ll=nn.Linear(in_dim, hidden_dim)
                # nn.init.normal_(ll.weight, std=0.5)
                # nn.init.zeros_(ll.bias)
                layers.append(ll)
            else:
                ll=nn.Linear(hidden_dim, hidden_dim)
                # nn.init.normal_(ll.weight, std=0.5)
                # nn.init.zeros_(ll.bias)
                layers.append(ll)
            #layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
        ll=nn.Linear(hidden_dim, out_dim)
        # nn.init.normal_(ll.weight, std=0.5)
        # nn.init.zeros_(ll.bias)
        layers.append(ll)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)

        return out

class G_FeatureTensor(nn.Module):
    def __init__(self, x_dim, y_dim, num_feats = 32, ds_factor = 1):
        super().__init__()
        self.x_dim, self.y_dim = x_dim, y_dim
        x_mode, y_mode = x_dim // ds_factor, y_dim // ds_factor
        self.num_feats = num_feats

        self.data = nn.Parameter(
            0.5 * torch.randn((x_mode, y_mode, num_feats)), requires_grad=True)

        self.data.data=torch.abs(self.data.data)

        half_dx, half_dy =  0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1-half_dx, x_dim)
        ys = torch.linspace(half_dx, 1-half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()

        xs = xy * torch.tensor([x_mode, y_mode], device=xs.device).float()
        indices = xs.long()
        self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)   #torch.Size([16384, 2]) =0.5

        self.x0 = nn.Parameter(indices[:, 0].clamp(min=0, max=x_mode-1), requires_grad=False)       #torch.Size([16384])
        self.y0 = nn.Parameter(indices[:, 1].clamp(min=0, max=y_mode-1), requires_grad=False)       #torch.Size([16384])
        self.x1 = nn.Parameter((self.x0 + 1).clamp(max=x_mode-1), requires_grad=False)              #torch.Size([16384])
        self.y1 = nn.Parameter((self.y0 + 1).clamp(max=y_mode-1), requires_grad=False)              #torch.Size([16384])

    def sample(self):

        return (self.data[self.x0, self.y0])        # no input mixing


    def forward(self):
        return self.sample()

class G_Tensor(G_FeatureTensor):
    def __init__(self, x_dim, y_dim=None):
        if y_dim is None:
            y_dim = x_dim
        super().__init__(x_dim, y_dim)
        self.renderer = G_Renderer()

    def forward(self):

        feats = self.sample()
        # torch.Size([16384, 32]) 128*128*32 requires_grad True

        return self.renderer(feats).reshape([-1, 1, self.x_dim, self.y_dim])

class G_PatchTensor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net1 = G_Tensor(width, width)

    def forward(self):
        p1 = self.net1()        # torch.Size([1, 1, 128, 128]) requires_grad=True

        return p1

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class G_SpaceTime(nn.Module):
    def __init__(self, x_width, y_width, bsize=8):
        super().__init__()

        hidden_dim, num_hidden_layers = 32, 3

        self.spatial_net = G_FeatureTensor(x_width, y_width, hidden_dim, ds_factor=1)

        self.x_width, self.y_width = x_width, y_width

        self.t0 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.t1 = nn.Parameter(torch.randn(1), requires_grad=True)

        num_t_freq = 5
        self.embedding_t = Embedding(1, num_t_freq) 
        #t_dim = 1 + num_t_freq * 2 + 2
        t_dim = 1 + 2

        act_fn = sine_act()   #nn.LeakyReLU(inplace=True)
        layers = []
        layers.append(nn.Linear(t_dim, hidden_dim))
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, 2))

        self.warp_net = nn.Sequential(*layers)

        xs = torch.linspace(-1, 1, steps=x_width)
        ys = torch.linspace(-1, 1, steps=y_width)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        self.xy_basis = nn.Parameter(
            torch.stack([x, y], axis=-1).unsqueeze(0).repeat(bsize, 1, 1, 1),
            requires_grad=False
        )
        self.renderer = G_Renderer(in_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=3)


    def forward(self, t):

        spatial_feats = self.spatial_net().unsqueeze(0).repeat(1, 1, 1)
        spatial_feats = spatial_feats.reshape(-1, self.x_width, self.y_width, spatial_feats.shape[-1])
        spatial_feats = spatial_feats.permute(0, 3, 1, 2)

        alpha = t
        t_emb = alpha * torch.ones_like(self.t1)
        #t_emb = (t).unsqueeze(-1) * torch.ones_like(self.t1)
        t_emb = t_emb.unsqueeze(-2).unsqueeze(-2).repeat(1, self.x_width, self.y_width, 1)
        t_emb = self.embedding_t(t_emb)
        t_emb = t_emb.permute(0, 3, 1, 2)

        t_input = torch.cat([spatial_feats, t_emb], dim=1)

        output = self.renderer(t_input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        output = F.leaky_relu(output, 0.001)

        return output

class TemporalZernNet(nn.Module):
    def __init__(self, width, PSF_size, phs_layers = 2, use_FFT=True, bsize=8, use_pe=False, static_phase=True, phs_draw=10):
        super().__init__()

        # self.g_im = G_PatchTensor(width)
        # self.dn_im = G_PatchTensor(width)

        self.g_im = G_SpaceTime(width, width, bsize)        # time is Z
        self.dn_im = G_SpaceTime(width, width, bsize)

        if not use_pe:
            print ('use Zernike')
            self.basis = nn.Parameter(compute_zernike_basis(
                num_polynomials=28,
                field_res=(PSF_size, PSF_size)).permute(1, 2, 0).unsqueeze(0).repeat(bsize, 1, 1, 1),
                requires_grad=False)
        else:
            print('use torch.sin')
            xs = torch.linspace(-1, 1, steps=PSF_size)
            ys = torch.linspace(-1, 1, steps=PSF_size)
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            basis = []
            for i in range(1, 16):
                basis.append(torch.sin(i * x))
                basis.append(torch.sin(i * y))
                basis.append(torch.cos(i * x))
                basis.append(torch.cos(i * y))
            self.basis = nn.Parameter(
                torch.stack(basis, axis=-1).unsqueeze(0).repeat(bsize, 1, 1, 1),
                requires_grad=False
            )

        hidden_dim = 32
        t_dim = 1 if not static_phase else 0
        in_dim = self.basis.shape[-1]

        self.t_grid = nn.Parameter(torch.ones_like(self.basis[..., 0:1]), requires_grad=False)
        self.use_FFT = use_FFT
        self.static_phase = static_phase

        self.bpm_config = {'nm': 1.33,
                        'na': 1.0556,
                        'dx': 0.1154,
                        'dz': 0.1,
                        'lbda': 0.5320,
                        'volume': [256, 256, 256],
                        'padding': False,
                        'dtype': torch.float32,
                        'device': 'cuda',
                        'phs_draw': phs_draw,
                        'load_data': False}

        self.Niter = phs_draw                                           # Numer of random draw for the phase of the fluorescence emitters
        self.bpm = bpm3Dfluo_all_volume(bpm_config=self.bpm_config)     # forward model of light propagation
        self.phiL = torch.rand([self.bpm.Nx, self.bpm.Ny, self.Niter*50], dtype=torch.float32, requires_grad=False, device='cuda:0') * 2 * np.pi        # random draw of phase images

    def forward(self, x_batch, t):

        F_estimated, Phi_estimated = self.get_estimates(t)

        for w in range(0, self.Niter):
            zoi = np.random.randint(self.Niter*50)
            if w==0:
                I=self.bpm(plane=int(t), phi=self.phiL[:, :, zoi], gamma=x_batch, fluo_unknown=F_estimated, dn_unknown=Phi_estimated)
            else:
                I += self.bpm(plane=int(t), phi=self.phiL[:, :, zoi], gamma=x_batch, fluo_unknown=F_estimated, dn_unknown=Phi_estimated)

        I = I/self.Niter

        return I, F_estimated, Phi_estimated

class StaticDiffuseNet(TemporalZernNet):
    def __init__(self, width, PSF_size, phs_layers = 2, use_FFT=True, bsize=8, use_pe=False, static_phase=True, phs_draw=10):
        super().__init__(width, PSF_size, phs_layers = phs_layers, use_FFT=use_FFT, bsize=bsize, use_pe=use_pe, phs_draw=phs_draw)

        hidden_dim = 32
        t_dim = 1 if not static_phase else 0
        in_dim = self.basis.shape[-1]
        act_fn = nn.LeakyReLU(inplace=True)

        self.static_phase = static_phase


    def get_estimates(self, t):

        I_est = self.g_im(t)     #torch.Size([1, 1, 256, 256])  requires_grad=True

        Phi_estimated = torch.zeros(256,256,256,device='cuda:0')

        for dn_plane in range (int(t*256),256):
                Phi_estimated[dn_plane,:,:] = torch.squeeze(self.dn_im(dn_plane/256))

        return torch.squeeze(I_est), torch.squeeze(Phi_estimated)




