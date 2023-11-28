# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

import numpy as np
import torchvision.transforms
from modules.utils import compute_zernike_basis, fft_2xPad_Conv2D

DEVICE = 'cuda'  # TODO: take device from the main script


class sine_act(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.sin(x)
        return out

# class G_Renderer(nn.Module):
#     def __init__(self, in_dim=32, hidden_dim=32, num_layers=2, out_dim=1):
#         super().__init__()
#         act_fn = nn.ReLU()
#         layers = []
#         for _ in range(num_layers):
#             if len(layers) == 0:
#                 layers.append(nn.Linear(in_dim, hidden_dim))
#             else:
#                 layers.append(nn.Linear(hidden_dim, hidden_dim))
#             # layers.append(nn.LayerNorm(hidden_dim))
#             layers.append(act_fn)
#         layers.append(nn.Linear(hidden_dim, out_dim))
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.net(x)
#         return out

class G_Renderer(nn.Module):
    def __init__(
        self, in_dim=32, hidden_dim=32, num_layers=2, out_dim=1, use_layernorm=False
    ):
        super().__init__()
        act_fn = nn.ReLU()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        layers.append(act_fn)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class G_Tensor3D(nn.Module):
    def __init__(
        self, x_mode, y_mode, z_mode, z_min, z_max, num_feats=32, use_layernorm=False
    ):
        super().__init__()
        self.x_mode, self.y_mode, self.num_feats = x_mode, y_mode, num_feats
        self.data = nn.Parameter(
            2e-4 * torch.randn((self.x_mode, self.y_mode, self.num_feats)),
            requires_grad=True,
        )
        self.renderer = G_Renderer(in_dim=self.num_feats, use_layernorm=use_layernorm)
        self.x0 = None

        self.z_mode = z_mode
        self.z_data = nn.Parameter(
            torch.randn((self.z_mode, self.num_feats)), requires_grad=True
        )
        self.z_min = z_min
        self.z_max = z_max
        self.z_mode = z_mode

    def create_coords(self, x_dim, y_dim, x_max, y_max):
        half_dx, half_dy = 0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dx, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        xs = xy * torch.tensor([x_max, y_max], device=xs.device).float()
        indices = xs.long()
        self.x_dim, self.y_dim = x_dim, y_dim
        self.xy_coords = nn.Parameter(
            xy[None],
            requires_grad=False,
        )

        if self.x0 is not None:
            device = self.x0.device
            self.x0.data = (indices[:, 0].clamp(min=0, max=x_max - 1)).to(device)
            self.y0.data = indices[:, 1].clamp(min=0, max=y_max - 1).to(device)
            self.x1.data = (self.x0 + 1).clamp(max=x_max - 1).to(device)
            self.y1.data = (self.y0 + 1).clamp(max=y_max - 1).to(device)
            self.lerp_weights.data = (xs - indices.float()).to(device)
        else:
            self.x0 = nn.Parameter(
                indices[:, 0].clamp(min=0, max=x_max - 1),
                requires_grad=False,
            )
            self.y0 = nn.Parameter(
                indices[:, 1].clamp(min=0, max=y_max - 1),
                requires_grad=False,
            )
            self.x1 = nn.Parameter(
                (self.x0 + 1).clamp(max=x_max - 1), requires_grad=False
            )
            self.y1 = nn.Parameter(
                (self.y0 + 1).clamp(max=y_max - 1), requires_grad=False
            )
            self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

    def normalize_z(self, z):
        return (self.z_mode - 1) * (z - self.z_min) / (self.z_max - self.z_min)

    def sample(self, z):
        z = self.normalize_z(z)
        z0 = z.long().clamp(min=0, max=self.z_mode - 1)
        z1 = (z0 + 1).clamp(max=self.z_mode - 1)
        zlerp_weights = (z - z.long().float())[:, None]

        xy_feat = (
            self.data[self.y0, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y0, self.x1]
            * self.lerp_weights[:, 0:1]
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y1, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * self.lerp_weights[:, 1:2]
            + self.data[self.y1, self.x1]
            * self.lerp_weights[:, 0:1]
            * self.lerp_weights[:, 1:2]
        )
        z_feat = (
            self.z_data[z0] * (1.0 - zlerp_weights) + self.z_data[z1] * zlerp_weights
        )
        z_feat = z_feat[:, None].repeat(1, xy_feat.shape[0], 1)
        feat = xy_feat[None].repeat(z.shape[0], 1, 1) * z_feat

        return feat

    def forward(self, z):
        feat = self.sample(z)

        out = self.renderer(feat)
        b = z.shape[0]
        w, h = self.x_dim, self.y_dim
        out = out.view(b, 1, w, h)
        return out

class G_Model3D(nn.Module):
    def __init__(
        self,
        w,
        h,
        num_feats,
        x_mode,
        y_mode,
        z_mode,
        z_min,
        z_max,
    ):
        """
        Args:
            w, h (int): image resolution
            num_feats (int): number of features (e.g. 32)
            x_mode, y_mode (int): feature grid resolution across x-y (e.g. 512)
            z_mode (int): feature grid resolution along z (e.g. 64)
            z_min (float): mininum value of z, used in function normalize_z
            z_max (float): maximum value of z, used in function normalize_z
        """
        super().__init__()
        self.img_1 = G_Tensor3D(
            x_mode=x_mode,
            y_mode=y_mode,
            z_mode=z_mode,
            z_min=z_min,
            z_max=z_max,
            num_feats=num_feats,
        )
        self.img_2 = G_Tensor3D(
            x_mode=x_mode,
            y_mode=y_mode,
            z_mode=z_mode,
            z_min=z_min,
            z_max=z_max,
            num_feats=num_feats,
        )
        self.w, self.h = w, h
        self.init_grids()

    def init_grids(self):
        self.img_1.create_coords(
            x_dim=self.w,
            y_dim=self.h,
            x_max=self.img_real.x_mode,
            y_max=self.img_real.y_mode,
        )
        self.img_2.create_coords(
            x_dim=self.w,
            y_dim=self.h,
            x_max=self.img_imag.x_mode,
            y_max=self.img_imag.y_mode,
        )

    def forward(self, z):
        # G_Tensor3D takes in a z value and returns the predicted 1-channel image of shape (w, h)
        img_1 = self.img_1(z)
        img_2 = self.img_2(z)

        return img_1, img_2


class G_FeatureTensor(nn.Module):
    def __init__(self, x_dim, y_dim, num_feats=32, ds_factor=1):
        super().__init__()
        self.x_dim, self.y_dim = x_dim, y_dim
        x_mode, y_mode = x_dim // ds_factor, y_dim // ds_factor
        self.num_feats = num_feats

        self.data = nn.Parameter(0 * torch.randn((x_mode, y_mode, num_feats)), requires_grad=True)
        # self.data = nn.Parameter(0.1 * torch.randn((x_mode, y_mode, num_feats)), requires_grad=True)

        half_dx, half_dy = 0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dx, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()

        xs = xy * torch.tensor([x_mode, y_mode], device=xs.device).float()
        indices = xs.long()
        self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

        self.x0 = nn.Parameter(indices[:, 0].clamp(min=0, max=x_mode - 1), requires_grad=False)
        self.y0 = nn.Parameter(indices[:, 1].clamp(min=0, max=y_mode - 1), requires_grad=False)
        self.x1 = nn.Parameter((self.x0 + 1).clamp(max=x_mode - 1), requires_grad=False)
        self.y1 = nn.Parameter((self.y0 + 1).clamp(max=y_mode - 1), requires_grad=False)

    def sample(self):
        # Note: mixing is implementing bilinear interpolation
        # (obtain the features for continuous (x, y) from a discrete feature grid)
        return (
                self.data[self.y0, self.x0] * (1.0 - self.lerp_weights[:, 0:1]) * (1.0 - self.lerp_weights[:, 1:2]) +
                self.data[self.y0, self.x1] * self.lerp_weights[:, 0:1] * (1.0 - self.lerp_weights[:, 1:2]) +
                self.data[self.y1, self.x0] * (1.0 - self.lerp_weights[:, 0:1]) * self.lerp_weights[:, 1:2] +
                self.data[self.y1, self.x1] * self.lerp_weights[:, 0:1] * self.lerp_weights[:, 1:2]
        )

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
        return self.renderer(feats).reshape([-1, 1, self.x_dim, self.y_dim])


class G_PatchTensor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = G_Tensor(width, width)

    def forward(self):
        p = self.net()
        return p


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
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

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
                out += [func(freq * x)]

        return torch.cat(out, -1)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(MLP, self).__init__()

        act_fn = nn.LeakyReLU(inplace=True)

        self.layers = nn.ModuleList()

        first_layer = nn.Linear(in_dim, hidden_dim)
        nn.init.normal_(first_layer.weight, std=0.4)
        nn.init.zeros_(first_layer.bias)
        self.layers.append(first_layer)
        for _ in range(num_layers):
            new_layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.normal_(new_layer.weight, std=0.4)
            nn.init.zeros_(new_layer.bias)
            self.layers.append(new_layer)
            # layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(act_fn)
        last_layer = nn.Linear(hidden_dim, 2)
        nn.init.normal_(last_layer.weight, std=0.4)
        nn.init.zeros_(last_layer.bias)
        self.layers.append(last_layer)

    def forward(self, x):

        for l in range(len(self.layers)):
            x = self.layers[l](x)
        return x


class ZernNet(nn.Module):
    def __init__(self, zernike, pupil, width, PSF_size, phs_layers=2, use_FFT=True, bsize=8, use_pe=False,
                 num_polynomials_gammas=20, static_phase=True, acquisition_data=[]):
        super().__init__()

        self.g_im = G_PatchTensor(width)

        if not use_pe:
            zernike_basis = zernike.calculate_polynomials(np.arange(3, 3 + num_polynomials_gammas))
            zernike_basis = torch.FloatTensor(zernike_basis).to(DEVICE)
            self.basis = nn.Parameter(zernike_basis.unsqueeze(0).repeat(bsize, 1, 1, 1), requires_grad=False)

        else:
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
        t_dim = 0  # 0 in static case # TODO: remove t_dim
        in_dim = self.basis.shape[-1]
        self.t_grid = nn.Parameter(torch.ones_like(self.basis[..., 0:1]), requires_grad=False)
        self.use_FFT = use_FFT
        print(f'Using FFT approximation of convolution: {self.use_FFT}')
        self.static_phase = static_phase

    def forward(self, object_gt, phase_aberration_gt):
        object_est, total_aberration_est, phase_aberration_est, gammas, gammas_raw = self.get_estimates()

        # Estimated_acquisition
        acquisition_est = self.calculate_image(object_est, phase_aberration_est + gammas, self.pupil)

        # Ground truth acquisition
        acquisition_gt = self.calculate_image(object_gt, phase_aberration_gt + gammas, self.pupil)

        total_aberration_est = torch.squeeze(total_aberration_est)
        kernel = []
        return acquisition_est, acquisition_gt, gammas, kernel, total_aberration_est, phase_aberration_est, object_est, gammas_raw

    def calculate_image(self, object, aberration, pupil):
        pupil_function = pupil.values * torch.exp(1j * aberration)
        prf = fftshift(ifft2(ifftshift(pupil_function, dim=(1, 2))), dim=(1, 2))
        PSF = torch.abs(prf) ** 2
        PSF = PSF / torch.sum(PSF, dim=(1, 2), keepdim=True)
        OTF = fft2(ifftshift(PSF, dim=(1, 2)))
        object_ft = fft2(object, dim=(1, 2))
        image_fourier_space = object_ft * OTF
        image = torch.real(ifft2(image_fourier_space, dim=(1, 2)))
        # Scale to [0, 1]
        # image = (image - torch.min(image, dim=(1,2), keepdim=True)[0]) / (torch.max(image, dim=(1,2), keepdim=True)[0] - torch.min(image, dim=(1,2), keepdim=True)[0])
        return image


class StaticNet(ZernNet):
    def __init__(self, zernike, pupil, width, PSF_size, phs_layers=2, use_FFT=True, bsize=8, use_pe=False,
                 static_phase=True, n_gammas=5, input_gammas_zernike=[], b_gamma_optimization=False,
                 num_polynomials_gammas=20,acquisition_data=[], optimize_phase_diversities_with_mlp=True):
        super().__init__(zernike, pupil, width, PSF_size, phs_layers=phs_layers, use_FFT=use_FFT, bsize=bsize,
                         use_pe=use_pe, num_polynomials_gammas=num_polynomials_gammas, acquisition_data=acquisition_data)
        self.n_gammas = n_gammas
        self.pupil = pupil
        self.zernike = zernike
        self.bfp_size = PSF_size
        self.acquisition_data=acquisition_data
        self.optimize_phase_diversities_with_mlp = optimize_phase_diversities_with_mlp

        hidden_dim = 32
        in_dim = self.basis.shape[-1]
        act_fn = nn.LeakyReLU(inplace=True)

        self.g_g = MLP(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=phs_layers)  # phase aberrations

        self.gammas_nnr = torch.nn.ModuleList(
            [MLP(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=phs_layers) for _ in range(n_gammas + 1)])

        self.static_phase = static_phase

        if len(input_gammas_zernike) > 0:
            self.gammas_zernike = nn.Parameter(torch.tensor(input_gammas_zernike, device=DEVICE).float(),
                                               requires_grad=b_gamma_optimization)
        else:
            self.gammas_zernike = []

    def get_estimates(self):

        object_est = self.g_im()  # Neural representation of estimated object


        object_est = object_est.squeeze().repeat(self.n_gammas + 1, 1, 1)

        g_in = self.basis[0]  # Zernike basis 1x15

        g_out_gammas = []
        g_out_gammas.append(self.gammas_nnr[0])  # Flat phase

        for i in range(1, self.n_gammas + 1):
            g_out_gammas.append(self.gammas_nnr[i](g_in))  # dim 1x256x256

        g_in = self.basis  # Zernike basis 6x15
        g_out_phi = self.g_g(
            g_in)  # Neural representation of aberration from Zernike basis  dim 6 256 256 all identical

        g_out_phi = g_out_phi.permute(0, 3, 1, 2)
        phase_aberration_est = g_out_phi[:, 1:2] * self.pupil.values
        amplitude_aberration_est = g_out_phi[:, 0:1]  # NOTE: Amplitude should be 1 in our case! Maybe remove it
        total_aberration_est = amplitude_aberration_est * torch.exp(1j * phase_aberration_est)
        phase_aberration_est = phase_aberration_est.squeeze()

        if self.optimize_phase_diversities_with_mlp:
            # Optimized phase diversities (gammas)
            gammas = torch.zeros(self.n_gammas + 1, 1, self.bfp_size, self.bfp_size, device=DEVICE)
            gammas_raw = []
            for i in range(1, self.n_gammas + 1):
                g = self.gammas_nnr[i](g_in[0])
                g = g.permute(2, 0, 1)
                gammas[i, :, :, :] = g[1:2] * self.pupil.values
                gammas_raw.append(g[1:2])
            gammas = gammas.squeeze()
        else:
            # TODO: assert that gammas_zernike is greater than 0 (len(self.gammas_zernike) > 0)
            # Optimize Zernike coefficients for gammas without MLP
            gammas = torch.zeros(self.n_gammas + 1, 1, self.bfp_size, self.bfp_size, device=DEVICE)
            gammas_raw = []
            for i in range(1, self.n_gammas + 1):
                g = g_in[0] * self.gammas_zernike[i].repeat(self.bfp_size, self.bfp_size, 1)
                g = torch.sum(g, axis=2).float()
                gammas[i, :, :, :] = g * self.pupil.values
                gammas_raw.append(g)
            gammas = gammas.squeeze()

        return object_est, total_aberration_est, phase_aberration_est, gammas, gammas_raw

    # TODO: the forward function is inherited from the parent class!
    # TODO: does it make sense to make StaticDiffuseNet a child class of TemporalZernNet
