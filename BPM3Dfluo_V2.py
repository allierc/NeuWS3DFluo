import math
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tifffile
import torch.nn as nn
import torch.nn.functional as nf
import torch
from numpy.fft import fft2
from tqdm import tqdm
from timeit import default_timer as timer
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io
import json
from tifffile import imwrite
from tifffile import imread
import logging
from shutil import copyfile
import glob
from astropy import units as u
from torch.fft import fft2, ifft2, fftshift, ifftshift

class G_Renderer(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim, device):
        super().__init__()

        act_fn = nn.ReLU()
        layers = []

        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(in_dim, hidden_dim,device=device))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim,device=device))
            #layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, out_dim,device=device))
        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)

        # table = PrettyTable(["Modules", "Parameters"])
        # total_params = 0
        # for name, parameter in self.net.named_parameters():
        #     if not parameter.requires_grad:
        #         continue
        #     param = parameter.numel()
        #     table.add_row([name, param])
        #     total_params += param
        # print(table)
        # print(f"Total Trainable Params: {total_params}")

class bpmPytorch(torch.nn.Module):

    def __init__(self, model_config):

        super(bpmPytorch, self).__init__()

        self.nm = model_config['nm']
        self.numerical_aperture = model_config['numerical_aperture']
        p_num, p_unit = model_config['pixel_size'].split()
        self.pixel_size = float(p_num)
        p_num, p_unit = model_config['dz'].split()
        self.dz = float(p_num)
        p_num, p_unit = model_config['wavelength'].split()
        self.wavelength = float(p_num)
        self.volume = model_config['volume']
        self.padding = model_config['padding']
        self.device = model_config['device']
        self.dtype = torch.float32
        self.bAber = model_config['bAber']
        self.num_feats = model_config['num_feats']
        self.bFit = model_config['bFit']
        self.dn_factor = model_config['dn_factor']
        self.n_gammas = model_config['n_gammas']

        self.renderer = G_Renderer(in_dim=self.num_feats, hidden_dim=self.num_feats, num_layers=2, out_dim=1, device=self.device)

        self.image_width = self.volume[0]
        self.Nz = self.volume[2]

        self.f = nn.Parameter(torch.tensor(np.zeros([256, 256]),  device=self.device, dtype=self.dtype, requires_grad=True))

        self.Npixels = self.image_width * self.image_width
        self.field_shape = (self.image_width, self.image_width)

        null_obj = np.zeros(( self.image_width, self.image_width, self.Nz,))
        self.dn0 = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
        self.fluo = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
        self.aber = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)

        new_obj = imread(model_config['dn_path']) * self.dn_factor
        new_obj = np.moveaxis(new_obj, 0, -1)
        self.dn0 = torch.tensor(new_obj, device=self.device, dtype=self.dtype, requires_grad=False)

        new_obj = imread(model_config['fluo_path'])
        if new_obj.ndim==2:
            self.fluo = torch.tensor(new_obj, device=self.device, dtype=self.dtype, requires_grad=False)
            self.fluo = self.fluo.repeat(self.Nz,1,1)
            self.fluo = torch.moveaxis(self.fluo, 0,-1)
        else:
            new_obj = np.moveaxis(new_obj, 0, -1)
            self.fluo = torch.tensor(new_obj, device=self.device, dtype=self.dtype, requires_grad=False)

        self.dn0_layers = self.dn0.unbind(dim=2)
        self.fluo_layers = self.fluo.unbind(dim=2)

        N_x = np.arange(-self.image_width / 2 + 0, self.image_width / 2 - 0)
        N_y = np.arange(-self.image_width / 2 + 0, self.image_width / 2 - 0)
        x_range = self.image_width * self.pixel_size
        y_range = self.image_width * self.pixel_size
        mux = np.fft.fftshift(N_x / x_range).reshape(1, -1)
        muy = np.fft.fftshift(N_y / y_range).reshape(-1, 1)
        self.mux = torch.tensor(mux, dtype=self.dtype, requires_grad=False, device=self.device)
        self.muy = torch.tensor(muy, dtype=self.dtype, requires_grad=False, device=self.device)

        self.Hz_down = self.FresnelPropag(dz=self.dz, fdir=[0,0])
        self.Hz_up = self.FresnelPropag(dz=-self.dz, fdir=[0,0])

        fdir= [0,0]
        mux_inc = (self.mux - fdir[0])
        muy_inc = (self.muy - fdir[1])
        munu = torch.sqrt(mux_inc ** 2 + muy_inc ** 2).reshape(self.image_width, self.image_width, 1)
        pupil = np.squeeze((munu < (self.numerical_aperture / self.wavelength)).float())
        imwrite('./pupil.tif', np.array(pupil.cpu()))
        self.pupil = torch.complex(
            torch.ones(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False) * pupil,
            torch.ones(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False) * 0)

        C = (self.wavelength/self.nm) * torch.sqrt((self.nm/self.wavelength)**2 - mux_inc ** 2 - muy_inc ** 2).reshape(self.image_width, self.image_width)
        C = 1/C*pupil
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        imwrite('./C.tif', np.array(C.cpu()))
        self.C = C

    def forward(self, plane=None, phi=None, naber=0):

        k0 = 2 * np.pi / self.wavelength

        self.field = torch.complex(
            torch.zeros(self.field_shape, dtype=torch.float32, device=self.device),
            torch.zeros(self.field_shape, dtype=torch.float32, device=self.device))

        coef=torch.tensor(self.dz*k0*1.j, dtype=torch.cfloat, requires_grad=False, device=self.device)

        self.fluo_layer = self.fluo_layers[plane]

        phi_layers=phi.unbind(dim=2)

        for i in range(plane,self.Nz):
            if (i==plane):
                S = torch.mul(self.fluo_layer,torch.exp(phi_layers[i]*1.j))
                S = ifft2(torch.mul(fft2(S),self.C))
                self.field = torch.mul(S, torch.exp(torch.mul(self.dn0_layers[i], coef)))
            else:
                self.field = torch.mul(self.field, torch.exp(torch.mul(self.dn0_layers[i], coef)))
            self.field = ifft2(torch.mul(fft2(self.field), self.Hz_down))

        self.field = fft2(self.field)

        for i in range(plane,self.Nz):
            self.field = torch.mul(self.field,self.Hz_up)

        self.field = self.field.repeat((self.n_gammas+1, 1, 1))
        return torch.abs(ifft2(self.field * self.gammas)) ** 2

    def FresnelPropag(self, dz=0, fdir=[0, 0]):

        K = (self.nm / self.wavelength) ** 2 - (self.mux - fdir[0]) ** 2 - (self.muy - fdir[1]) ** 2
        if dz <= 0:
            K = torch.complex(torch.sqrt(nf.relu(-K)), torch.sqrt(nf.relu(K)))
        else:
            K = torch.complex(-torch.sqrt(nf.relu(-K)), torch.sqrt(nf.relu(K)))
        hz = torch.exp(2 * np.pi * dz * K)

        return hz
