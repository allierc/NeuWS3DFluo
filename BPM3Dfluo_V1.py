import math
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tifffile
import torch.nn as nn
import torch.nn.functional as nf
import torch
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


class bpm3Dfluo(torch.nn.Module):

    def __init__(self, bpm_config):

        super(bpm3Dfluo, self).__init__()
        self.mc = bpm_config

        self.device = self.mc['device']
        self.dtype = self.mc['dtype']
        self.nm = self.mc['nm']
        self.na = self.mc['na']
        self.dx = self.mc['dx']
        self.dz = self.mc['dz']
        self.lbda = self.mc['lbda']
        self.volume = self.mc['volume']
        self.padding = self.mc['padding']
        self.load_data  = self.mc['load_data']

        self.Nx = self.volume[0]
        self.Ny = self.volume[1]
        self.Nz = self.volume[2]

        self.Npixels = self.Nx * self.Ny
        self.field_shape = (self.Nx, self.Ny)

        null_obj = np.zeros((self.Nx, self.Ny, self.Nz,))
        self.dn0 = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
        self.fluo = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)

        if self.load_data:
            new_obj = imread(f'./Pics_input/target.tif')
            new_obj = np.moveaxis(new_obj, 0, -1)
            self.fluo = torch.tensor(new_obj, device=self.device, dtype=self.dtype, requires_grad=False)
            new_obj = imread(f'./Pics_input/dn.tif')
            new_obj = np.moveaxis(new_obj, 0, -1)
            self.dn0 = torch.tensor(new_obj, device=self.device, dtype=self.dtype, requires_grad=False)

        self.dn0_layers = self.dn0.unbind(dim=2)
        self.fluo_layers = self.fluo.unbind(dim=2)

        N_x = np.arange(-self.Nx / 2 + 0, self.Nx / 2 - 0)
        N_y = np.arange(-self.Ny / 2 + 0, self.Ny / 2 - 0)
        x_range = self.Nx * self.dx
        y_range = self.Ny * self.dx
        mux = np.fft.fftshift(N_x / x_range).reshape(1, -1)
        muy = np.fft.fftshift(N_y / y_range).reshape(-1, 1)
        self.mux = torch.tensor(mux, dtype=self.dtype, requires_grad=False, device=self.device)
        self.muy = torch.tensor(muy, dtype=self.dtype, requires_grad=False, device=self.device)

        self.Hz1 = self.FresnelPropag(dz=self.dz, fdir=[0, 0])
        self.Hz2 = self.FresnelPropag(dz=-self.dz, fdir=[0, 0])
        self.Hz3 = self.FresnelPropag(dz=self.dz * 50, fdir=[0, 0])
        self.Hz4 = self.FresnelPropag(dz=-self.dz * 50, fdir=[0, 0])

        fdir = [0, 0]
        mux_inc = (self.mux - fdir[0])
        muy_inc = (self.muy - fdir[1])
        munu = torch.sqrt(mux_inc ** 2 + muy_inc ** 2).reshape(self.Nx, self.Ny, 1)
        pupil = np.squeeze((munu < (self.na / self.lbda)).float())
        # imwrite('./vis/pupil.tif', np.array(pupil.cpu()))
        self.pupil = torch.complex(
            torch.ones(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False) * pupil,
            torch.ones(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False) * 0)

        C = (self.lbda / self.nm) * torch.sqrt((self.nm / self.lbda) ** 2 - mux_inc ** 2 - muy_inc ** 2).reshape(self.Nx, self.Ny)
        C = 1 / C * pupil
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        imwrite('./vis/C.tif', np.array(C.cpu()))
        self.C = C

    def forward(self, plane=None, phi=None, gamma=None, fluo_unknown=None, dn_unknown=None):

        k0 = 2 * np.pi / self.lbda

        self.field = torch.complex(
            torch.zeros(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False),
            torch.zeros(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False))

        coef = torch.tensor(self.dz * k0 * 1.j, dtype=torch.cfloat, requires_grad=False, device=self.device)

        for i in range(plane,self.Nz):
            if (i==plane):
                S = torch.mul(fluo_unknown,torch.exp(phi*1.j))
                S = torch.fft.ifftn(torch.mul(torch.fft.fftn(S),self.C))
                self.field = torch.mul(S, torch.exp(torch.mul(dn_unknown, coef)))
            else:
                self.field = torch.mul(self.field, torch.exp(torch.mul(self.dn0_layers[i], coef)))
            self.field = torch.fft.ifftn(torch.mul(torch.fft.fftn(self.field), self.Hz1))

        self.field = torch.fft.fftn(self.field)

        for i in range(plane, self.Nz):
            self.field = torch.mul(self.field, self.Hz2)

        I = torch.abs(torch.fft.ifftn(self.field * gamma)) ** 2  # aberration is defined per volume (x256)


        return I

    def FresnelPropag(self, dz=0, fdir=[0, 0]):

        K = (self.nm / self.lbda) ** 2 - (self.mux - fdir[0]) ** 2 - (self.muy - fdir[1]) ** 2
        if dz <= 0:
            K = torch.complex(torch.sqrt(nf.relu(-K)), torch.sqrt(nf.relu(K)))
        else:
            K = torch.complex(-torch.sqrt(nf.relu(-K)), torch.sqrt(nf.relu(K)))
        hz = torch.exp(2 * np.pi * dz * K)

        return hz

def TV(o):
    nb_voxel = (o.shape[0]) * (o.shape[1])
    sx,sy= grads(o)
    TVloss = torch.sqrt(sx ** 2 + sy ** 2 + 1e-8).sum()
    return TVloss / (nb_voxel)

def grads(o):

    o=torch.squeeze(o)

    if len(o.shape)==2:
        o_sx = torch.roll(o, -1, 0)
        o_sy = torch.roll(o, -1, 1)

        sx = -(o - o_sx)
        sy = -(o - o_sy)

        sx[-1, :] = 0
        sy[:, -1] = 0

        return [sx,sy]

    elif len(o.shape)==3:
        o_sx = torch.roll(o, -1, 0)
        o_sy = torch.roll(o, -1, 1)
        o_sz = torch.roll(o, -1, 2)

        sx = -(o - o_sx)
        sy = -(o - o_sy)
        sz = -(o - o_sz)

        sx[-1, :, :] = 0
        sy[:, -1, :] = 0
        sz[:, :, -1] = 0

        return [sx,sy,sz]
