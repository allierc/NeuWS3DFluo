import math
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tifffile
import torch.nn as nn
import torch.nn.functional as nf
import torch
from torch.fft import fft2, fftshift, irfftn, rfftn, ifftshift
from tqdm import tqdm
from timeit import default_timer as timer
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io
import json
from tifffile import imwrite
from tifffile import imread
import logging

from utils import compute_zernike_basis, fft_2xPad_Conv2D
class bpmPytorch(torch.nn.Module):

    def __init__(self, model_config, coeff_RI=False):

        super(bpmPytorch, self).__init__()
        self.mc = model_config

        self.nm = self.mc['nm']
        self.na = self.mc['na']
        self.dx = self.mc['dx']
        self.dz = self.mc['dz']
        self.lbda = self.mc['lbda']

        self.volume = self.mc['volume']
        self.padding = self.mc['padding']

        self.device = self.mc['device']
        self.dtype = self.mc['dtype']

        if self.padding:
            self.Nx = self.volume[0] *2
            self.Ny = self.volume[1] *2
            self.Nz = self.volume[2]

            self.Npixels = self.Nx * self.Ny
            self.field_shape = (self.Nx, self.Ny)

            null_obj = np.zeros(( self.Nx, self.Ny, self.Nz,))
            self.dn0 = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
            self.fluo = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=True)
            self.aber = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=True)

        else:
            self.Nx = self.volume[0]
            self.Ny = self.volume[1]
            self.Nz = self.volume[2]

            self.Npixels = self.Nx * self.Ny
            self.field_shape = (self.Nx, self.Ny)

            null_obj = np.zeros(( self.Nx, self.Ny, self.Nz,))
            self.dn0 = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
            self.fluo = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=True)
            self.aber = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=True)

            # new_obj = io.imread(self.data_path)
            # new_obj = np.moveaxis(new_obj, 0, -1)
            # self.dn0 = torch.tensor(new_obj * coeff_RI, device=self.device, dtype=self.dtype, requires_grad=False)
            # self.aber = torch.tensor(new_obj, device=self.device, dtype=self.dtype, requires_grad=True)
            # fluo_obj = io.imread(self.fluo_data_path)
            # fluo_obj = np.moveaxis(fluo_obj, 0, -1)
            # self.fluo = torch.tensor(fluo_obj, device=self.device, dtype=self.dtype, requires_grad=True)
            # self.fluo = torch.sqrt(self.fluo)

        N_x = np.arange(-self.Nx / 2 + 0, self.Nx / 2 - 0)
        N_y = np.arange(-self.Ny / 2 + 0, self.Ny / 2 - 0)
        x_range = self.Nx * self.dx
        y_range = self.Ny * self.dx
        mux = np.fft.fftshift(N_x / x_range).reshape(1, -1)
        muy = np.fft.fftshift(N_y / y_range).reshape(-1, 1)
        self.mux = torch.tensor(mux, dtype=self.dtype, requires_grad=False, device=self.device)
        self.muy = torch.tensor(muy, dtype=self.dtype, requires_grad=False, device=self.device)

        self.Hz1 = self.FresnelPropag(dz=self.dz, fdir=[0,0])
        self.Hz2 = self.FresnelPropag(dz=-self.dz, fdir=[0,0])
        self.Hz3 = self.FresnelPropag(dz=self.dz*50, fdir=[0,0])
        self.Hz4 = self.FresnelPropag(dz=-self.dz*50, fdir=[0,0])

        fdir= [0,0]
        mux_inc = (self.mux - fdir[0])
        muy_inc = (self.muy - fdir[1])
        munu = torch.sqrt(mux_inc ** 2 + muy_inc ** 2).reshape(self.Nx, self.Ny, 1)
        pupil = np.squeeze((munu < (self.na / self.lbda)).float())
        # imwrite('./temp/pupil.tif', np.array(pupil.cpu()))
        self.pupil = torch.complex(
            torch.ones(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False) * pupil,
            torch.ones(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False) * 0)

        C = (self.lbda/self.nm) * torch.sqrt((self.nm/self.lbda)**2 - mux_inc ** 2 - muy_inc ** 2).reshape(self.Nx, self.Ny)
        C = 1/C*pupil
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        self.C = C

    def forward(self, phi=None, plane=None, verbose=False):

        k0 = 2 * np.pi / self.lbda

        self.field = torch.complex(
            torch.zeros(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False),
            torch.zeros(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False))

        coef=torch.tensor(self.dz*k0*1.j, dtype=torch.cfloat, requires_grad=False, device=self.device)

        self.dn0_layers = self.dn0.unbind(dim=2)
        self.fluo_layers = self.fluo.unbind(dim=2)
        self.aber_layers = self.aber.unbind(dim=0)

        phi_layers=phi.unbind(dim=2)

        for i in range(plane,self.Nz):
            depha0 = torch.mul(self.dn0_layers[i], coef)
            depha = torch.mul(self.field, torch.exp(depha0))
            if (i==plane):
                S = torch.mul(self.fluo_layers[i],torch.exp(phi_layers[i]*1.j))
                S = torch.fft.ifftn(torch.mul(torch.fft.fftn(S),self.C))
                self.field = torch.fft.ifftn(torch.mul(torch.fft.fftn(depha+S),self.Hz1))
            else:
                self.field = torch.fft.ifftn(torch.mul(torch.fft.fftn(depha), self.Hz1))

        self.field = torch.fft.fftn(self.field)

        for i in range(plane,self.Nz):
            self.field = torch.mul(self.field,self.Hz2)

        # plt.ion()
        # plt.imshow(torch.abs(bpm.pupil).detach().cpu().numpy())
        # plt.imshow(self.dn0_layers[plane].detach().cpu().numpy())
        # plt.imshow(self.fluo_layers[plane].detach().cpu().numpy())
        # plt.imshow(torch.abs(self.aber_layers[plane]).detach().cpu().numpy())
        # plt.imshow(torch.angle(self.aber_layers[plane]).detach().cpu().numpy())

        if bAber:
            I = torch.abs(torch.fft.ifftn(self.field * self.aber_layers[plane])) ** 2
        else:
            I=torch.abs(torch.fft.ifftn(self.field * self.pupil)) ** 2

        return  I

    def FresnelPropag(self, dz=0, fdir=[0, 0]):

        K = (self.nm / self.lbda) ** 2 - (self.mux - fdir[0]) ** 2 - (self.muy - fdir[1]) ** 2
        if dz <= 0:
            K = torch.complex(torch.sqrt(nf.relu(-K)), torch.sqrt(nf.relu(K)))
        else:
            K = torch.complex(-torch.sqrt(nf.relu(-K)), torch.sqrt(nf.relu(K)))
        hz = torch.exp(2 * np.pi * dz * K)
        return hz

if __name__ == '__main__':

    print('Init ...')

    model_config = {'nm': 1.33,
                    'na': 1.0556,
                    'dx': 0.1154,
                    'dz': 0.1,
                    'lbda': 0.5320,
                    'volume': [256, 256, 256],
                    'padding': False,
                    'dtype': torch.float32,
                    'device': 'cuda:0',
                    'out_path': "./Recons3D/"}

    coeff_RI = 0
    Niter = 100

    torch.cuda.empty_cache()
    bpm = bpmPytorch(model_config, coeff_RI=coeff_RI)

    new_obj = np.zeros((bpm.Nz, bpm.Nx, bpm.Ny))
    tmp=imread(f'./Pics_input/target.tif')
    new_obj=tmp
    # new_obj[:, int(bpm.Nx / 4):int(bpm.Nx * 3 / 4), int(bpm.Ny / 4):int(bpm.Ny * 3 / 4)] = tmp
    new_obj = np.moveaxis(new_obj, 0, -1)
    bpm.fluo = torch.tensor(new_obj, device=bpm.device, dtype=bpm.dtype, requires_grad=False)

    new_obj = np.zeros((bpm.Nz, bpm.Nx, bpm.Ny))
    tmp=imread(f'./Pics_input/dn.tif')
    new_obj=tmp
    # new_obj[:, int(bpm.Nx / 4):int(bpm.Nx * 3 / 4), int(bpm.Ny / 4):int(bpm.Ny * 3 / 4)] = tmp
    new_obj = np.moveaxis(new_obj, 0, -1)
    bpm.dn0 = torch.tensor(new_obj, device=bpm.device, dtype=bpm.dtype, requires_grad=False)

    bAber = False

    # Istack = np.zeros([bpm.Nx, bpm.Ny, bpm.Nz])
    #
    # for plane in tqdm(range(0, bpm.Nz)):
    #     I = torch.tensor(np.zeros((bpm.Nx, bpm.Ny)), device=bpm.device, dtype=bpm.dtype, requires_grad=False)
    #     phiL = torch.rand([bpm.Nx, bpm.Ny, 1000], dtype=torch.float32, requires_grad=False,device='cuda:0') * 2 * np.pi
    #     for w in range(0, Niter):
    #         zoi = np.random.randint(1000 - bpm.Nz)
    #         with torch.no_grad():
    #             I = I + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz ])
    #     Istack[:, :, plane] = I.detach().cpu().numpy() / Niter
    #
    #     # imwrite(f'./Recons3D/fluo_LSM_RIx{np.round(coeff_RI)/100}_{plane}.tif',
    #     #         np.moveaxis(np.array(I.cpu()), -1, 0) / Niter * 5 )
    #
    # tmp = np.moveaxis(Istack, -1, 0)
    # tmp = tmp[:, 128:128 + 256, 128:128 + 256]
    # imwrite(f'./Pics_input/input_aberration_null.tif', tmp )

    t=torch.load('./Pics_input/aberration.pt')
    t=t.to('cuda:0')
    bpm.aber=t

    bAber=True

    # new_obj = np.zeros((bpm.Nz, bpm.Nx, bpm.Ny))
    # tmp = imread(f'./Pics_input/aberration.tif')
    # tmp = tmp * np.pi
    # new_obj=tmp
    # # new_obj[:, int(bpm.Nx / 4):int(bpm.Nx * 3 / 4), int(bpm.Ny / 4):int(bpm.Ny * 3 / 4)] = tmp
    # new_obj = np.moveaxis(new_obj, 0, -1)
    # bpm.aber=torch.tensor(new_obj, device=bpm.device, dtype=bpm.dtype, requires_grad=False)

    Istack = np.zeros([bpm.Nx, bpm.Ny, bpm.Nz])

    for plane in tqdm(range(0, bpm.Nz)):
        I = torch.tensor(np.zeros((bpm.Nx, bpm.Ny)), device=bpm.device, dtype=bpm.dtype, requires_grad=False)
        phiL = torch.rand([bpm.Nx, bpm.Ny, 1000], dtype=torch.float32, requires_grad=False,device='cuda:0') * 2 * np.pi
        for w in range(0, Niter):
            zoi = np.random.randint(1000 - bpm.Nz)
            with torch.no_grad():
                I = I + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz ])
        Istack[:, :, plane] = I.detach().cpu().numpy() / Niter

        # imwrite(f'./Recons3D/fluo_LSM_RIx{np.round(coeff_RI)/100}_{plane}.tif',
        #         np.moveaxis(np.array(I.cpu()), -1, 0) / Niter * 5 )

    tmp = np.moveaxis(Istack, -1, 0)
    imwrite(f'./Pics_input/input_aberration.tif', tmp )








