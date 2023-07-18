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

        self.fluo_data_path = self.mc['fluo_data_path']
        self.data_path = self.mc['data_path']
        self.out_path = self.mc['out_path']

        if self.padding:
            self.Nx = self.volume[0] *2
            self.Ny = self.volume[1] *2
            self.Nz = self.volume[2]

            self.Npixels = self.Nx * self.Ny
            self.field_shape = (self.Nx, self.Ny)

            new_obj = np.zeros(( self.Nx, self.Ny, self.Nz,))
            self.dn0 = torch.tensor(new_obj, device=self.device, dtype=self.dtype, requires_grad=False)
            self.fluo = torch.tensor(new_obj, device=self.device, dtype=self.dtype, requires_grad=True)
            # self.fluo = torch.sqrt(self.fluo)
        else:
            self.Nx = self.volume[0]
            self.Ny = self.volume[1]
            self.Nz = self.volume[2]

            self.Npixels = self.Nx * self.Ny
            self.field_shape = (self.Nx, self.Ny)
            new_obj = io.imread(self.data_path)
            new_obj = np.moveaxis(new_obj, 0, -1)
            self.dn0 = torch.tensor(new_obj * coeff_RI, device=self.device, dtype=self.dtype, requires_grad=False)

            fluo_obj = io.imread(self.fluo_data_path)
            fluo_obj = np.moveaxis(fluo_obj, 0, -1)
            self.fluo = torch.tensor(fluo_obj, device=self.device, dtype=self.dtype, requires_grad=True)
            self.fluo = torch.sqrt(self.fluo)

        N_x = np.arange(-self.Nx / 2 + 0, self.Nx / 2 - 0)
        N_y = np.arange(-self.Ny / 2 + 0, self.Ny / 2 - 0)
        x_range = self.Nx * self.dx
        y_range = self.Ny * self.dx
        mux = np.fft.fftshift(N_x / x_range).reshape(1, -1)
        muy = np.fft.fftshift(N_y / y_range).reshape(-1, 1)
        self.mux = torch.tensor(mux, dtype=self.dtype, requires_grad=False, device=self.device)
        self.muy = torch.tensor(muy, dtype=self.dtype, requires_grad=False, device=self.device)

        Hz = self.FresnelPropag(dz=self.dz, fdir=[0,0])
        self.Hz1 =Hz
        Hz = self.FresnelPropag(dz=-self.dz, fdir=[0,0])
        self.Hz2 =Hz

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

        self.field=torch.fft.fftn(self.field)
        self.field = torch.mul(self.field,self.pupil)

        I=self.dn0*0
        for i in range(self.Nz):
            self.field = torch.mul(self.field,self.Hz2)
            I[:,:,i]=torch.abs(torch.fft.ifftn(self.field)) ** 2

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

    # flist=['Recons3D']
    # for folder in flist:
    #     files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/Propag/{folder}/*")
    #     for f in files:
    #         os.remove(f)

    model_config = {'nm': 1.33,
                    'na': 1.0556,
                    'dx': 0.1154,
                    'dz': 0.1,
                    'lbda': 0.5320,
                    'volume': [256, 256, 256],
                    'padding': True,
                    'dtype': torch.float32,
                    'device': 'cuda:0',
                    'data_path': "./Pics_input/KidneyL_30_2.tif",
                    'fluo_data_path': "./Pics_input/Fluo_6_30.tif",
                    'raw_data_path': "./Pics_input/",
                    'out_path': "./Recons3D/"}

    coeff_RI = 0

    torch.cuda.empty_cache()
    bpm = bpmPytorch(model_config, coeff_RI=coeff_RI)

    Istack = np.array([512, 512, bpm.Nz])


    large_object = io.imread('./Pics_input/KidneyLL_512_8bits_res2.tif')
    large_object = large_object

    large_fluo_empty = io.imread('./Pics_input/KidneyLL_512_8bits_full_blur_res2.tif')
    large_fluo_full = io.imread('./Pics_input/KidneyLL_512_8bits_empty_blur_res2.tif')

    coeff_RI_list=[]

    for run in range(85,200):

        coeff_RI = np.abs(0.5 + np.random.randn())

        print('coeff_RI: ',coeff_RI)

        coeff_RI_list.append(coeff_RI)

        size = [large_object.shape[0], large_object.shape[1], large_object.shape[2]]
        zoi = (np.random.randint(size[0] - 256), np.random.randint(size[1] - 256), np.random.randint(size[2] - 256))

        new_obj = np.zeros((bpm.Nz, bpm.Nx, bpm.Ny))
        if (run%2==0):
            tmp = large_fluo_empty[zoi[0]:zoi[0] + 256, zoi[1]:zoi[1] + 256, zoi[2]:zoi[2] + 256]
            tmp = tmp / 3700
        else:
            tmp = large_fluo_full[zoi[0]:zoi[0] + 256, zoi[1]:zoi[1] + 256, zoi[2]:zoi[2] + 256]
            tmp = tmp / 236
        imwrite(f'./Pics_CNN/Diffusion_forward_RI_230306/target_{run}.tif', tmp)
        new_obj[:, int(bpm.Nx / 4):int(bpm.Nx * 3 / 4), int(bpm.Ny / 4):int(bpm.Ny * 3 / 4)] = tmp
        new_obj = np.moveaxis(new_obj, 0, -1)
        bpm.fluo = torch.tensor(new_obj, device=bpm.device, dtype=bpm.dtype, requires_grad=False)

        new_obj = np.zeros((bpm.Nz, bpm.Nx, bpm.Ny))
        tmp = large_object[zoi[0]:zoi[0] + 256, zoi[1]:zoi[1] + 256, zoi[2]:zoi[2] + 256]
        tmp = tmp / 255/25*5 * coeff_RI
        imwrite(f'./Pics_CNN/Diffusion_forward_RI_230306/dn_{run}.tif', tmp)
        new_obj[:, int(bpm.Nx / 4):int(bpm.Nx * 3 / 4), int(bpm.Ny / 4):int(bpm.Ny * 3 / 4)] = tmp
        new_obj = np.moveaxis(new_obj, 0, -1)
        bpm.dn0 = torch.tensor(new_obj, device=bpm.device, dtype=bpm.dtype, requires_grad=False)

        Istack = np.array([bpm.Nx, bpm.Ny, bpm.Nz])

        for plane in tqdm(range(0, bpm.Nz)):

            I = bpm.dn0 * 0

            phiL = torch.rand([bpm.Nx, bpm.Ny, 1000], dtype=torch.float32, requires_grad=False,
                              device='cuda:0') * 2 * np.pi
            Niter = 100
            for w in range(0, Niter):
                zoi = np.random.randint(1000 - bpm.Nz)
                with torch.no_grad():
                    I = I + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz ])

            temp = np.array(I.cpu())
            if plane == 0:
                Istack = temp * 0
            Istack[:, :, plane] = temp[:, :, bpm.Nz - plane -1] / Niter * 5

            # imwrite(f'./Recons3D/fluo_LSM_RIx{np.round(coeff_RI)/100}_{plane}.tif',
            #         np.moveaxis(np.array(I.cpu()), -1, 0) / Niter * 5 )

        tmp = np.moveaxis(Istack, -1, 0)
        tmp = tmp[:, 128:128 + 256, 128:128 + 256]
        imwrite(f'./Pics_CNN/Diffusion_forward_RI_230306/input_{run}.tif', tmp )

    np.save('./Pics_CNN/Diffusion_forward_RI_230306/coeff_RI_list.npy',coeff_RI_list)

