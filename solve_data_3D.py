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
        out = self.net(x)
        return out

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

        self.num_feats = self.mc['num_feats']

        self.renderer = G_Renderer(in_dim=self.num_feats, hidden_dim=self.num_feats, num_layers=2, out_dim=2, device=self.device)


        if self.padding:
            self.Nx = self.volume[0] *2
            self.Ny = self.volume[1] *2
            self.Nz = self.volume[2]

            self.data = nn.Parameter(0.1 * torch.randn((self.Nx, self.Ny, self.Nz, self.num_feats), device=self.device, requires_grad=True))

            self.Npixels = self.Nx * self.Ny
            self.field_shape = (self.Nx, self.Ny)

            null_obj = np.zeros(( self.Nx, self.Ny, self.Nz,))
            self.dn0 = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
            self.fluo = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
            self.aber = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)

        else:
            self.Nx = self.volume[0]
            self.Ny = self.volume[1]
            self.Nz = self.volume[2]

            self.data = nn.Parameter(0.1 * torch.randn((self.Nx, self.Ny, self.Nz, self.num_feats), device=self.device, requires_grad=True))

            self.Npixels = self.Nx * self.Ny
            self.field_shape = (self.Nx, self.Ny)

            # null_obj = np.zeros(( self.Nx, self.Ny, self.Nz,))
            # self.dn0 = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
            # self.fluo = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
            # self.aber = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)

            new_obj = np.zeros((self.Nz, self.Nx, self.Ny))
            tmp = imread(f'./Pics_input/target.tif')
            new_obj = np.moveaxis(tmp, 0, -1)
            self.fluo = torch.tensor(new_obj, device=self.device, dtype=self.dtype, requires_grad=False)

            new_obj = np.zeros((self.Nz, self.Nx, self.Ny))
            tmp = imread(f'./Pics_input/dn.tif')
            new_obj = np.moveaxis(tmp, 0, -1)
            self.dn0 = torch.tensor(new_obj, device=self.device, dtype=self.dtype, requires_grad=False)

            new_obj = np.zeros((self.Nz, self.Nx, self.Ny))
            tmp = imread(f'./Pics_input/aberration.tif')
            tmp = tmp * np.pi
            new_obj = np.moveaxis(tmp, 0, -1)
            self.aber = torch.tensor(new_obj, device=self.device, dtype=self.dtype, requires_grad=False)
            self.aber_layers = self.aber.unbind(dim=2)

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

        sample=self.renderer(self.data)

        self.dn0_layers = sample[:,:,:,0].unbind(dim=2)
        self.fluo_layers = sample[:,:,:,1].unbind(dim=2)

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

        self.field = torch.mul(self.field, torch.exp(self.aber_layers [plane] ))
        self.field = torch.fft.fftn(self.field)

        for i in range(plane,self.Nz):
            self.field = torch.mul(self.field,self.Hz2)

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
                    'num_feats': 8,
                    'out_path': "./Recons3D/"}

    coeff_RI = 0
    Niter = 10

    torch.cuda.empty_cache()
    bpm = bpmPytorch(model_config)
    bpm.train()

    # bpm.dn0 = nn.Parameter(0.0001*torch.rand([bpm.Nx, bpm.Ny, bpm.Nz], dtype=torch.float32, device=bpm.device, requires_grad=True))
    # bpm.fluo = nn.Parameter(0.1 * torch.rand([bpm.Nx, bpm.Ny, bpm.Nz], dtype=torch.float32, device=bpm.device, requires_grad=True))

    raw_data = io.imread('./Pics_input/input_aberration_null.tif')
    raw_data = torch.tensor(raw_data, device=bpm.device, dtype=bpm.dtype, requires_grad=False)

    optimizer = torch.optim.Adam(bpm.parameters(), lr=1E-5)

    phiL = torch.rand([bpm.Nx, bpm.Ny, 1000], dtype=torch.float32, requires_grad=False, device=bpm.device) * 2 * np.pi

    optimizer.zero_grad()

    loss_list=[]

    for epoch in range(10000):

        for plane in range(255,-1,-1):

            I = torch.ones(bpm.Nx, bpm.Ny, dtype=torch.float32, requires_grad=False, device=bpm.device)
            for w in range(0, Niter):
                zoi = np.random.randint(1000 - bpm.Nz)
                I = I + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz ])

            I = I / Niter

            loss = (I-raw_data[plane]).norm(2)
            loss_list.append(loss.item())

            if (plane%4==0):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print(f"     plane: {plane} Loss: {np.round(loss.item(), 4)}")

        print(f"Epoch: {epoch} Loss: {np.round(np.mean(loss_list), 4)}")

        if (epoch%10==0):
            with torch.no_grad():
                sample=bpm.renderer(bpm.data)
            obj = sample[:,:,:,0].cpu().detach().numpy().astype('float32')
            imwrite(f'./Recons3D/dn_recons_{epoch}.tif', np.moveaxis(obj, -1, 0))
            obj = sample[:,:,:,1].cpu().detach().numpy().astype('float32')
            imwrite(f'./Recons3D/fluo_recons_{epoch}.tif', np.moveaxis(obj, -1, 0))


