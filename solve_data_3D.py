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
        self.bAber = self.mc['bAber']
        self.num_feats = self.mc['num_feats']
        self.bFit = self.mc['bFit']

        self.renderer = G_Renderer(in_dim=self.num_feats, hidden_dim=self.num_feats, num_layers=2, out_dim=1, device=self.device)

        self.Nx = self.volume[0]
        self.Ny = self.volume[1]
        self.Nz = self.volume[2]

        self.f = nn.Parameter(torch.tensor(np.zeros([256, 256]),  device=self.device, dtype=self.dtype, requires_grad=True))

        self.data = nn.Parameter(torch.zeros((self.Nx, self.Ny, self.Nz, self.num_feats), device=self.device, requires_grad=True))

        self.Npixels = self.Nx * self.Ny
        self.field_shape = (self.Nx, self.Ny)

        null_obj = np.zeros(( self.Nx, self.Ny, self.Nz,))
        self.dn0 = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
        self.fluo = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
        self.aber = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)

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

        self.Hz1 = self.FresnelPropag(dz=self.dz, fdir=[0,0])
        self.Hz2 = self.FresnelPropag(dz=-self.dz, fdir=[0,0])
        self.Hz3 = self.FresnelPropag(dz=self.dz*50, fdir=[0,0])
        self.Hz4 = self.FresnelPropag(dz=-self.dz*50, fdir=[0,0])

        fdir= [0,0]
        mux_inc = (self.mux - fdir[0])
        muy_inc = (self.muy - fdir[1])
        munu = torch.sqrt(mux_inc ** 2 + muy_inc ** 2).reshape(self.Nx, self.Ny, 1)
        pupil = np.squeeze((munu < (self.na / self.lbda)).float())
        imwrite('./pupil.tif', np.array(pupil.cpu()))
        self.pupil = torch.complex(
            torch.ones(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False) * pupil,
            torch.ones(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False) * 0)

        C = (self.lbda/self.nm) * torch.sqrt((self.nm/self.lbda)**2 - mux_inc ** 2 - muy_inc ** 2).reshape(self.Nx, self.Ny)
        C = 1/C*pupil
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        imwrite('./C.tif', np.array(C.cpu()))
        self.C = C

    def forward(self, plane=None, phi=None, naber=0):

        k0 = 2 * np.pi / self.lbda

        self.field = torch.complex(
            torch.zeros(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False),
            torch.zeros(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False))

        coef=torch.tensor(self.dz*k0*1.j, dtype=torch.cfloat, requires_grad=False, device=self.device)

        if self.bFit:
            sample=self.renderer(self.data)
            self.dn0_layers = sample[:,:,:,0].unbind(dim=2)
            self.fluo_layers = sample[:,:,:,1].unbind(dim=2)
            self.fluo_layer = torch.squeeze(self.data[plane,:,:,0])
        else:
             self.fluo_layer = self.fluo_layers[plane]

        phi_layers=phi.unbind(dim=2)

        for i in range(plane,self.Nz):
            if (i==plane):
                S = torch.mul(self.fluo_layer,torch.exp(phi_layers[i]*1.j))
                S = torch.fft.ifftn(torch.mul(torch.fft.fftn(S),self.C))
                self.field = torch.mul(S, torch.exp(torch.mul(self.dn0_layers[i], coef)))
            else:
                self.field = torch.mul(self.field, torch.exp(torch.mul(self.dn0_layers[i], coef)))
            self.field = torch.fft.ifftn(torch.mul(torch.fft.fftn(self.field), self.Hz1))

        self.field = torch.fft.fftn(self.field)

        for i in range(plane,self.Nz):
            self.field = torch.mul(self.field,self.Hz2)

        if self.bAber == 0:
            I = torch.abs(torch.fft.ifftn(self.field * self.pupil)) ** 2
        if self.bAber == 1:
            I = torch.abs(torch.fft.ifftn(self.field * self.aber_layers[plane])) ** 2       # aberration is defined per volume (x256)
        if self.bAber == 2:
            I = torch.abs(torch.fft.ifftn(self.field * self.aber_layers[naber])) ** 2       # aberration is defined per plane (x100)

        return I

    def FresnelPropag(self, dz=0, fdir=[0, 0]):

        K = (self.nm / self.lbda) ** 2 - (self.mux - fdir[0]) ** 2 - (self.muy - fdir[1]) ** 2
        if dz <= 0:
            K = torch.complex(torch.sqrt(nf.relu(-K)), torch.sqrt(nf.relu(K)))
        else:
            K = torch.complex(-torch.sqrt(nf.relu(-K)), torch.sqrt(nf.relu(K)))
        hz = torch.exp(2 * np.pi * dz * K)

        return hz

class bpmPytorch_defocus(torch.nn.Module):

    def __init__(self, model_config):

        super(bpmPytorch_defocus, self).__init__()
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
        self.bAber = self.mc['bAber']
        self.num_feats = self.mc['num_feats']
        self.bFit = self.mc['bFit']

        self.renderer = G_Renderer(in_dim=self.num_feats, hidden_dim=self.num_feats, num_layers=2, out_dim=1, device=self.device)

        self.Nx = self.volume[0]
        self.Ny = self.volume[1]
        self.Nz = self.volume[2]

        self.f = nn.Parameter(torch.tensor(np.zeros([256, 256]),  device=self.device, dtype=self.dtype, requires_grad=True))

        self.data = nn.Parameter(torch.zeros((self.Nx, self.Ny, self.Nz, self.num_feats), device=self.device, requires_grad=True))

        self.Npixels = self.Nx * self.Ny
        self.field_shape = (self.Nx, self.Ny)

        null_obj = np.zeros(( self.Nx, self.Ny, self.Nz,))
        self.dn0 = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
        self.fluo = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)
        self.aber = torch.tensor(null_obj, device=self.device, dtype=self.dtype, requires_grad=False)

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

        self.Hz1 = self.FresnelPropag(dz=self.dz, fdir=[0,0])
        self.Hz2 = self.FresnelPropag(dz=-self.dz, fdir=[0,0])
        self.Hz3 = self.FresnelPropag(dz=self.dz*50, fdir=[0,0])
        self.Hz4 = self.FresnelPropag(dz=-self.dz*50, fdir=[0,0])

        fdir= [0,0]
        mux_inc = (self.mux - fdir[0])
        muy_inc = (self.muy - fdir[1])
        munu = torch.sqrt(mux_inc ** 2 + muy_inc ** 2).reshape(self.Nx, self.Ny, 1)
        pupil = np.squeeze((munu < (self.na / self.lbda)).float())
        imwrite('./pupil.tif', np.array(pupil.cpu()))
        self.pupil = torch.complex(
            torch.ones(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False) * pupil,
            torch.ones(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False) * 0)

        C = (self.lbda/self.nm) * torch.sqrt((self.nm/self.lbda)**2 - mux_inc ** 2 - muy_inc ** 2).reshape(self.Nx, self.Ny)
        C = 1/C*pupil
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        imwrite('./C.tif', np.array(C.cpu()))
        self.C = C

    def forward(self, plane=None, phi=None, naber=0):

        k0 = 2 * np.pi / self.lbda

        self.field = torch.complex(
            torch.zeros(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False),
            torch.zeros(self.field_shape, dtype=torch.float32, device=self.device, requires_grad=False))

        coef=torch.tensor(self.dz*k0*1.j, dtype=torch.cfloat, requires_grad=False, device=self.device)

        if self.bFit:
            sample=self.renderer(self.data)
            self.dn0_layers = sample[:,:,:,0].unbind(dim=2)
            self.fluo_layers = sample[:,:,:,1].unbind(dim=2)
            self.fluo_layer = torch.squeeze(self.data[plane,:,:,0])
        else:
             self.fluo_layer = self.fluo_layers[plane]

        phi_layers=phi.unbind(dim=2)

        for i in range(plane,self.Nz):
            if (i==plane):
                S = torch.mul(self.fluo_layer,torch.exp(phi_layers[i]*1.j))
                S = torch.fft.ifftn(torch.mul(torch.fft.fftn(S),self.C))
                self.field = torch.mul(S, torch.exp(torch.mul(self.dn0_layers[i], coef)))
            else:
                self.field = torch.mul(self.field, torch.exp(torch.mul(self.dn0_layers[i], coef)))
            self.field = torch.fft.ifftn(torch.mul(torch.fft.fftn(self.field), self.Hz1))

        self.field = torch.fft.fftn(self.field)

        for i in range(plane,self.Nz):
            self.field = torch.mul(self.field,self.Hz2)

        self.field = torch.mul(self.field, self.Hz4)

        if self.bAber == 0:
            I = torch.abs(torch.fft.ifftn(self.field * self.pupil)) ** 2
        if self.bAber == 1:
            I = torch.abs(torch.fft.ifftn(self.field * self.aber_layers[plane])) ** 2       # aberration is defined per volume (x256)
        if self.bAber == 2:
            I = torch.abs(torch.fft.ifftn(self.field * self.aber_layers[naber])) ** 2       # aberration is defined per plane (x100)

        return I

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
    torch.cuda.empty_cache()

    model_config = {'nm': 1.33,
                    'na': 1.0556,
                    'dx': 0.1154,
                    'dz': 0.1,
                    'lbda': 0.5320,
                    'volume': [256, 256, 256],
                    'padding': False,
                    'dtype': torch.float32,
                    'device': 'cuda:0',
                    'num_feats': 4,
                    'bAber': True,
                    'bFit': True,
                    'out_path': "./Recons3D/"}

    flist = ['Recons3D']
    for folder in flist:
        files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/NeuWS3D/{folder}/*")
        for f in files:
            os.remove(f)

    ntry=1

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)
    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

    torch.cuda.empty_cache()
    bpm = bpmPytorch(model_config)

    # bpm.dn0 = nn.Parameter(0.0001*torch.rand([bpm.Nx, bpm.Ny, bpm.Nz], dtype=torch.float32, device=bpm.device, requires_grad=True))
    # bpm.fluo = nn.Parameter(0.1 * torch.rand([bpm.Nx, bpm.Ny, bpm.Nz], dtype=torch.float32, device=bpm.device, requires_grad=True))

    raw_data = io.imread('./Pics_input/input_aberration.tif')
    raw_data = torch.tensor(raw_data, device=bpm.device, dtype=bpm.dtype, requires_grad=False)

    t=torch.load('./Pics_input/aberration.pt')
    bpm.aber=t.to('cuda:0')
    bpm.aber_layers = bpm.aber.unbind(dim=0)

    phiL = torch.rand([bpm.Nx, bpm.Ny, 1000], dtype=torch.float32, requires_grad=False, device=bpm.device) * 2 * np.pi

    # im_opt = torch.optim.Adam(bpm.data.parameters(), lr=1E-1)

    optimizer = torch.optim.Adam(bpm.parameters(), lr=1E-2)  # , weight_decay=5e-3)

    criteria = nn.MSELoss()
    bpm.train()

    Niter = 5
    batch = 1

    best_loss = np.inf

    # for plane in range(bpm.Nz):

    plane=0
    loss_list = []

    for epoch in range(100):

        optimizer.zero_grad()

        # zoi = np.random.randint(1000 - bpm.Nz)
        # out = bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz ])

        out = bpm(plane=int(plane))

        loss = criteria(out,raw_data[plane]*25) # + I.norm(1) * 1E-4
        loss_list.append(loss.item())


        loss.backward()

        optimizer.step()

        print(f"Epoch: {epoch} Loss: {np.round(loss.item(), 4)}")

    imwrite(f'./Recons3D/fluo_recons_{epoch}.tif', out.detach().cpu().numpy())
    a=1






    # loss = 0
    #
    # n_grad=0
    # plane_list=np.random.permutation(bpm.Nz)
    #
    # # for plane in range(0,1):
    #
    # plane=0
    #
    # I = torch.ones(bpm.Nx, bpm.Ny, dtype=torch.float32, requires_grad=False, device=bpm.device)

    # for w in range(0, Niter):
    #     zoi = np.random.randint(1000 - bpm.Nz)
    #     I = I + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz ])
    # I = I / Niter

    # zoi = np.random.randint(1000 - bpm.Nz)
    # I = bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz])



    # if (np.mean(loss_list) < best_loss):
    #     torch.save({'model_state_dict': bpm.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},os.path.join(log_dir, 'models', 'best_model.pt'))
    #     # with torch.no_grad():
    #     #     sample = bpm.renderer(bpm.data)
    #     # # obj = sample[:, :, :, 0].cpu().detach().numpy().astype('float32')
    #     # # imwrite(f'./Recons3D/dn_recons_{epoch}.tif', np.moveaxis(obj, -1, 0))
    #     # obj = sample[:, :, :, 1].cpu().detach().numpy().astype('float32')



    # plt.ion()
    # plt.imshow(25 * raw_data[plane].detach().cpu().numpy(),vmin=0,vmax=1)
    # plt.colorbar()
    #
    # plt.ion()
    # plt.imshow(I.detach().cpu().numpy(),vmin=0,vmax=1)
    # plt.colorbar()
    #
    # plt.imshow(self.fluo_layers[plane].detach().cpu().numpy())
    # plt.imshow(I.detach().cpu().numpy())
    # plt.imshow(raw_data[plane].detach().cpu().numpy())
    # plt.imshow((I - raw_data[plane]).detach().cpu().numpy())
    # plt.imshow(torch.abs(bpm.pupil).detach().cpu().numpy())
    # plt.imshow(self.dn0_layers[plane].detach().cpu().numpy())
    # plt.imshow(self.fluo_layers[plane].detach().cpu().numpy())
    # plt.imshow(torch.abs(self.aber_layers[plane]).detach().cpu().numpy())
    # plt.imshow(torch.angle(self.aber_layers[plane]).detach().cpu().numpy())
    # plt.imshow(torch.angle(self.aber_layers[naber]).detach().cpu().numpy())




