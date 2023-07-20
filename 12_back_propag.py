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

def TV(params):
    nb_voxel = (params.shape[0]) * (params.shape[1])
    sx,sy= grads(params)
    TVloss = torch.sqrt(sx ** 2 + sy ** 2 + 1e-8).sum()
    return TVloss / (nb_voxel)

def grads(params):
    if len(params.shape)==2:
        params_sx = torch.roll(params, -1, 0)
        params_sy = torch.roll(params, -1, 1)

        sx = -(params - params_sx)
        sy = -(params - params_sy)

        sx[-1, :] = 0
        sy[:, -1] = 0

        return [sx,sy]

    elif len(params.shape)==3:
        params_sx = torch.roll(params, -1, 0)
        params_sy = torch.roll(params, -1, 1)
        params_sz = torch.roll(params, -1, 2)

        sx = -(params - params_sx)
        sy = -(params - params_sy)
        sz = -(params - params_sz)

        sx[-1, :, :] = 0
        sy[:, -1, :] = 0
        sz[:, :, -1] = 0

        return [sx,sy,sz]

def sqrt_cpx(t1):
    bs_pos = nf.relu(t1)
    bs_neg = nf.relu(-t1)
    return torch.stack((torch.sqrt(bs_pos), torch.sqrt(bs_neg)), dim=len(t1.shape))

def exp_cpx(input, conj=False):
    output = input.clone()
    if conj:
        amplitude = torch.exp(input[..., 1])
    else:
        amplitude = torch.exp(-input[..., 1])
    output[..., 0] = amplitude * torch.cos(input[..., 0])
    output[..., 1] = amplitude * torch.sin(input[..., 0])
    return output

def module_carre(t1):
    real1 = t1[:, :, 0]
    imag1 = t1[:, :, 1]
    return real1 * real1 + imag1 * imag1

class bpmPytorch(torch.nn.Module):

    def __init__(self, model_config, ini_obj=None):
        super(bpmPytorch, self).__init__()
        self.mc = model_config

        self.dx = self.mc['dx']
        self.lbda = self.mc['lbda']
        self.nm = 1
        self.file_name = self.mc['file_name']
        self.recons_abs = self.mc['recons_abs']
        self.init_type = self.mc['init_type']
        self.device = self.mc['device']

        self.pic = tifffile.imread(self.file_name)

        if (self.pic.ndim==3):
            temp = self.pic[np.random.randint(self.pic.shape[0]), :, :]
            self.pic=np.squeeze(temp)

        if (self.pic.shape[0] != self.pic.shape[1]):
            self.pic = resize(self.pic, (1024, 1024),
                                   anti_aliasing=True)

        if(np.mean(self.pic)>0.1):
            self.pic=self.pic/2

        self.Nx = self.pic.shape[0]
        self.Ny = self.pic.shape[1]

        self.field_shape = (self.Nx, self.Ny)
        self.z = nn.Parameter(torch.tensor(self.mc['z'], dtype=self.mc['dtype'], device=self.mc['device'], requires_grad=False))

        self.Hz = self.FresnelPropag()
        self.Hz = self.Hz.detach()
        self.Hz = torch.view_as_complex(self.Hz)

        self.k0 = 2 * np.pi / self.lbda

        self.dn0 = nn.Parameter(
            torch.tensor(self.pic, dtype=self.mc['dtype'], device=self.mc['device'], requires_grad=True))


    def forward(self, incoming_field=None, z_detector=None, verbose=False):

        # return torch.pow(self.dn0,2)

        depha0=torch.mul(self.dn0,self.k0)
        cpx=torch.complex(torch.cos(depha0),torch.sin(depha0))
        tt = torch.fft.ifftn(torch.fft.fftn(cpx)*self.Hz)
        return torch.abs(tt)**2


    def FresnelPropag(self, field_dir=[0, 0, 1]):

        int_x = np.arange(-self.Nx / 2 + 0, self.Nx / 2 - 0)
        int_y = np.arange(-self.Ny / 2 + 0, self.Ny / 2 - 0)
        x_range = self.Nx * self.dx
        y_range = self.Ny * self.dx
        mux = np.fft.fftshift(int_x / x_range).reshape(1, -1)
        muy = np.fft.fftshift(int_y / y_range).reshape(-1, 1)

        self.mux = torch.tensor(mux, dtype=self.mc['dtype'], requires_grad=False, device=self.device)
        self.muy = torch.tensor(muy, dtype=self.mc['dtype'], requires_grad=False, device=self.device)

        K = (self.nm / self.lbda) ** 2 - (self.mux - field_dir[0]) ** 2 - (self.muy - field_dir[1]) ** 2
        if self.z <= 0:
            hz = exp_cpx(2 * np.pi * self.z * sqrt_cpx(K), conj=True)
        else:
            hz = exp_cpx(2 * np.pi * self.z * sqrt_cpx(K), conj=False)
        return hz

if __name__ == '__main__':

    dx = 1.67

    print('Init ...')

    Nfig=0



    model_config0 = {'nm': 1.00,
                    'na': 1.00,
                    'dx': dx,
                    'lbda': 0.5320,
                    'z': 5420,
                    'init_type': 'files',
                    'dtype': torch.float32,
                    'file_name': "./Pics_input/L_demo_7.tif", #6=fly 7=COS7 8=Lensfree 8b=Lensfree thin "./Pics_input/L_demo_6.tif",  #
                    'device': 'cuda:0',
                    'recons_abs': False,
                    'abs_init_type': 'empty'}

    bpm = bpmPytorch(model_config0)
    with torch.no_grad():
        out = bpm()
        I_out = out.detach()

    tifffile.imwrite("./tmp/I_out.tif", np.array(I_out.cpu()))

    bpm.dn0.data = bpm.dn0.data *0

    out = bpm()
    loss = (out - I_out).norm(2)
    loss.backward()
    dn0grad = nn.Parameter(-bpm.dn0.grad.clone().detach().requires_grad_(True))
    dn0grad = np.array(dn0grad.detach().cpu())

    optimizer = torch.optim.Adam(bpm.parameters(), lr=0.01)
    losses = []
    datafits = []

    print('Gradient descent ...')
    start = timer()
    loss_best=1E8
    for i in tqdm(range(0, 150)):
        out = bpm()
        datafit = (out - I_out).norm(2)

        if (i>90):
            loss = datafit
        else:
            loss = datafit + TV(bpm.dn0).norm(2) * 5E3 + (bpm.dn0-bpm.dn0.data.relu()).norm(2) * 10

        if (loss<loss_best):
            loss_best=loss
            trans_0_best = bpm.dn0.detach().cpu()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        trans_0 = bpm.dn0.detach().cpu()
        tifffile.imwrite(f"./tmp/dn_{i}.tif", np.array(trans_0))


        losses.append(loss.item())
        datafits.append(datafit.item())

    end = timer()
    print(end - start)

    trans_0 = bpm.dn0.detach().cpu()

    fig = plt.figure(figsize=(15, 9))
    plt.suptitle('back_propag_12.py')
    ax = fig.add_subplot(2, 4, 1)
    imgplot = plt.imshow(np.array(I_out.cpu()),clim=(0.0, 2),cmap='Greys')
    ax.set_title('Diffraction')
    ax = fig.add_subplot(2, 4, 2)
    imgplot = plt.imshow(-dn0grad,cmap='Greys')
    ax.set_title('First back-propagation')
    ax = fig.add_subplot(2, 4, 5)
    imgplot = plt.imshow(-trans_0,clim=(-0.2, 0.0),cmap='Greys')
    ax.set_title('Final reconstruction')
    ax = fig.add_subplot(2, 4, 6)
    imgplot = plt.imshow(-trans_0_best,clim=(-0.2, 0.0),cmap='Greys')
    ax.set_title('Reconstruction with best loss')
    ax = fig.add_subplot(2, 4, 7)
    imgplot = plt.imshow(-bpm.pic,clim=(-0.2, 0.0),cmap='Greys')
    ax.set_title('Ground truth OPD')
    ax = fig.add_subplot(2, 4, 8)
    imgplot = plt.imshow(bpm.pic-np.array(trans_0),clim=(-0.2, 0.2),cmap='Greys')
    ax.set_title(f'Difference_{np.round(100*np.array(datafit.detach().cpu()))/100}')

    ax = fig.add_subplot(2, 4, 3)
    plt.plot(losses, label='loss')
    plt.plot(datafits, label='datafit')
    plt.yscale('log')
    plt.xlabel('Iterations [a.u]')
    plt.ylabel('Cost function [a.u]')
    plt.legend()
    plt.tight_layout()
    plt.show()


    Nfig += 1