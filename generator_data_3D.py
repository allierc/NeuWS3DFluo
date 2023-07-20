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
from solve_data_3D import bpmPytorch
import torch.nn.functional as f
from torch.fft import fft2, fftshift

if __name__ == '__main__':

    print('Init ...')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'devicew: {device}')

    PSF_size = 256
    num_polynomials = 48
    Niter = 50

    print(f'PSF_size: {PSF_size}')
    print(f'num_polynomials: {num_polynomials}')
    print(f'Niter: {Niter}')

    model_config = {'nm': 1.33,
                    'na': 1.0556,
                    'dx': 0.1154,
                    'dz': 0.1,
                    'lbda': 0.5320,
                    'volume': [256, 256, 256],
                    'padding': False,
                    'dtype': torch.float32,
                    'device': 'cuda:0',
                    'bAber': False,
                    'bFit': False,
                    'num_feats': 4,
                    'out_path': "./Recons3D/"}

    torch.cuda.empty_cache()
    bpm = bpmPytorch(model_config)      # just to get the pupil function

    new_obj = np.zeros((bpm.Nz, bpm.Nx, bpm.Ny))
    tmp = imread(f'./Pics_input/target.tif')
    new_obj = np.moveaxis(tmp, 0, -1)
    bpm.fluo = torch.tensor(new_obj, device=bpm.device, dtype=bpm.dtype, requires_grad=False)

    new_obj = np.zeros((bpm.Nz, bpm.Nx, bpm.Ny))
    tmp = imread(f'./Pics_input/dn.tif')
    new_obj = np.moveaxis(tmp, 0, -1)
    bpm.dn0 = torch.tensor(new_obj, device=bpm.device, dtype=bpm.dtype, requires_grad=False)

    bpm.dn0_layers = bpm.dn0.unbind(dim=2)
    bpm.fluo_layers = bpm.fluo.unbind(dim=2)

    print(' ')

    for plane in range(0, bpm.Nz):

        print(f'plane: {plane}')

        Istack = np.zeros([bpm.Nx, bpm.Ny, 100])

        z_in = torch.abs(torch.randn(100, num_polynomials, device=DEVICE))
        z_in = f.normalize(z_in, p=1, dim=1)
        z_in = z_in.repeat(256, 256, 1, 1).permute(2, 3, 0, 1)
        basis = compute_zernike_basis(num_polynomials=num_polynomials, field_res=(PSF_size, PSF_size)).unsqueeze(0).repeat(100, 1, 1, 1)
        basis = basis.to(DEVICE)
        out = basis * z_in
        out = torch.sum(out, dim=1) * torch.pi

        out_cpx = torch.zeros(100, 256, 256, dtype=torch.cfloat, device=DEVICE)
        t = fftshift(bpm.pupil)

        for k in range(100):
            tt = torch.exp(torch.mul(torch.abs(t) * out[k, :, :], 20j))
            tt = fftshift(tt)
            out_cpx[k, :, :] = tt

        torch.save(out_cpx, f'./Pics_input/stack/aberration_plane{plane}.pt')
        imwrite(f'./Pics_input/stack/aberration_plane{plane}.tif', torch.angle(out_cpx).detach().cpu().numpy())

        bpm.aber = out_cpx
        bpm.aber_layers = bpm.aber.unbind(dim=0)
        bpm.bAber = 2

        phiL = torch.rand([bpm.Nx, bpm.Ny, 1000], dtype=torch.float32, requires_grad=False, device='cuda:0') * 2 * np.pi

        for naber in tqdm(range(100)):

            I = torch.tensor(np.zeros((bpm.Nx, bpm.Ny)), device=bpm.device, dtype=bpm.dtype, requires_grad=False)
            for w in range(0, Niter):
                zoi = np.random.randint(1000 - bpm.Nz)
                with torch.no_grad():
                    I = I + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz ], naber=naber)
            Istack[:, :, naber] = I.detach().cpu().numpy() / Niter

        tmp = np.moveaxis(Istack, -1, 0)
        imwrite(f'./Pics_input/stack/input_aberration_plane{plane}.tif', tmp )









