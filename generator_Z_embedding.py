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
from solve_data_3D import bpmPytorch, bpmPytorch_defocus
import torch.nn.functional as f
from torch.fft import fft2, fftshift
from shutil import copyfile

if __name__ == '__main__':

    print('Init ...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L=5
    dim=256

    z_in = torch.arange(dim, device=device)
    z_in = z_in.repeat(dim, dim, 1)
    z_out = torch.zeros(dim,dim,dim,2*L, device=device)

    for k in tqdm(range(L)):
        temp1= torch.sin(2**k*np.pi*z_in/dim)
        temp2 = torch.cos(2 ** k * np.pi * z_in / dim)
        z_out[:,:,:,2*k:2*k+1]=temp1[:,:,:,None]
        z_out[:,:,:,2*k+1:2*k+2] = temp2[:,:,:,None]

    torch.save(z_out, f'./Pics_input/z_embedding_L_{2*L}.pt')





