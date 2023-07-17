# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import os, time, imageio, tqdm, argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

import torch
print(f"Using PyTorch Version: {torch.__version__}")
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

import torch.nn.functional as F
from torch.fft import fft2, fftshift
from networks import *
from utils import *
from dataset import *
import matplotlib.pyplot as plt

from tifffile import imwrite
from tifffile import imread
import torch.nn.functional as f

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

# python3 ./recon_exp_data.py --static_phase True --num_t 100 --data_dir ./DATA_DIR/static_objects_static_aberrations/dog_esophagus_0.5diffuser/Zernike_SLM_data  --scene_name dog_esophagus_0.5diffuser --phs_layers 4 --num_epochs 1000 --save_per_fram

# python ./recon_exp_data.py --dynamic_scene --num_t 100 --data_dir ./DATA_DIR/NeuWS_experimental_data-selected/dynamic_objects_dynamic_aberrations/owlStamp_onionSkin/Zernike_SLM_data --scene_name owlStamp_onionSkin --phs_layers 4 --num_epochs 1000 --save_per_frame

DEVICE = 'cuda'

if __name__ == "__main__":

    bDynamic = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    if bDynamic:
        parser.add_argument('--data_dir', default='DATA_DIR/NeuWS_experimental_data-selected/dynamic_objects_dynamic_aberrations/owlStamp_onionSkin/Zernike_SLM_data/', type=str)
    else:
        parser.add_argument('--data_dir',default='DATA_DIR/static_objects_static_aberrations/dog_esophagus_0.5diffuser/Zernike_SLM_data/',type=str)
    parser.add_argument('--scene_name', default='dog_esophagus_0.5diffuser', type=str)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_t', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--init_lr', default=1e-3, type=float)
    parser.add_argument('--final_lr', default=1e-3, type=float)
    parser.add_argument('--silence_tqdm', action='store_true')
    parser.add_argument('--save_per_frame', action='store_true')
    parser.add_argument('--static_phase', default=True, type=bool)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_intensity', default=0, type=float)
    parser.add_argument('--im_prefix', default='SLM_raw', type=str)
    parser.add_argument('--zero_freq', default=-1, type=int)
    parser.add_argument('--phs_layers', default=4, type=int)
    parser.add_argument('--dynamic_scene', action='store_true')

    args = parser.parse_args()
    PSF_size = args.width

    ############
    # Setup output folders
    data_dir = f'{args.root_dir}/{args.data_dir}'
    vis_dir = f'{args.root_dir}/vis/{args.scene_name}'

    dset = BatchDataset(data_dir, num=args.num_t, im_prefix=args.im_prefix, max_intensity=args.max_intensity, zero_freq=args.zero_freq)
    x_batches = torch.cat(dset.xs, axis=0).unsqueeze(1).to(DEVICE)
    y_batches = torch.stack(dset.ys, axis=0).to(DEVICE)

    if bDynamic:
        args.dynamic_scene = True
        args.static_phase = False
    else:
        args.dynamic_scene = False
        args.static_phase = True

    print('x_batches', x_batches.shape)
    print('y_batches', y_batches.shape)
    print('static_phase', args.static_phase)
    print('dynamic_scene',args.dynamic_scene)

    net = StaticDiffuseNet(width=args.width, PSF_size=PSF_size, use_pe=False, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase)
    net = net.to(DEVICE)

    z_in = torch.abs(torch.randn(256,28,device=DEVICE))
    z_in = f.normalize(z_in, p=1, dim=1)
    z_in = z_in.repeat(256, 256, 1, 1).permute(2,3,0,1)
    basis = compute_zernike_basis(num_polynomials=28,field_res=(PSF_size, PSF_size)).unsqueeze(0).repeat(256, 1, 1, 1)
    basis = basis.to(DEVICE)
    out=basis*z_in
    out=torch.sum(out, dim=1)

    imwrite(f'./Pics_input/aberration.tif', out.detach().cpu().numpy())
