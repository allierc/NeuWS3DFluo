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
from tifffile import imwrite
from tifffile import imread
import logging
from BPM3Dfluo_V2 import bpmPytorch
import torch.nn.functional as F
from torch.fft import fft2, fftshift
from shutil import copyfile
from modules.pupil import Pupil
from modules.zernike import ZernikePolynomials, wavefront_to_coefficients
from astropy import units as u
import yaml # need to install pyyaml
import datetime
from torch.fft import fft2, ifft2, fftshift, ifftshift
from networks_3D import *

def data_generate():
    print(' ')

    y_batches = torch.zeros((bpm.n_gammas + 1, 30, image_width, image_width), device=device)

    start = timer()
    phiL = torch.rand([bpm.image_width, bpm.image_width, 1000], dtype=torch.float32, requires_grad=False,
                      device='cuda:0') * 2 * np.pi

    for plane in tqdm(range(0, bpm.Nz)):
        I = torch.tensor(np.zeros((bpm.n_gammas + 1, bpm.image_width, bpm.image_width)), device=bpm.device,
                         dtype=bpm.dtype, requires_grad=False)

        for w in range(0, Niter):
            zoi = np.random.randint(1000 - bpm.Nz)
            with torch.no_grad():
                I = I + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz], naber=0)
        y_batches[:, plane:plane + 1, :, :] = I[:, None, :, :]

    end = timer()
    print(f'elapsed time : {np.round(end - start, 2)}')

    for k in range(bpm.n_gammas + 1):
        imwrite(f'./{log_dir}/stack_{k}.tif', y_batches[k].detach().cpu().numpy().squeeze())

    torch.save(y_batches, f'./{log_dir}/y_batches.pt')

def data_train():

    y_batches = torch.load(model_config['input_fluo_simulated_acquisition'], map_location=device)

    net = StaticNet(zernike=zernike_instance,
                    pupil=pupil,
                    width=image_width,
                    PSF_size=image_width,
                    use_FFT=True,
                    bsize=model_config['batch_size'],
                    phs_layers=model_config['phs_layers'],
                    static_phase=model_config['static_phase'],
                    n_gammas=n_gammas,
                    input_gammas_zernike=input_gammas_zernike,
                    b_gamma_optimization=False,
                    num_polynomials_gammas=model_config['num_polynomials_gammas'] - 3,
                    # NOTE: don't include piston and x/y tilts
                    acquisition_data=[],
                    optimize_phase_diversities_with_mlp=False,
                    z_mode=bpm.Nz,
                    bpm=bpm)

    net = net.to(DEVICE)

    # Create optimizer
    optimizer_fluo_3D = torch.optim.Adam(net.g_fluo_3D.parameters(), lr=model_config['init_lr'])  # Object optimizer
    optimizer_dn_3D = torch.optim.Adam(net.g_dn_3D.parameters(), lr=model_config['init_lr'])

    ### Training loop

    total_it = 0

    loss0_list = []
    loss1_list = []  # Iest - I_gt
    loss2_list = []  # Iest - acquisition
    loss3_list = []  # I_gt - acquisition

    for epoch in range(model_config['num_epochs']):

        optimizer_fluo_3D.zero_grad();
        optimizer_dn_3D.zero_grad();

        plane = np.random.randint(bpm.Nz)

        pred, dn_est, fluo_est = net(plane)

        loss_image_negative = 1E4*torch.relu(-fluo_est).norm(2)

        loss = F.mse_loss(pred[:,plane], y_batches[:,plane]) + loss_image_negative

        loss.backward()

        optimizer_fluo_3D.step()
        optimizer_dn_3D.step()

        print (f'epoch: {epoch} loss: {np.round(loss.item()/1E6,2)}')

        if epoch%100==0:

            # torch.save(pred, f'./{log_dir}/pred_{epoch}.pt')
            # torch.save(dn_est, f'./{log_dir}/dn_est{epoch}.pt')
            # torch.save(fluo_est, f'./{log_dir}/fluo_est{epoch}.pt')
            imwrite(f'./{log_dir}/pred_{epoch}.tif', y_batches[0,29].detach().cpu().numpy().squeeze())
            fluo_est = torch.moveaxis(fluo_est, 0, -1)
            fluo_est = torch.moveaxis(fluo_est, 0, -1)
            dn_est = torch.moveaxis(dn_est, 0, -1)
            dn_est = torch.moveaxis(dn_est, 0, -1)
            imwrite(f'./{log_dir}/dn_est_{epoch}.tif', dn_est.detach().cpu().numpy().squeeze())
            imwrite(f'./{log_dir}/fluo_est_{epoch}.tif', fluo_est.detach().cpu().numpy().squeeze())













if __name__ == '__main__':

    print('Init ...')

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = DEVICE

    print(f'device: {device}')

    PSF_size = 256
    num_polynomials = 48
    Niter = 500

    print(f'PSF_size: {PSF_size}')
    print(f'num_polynomials: {num_polynomials}')
    print(f'Niter: {Niter}')

    config_list = ['config_recons_from_beads'] #['config_grid', 'config_beads', 'config_beads_cropped','config_boats']

    for config in config_list:

        print (f'run :{config}')

        # Create log directory
        l_dir = os.path.join('.', 'log', config)
        log_dir = os.path.join(l_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print('log_dir: {}'.format(log_dir))
        os.makedirs(log_dir, exist_ok=True)
        copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'generating_code.py'))

        with open(f'./config/{config}.yaml', 'r') as file:
            model_config = yaml.safe_load(file)

        torch.cuda.empty_cache()
        bpm = bpmPytorch(model_config)      # just to get the pupil function

        num_polynomials = model_config['num_polynomials_gammas'] - 3  # NOTE: don't include piston and tip/tilt

        conversion_factor_rms_pi = 2 * np.pi / bpm.wavelength  # Conversion factor to convert RMS between phase and length units
        ansi_indices = np.arange(3, num_polynomials + 3)  # NOTE: don't include piston and tip/tilt
        zernike_coefficients_gt = model_config['zernike_coefficients_gt']
        zernike_coefficients_gt = [z * conversion_factor_rms_pi for z in zernike_coefficients_gt]

        input_gammas_zernike = conversion_factor_rms_pi * np.array(model_config['input_gammas_zernike'])
        n_gammas = model_config['n_gammas']
        p_num, p_unit = model_config['pixel_size'].split()
        pixel_size = float(p_num) * u.Unit(p_unit)
        numerical_aperture = model_config['numerical_aperture']
        w_num, w_unit = model_config['wavelength'].split()
        wavelength = float(w_num) * u.Unit(w_unit)
        image_width=bpm.image_width

        pupil = Pupil(numerical_aperture=numerical_aperture, wavelength=wavelength, pixel_size=pixel_size,
                      size_fourier_space=(image_width, image_width), device=DEVICE)
        zernike_instance = ZernikePolynomials(pupil, index_convention='ansi', normalization=False)
        phase_aberration_gt = zernike_instance.get_aberration(ansi_indices, zernike_coefficients_gt)
        phase_aberration_gt = torch.FloatTensor(phase_aberration_gt).to(DEVICE)
        # phase_aberration_gt = phase_aberration_gt.repeat(n_gammas + 1, 1, 1)

        gammas = torch.zeros(n_gammas + 1, 1, image_width, image_width, device=DEVICE, dtype=torch.complex128)
        for i in range(n_gammas + 1):
           # phase_shift = fftshift ( torch.tensor(zernike_instance.get_aberration(ansi_indices, input_gammas_zernike[i]),device=DEVICE) * pupil.values + phase_aberration_gt)
           phase_shift = fftshift(torch.tensor(zernike_instance.get_aberration(ansi_indices, input_gammas_zernike[i]),device=DEVICE) * pupil.values)
           gammas[i, :, :, :] = torch.exp(1j * phase_shift)
        gammas = gammas.squeeze()
        bpm.gammas = gammas


        if 'y_batches.pt' in model_config['input_fluo_simulated_acquisition']:

            data_train()

        else:

            data_generate ()



