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
import torch.nn.functional as f
from torch.fft import fft2, fftshift
from shutil import copyfile
from modules.pupil import Pupil
from modules.zernike import ZernikePolynomials, wavefront_to_coefficients
from astropy import units as u
import yaml # need to install pyyaml
import datetime
from torch.fft import fft2, ifft2, fftshift, ifftshift

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

    config = 'config_beads'

    # Create log directory
    l_dir = os.path.join('.', 'log', config)
    log_dir = os.path.join(l_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print('log_dir: {}'.format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    # os.makedirs(os.path.join(log_dir, 'viz'), exist_ok=True)

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

    print(' ')
    y_batches = torch.zeros((1, 30, image_width, image_width), device=device)

    bpm.bAber=1

    for k in range(5):

        bpm.gamma = gammas[k]

        start = timer()
        phiL = torch.rand([bpm.image_width, bpm.image_width, 1000], dtype=torch.float32, requires_grad=False, device='cuda:0') * 2 * np.pi

        for plane in tqdm(range(0, bpm.Nz)):
            I = torch.tensor(np.zeros((bpm.image_width, bpm.image_width)), device=bpm.device, dtype=bpm.dtype, requires_grad=False)
            for w in range(0, Niter):
                zoi = np.random.randint(1000 - bpm.Nz)
                with torch.no_grad():
                    I = I + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz], naber=0)
            y_batches[:, plane:plane + 1, :, :] = I

        end = timer()
        print(f'elapsed time : {np.round(end - start,2)}')

        imwrite(f'./{log_dir}/stack_{k}.tif',y_batches.detach().cpu().numpy().squeeze() )



    #
    #
    # else :
    #
    #     for plane in range(0, bpm.Nz):
    #
    #         print(f'plane: {plane}')
    #
    #         z_in = torch.abs(torch.randn(100, num_polynomials, device=DEVICE))
    #         z_in = f.normalize(z_in, p=1, dim=1)
    #         z_in = z_in.repeat(256, 256, 1, 1).permute(2, 3, 0, 1)
    #         basis = compute_zernike_basis(num_polynomials=num_polynomials, field_res=(PSF_size, PSF_size)).unsqueeze(0).repeat(100, 1, 1, 1)
    #         basis = basis.to(DEVICE)
    #         out = basis * z_in
    #         out = torch.sum(out, dim=1) * torch.pi
    #
    #         out_cpx = torch.zeros(100, 256, 256, dtype=torch.cfloat, device=DEVICE)
    #         t = fftshift(bpm.pupil)
    #
    #         for k in range(100):
    #             tt = torch.exp(torch.mul(torch.abs(t) * out[k, :, :], 20j))
    #             tt = fftshift(tt)
    #             out_cpx[k, :, :] = tt
    #
    #         x_batches[:, plane:plane+1, :, :] = out_cpx[:,None,:,:]
    #
    #         imwrite(f'./Pics_input/stack/aberration_plane{plane}.tif',
    #                     torch.angle(out_cpx).detach().cpu().numpy())
    #
    #         bpm.aber = out_cpx
    #         bpm.aber_layers = bpm.aber.unbind(dim=0)
    #         bpm.bAber = 2
    #
    #         Istack = np.zeros([bpm.image_width, bpm.image_width, 100])
    #         phiL = torch.rand([bpm.image_width, bpm.image_width, 1000], dtype=torch.float32, requires_grad=False, device='cuda:0') * 2 * np.pi
    #
    #         for naber in tqdm(range(100)):
    #
    #             I = torch.tensor(np.zeros((bpm.image_width, bpm.image_width)), device=bpm.device, dtype=bpm.dtype, requires_grad=False)
    #             for w in range(0, Niter):
    #                 zoi = np.random.randint(1000 - bpm.Nz)
    #                 with torch.no_grad():
    #                     I = I + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz ], naber=naber)
    #             Istack[:, :, naber] = I.detach().cpu().numpy() / Niter
    #
    #         tmp = torch.tensor(Istack,device=device)
    #         tmp = torch.permute(tmp, (2, 0, 1))
    #         y_batches[:, plane:plane + 1, :, :] = tmp[:, None, :, :]
    #
    #         tmp = np.moveaxis(Istack, -1, 0)
    #         imwrite(f'./Pics_input/stack/input_aberration_plane{plane}.tif', tmp )
    #
    #
    # if bSingle:
    #     torch.save(x_batches, f'./Pics_input/x_single_batches.pt')
    #     torch.save(y_batches, f'./Pics_input/y_single_batches.pt')
    # else:
    #     if bDefocus:
    #         torch.save(x_batches,f'./Pics_input/defocus_x_batches.pt')
    #         torch.save(y_batches, f'./Pics_input/defocus_y_batches.pt')
    #     else:
    #         torch.save(x_batches,f'./Pics_input/x_batches.pt')
    #         torch.save(y_batches, f'./Pics_input/y_batches.pt')
    #
    #
    #
    #
    #
    #
    #
