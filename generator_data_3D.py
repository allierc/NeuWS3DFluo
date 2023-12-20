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
from modules.plots import create_epoch_plot
import imageio.v2 as imageio

def TV(params):
    if len(params.shape) == 2:
        nb_voxel = (params.shape[0]) * (params.shape[1])
        sx,sy= grads(params)
        TVloss = torch.sqrt(sx ** 2 + sy ** 2 + 1e-8).sum()
    elif len(params.shape)==3:
        nb_voxel = (params.shape[0]) * (params.shape[1]) * (params.shape[2])
        [sx, sy, sz] = grads(params)
        TVloss = torch.sqrt(sx ** 2 + sy ** 2 + sz ** 2 + 1e-8).sum()

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

def data_generate():
    print(' ')

    y_batches = torch.zeros((bpm.n_gammas + 1, 30, image_width, image_width), device=device)

    start = timer()
    phiL = torch.rand([bpm.image_width, bpm.image_width, 1000], dtype=torch.float32, requires_grad=False, device=device) * 2 * np.pi

    object_est, total_aberration_est, phase_aberration_est, gammas, gammas_raw = net.get_estimates()
    pupil_function = pupil.values * torch.exp(1j * (phase_aberration_gt + gammas))
    prf = fftshift(ifft2(ifftshift(pupil_function, dim=(1, 2))), dim=(1, 2))
    PSF = torch.abs(prf) ** 2
    PSF = PSF / torch.sum(PSF, dim=(1, 2), keepdim=True)
    OTF = fft2(ifftshift(PSF, dim=(1, 2)))


    for plane in tqdm(range(0, bpm.Nz)):
        object_gt = torch.tensor(np.zeros((bpm.image_width, bpm.image_width)), device=bpm.device,dtype=bpm.dtype, requires_grad=False)
        for w in range(0, Niter):
            zoi = np.random.randint(1000 - bpm.Nz)
            with torch.no_grad():
                object_gt = object_gt + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz])
        object_gt=object_gt.repeat(bpm.n_gammas + 1,1,1)

        object_ft = fft2(object_gt, dim=(1, 2))
        image_fourier_space = object_ft * OTF
        image = torch.real(ifft2(image_fourier_space, dim=(1, 2)))

        y_batches[:, plane:plane + 1, :, :] = image[:, None, :, :]

    end = timer()
    print(f'elapsed time : {np.round(end - start, 2)}')

    torch.save(y_batches, f'./{log_dir}/y_batches.pt')
    y_norm = torch.std(y_batches)
    dn_norm=torch.std(bpm.dn)
    torch.save(y_norm, f'./{log_dir}/y_norm.pt')
    torch.save(dn_norm, f'./{log_dir}/dn_norm.pt')

    for k in range(bpm.n_gammas + 1):
        imwrite(f'./{log_dir}/fluo_{k}.tif', (y_batches[k]/y_norm).detach().cpu().numpy().squeeze())


def data_train():

    y_batches = torch.load(model_config['input_fluo_simulated_acquisition'], map_location=device)
    f=model_config['input_fluo_simulated_acquisition']
    filename=f'{f[:-12]}/y_norm.pt'
    y_norm = torch.load(filename, map_location=device)
    y_batches = y_batches / y_norm
    filename=f'{f[:-12]}/dn_norm.pt'
    dn_norm = torch.load(filename, map_location=device)

    ### Training loop

    total_it = 0

    regul_0, regul_1, regul_2, regul_3, regul_4, regul_5 = [torch.tensor(float(model_config[f'regul_{i}']),device=device) for i in range(6)]

    Npixels = torch.tensor(bpm.Npixels,device=device)

    object_est_init = torch.zeros((bpm.n_gammas + 1,bpm.Nz,bpm.image_width,bpm.image_width),device=device)

    for plane in range(6,24):

        print(plane)

        object_gt_ = object_gt[:,plane]

        optimizer_object = torch.optim.Adam(net.g_im.parameters(), lr=model_config['init_lr'])  # Object optimizer
        optimizer_phase = torch.optim.Adam(net.g_g.parameters(),lr=model_config['init_lr'])  # Phase aberration optimizer (Zernike + MLP)

        ### Training loop

        total_it = 0

        loss0_list = []
        loss1_list = []  # Iest - I_gt
        loss2_list = []  # Iest - acquisition
        loss3_list = []  # I_gt - acquisition

        for epoch in range(model_config['num_epochs_phase_diversity']):

            optimizer_object.zero_grad();
            optimizer_phase.zero_grad();

            acquisition_est, acquisition_gt, gammas, _kernel, total_aberration_est, phase_aberration_est, object_est, gammas_raw = net(y_batches[:,plane], phase_aberration_gt)

            acquisition_gt = y_batches[:,plane] / 10

            loss_phase_est = torch.relu((torch.abs(phase_aberration_est) - 4 * np.pi)).norm(2)
            loss_image_negative = torch.relu(-object_est[0]).norm(2)
            loss_image_TV = TV(object_est) * 1E4
            loss_image_norm1 = object_est.norm(1)/Npixels

            loss = regul_0 * (acquisition_est-acquisition_gt).norm(2)/Npixels + regul_3 * loss_phase_est + 0 * regul_4 * loss_image_negative + loss_image_TV + loss_image_norm1 * 1E4

            loss.backward()

            optimizer_object.step()
            optimizer_phase.step()

            loss0_list.append(loss.item())
            loss1_list.append(F.mse_loss(object_est[0], object_gt_[0]).item())
            loss2_list.append(F.mse_loss(object_est[0], acquisition_gt[0]).item())
            loss3_list.append(F.mse_loss(object_gt_[0], acquisition_gt[0]).item())

            print(f'epoch {epoch } {loss.item():.4e}')

            if epoch % 5 == 0:
                # Create plot (call function)
                create_epoch_plot(epoch, plane, object_gt_, object_est, acquisition_gt, acquisition_est,
                                  phase_aberration_gt, phase_aberration_est, gammas, gammas_raw,
                                  loss0_list, loss1_list, loss2_list, loss3_list, n_gammas, f'{log_dir}/viz/')

        object_est_init[:,plane]=object_est


    for k in range(bpm.n_gammas + 1):
        imwrite(f'./{log_dir}/fluo_init_{k}.tif', (object_est_init[k]).detach().cpu().numpy().squeeze())


    fig = plt.figure(figsize=(8, 8))
    # plt.ion()
    plt.yscale("log")
    plt.plot(loss0_list)
    plt.savefig(f"./{log_dir}/Loss_fluo_init.tif")
    plt.close()

    loss0_list = []

    #### Initialisation fluo_est  ##################################################

    optimizer_fluo_3D = torch.optim.Adam(net.g_fluo_3D.parameters(), lr=0.01)  # Object optimizer
    optimizer_dn_3D = torch.optim.Adam(net.g_dn_3D.parameters(), lr=0.01)
    for epoch in range(-100,0):

        optimizer_fluo_3D.zero_grad();
        loss = 0
        for batch in range(8):
            fluo_est = net.init_fluo()
            loss += F.mse_loss(fluo_est, object_est_init[0])
        loss.backward()
        optimizer_fluo_3D.step()

        loss0_list.append(loss.item())

        if epoch % 20 == 0:
            print (f'epoch: {epoch} loss: {np.round(loss.item(),6)}')

    object_est = net.forward_volume_eval()
    for k in range(bpm.n_gammas + 1):
        imwrite(f'./{log_dir}/fluo_intermediate{k}.tif', (object_est[k]).detach().cpu().numpy().squeeze())

    fig = plt.figure(figsize=(8, 8))
    plt.ion()
    plt.yscale("log")
    plt.plot(loss_list)
    plt.savefig(f"./{log_dir}/Loss_fluo_intermediate.tif")
    plt.close()

    torch.save({'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer_fluo_3D.state_dict()},
               os.path.join(log_dir, 'model_intermediate.pt'))

    # optimizer_fluo_3D = torch.optim.Adam(net.g_fluo_3D.parameters(), lr=model_config['init_lr'])  # Object optimizer
    optimizer_dn_3D = torch.optim.Adam(net.g_dn_3D.parameters(), lr=model_config['init_lr'])
    loss_list = []
    loss1_list = []  # Iest - I_gt
    loss2_list = []  # Iest - acquisition
    loss3_list = []  # I_gt - acquisition

    if False:

        for epoch in range(model_config['num_epochs']):

            # optimizer_fluo_3D.zero_grad();
            optimizer_dn_3D.zero_grad();

            loss = 0

            for batch in range(model_config['batch_size']):
                plane = np.random.randint(bpm.Nz)
                pred, dn_est, fluo_est = net.forward_3D (plane, dn_norm=0)

                # loss_fluo_est_negative = regul_0 * torch.relu(-fluo_est).norm(2) / Npixels
                # loss_fluo_est_var = -regul_1 * torch.std(fluo_est)**2
                # loss_fluo_est_TV = regul_2 * TV(fluo_est)
                # loss_dn_est_TV = regul_3 * TV(dn_est)

                loss += F.mse_loss(pred[:,plane], y_batches[:,plane]) + 1E4 * TV(dn_est) # + loss_fluo_est_negative + loss_fluo_est_var + loss_fluo_est_TV +

            loss.backward()

            # optimizer_fluo_3D.step()
            optimizer_dn_3D.step()

            loss_list.append(loss.item() / model_config['batch_size'])
            # loss1_list.append(loss_fluo_est_var.item() / model_config['batch_size'])
            # loss2_list.append(loss_fluo_est_TV.item() / model_config['batch_size'])
            # loss3_list.append(loss_dn_est_TV.item() / model_config['batch_size'])

            if epoch % 20 == 0:
                print(f'epoch: {epoch} loss: {np.round(loss.item(), 6)}')
            if epoch % 800 == 0:
                fluo_est = torch.moveaxis(fluo_est, 0, -1)
                fluo_est = torch.moveaxis(fluo_est, 0, -1)
                imwrite(f'./{log_dir}/fluo_est_{epoch}_{regul_1}_{regul_2}_{regul_3}.tif', fluo_est.detach().cpu().numpy().squeeze())
                dn_est = torch.moveaxis(dn_est, 0, -1)
                dn_est = torch.moveaxis(dn_est, 0, -1)
                imwrite(f'./{log_dir}/dn_est_{epoch}_{regul_1}_{regul_2}_{regul_3}.tif', dn_est.detach().cpu().numpy().squeeze())

        y_pred = net.forward_volume_eval()
        for k in range(bpm.n_gammas + 1):
            imwrite(f'./{log_dir}/fluo_final_{k}.tif', (y_pred[k]).detach().cpu().numpy().squeeze())


    # torch.save(pred, f'./{log_dir}/pred_{epoch}.pt')
    # torch.save(dn_est, f'./{log_dir}/dn_est{epoch}.pt')
    # torch.save(fluo_est, f'./{log_dir}/fluo_est{epoch}.pt')
    # imwrite(f'./{log_dir}/pred_{epoch}.tif', pred[:,plane].detach().cpu().numpy().squeeze())
    # fluo_est = torch.moveaxis(fluo_est, 0, -1)
    # fluo_est = torch.moveaxis(fluo_est, 0, -1)
    # dn_est = torch.moveaxis(dn_est, 0, -1)
    # dn_est = torch.moveaxis(dn_est, 0, -1)
    # imwrite(f'./{log_dir}/dn_est_{epoch}.tif', dn_est.detach().cpu().numpy().squeeze())



    fig = plt.figure(figsize=(8, 8))
    plt.yscale("log")
    plt.plot(loss_list)
    plt.savefig(f"./{log_dir}/Loss.tif")
    plt.close()
    # fig = plt.figure(figsize=(8, 8))
    # plt.yscale("log")
    # plt.plot(loss1_list)
    # plt.savefig(f"./{log_dir}/Loss1.tif")
    # plt.close()
    # fig = plt.figure(figsize=(8, 8))
    # plt.yscale("log")
    # plt.plot(loss2_list)
    # plt.savefig(f"./{log_dir}/Loss2.tif")
    # plt.close()
    # fig = plt.figure(figsize=(8, 8))
    # plt.yscale("log")
    # plt.plot(loss3_list)
    # plt.savefig(f"./{log_dir}/Loss3.tif")
    # plt.close()


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

    # config_list = ['config_beads_GT']
    # config_list = ['config_recons_beads_GT']
    # config_list = ['config_CElegans']
    # config_list = ['config_recons_CElegans']
    # config_list = ['config_test']
    # config_list = ['config_recons_test']
    # config_list = ['config_CElegans_RI']
    config_list = ['config_beads_PSF']


    regul_list = ["0"] #, "5", "10", "50", "100", "200", "500", "1000", "5000", "10000"]

    for regul_ in regul_list:

                for config in config_list:

                    print (f'run :{config}')
                    print(regul_)

                    # Create log directory
                    l_dir = os.path.join('.', 'log', config)
                    log_dir = os.path.join(l_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                    print('log_dir: {}'.format(log_dir))
                    os.makedirs(log_dir, exist_ok=True)
                    os.makedirs(os.path.join(log_dir, 'viz'), exist_ok=True)
                    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'generating_code.py'))
                    copyfile(f'./config/{config}.yaml', os.path.join(log_dir, 'config.yaml'))

                    with open(f'./config/{config}.yaml', 'r') as file:
                        model_config = yaml.safe_load(file)

                    # model_config['regul_1'] = regul_
                    # model_config['regul_2'] = regul_
                    # model_config['regul_3'] = regul_

                    torch.cuda.empty_cache()
                    bpm = bpmPytorch(model_config, device=device)      # just to get the pupil function

                    num_polynomials = model_config['num_polynomials_gammas'] - 3  # NOTE: don't include piston and tip/tilt

                    conversion_factor_rms_pi = 2 * np.pi / bpm.wavelength  # Conversion factor to convert RMS between phase and length units
                    ansi_indices = np.arange(3, num_polynomials + 3)  # NOTE: don't include piston and tip/tilt
                    zernike_coefficients_gt = model_config['zernike_coefficients_gt']
                    zernike_coefficients_gt = [z * conversion_factor_rms_pi for z in zernike_coefficients_gt]

                    if 'input_gammas_zernike' in model_config and isinstance(model_config['input_gammas_zernike'],str):  # if input_gammas_zernike exists and is string, load file
                        model_config['input_gammas_zernike'] = np.loadtxt(model_config['input_gammas_zernike'],delimiter=',')
                    if 'input_gammas_zernike' not in model_config:
                        input_gammas_zernike = []
                    else:
                        input_gammas_zernike = conversion_factor_rms_pi * np.array(model_config['input_gammas_zernike'])

                    n_gammas = model_config['n_gammas']
                    p_num, p_unit = model_config['pixel_size'].split()
                    pixel_size = float(p_num) * u.Unit(p_unit)
                    numerical_aperture = model_config['numerical_aperture']
                    w_num, w_unit = model_config['wavelength'].split()
                    wavelength = float(w_num) * u.Unit(w_unit)
                    image_width=bpm.image_width

                    # Load data - ground truth object
                    if 'acquisition_GT_data' in model_config:
                        object_gt = imageio.imread(model_config['acquisition_GT_data'])
                        object_gt = object_gt / np.max(object_gt)
                        object_gt = torch.FloatTensor(object_gt).to(device)
                        if object_gt.ndim == 2:
                            object_gt = object_gt.repeat(bpm.Nz, 1, 1)
                        object_gt = object_gt.repeat(n_gammas + 1, 1, 1,1)
                    else:
                        object_gt = torch.zeros((n_gammas + 1, bpm.Nz, bpm.image_width, bpm.image_width))

                    pupil = Pupil(numerical_aperture=numerical_aperture, wavelength=wavelength, pixel_size=pixel_size,
                                  size_fourier_space=(image_width, image_width), device=DEVICE)
                    zernike_instance = ZernikePolynomials(pupil, index_convention='ansi', normalization=False)
                    phase_aberration_gt = zernike_instance.get_aberration(ansi_indices, zernike_coefficients_gt)
                    phase_aberration_gt = torch.FloatTensor(phase_aberration_gt).to(DEVICE)
                    phase_aberration_gt = phase_aberration_gt.repeat(n_gammas + 1, 1, 1)

                    if 'input_gammas_zernike' in model_config and isinstance(model_config['input_gammas_zernike'],str):  # if input_gammas_zernike exists and is string, load file
                        model_config['input_gammas_zernike'] = np.loadtxt(model_config['input_gammas_zernike'],
                                                                          delimiter=',')
                    if 'input_gammas_zernike' not in model_config:
                        input_gammas_zernike = []
                    else:
                        input_gammas_zernike = conversion_factor_rms_pi * np.array(model_config['input_gammas_zernike'])


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
                                    bpm=bpm,
                                    device=device)

                    net = net.to(DEVICE)

                    if 'y_batches.pt' in model_config['input_fluo_simulated_acquisition']:

                        data_train()

                    else:

                        data_generate()



