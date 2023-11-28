import os, time, tqdm, argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
import numpy as np
from astropy import units as u
import yaml # need to install pyyaml

import torch
print(f"Using PyTorch Version: {torch.__version__}")
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftshift
from networks import *
from modules.utils import *
from dataset import *
from prettytable import PrettyTable
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable

from modules.pupil import Pupil
from modules.zernike import ZernikePolynomials, wavefront_to_coefficients
from modules.losses import TV
from modules.plots import create_epoch_plot
import imageio.v2 as imageio
from shutil import copyfile
import datetime

DEVICE = 'cuda'

config_list = ['config_13']


if __name__ == "__main__":

    for config in config_list:

        ### Load data and set parameters

        # Load parameters from config file
        with open(f'./config/{config}.yaml', 'r') as file:
            model_config = yaml.safe_load(file)

        # Assert that config file is valid
        assert model_config['num_polynomials_gammas'] >= 3, 'num_polynomials_gammas must be >= 3'
        # If optimize_phase_diversities_with_mlp is True, assert that input_gammas_zernike either does not exist as parameter in config file or is empty string
        assert not (model_config['optimize_phase_diversities_with_mlp'] and 'input_gammas_zernike' in model_config and model_config['input_gammas_zernike']), 'input_gammas_zernike must be empty string if optimize_phase_diversities_with_mlp is True'


        n_gammas = model_config['n_gammas']
        b_gamma_optimization = model_config['b_gamma_optimization']
        regul_0, regul_1, regul_2, regul_3, regul_4, regul_5 = [float(model_config[f'regul_{i}']) for i in range(6)]
        p_num, p_unit = model_config['pixel_size'].split()
        pixel_size = float(p_num) * u.Unit(p_unit)
        numerical_aperture = model_config['numerical_aperture']
        w_num, w_unit = model_config['wavelength'].split()
        wavelength = float(w_num) * u.Unit(w_unit)
        num_polynomials = model_config['num_polynomials_gammas'] - 3 # NOTE: don't include piston and tip/tilt

        conversion_factor_rms_pi = 2*np.pi / wavelength.to(u.um).value # Conversion factor to convert RMS between phase and length units
        ansi_indices = np.arange(3, num_polynomials+3) # NOTE: don't include piston and tip/tilt
        zernike_coefficients_gt = model_config['zernike_coefficients_gt']
        zernike_coefficients_gt = [z * conversion_factor_rms_pi for z in zernike_coefficients_gt]
        
        if 'input_gammas_zernike' in model_config and isinstance(model_config['input_gammas_zernike'], str): # if input_gammas_zernike exists and is string, load file
            model_config['input_gammas_zernike'] = np.loadtxt(model_config['input_gammas_zernike'], delimiter=',')
        if 'input_gammas_zernike' not in model_config:
            input_gammas_zernike = []
        else:
            input_gammas_zernike = conversion_factor_rms_pi * np.array(model_config['input_gammas_zernike'])

        # Create log directory
        l_dir = os.path.join('.', 'log', config)
        log_dir = os.path.join(l_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print('log_dir: {}'.format(log_dir))
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'viz'), exist_ok=True)

        # Save config file and git hash
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as file:
            yaml.dump(model_config, file)
        os.system('git rev-parse HEAD > {}'.format(os.path.join(log_dir, 'git_hash.txt')))

        # Set random seeds for reproducibility
        torch.manual_seed(model_config['random_seed'])
        torch.cuda.manual_seed(model_config['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load data - ground truth object
        if 'acquisition_GT_data' not in model_config or not model_config['acquisition_GT_data']:
            data_path = '/groups/saalfeld/saalfeldlab/magdalena/AdaptiveOptics/ExperimentalData/230829 AO0807 AO_RNG_0p5_Div_1p0_Field3_reseatsample/'
            filename = 'GT_cropped.tif'
            object_gt = imageio.imread(os.path.join(data_path, filename))
        else:
            object_gt = imageio.imread(model_config['acquisition_GT_data'])
        object_gt = object_gt / np.max(object_gt)
        object_gt = torch.FloatTensor(object_gt).to(DEVICE)
        object_gt = object_gt.repeat(n_gammas+1, 1, 1)
        image_width = object_gt.shape[-1] 

        # Load data - acquisition data
        if 'acquisition_diversity_data' in model_config and model_config['acquisition_diversity_data']:
            acquisition_data = imageio.imread(model_config['acquisition_diversity_data'])
            acquisition_data = acquisition_data / np.max(acquisition_data)
            acquisition_data = torch.FloatTensor(acquisition_data).to(DEVICE)
        else:
            acquisition_data = []


        ### Create network and optimizer

        # Create pupil and Zernike polynomials
        pupil = Pupil(numerical_aperture=numerical_aperture, wavelength=wavelength, pixel_size=pixel_size, size_fourier_space=(image_width, image_width), device=DEVICE)
        zernike_instance = ZernikePolynomials(pupil, index_convention='ansi', normalization=False)

        phase_aberration_gt = zernike_instance.get_aberration(ansi_indices, zernike_coefficients_gt)
        phase_aberration_gt = torch.FloatTensor(phase_aberration_gt).to(DEVICE)
        phase_aberration_gt = phase_aberration_gt.repeat(n_gammas+1, 1, 1)

        net = StaticNet(zernike=zernike_instance,
                        pupil=pupil,
                        width=image_width,
                        PSF_size=image_width,
                        use_FFT=True,
                        bsize=model_config['batch_size'],
                        phs_layers=model_config['phs_layers'],
                        static_phase=model_config['static_phase'],
                        n_gammas=n_gammas,
                        input_gammas_zernike = input_gammas_zernike,
                        b_gamma_optimization = model_config['b_gamma_optimization'],
                        num_polynomials_gammas = model_config['num_polynomials_gammas'] - 3, # NOTE: don't include piston and x/y tilts
                        acquisition_data=acquisition_data,
                        optimize_phase_diversities_with_mlp=model_config['optimize_phase_diversities_with_mlp'])
        net = net.to(DEVICE)

        # Create table with network parameters
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in net.named_parameters():
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        #print(table)
        print(f"Total Trainable Params: {total_params}")

        # Create optimizer
        optimizer_object = torch.optim.Adam(net.g_im.parameters(), lr=model_config['init_lr']) # Object optimizer
        optimizer_phase = torch.optim.Adam(net.g_g.parameters(), lr=model_config['init_lr']) # Phase aberration optimizer (Zernike + MLP)
        
        # TODO: check if this variable exists in config file and if b_gamma_optimization is True
        if model_config['optimize_phase_diversities_with_mlp']:
            optimizer_phase_diversities = torch.optim.Adam(net.gammas_nnr[:].parameters(), lr=model_config['init_lr']) # Phase diversity optimizer (Zernike + MLP)
        else:
            optimizer_phase_diversities = torch.optim.Adam([net.gammas_zernike], lr=model_config['init_lr'])  # Phase diversity optimizer (Zernike only, no MLP)



        ### Training loop

        total_it = 0
        t = tqdm.trange(model_config['num_epochs'])
        t0 = time.time()

        loss0_list=[]
        loss1_list=[]   # Iest - I_gt
        loss2_list=[]   # Iest - acquisition
        loss3_list=[]   # I_gt - acquisition

        for epoch in t:

            ## Stop optimizing phase diversities and reset optimizers for object and phase
            if (epoch == model_config['epoch_freeze_phase_diversities']) & model_config['do_freeze_phase_diversity_optimization']:
                b_gamma_optimization = False
                net.g_im = G_PatchTensor(image_width)
                net.g_im.net.data.data = 0 * net.g_im.net.data.data
                for layer in [net.g_g.layers[0], net.g_g.layers[1], net.g_g.layers[3], net.g_g.layers[5]]: # NOTE: layers 0,1,3,5 are linear layers, layers 2,4, are LaekyReLU
                    nn.init.normal_(layer.weight, std=0.1)
                    nn.init.zeros_(layer.bias)
                net = net.to(DEVICE)
                optimizer_object = torch.optim.Adam(net.g_im.parameters(), lr=model_config['init_lr'])  # Reset object optimizer
                optimizer_phase = torch.optim.Adam(net.g_g.parameters(), lr=model_config['init_lr'])  # Rest phase aberration optimizer Zernike + MLP

            ## Run optimization for object, phase (and phase diversities)
            # Set gradients to zero for next iteration
            optimizer_object.zero_grad();
            optimizer_phase.zero_grad();
            if b_gamma_optimization:
                optimizer_phase_diversities.zero_grad();

            acquisition_est, acquisition_gt, gammas, _kernel, total_aberration_est, phase_aberration_est, object_est, gammas_raw = net(object_gt, phase_aberration_gt)

            if 'acquisition_diversity_data' in model_config and model_config['acquisition_diversity_data']:
                acquisition_gt = acquisition_data

            mse_loss = F.mse_loss(acquisition_est, acquisition_gt)
            loss_ortho = 0
            loss_gamma_norm = 0
            loss_gamma_std = 0

            for i in range(1, n_gammas):
                for j in range(i+1, n_gammas+1):
                    loss_ortho += (gammas[i,:,:] * gammas[j,:,:]).norm(2)
            for i in range(1, n_gammas+1):
                loss_gamma_std -= torch.log(torch.std(gammas[i,:,:])+1e-8)
            for i in range(0, n_gammas):
                loss_gamma_norm += torch.relu((torch.abs(gammas_raw[i]) - 4*np.pi)).norm(2)

            loss_phase_est = torch.relu((torch.abs(phase_aberration_est) - 4*np.pi)).norm(2)
            loss_image_negative = torch.relu(-object_est[0]).norm(2)

            loss = regul_0 * F.mse_loss(acquisition_est, acquisition_gt) + regul_1 * loss_ortho + regul_2 * loss_gamma_norm + regul_3 * loss_phase_est + regul_4 * loss_image_negative + regul_5 *loss_gamma_std

            loss0_list.append(mse_loss.item())
            loss1_list.append(F.mse_loss(object_est[0], object_gt[0]).item())
            loss2_list.append(F.mse_loss(object_est[0], acquisition_gt[0]).item())
            loss3_list.append(F.mse_loss(object_gt[0], acquisition_gt[0]).item())

            if epoch%50==0:
                np.save(f'{log_dir}/loss0_list',loss0_list)
                np.save(f'{log_dir}/loss1_list', loss1_list)
                np.save(f'{log_dir}/loss2_list', loss2_list)
                np.save(f'{log_dir}/loss3_list', loss3_list)

            loss.backward()

            optimizer_object.step()
            optimizer_phase.step()

            if b_gamma_optimization:
                optimizer_phase_diversities.step()

            t.set_postfix(MSE=f'{mse_loss.item():.4e}')

            if epoch%5 == 0:
                # Create plot (call function)
                create_epoch_plot(epoch, object_gt, object_est, acquisition_gt, acquisition_est,
                                  phase_aberration_gt, phase_aberration_est, gammas, gammas_raw,
                                  loss0_list, loss1_list, loss2_list, loss3_list, n_gammas, f'{log_dir}/viz/')

            if (epoch == model_config['epoch_freeze_phase_diversities']) & model_config['do_freeze_phase_diversity_optimization']:
                gammas_zernike_coefficients = np.zeros((n_gammas+1, num_polynomials))

                gammas = gammas.detach().cpu().numpy()
                gammas = np.squeeze(gammas)
                n_zernike_fit = model_config['num_polynomials_gammas']
                zernike_basis_fit = zernike_instance.calculate_polynomials(np.arange(3, n_zernike_fit))
                for i in range(1, n_gammas+1): # don't include first gamma (no phase diversity)
                    zernike_coefficients = wavefront_to_coefficients(gammas[i,:,:], np.arange(3, n_zernike_fit), pupil, index_convention='ansi')
                    zernike_coefficients_rms = zernike_coefficients / conversion_factor_rms_pi
                    np.savetxt(f'{log_dir}/gammas_{i}.csv', zernike_coefficients_rms, delimiter=',')
                    print(f'Gamma {i}: {np.round(zernike_coefficients_rms,5)}')

                    gammas_zernike_coefficients[i,:] = zernike_coefficients_rms

                    imageio.imsave(f'{log_dir}/gammas_{i}.tif', gammas[i])
                    g = zernike_basis_fit * zernike_coefficients
                    g = np.sum(g,axis=2)
                    imageio.imsave(f'{log_dir}/gammas_zernike_fit_{i}.tif', g)

                # Save all gammas (except first gamma) as .mat file
                sio.savemat(f'{log_dir}/gammas.mat', {'gammas': np.transpose(gammas[1:,:,:], (1,2,0))})
                # Save all gammas coefficients as csv
                np.savetxt(f'{log_dir}/gammas_zernike_coefficients.csv', gammas_zernike_coefficients, delimiter=',')

            total_it += 1

        t1 = time.time()
        print(f'Training takes {t1 - t0} seconds.')
        imageio.imsave(f'{log_dir}/phase_aberration_est.tif', phase_aberration_est[0].detach().cpu().numpy())
        imageio.imsave(f'{log_dir}/phase_aberration_gt.tif', phase_aberration_gt[0].detach().cpu().numpy())

        phase_aberration_zernike = phase_aberration_est.detach().cpu().numpy()
        phase_aberration_zernike = np.squeeze(phase_aberration_zernike[0])
        n_zernike_fit = model_config['num_polynomials_gammas']
        zernike_basis_fit = zernike_instance.calculate_polynomials(np.arange(3, n_zernike_fit))

        zernike_coefficients = wavefront_to_coefficients(phase_aberration_zernike, np.arange(3, n_zernike_fit), pupil,index_convention='ansi')

        zernike_coefficients_rms = zernike_coefficients / conversion_factor_rms_pi

        print(f'phase_aberration_zernike {i}: {np.round(zernike_coefficients_rms, 5)}')

        gammas_zernike_coefficients[i, :] = zernike_coefficients_rms

        g = zernike_basis_fit * zernike_coefficients
        g = np.sum(g, axis=2)
        imageio.imsave(f'{log_dir}/estimated_aberration_zernike_fit.tif', g)