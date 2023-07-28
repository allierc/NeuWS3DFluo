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
from networks3D_PSF import *
from utils import *
from dataset import *
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import tifffile
import tqdm
import time

# python3 ./recon_exp_data.py --static_phase True --num_t 100 --data_dir ./DATA_DIR/static_objects_static_aberrations/dog_esophagus_0.5diffuser/Zernike_SLM_data  --scene_name dog_esophagus_0.5diffuser --phs_layers 4 --num_epochs 1000 --save_per_fram

# python ./recon_exp_data.py --dynamic_scene --num_t 100 --data_dir ./DATA_DIR/NeuWS_experimental_data-selected/dynamic_objects_dynamic_aberrations/owlStamp_onionSkin/Zernike_SLM_data --scene_name owlStamp_onionSkin --phs_layers 4 --num_epochs 1000 --save_per_frame

DEVICE = 'cuda'

if __name__ == "__main__":


    bDynamic = False

    flist = ['Recons3D','Recons3D_torch']
    for folder in flist:
        files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/NeuWS3D/{folder}/*")
        for f in files:
            os.remove(f)


    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    if bDynamic:
        parser.add_argument('--data_dir', default='DATA_DIR/NeuWS_experimental_data-selected/dynamic_objects_dynamic_aberrations/owlStamp_onionSkin/Zernike_SLM_data/', type=str)
    else:
        parser.add_argument('--data_dir',default='DATA_DIR/static_objects_static_aberrations/dog_esophagus_0.5diffuser/Zernike_SLM_data/',type=str)

    parser.add_argument('--scene_name', default='simu', type=str)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_t', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--vis_freq', default=200, type=int)
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

    # os.makedirs(f'{args.root_dir}/vis', exist_ok=True)
    # os.makedirs(vis_dir, exist_ok=True)
    # os.makedirs(f'{vis_dir}/final', exist_ok=True)
    # print(f'Saving output at: {vis_dir}')
    # if args.save_per_frame:
    #     os.makedirNeuWSs(f'{vis_dir}/final/per_frame', exist_ok=True)

    ############
    # Old training preparations
    # dset = BatchDataset(data_dir, num=args.num_t, im_prefix=args.im_prefix, max_intensity=args.max_intensity, zero_freq=args.zero_freq)
    # x_batches = torch.cat(dset.xs, axis=0).unsqueeze(1).to(DEVICE)
    # y_batches = torch.stack(dset.ys, axis=0).to(DEVICE)

    t0 = time.time()

    N_image=50
    N_acqui=100

    dset = BatchDataset(data_dir, num=args.num_t, im_prefix=args.im_prefix, max_intensity=args.max_intensity,zero_freq=args.zero_freq)

    print('Loading x_batches y_batches ...')
    x_batches = torch.load(f'./Pics_input/x_batches.pt')
    y_batches = torch.load(f'./Pics_input/y_batches.pt')
    print('done')

    print('x_batches', x_batches.shape)
    print('y_batches', y_batches.shape)
    print('static_phase', args.static_phase)
    print('dynamic_scene',args.dynamic_scene)

    if bDynamic:
        args.dynamic_scene = True
        args.static_phase = False
    else:
        args.dynamic_scene = False
        args.static_phase = True

    if args.dynamic_scene:
        net = MovingDiffuse(width=args.width, PSF_size=PSF_size, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase)
    else:
        net = StaticDiffuseNet(width=args.width, PSF_size=PSF_size, use_pe=False, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase)

    net = net.to(DEVICE)
    net.train()

    im_opt = torch.optim.Adam(net.g_im.parameters(), lr=args.init_lr*10)
    ph_opt = torch.optim.Adam(net.g_g.parameters(), lr=args.init_lr)
    #ph_opt = torch.optim.Adam(net.dn_im.parameters(), lr=args.init_lr)

    # im_sche = torch.optim.lr_scheduler.CosineAnnealingLR(im_opt, T_max = args.num_epochs, eta_min=args.final_lr)
    # ph_sche = torch.optim.lr_scheduler.CosineAnnealingLR(ph_opt, T_max = args.num_epochs, eta_min=args.final_lr)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr)  # , weight_decay=5e-3)


    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")

    total_it = 0
    print('Reconstructing ...  ')
    print(' ')
    time.sleep(0.5)

    target = imread(f'./Pics_input/target.tif')
    dn = imread(f'./Pics_input/dn.tif')
    loss = 0

    for plane in range(255, -1, -1):
        total_it = 0
        if plane==255:
            nepoch=25
        else:
            nepoch=10
        for epoch in range(nepoch):
            if (plane==255) & (epoch>0):
                print(f'    {epoch}     loss:{np.round(loss.item(), 6)}  ')
            for it in range(100):

                total_it += 1

                x_batch, y_batch = x_batches[it,plane,:,:], y_batches[it,plane,:,:]

                y_batch = torch.squeeze(y_batch) * 10

                cur_t = plane

                im_opt.zero_grad();  ph_opt.zero_grad()

                y, S_est, dn_est = net(torch.squeeze(x_batch), cur_t)

                loss = F.mse_loss(y, y_batch) + TV(S_est)*1E-4 + torch.abs(torch.std(dn_est)-3.1415) # + torch.abs(torch.mean(dn_est)-0.05) + TV(dn_est)*1E-2

                loss.backward()

                im_opt.step()
                ph_opt.step()


                if ((total_it%10==0) & (epoch<2)) | (it==99):

                    fig = plt.figure(figsize=(24, 6))
                    # plt.ion()
                    ax = fig.add_subplot(1, 6, 1)
                    plt.imshow(y_batch.detach().cpu().squeeze(), vmin=0, vmax=0.5, cmap='gray')
                    plt.axis('off')
                    plt.title('Simulated measurement')
                    ax = fig.add_subplot(1, 6, 2)
                    plt.imshow(y.detach().cpu().squeeze(), vmin=0, vmax=0.5, cmap='gray')
                    plt.axis('off')
                    plt.title('Reconstructed measurement')
                    ax = fig.add_subplot(1, 6, 3)
                    plt.imshow((S_est ** 2).detach().cpu().squeeze(), vmin=0, vmax=2, cmap='gray')
                    plt.axis('off')
                    plt.title('fluo_est')
                    ax = fig.add_subplot(1, 6, 4)
                    plt.imshow(dn_est.detach().cpu().squeeze(), vmin=-3.14, vmax=3.14, cmap='rainbow')
                    plt.axis('off')
                    plt.title(f'dn_est')
                    mmin = torch.min(dn_est).item()
                    mmax = torch.max(dn_est).item()
                    mstd = torch.std(dn_est).item()
                    mmean = torch.mean(dn_est).item()
                    # plt.text(10,15,f'min: {np.round(mmin,2)}   max: {np.round(mmax,2)}   {np.round(mmean,3)}+/-{np.round(mstd,3)}')
                    ax = fig.add_subplot(1, 6, 5)
                    plt.imshow(target[plane, :, :], vmin=0, vmax=0.5, cmap='gray')
                    plt.title(f'fluo target')
                    plt.axis('off')
                    ax = fig.add_subplot(1, 6, 6)
                    plt.imshow(dn[plane, :, :], vmin=0, vmax=0.1, cmap='gray')
                    plt.title(f'dn target')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(f'./Recons3D/plane_{plane}_it_{total_it}.jpg')
                    plt.clf()


        torch.save(S_est, f'./Recons3D_torch/S_est_plane_{plane}_it_{total_it}.pt')
        torch.save(dn_est, f'./Recons3D_torch/dn_est_plane_{plane}_it_{total_it}.pt')

        print(f'   plane: {plane} it: {total_it} loss:{np.round(loss.item(), 6)}')





    # t1 = time.time()
    # print(f'Training takes {t1 - t0} seconds.')
    #
    # ############
    # # Export final results
    # out_errs = []
    # out_abes = []
    # out_Iest = []
    #
    # for t in range(args.num_t):
    #     cur_t = (t / (args.num_t - 1)) - 0.5
    #     cur_t = torch.FloatTensor([cur_t]).to(DEVICE)
    #
    #     I_est, sim_g, sim_phs = net.get_estimates(cur_t)
    #     I_est = torch.clamp(I_est, 0, 1).squeeze().detach().cpu().numpy()
    #
    #     out_Iest.append(I_est)
    #
    #     est_g = sim_g.detach().cpu().squeeze().numpy()
    #     out_errs.append(np.uint8(ang_to_unit(np.angle(est_g)) * 255))
    #     abe = sim_phs[0].detach().cpu().squeeze()
    #     abe = (abe - abe.min()) / (abe.max() - abe.min())
    #     out_abes.append(np.uint8(abe * 255))
    #     if args.save_per_frame and not args.static_phase:
    #       sio.savemat(f'{vis_dir}/final/per_frame/sim_phase_{t}.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})
    #
    # if args.dynamic_scene:
    #     out_Iest = [np.uint8(im * 255) for im in out_Iest]
    #     imageio.mimsave(f'{vis_dir}/final/final_I.gif', out_Iest, duration=1000*1./30)
    # else:
    #     I_est = np.uint8(I_est.squeeze() * 255)
    #     imageio.imsave(f'{vis_dir}/final/final_I_est.png', I_est)
    #
    # if args.static_phase:
    #     imageio.imsave(f'{vis_dir}/final/final_aberrations_angle.png', out_errs[0])
    #     imageio.imsave(f'{vis_dir}/final/final_aberrations.png', out_abes[0])
    # else:
    #     imageio.mimsave(f'{vis_dir}/final/final_aberrations_angle_grey.gif', out_errs, duration=1000*1./30)
    #     imageio.mimsave(f'{vis_dir}/final/final_aberrations.gif', out_abes, duration=1000*1./30)
    #
    # print("Training concludes.")
    #
    # colored_err = []
    # for i, a in enumerate(out_errs):
    #     plt.imsave(f'{vis_dir}/final/per_frame/{i:03d}.jpg', a, cmap='rainbow')
    #     colored_err.append(imageio.imread(f'{vis_dir}/final/per_frame/{i:03d}.jpg'))
    # imageio.mimsave(f'{vis_dir}/final/final_aberrations_angle.gif', colored_err, duration=1000*1./30)