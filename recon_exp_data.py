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
from prettytable import PrettyTable

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

    # os.makedirs(f'{args.root_dir}/vis', exist_ok=True)
    # os.makedirs(vis_dir, exist_ok=True)
    # os.makedirs(f'{vis_dir}/final', exist_ok=True)
    # print(f'Saving output at: {vis_dir}')
    # if args.save_per_frame:
    #     os.makedirNeuWSs(f'{vis_dir}/final/per_frame', exist_ok=True)

    ############
    # Training preparations
    dset = BatchDataset(data_dir, num=args.num_t, im_prefix=args.im_prefix, max_intensity=args.max_intensity, zero_freq=args.zero_freq)
    x_batches = torch.cat(dset.xs, axis=0).unsqueeze(1).to(DEVICE)  #100x256x256 SLM images (phase)
    y_batches = torch.stack(dset.ys, axis=0).to(DEVICE)             #100x256c256 acquisitions
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


    # model initialization

    if args.dynamic_scene:
        net = MovingDiffuse(width=args.width, PSF_size=PSF_size, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase)
    else:
        net = StaticDiffuseNet(width=args.width, PSF_size=PSF_size, use_pe=False, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase)

    net = net.to(DEVICE)

    im_opt = torch.optim.Adam(net.g_im.parameters(), lr=args.init_lr)       # optimizer for unknown object
    ph_opt = torch.optim.Adam(net.g_g.parameters(), lr=args.init_lr)        # opitmizer for unknown aberration

    im_sche = torch.optim.lr_scheduler.CosineAnnealingLR(im_opt, T_max = args.num_epochs, eta_min=args.final_lr)
    ph_sche = torch.optim.lr_scheduler.CosineAnnealingLR(ph_opt, T_max = args.num_epochs, eta_min=args.final_lr)

    total_it = 0
    t = tqdm.trange(args.num_epochs, disable=args.silence_tqdm)

    ############
    # Training loop
    t0 = time.time()

    for epoch in t:
        idxs = torch.randperm(len(dset)).long().to(DEVICE)

        for it in range(0, len(dset), args.batch_size):
            idx = idxs[it:it+args.batch_size]

            x_batch, y_batch = x_batches[idx], y_batches[idx]               # taking batches of training data, introduced aberration and corresponding acquisition



            cur_t = (idx / (args.num_t - 1)) - 0.5          # time information -0.5 0.5

            im_opt.zero_grad();  ph_opt.zero_grad()
            # optimizer.zero_grad()

            y, _kernel, sim_g, sim_phs, I_est = net(x_batch, cur_t)         # run model plugs into network.py goes into TemporalZernNet (forward)  and StaticDiffuseNet (get all estimations)
                                                                            # y simulated measurement (I in eq.) forward model with unknown obejct unknown aberration and known SLM aberration to be compared with y_batch
                                                                            # I_est the estimated unknowm object (O in eq.)
                                                                            # sim_g modulus of estimated aberration ( M in eq.)
                                                                            # sim_phs phase of estimated aberration ( Phi in eq.)
            mse_loss = F.mse_loss(y, y_batch)

            # if epoch>100:
            #     loss = mse_loss + TV(I_est)/1E9
            # else:
            #     loss = mse_loss

            loss = mse_loss

            loss.backward()

            ph_opt.step()
            im_opt.step()
            # aber_opt.step()
            # optimizer.step()

            t.set_postfix(MSE=f'{mse_loss.item():.4e}')

            if args.vis_freq > 0 and (total_it % args.vis_freq) == 0:
                y, _kernel, sim_g, sim_phs, I_est = net(x_batch, torch.zeros_like(cur_t) - 0.5)

                Abe_est = fftshift(fft2(dset.a_slm.to(DEVICE) * sim_g, norm="forward"), dim=[-2, -1]).abs() ** 2
                if I_est.shape[0] > 1:
                    I_est = I_est[0:1]
                I_est = torch.clamp(I_est, 0, 1)
                yy = F.conv2d(I_est, Abe_est, padding='same').squeeze(0)

                #fig, ax = plt.subplots(1, 6, figsize=(48, 8))
                fig = plt.figure(figsize=(48, 8))
                # plt.ion()

                ax = fig.add_subplot(1, 6, 1)
                plt.imshow(y_batch[0].detach().cpu().squeeze(), vmin=0, vmax=1, cmap='gray')
                plt.axis('off')
                plt.title('Real Measurement')

                ax = fig.add_subplot(1, 6, 2)
                plt.imshow(y[0].detach().cpu().squeeze(), vmin=0, vmax=1, cmap='gray')
                plt.axis('off')
                plt.title('Sim Measurement')

                ax = fig.add_subplot(1, 6, 3)
                plt.imshow(I_est.detach().cpu().squeeze(), cmap='gray')
                plt.axis('off')
                plt.title('I_est')

                ax = fig.add_subplot(1, 6, 4)
                plt.imshow(torch.abs(sim_g[0]).detach().cpu().squeeze(), cmap='rainbow')
                plt.colorbar()
                plt.axis('off')
                plt.title(f'Sim Amp Error at t={idx[0]}')

                ax = fig.add_subplot(1, 6, 5)
                plt.imshow(sim_phs[0].detach().cpu().squeeze() % np.pi, cmap='rainbow')
                plt.colorbar()
                plt.axis('off')
                plt.title(f'Sim Phase Error at t={idx[0]}')

                ax = fig.add_subplot(1, 6, 6)
                plt.imshow(yy[0].squeeze().detach().cpu(), vmin=0, vmax=1, cmap='gray')
                plt.axis('off')
                plt.title(f'Abe_est * I_est at t={idx[0]}')

                plt.savefig(f'{vis_dir}/e_{epoch}_it_{it}.jpg')
                plt.clf()

                sio.savemat(f'{vis_dir}/Sim_Phase.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})

            total_it += 1

        im_sche.step()
        ph_sche.step()

    t1 = time.time()
    print(f'Training takes {t1 - t0} seconds.')

    ############
    # Export final results
    out_errs = []
    out_abes = []
    out_Iest = []
    for t in range(args.num_t):
        cur_t = (t / (args.num_t - 1)) - 0.5
        cur_t = torch.FloatTensor([cur_t]).to(DEVICE)

        I_est, sim_g, sim_phs = net.get_estimates(cur_t)
        I_est = torch.clamp(I_est, 0, 1).squeeze().detach().cpu().numpy()

        out_Iest.append(I_est)

        est_g = sim_g.detach().cpu().squeeze().numpy()
        out_errs.append(np.uint8(ang_to_unit(np.angle(est_g)) * 255))
        abe = sim_phs[0].detach().cpu().squeeze()
        abe = (abe - abe.min()) / (abe.max() - abe.min())
        out_abes.append(np.uint8(abe * 255))
        if args.save_per_frame and not args.static_phase:
          sio.savemat(f'{vis_dir}/final/per_frame/sim_phase_{t}.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})

    if args.dynamic_scene:
        out_Iest = [np.uint8(im * 255) for im in out_Iest]
        imageio.mimsave(f'{vis_dir}/final/final_I.gif', out_Iest, duration=1000*1./30)
    else:
        I_est = np.uint8(I_est.squeeze() * 255)
        imageio.imsave(f'{vis_dir}/final/final_I_est.png', I_est)

    if args.static_phase:
        imageio.imsave(f'{vis_dir}/final/final_aberrations_angle.png', out_errs[0])
        imageio.imsave(f'{vis_dir}/final/final_aberrations.png', out_abes[0])
    else:
        imageio.mimsave(f'{vis_dir}/final/final_aberrations_angle_grey.gif', out_errs, duration=1000*1./30)
        imageio.mimsave(f'{vis_dir}/final/final_aberrations.gif', out_abes, duration=1000*1./30)

    print("Training concludes.")

    colored_err = []
    for i, a in enumerate(out_errs):
        plt.imsave(f'{vis_dir}/final/per_frame/{i:03d}.jpg', a, cmap='rainbow')
        colored_err.append(imageio.imread(f'{vis_dir}/final/per_frame/{i:03d}.jpg'))
    imageio.mimsave(f'{vis_dir}/final/final_aberrations_angle.gif', colored_err, duration=1000*1./30)

    # for k in range(0,32):
    #     fig = plt.figure(figsize=(24, 8))
    #     ax = fig.add_subplot(1, 3, 1)
    #     im = torch.abs(x_batches[k,0, :, :])
    #     im = im.detach().cpu().numpy()
    #     plt.imshow(np.squeeze(im))
    #     plt.colorbar()
    #     plt.title('Amplitude SLM')
    #     ax = fig.add_subplot(1, 3, 2)
    #     im = torch.angle(x_batches[k,0, :, :])
    #     im = im.detach().cpu().numpy()
    #     plt.imshow(np.squeeze(im))
    #     plt.colorbar()
    #     plt.title('Phase SLM')
    #     ax = fig.add_subplot(1, 3, 3)
    #     im = torch.abs(y_batches[k, :, :])
    #     im = im.detach().cpu().numpy()
    #     plt.imshow(np.squeeze(im))
    #     plt.colorbar()
    #     plt.title(f'Measurement {k}/100')
    #     plt.savefig(f"./tmp/data_{k}.tif")
    #     plt.close()

    # t1 = net.g_im.net1.data
    # t2 = net.g_im.net2.data
    # t3 = net.g_im.net3.data
    # t4 = net.g_im.net4.data


    # for k in range(0,32):
    #     fig = plt.figure(figsize=(12, 12))
    #     ax = fig.add_subplot(2, 2, 1)
    #     im = t1[:,:,k]
    #     im = im.detach().cpu().numpy()
    #     plt.imshow(np.squeeze(im),vmin=-0.5,vmax=0.5)
    #     plt.colorbar()
    #     plt.title('p1')
    #     ax = fig.add_subplot(2, 2, 2)
    #     im = t2[:,:,k]
    #     im = im.detach().cpu().numpy()
    #     plt.imshow(np.squeeze(im),vmin=-0.5,vmax=0.5)
    #     plt.colorbar()
    #     plt.title('p2')
    #     ax = fig.add_subplot(2, 2, 3)
    #     im = t3[:,:,k]
    #     im = im.detach().cpu().numpy()
    #     plt.imshow(np.squeeze(im),vmin=-0.5,vmax=0.5)
    #     plt.colorbar()
    #     plt.title('p3')
    #     ax = fig.add_subplot(2, 2, 4)
    #     im = t4[:,:,k]
    #     im = im.detach().cpu().numpy()
    #     plt.imshow(np.squeeze(im),vmin=-0.5,vmax=0.5)
    #     plt.colorbar()
    #     plt.title('p4')
    #     plt.savefig(f"./tmp/data_{k}.tif")
    #     plt.close()