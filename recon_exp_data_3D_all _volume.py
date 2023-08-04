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
from networks3D_all_volume import *
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

    flist = ['Recons3D']
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
    parser.add_argument('--phs_draw', default=25, type=int)
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

    net = StaticDiffuseNet(width=args.width, PSF_size=PSF_size, use_pe=False, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase, phs_draw=args.phs_draw)
    net = net.to(DEVICE)
    state_dict = torch.load(os.path.join('./Recons3D/', f'model_epoch_{best_epoch}.pt'))
    net.load_state_dict(state_dict['model_state_dict'])


    net.train()

    im_opt = torch.optim.Adam(net.g_im.parameters(), lr=args.init_lr)
    # ph_opt = torch.optim.Adam(net.g_g.parameters(), lr=args.init_lr)
    ph_opt = torch.optim.Adam(net.dn_im.parameters(), lr=args.init_lr)

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
    dn = imread(f'./Pics_input/dn_reversed.tif')
    loss = 0


    im_opt = torch.optim.Adam(net.g_im.parameters(), lr=1E-3)
    ph_opt = torch.optim.Adam(net.dn_im.parameters(), lr=1E-3)


    print ('Initialisation of the neural representation')

    total_it = 0

    for epoch in range(100000):

        plane=np.random.randint(256)
        # plane = 255

        it_list = np.random.permutation(np.arange(100))
        it_list = it_list[0:20]

        for it in it_list:

            total_it += 1

            x_batch, y_batch = x_batches[it, plane, :, :], y_batches[it, plane, :, :]

            y_batch = torch.squeeze(y_batch) * 10

            cur_t = plane/256

            im_opt.zero_grad()
            ph_opt.zero_grad()

            y, F_estimated, Phi_estimated = net(torch.squeeze(x_batch), cur_t)

            loss = 2 * F.mse_loss(y, y_batch) + 1E-4 * TV(F_estimated)  # - torch.log(Phi_estimated.norm(1))

            loss.backward()

            im_opt.step()
            # ph_opt.step()

            if total_it%2000==0:

                fig = plt.figure(figsize=(24, 6))
                # plt.ion()
                ax = fig.add_subplot(1, 6, 1)
                plt.imshow(y_batch.detach().cpu().squeeze(), vmin=0, vmax=0.5, cmap='gray')
                plt.axis('off')
                plt.title('Simulated measurement')
                ax = fig.add_subplot(1, 6, 2)
                plt.imshow(y.detach().cpu().squeeze(), cmap='gray')
                plt.axis('off')
                plt.title('Reconstructed measurement')
                ax = fig.add_subplot(1, 6, 3)
                plt.imshow((F_estimated ** 2).detach().cpu().squeeze(), vmin=0, vmax=0.5, cmap='gray')
                plt.axis('off')
                plt.title('fluo_est')
                ax = fig.add_subplot(1, 6, 4)
                plt.imshow(Phi_estimated[plane].detach().cpu().squeeze(), cmap='rainbow')
                plt.axis('off')
                plt.title(f'Phi_estimated')
                mmin = torch.min(Phi_estimated).item()
                mmax = torch.max(Phi_estimated).item()
                mstd = torch.std(Phi_estimated).item()
                mmean = torch.mean(Phi_estimated).item()
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
                plt.savefig(f'./Recons3D/it_{total_it}_plane_{plane}.jpg')
                plt.clf()

        print(f'     total_it: {total_it}  loss: {np.round(loss.item(), 5)}')

        if epoch%256==0:

            torch.save({'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': im_opt.state_dict()},
                       os.path.join('./Recons3D/', f'model_epoch_{epoch}.pt'))

            Fluo_all = torch.zeros(256, 256, 256, device='cuda:0')

            print('Saving volume ...')

            for plane in range(255, -1, -1):
                F_est = torch.squeeze(net.g_im(t=plane/256))  # torch.Size([1, 1, 256, 256])  requires_grad=True

                with torch.no_grad():
                    Fluo_all[plane, :, :] = F_est

            imwrite(f'./Recons3D/fluo_epoch_{epoch}.tif', Fluo_all.detach().cpu().numpy())

            Fluo_all = []


    if False:
        for epoch in range(1000):

            print(f'epoch: {epoch}')

            plane_list=np.arange(256)
            plane_list=np.random.permutation(plane_list)

            loss_list=np.ones(256)

            for plane in plane_list:

                total_it = 0

                it_list = np.arange(100)
                it_list = np.random.permutation(it_list)
                it_list = it_list[0:10]

                for it in it_list:

                    total_it += 1

                    x_batch, y_batch = x_batches[it,plane,:,:], y_batches[it,plane,:,:]

                    y_batch = torch.squeeze(y_batch) * 10

                    cur_t = plane

                    im_opt.zero_grad();  ph_opt.zero_grad()

                    y, F_estimated, Phi_estimated = net(torch.squeeze(x_batch), cur_t)

                    loss = 2 * F.mse_loss(y, y_batch) # + 2E-4 * TV(F_estimated) # - torch.log(Phi_estimated.norm(1))

                    # loss = F.mse_loss(y, y_batch) + torch.abs(torch.std(Phi_estimated)-1E-1) + torch.abs(torch.mean(Phi_estimated)-0.05) + 0*TV(Phi_estimated)*1E-2 + TV(F_estimated)*1E-4

                    if epoch<5:
                        print(f'     plane: {plane}  it: {it}  loss: {loss}')

                    loss.backward()

                    loss_list[plane]=loss.item()

                    im_opt.step()
                    ph_opt.step()

            # Phi_estimated = torch.zeros(256,256,256,device='cuda:0')
            #
            # for dn_plane in range (t,256):
            #         z_embedding = torch.squeeze(self.z_embedding[:, dn_plane, :])
            #         Phi_estimated[dn_plane,:,:] = torch.squeeze(self.dn_im(z_embedding=z_embedding))

            Phi_all = torch.zeros(256, 256, 256, device='cuda:0')
            Fluo_all = torch.zeros(256, 256, 256, device='cuda:0')

            np.save(f'./Recons3D/loss_epoch_{epoch}.npy',loss_list)

            print(f'     epoch: {epoch}  mean loss: {np.round(np.mean(loss_list),5)}')

            print('Saving volume ...')

            for plane in range(255,-1,-1):

                z_embedding = torch.squeeze(net.z_embedding[:, plane, :])
                F_est = torch.squeeze(net.g_im(z_embedding=z_embedding))  # torch.Size([1, 1, 256, 256])  requires_grad=True
                dn_est = torch.squeeze(net.dn_im(z_embedding=z_embedding))

                with torch.no_grad():
                    Fluo_all[plane,:,:] = F_est
                    Phi_all[plane,:,:] = dn_est

            imwrite(f'./Recons3D/fluo_epoch_{epoch}.tif',Fluo_all.detach().cpu().numpy())
            imwrite(f'./Recons3D/dn_epoch_{epoch}.tif', Phi_all.detach().cpu().numpy())

            Phi_all=[]
            Fluo_all=[]


            fig = plt.figure(figsize=(24, 6))
            plt.ion()
            ax = fig.add_subplot(1, 6, 1)
            plt.imshow(y_batch.detach().cpu().squeeze(), vmin=0, vmax=0.5, cmap='gray')
            plt.axis('off')
            plt.title('Simulated measurement')
            ax = fig.add_subplot(1, 6, 2)
            plt.imshow(y.detach().cpu().squeeze(), cmap='gray')
            plt.axis('off')
            plt.title('Reconstructed measurement')
            ax = fig.add_subplot(1, 6, 3)
            plt.imshow((F_estimated ** 2).detach().cpu().squeeze(), cmap='gray')
            plt.axis('off')
            plt.title('fluo_est')
            ax = fig.add_subplot(1, 6, 4)
            plt.imshow(Phi_estimated[plane].detach().cpu().squeeze(), cmap='rainbow')
            plt.axis('off')
            plt.title(f'Phi_estimated')
            mmin = torch.min(Phi_estimated).item()
            mmax = torch.max(Phi_estimated).item()
            mstd = torch.std(Phi_estimated).item()
            mmean = torch.mean(Phi_estimated).item()
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


