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

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os

import glob
import torch_geometric as pyg
import torch_geometric.data as data
import math
import torch_geometric.utils as pyg_utils
import torch.nn as nn
from torch.nn import functional as F
from shutil import copyfile
from prettytable import PrettyTable
import time
import networkx as nx
from torch_geometric.utils.convert import to_networkx

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

    target = imread(f'./Pics_input/target.tif')
    dn = imread(f'./Pics_input/dn_reversed.tif')
    loss = 0


    for gridsearch in range(10):

        print(f'Gridsearch {gridsearch} #######################################')

        net = StaticDiffuseNet(width=args.width, PSF_size=PSF_size, use_pe=False, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase, phs_draw=args.phs_draw)
        net = net.to(DEVICE)
        state_dict = torch.load(os.path.join('./Recons3D_NNR_0/', f'model_epoch_3584.pt'))
        net.load_state_dict(state_dict['model_state_dict'])
        net.train()

        im_opt = torch.optim.Adam(net.g_im.parameters(), lr=args.init_lr)
        im_opt.load_state_dict(state_dict['optimizer_state_dict'])

        ph_opt = torch.optim.Adam(net.dn_im.parameters(), lr=args.init_lr)

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


        im_opt = torch.optim.Adam(net.g_im.parameters(), lr=1E-3)
        ph_opt = torch.optim.Adam(net.dn_im.parameters(), lr=1E-3)

        total_it = 0

        for epoch in range(512):

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

                if gridsearch==0:
                    loss = 2 * F.mse_loss(y, y_batch) + 5E-4 * TV(F_estimated)  # - torch.log(Phi_estimated.norm(1))
                if gridsearch==1:
                    loss = 2 * F.mse_loss(y, y_batch) + 2.5E-4 * TV(F_estimated)  # - torch.log(Phi_estimated.norm(1))
                if gridsearch==2:
                    loss = 2 * F.mse_loss(y, y_batch) + 2E-4 * TV(F_estimated)  - 1E-4* torch.log(Phi_estimated.norm(1))
                if gridsearch==3:
                    loss = 2 * F.mse_loss(y, y_batch) + 2E-4 * TV(F_estimated)  - 1E-3* torch.log(Phi_estimated.norm(1))
                if gridsearch==4:
                    loss = 2 * F.mse_loss(y, y_batch) + 2E-4 * TV(F_estimated)  - 1E-2* torch.log(Phi_estimated.norm(1))
                if gridsearch==5:
                    loss = 2 * F.mse_loss(y, y_batch) + 2E-4 * TV(F_estimated)  - 1E-1 * torch.log(Phi_estimated.norm(1))
                if gridsearch==6:
                    loss = 2 * F.mse_loss(y, y_batch) + 2E-4 * TV(F_estimated) + 1E-7 * F_estimated.norm(1)  # - torch.log(Phi_estimated.norm(1))
                if gridsearch==7:
                    loss = 2 * F.mse_loss(y, y_batch) + 2E-4 * TV(F_estimated) + 1E-6 * F_estimated.norm(1)  # - torch.log(Phi_estimated.norm(1))
                if gridsearch==8:
                    loss = 2 * F.mse_loss(y, y_batch) + 2E-4 * TV(F_estimated) + 1E-5 * F_estimated.norm(1)  # - torch.log(Phi_estimated.norm(1))
                if gridsearch==9:
                    loss = 2 * F.mse_loss(y, y_batch) + 2E-4 * TV(F_estimated) + 1E-4 * F_estimated.norm(1)  # - torch.log(Phi_estimated.norm(1))

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
                    plt.savefig(f'./Recons3D/it_{total_it}_plane_{plane}_grid_{gridsearch}.jpg')
                    plt.clf()

            print(f'     total_it: {total_it}  loss: {np.round(loss.item(), 5)}')

            if epoch%256==0:

                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': im_opt.state_dict()},
                           os.path.join('./Recons3D/', f'model_epoch_{epoch}_grid_{gridsearch}.pt'))

                Fluo_all = torch.zeros(256, 256, 256, device='cuda:0')

                print('Saving volume ...')

                plane=0
                with torch.no_grad():
                    y, F_estimated, Phi_estimated = net(torch.squeeze(x_batch), 0)
                imwrite(f'./Recons3D/phi_epoch_{epoch}_grid_{gridsearch}.tif', Phi_estimated.detach().cpu().numpy())

                for plane in range(255, -1, -1):
                    F_est = torch.squeeze(net.g_im(t=plane/256))  # torch.Size([1, 1, 256, 256])  requires_grad=True
                    with torch.no_grad():
                        Fluo_all[plane, :, :] = F_est

                imwrite(f'./Recons3D/fluo_epoch_{epoch}_grid_{gridsearch}.tif', Fluo_all.detach().cpu().numpy())

                Fluo_all = []



###########################################################################################3



def normalize99(Y, lower=1,upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return x01, x99


def display_frame(t=20):
    s = t/(niter//save_per-1)
    plt.scatter(Zsvg[:,0,t], Zsvg[:,1,t], color=[s,0,1-s])
    plt.axis('equal')
    plt.axis([0,1,0,1])

def distmat_square(X,Y):
    return torch.sum( bc_diff(X[:,None,:] - Y[None,:,:])**2, axis=2 )

def distmat_square2(X, Y):
    X_sq = (X ** 2).sum(axis=-1)
    Y_sq = (Y ** 2).sum(axis=-1)
    cross_term = X.matmul(Y.T)
    return X_sq[:, None] + Y_sq[None, :] - 2 * cross_term

def kernel(X,Y):
    return -torch.sqrt( distmat_square(X,Y) )

def MMD(X,Y):
    n = X.shape[0]
    m = Y.shape[0]
    a = torch.sum( kernel(X,X) )/n**2 + \
      torch.sum( kernel(Y,Y) )/m**2 - \
      2*torch.sum( kernel(X,Y) )/(n*m)
    return a.item()

def psi(r,p):
    sigma = .05;
    return -p[2]*torch.exp(-r**p[0] / (2 * sigma ** 2)) + p[3]* torch.exp(-r**p[1] / (2 * sigma ** 2))
def Speed(X,Y,p):
    sigma = .05;

    temp=distmat_square(X,Y)
    return 0.25/X.shape[0] * 1/sigma**2 * torch.sum(psi(distmat_square(X,Y),p)[:,:,None] * bc_diff( X[:,None,:] - Y[None,:,:] ), axis=1 )

def Edge_index(X,Y):

    return torch.sum( bc_diff(X[:,None,:] - Y[None,:,:])**2, axis=2 )


class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, nlayers, device):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layernorm = False

        for i in range(nlayers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == nlayers - 1 else hidden_size, device=device, dtype=torch.float64
            ))
            if i != nlayers - 1:
                self.layers.append(torch.nn.ReLU())
                # self.layers.append(torch.nn.Dropout(p=0.0))
        if self.layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size, device=device, dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class InteractionParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, in_feats=9, out_feats=2, num_layers=2, hidden=16):

        super(InteractionParticles, self).__init__(aggr='add')  # "Add" aggregation.

        self.lin_edge = MLP(in_feats=11, out_feats=2, num_layers=3, hidden=32)
        # self.lin_node = MLP(in_feats=4, out_feats=1, num_layers=2, hidden=16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=(x,x))
        return acc

    def message(self, x_i, x_j):

        r = torch.sqrt(torch.sum((x_i[:,0:2] - x_j[:,0:2])**2,axis=1)) / radius  # squared distance
        r = r[:, None]

        delta_pos=(x_i[:,0:2]-x_j[:,0:2]) / radius
        x_i_vx = x_i[:, 2:3]  / vnorm[4]
        x_i_vy = x_i[:, 3:4]  / vnorm[5]
        x_i_type= x_i[:,4]
        x_j_vx = x_j[:, 2:3]  / vnorm[4]
        x_j_vy = x_j[:, 3:4]  / vnorm[5]

        in_features = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type[:,None].repeat(1,4)), dim=-1)

        return self.lin_edge(in_features)
    def update(self, aggr_out):

        return aggr_out     #self.lin_node(aggr_out)

class EdgeNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        return self.new_edges

    def message(self, x_i, x_j, edge_feature):

        r = torch.sqrt(torch.sum((x_i[:,0:2] - x_j[:,0:2])**2,axis=1)) / radius  # squared distance
        r = r[:, None]

        delta_pos=(x_i[:,0:2]-x_j[:,0:2]) / radius
        x_i_vx = x_i[:, 2:3]  / vnorm[4]
        x_i_vy = x_i[:, 3:4]  / vnorm[5]
        x_i_type= x_i[:,4]
        x_j_vx = x_j[:, 2:3]  / vnorm[4]
        x_j_vy = x_j[:, 3:4]  / vnorm[5]

        d = r

        self.new_edges = torch.cat((delta_pos, r, x_i_vx, x_i_vy, x_j_vx, x_j_vy, x_i_type[:,None].repeat(1,4)), dim=-1)

        return d

class InteractionNetworkEmb(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, nlayers, embedding, device):
        super().__init__(aggr='add')  # "Add" aggregation.

        self.nlayers = nlayers
        self.device = device
        self.embedding = embedding

        self.lin_edge = MLP(input_size=3*self.embedding, hidden_size=3*self.embedding, output_size=self.embedding, nlayers= self.nlayers, device=self.device)
        self.lin_node = MLP(input_size=2*self.embedding, hidden_size=2*self.embedding, output_size=self.embedding, nlayers= self.nlayers, device=self.device)


    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        node_out = x + node_out
        edge_out = edge_feature + self.new_edges

        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):

        x = torch.cat((edge_feature, x_i, x_j ), dim=-1)

        x = self.lin_edge(x)
        self.new_edges = x

        return x

class ReccurentGNN(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(self,model_config, device):
        super().__init__()

        self.hidden_size = model_config['hidden_size']
        self.embedding = model_config['embedding']
        self.nlayers = model_config['n_mp_layers']
        self.device = device
        self.noise_level = model_config['noise_level']

        self.edge_init = EdgeNetwork()

        self.layer = torch.nn.ModuleList([InteractionNetworkEmb(nlayers=3, embedding=self.embedding, device=self.device) for _ in range(self.nlayers)])
        self.node_out = MLP(input_size=self.embedding, hidden_size=self.hidden_size, output_size=2, nlayers=3, device=self.device)

        self.embedding_node = MLP(input_size=8, hidden_size=self.embedding, output_size=self.embedding, nlayers=3, device=self.device)
        self.embedding_edges = MLP(input_size=11, hidden_size=self.embedding, output_size=self.embedding, nlayers=3, device=self.device)

    def forward(self, data):

        node_feature = torch.cat((data.x[:,0:4],data.x[:,4:5].repeat(1,4)), dim=-1)

        noise = torch.randn((node_feature.shape[0], node_feature.shape[1]),requires_grad=False, device='cuda:0') * self.noise_level
        node_feature= node_feature+noise
        edge_feature = self.edge_init(node_feature, data.edge_index, edge_feature=data.edge_attr)

        node_feature = node_feature.to(dtype=torch.float64)
        edge_feature = edge_feature.to(dtype=torch.float64)

        node_feature = self.embedding_node(node_feature)
        edge_feature = self.embedding_edges(edge_feature)

        for i in range(self.nlayers):
            node_feature, edge_feature = self.layer[i](node_feature, data.edge_index, edge_feature=edge_feature)

        pred = self.node_out(node_feature)

        return pred


















    print(f"Using PyTorch Version: {torch.__version__}")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()



    niter = 400
    d = 2  # dimension
    radius = 0.075

    p=torch.load('./p_list_simu_N5.pt')


    # datum = '230721'
    # print(datum)
    # ntypes_list=[5]
    # n_list=[250]

    # datum = '230724'
    # print(datum)
    # ntypes_list=[5]
    # n_list=[500]

    datum = '230802'
    print(datum)
    ntypes_list=[2]
    n_list=[2000]

    ntypes=ntypes_list[0]
    nparticles=n_list[0]

    boundary = 'no'  # no boundary condition
    # boundary = 'per' # periodic
    if boundary == 'no':
        tau = 1 / 1000  # time step
    else:
        tau = 1 / 200
    if boundary == 'no':  # change this for usual BC
        def bc_pos(X):
            return X
        def bc_diff(D):
            return D
    else:
        def bc_pos(X):
            return torch.remainder(X, 1.0)
        def bc_diff(D):
            return torch.remainder(D - .5, 1.0) - .5

    step = 1

    # train
    if step==1:

    # train data

        graph_files = glob.glob(f"../Graph/graphs_data/graphs_particles_{datum}/edge*")
        NGraphs=int(len(graph_files)/niter)
        print ('Graph files N: ',NGraphs)
        print('Normalize ...')
        time.sleep(0.5)

        arr = np.arange(0,NGraphs-1,2)
        for run in tqdm(arr):
            x=torch.load(f'../Graph/graphs_data/graphs_particles_{datum}/x_list_{run}.pt')
            acc=torch.load(f'../Graph/graphs_data/graphs_particles_{datum}/acc_list_{run}.pt')
            if run == 0:
                xx = x
                aacc = acc
            else:
                xx = torch.concatenate((x, xx))
                aacc = torch.concatenate((aacc, acc))

        mvx = torch.mean(xx[:,:,0,:])
        mvy = torch.mean(xx[:,:,1,:])
        vx = torch.std(xx[:,:,0,:])
        vy = torch.std(xx[:,:,1,:])
        nvx = np.array(xx[:,:,0,:].detach().cpu())
        vx01, vx99 = normalize99(nvx)
        nvy = np.array(xx[:,:,1,:].detach().cpu())
        vy01, vy99 = normalize99(nvy)
        vnorm = torch.tensor([vx01, vx99, vy01, vy99, vx, vy], device=device)


        print(f'v_x={mvx} +/- {vx}')
        print(f'v_y={mvy} +/- {vy}')
        print(f'vx01={vx01} vx99={vx99}')
        print(f'vy01={vy01} vy99={vy99}')

        max = torch.mean(aacc[:,:,0,:])
        may = torch.mean(aacc[:,:,1,:])
        ax = torch.std(aacc[:,:,0,:])
        ay = torch.std(aacc[:,:,1,:])
        nax = np.array(aacc[:,:,0,:].detach().cpu())
        ax01, ax99 = normalize99(nax)
        nay = np.array(aacc[:,:,1,:].detach().cpu())
        ay01, ay99 = normalize99(nay)

        ynorm = torch.tensor([ax01, ax99, ay01, ay99, ax, ay], device=device)

        print(f'acc_x={max} +/- {ax}')
        print(f'acc_y={may} +/- {ay}')
        print(f'ax01={ax01} ax99={ax99}')
        print(f'ay01={ay01} ay99={ay99}')

        best_loss = np.inf

        batch_size = 1

        model_config = {'ntry': 513,
                        'embedding': 128,
                        'hidden_size': 32,
                        'n_mp_layers': 4,
                        'noise_level': 0}

        ntry=model_config['ntry']

        print(f"ntry :{ntry}")

        l_dir = os.path.join('..','Graph','log')
        log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
        print('log_dir: {}'.format(log_dir))

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)

        copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))
        torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
        torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))

        grid_list = [1, 2, 5, 10, 20, 100, 200]

        for gridsearch in grid_list:

            model = ReccurentGNN(model_config=model_config, device=device)

            table = PrettyTable(["Modules", "Parameters"])
            total_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                param = parameter.numel()
                table.add_row([name, param])
                total_params += param
            print(table)
            print(f"Total Trainable Params: {total_params}")

            optimizer= torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            model.train()

            for epoch in range(24):

                model.train()
                total_loss = []

                for N in tqdm(range(10000)):

                    run = 1 + np.random.randint(gridsearch)

                    x_list=torch.load(f'graphs_data/graphs_particles_{datum}/x_list_{run}.pt')
                    acc_list=torch.load(f'graphs_data/graphs_particles_{datum}/acc_list_{run}.pt')

                    acc_list[:,:, 0,:] = acc_list[:,:, 0,:] / ynorm[4]
                    acc_list[:,:, 1,:] = acc_list[:,:, 1,:] / ynorm[5]

                    optimizer.zero_grad()
                    loss = 0

                    for loop in range(batch_size):

                        k = np.random.randint(niter - 1)

                        edges = torch.load(f'graphs_data/graphs_particles_{datum}/edge_{run}_{k}.pt')

                        x=torch.squeeze(x_list[k,:,:,:])
                        x = torch.permute(x, (2, 0, 1))
                        x = torch.reshape (x,(nparticles*ntypes,5))
                        dataset = data.Data(x=x, edge_index=edges)
                        y = torch.squeeze(acc_list[k,:,:,:])
                        y = torch.permute(y, (2, 0, 1))
                        y = torch.reshape (y,(nparticles*ntypes,2))

                        pred = model(dataset)

                        datafit = (pred-y).norm(2)

                        loss += datafit

                        total_loss.append(datafit.item())

                    loss.backward()
                    optimizer.step()

                    if (epoch==0) & (N%10==0):
                        print("   N {} Loss: {:.4f}".format(N, np.mean(total_loss) / nparticles))
                    if N%250==0:
                        print("   N {} Loss: {:.4f}".format(N,np.mean(total_loss)/nparticles ))

                print("Epoch {}. Loss: {:.4f}".format(epoch, np.mean(total_loss)/nparticles ))

                if (np.mean(total_loss) < best_loss):
                    best_loss = np.mean(total_loss)
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(log_dir, 'models', f'best_model_trained_with_{gridsearch}_graph.pt'))
                    print('\t\t Saving model')


    # test
    if step==2:


        # model_config = {'ntry': 502,
        #                 'embedding': 32,
        #                 'n_mp_layers': 4,
        #                 'hidden_size': 32,
        #                 'noise_level': 0}
        #
        # model_config = {'ntry': 503,
        #                 'embedding': 128,
        #                 'n_mp_layers': 4,
        #                 'hidden_size': 32,
        #                 'noise_level': 0}
        #
        # model_config = {'ntry': 504,
        #                 'embedding': 64,
        #                 'n_mp_layers': 4,
        #                 'hidden_size': 32,
        #                 'noise_level': 0}
        #
        model_config = {'ntry': 505,
                        'embedding': 32,
                        'n_mp_layers': 4,
                        'hidden_size': 32,
                        'noise_level': 0}

        datum = '230724'
        print(datum)

        ntry=model_config['ntry']
        print(f"ntry :{ntry}")
        print(f"embedding: {model_config['embedding']}")


        model = ReccurentGNN(model_config=model_config, device=device)
        state_dict = torch.load(f"../Graph/log/try_{ntry}/models/best_model.pt")
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()


        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")

        GT = torch.load(f'../Graph/graphs_data/graphs_particles_{datum}/x_list_0.pt')
        GT0 = torch.squeeze (GT[0])

        ynorm = torch.load(f'../Graph/log/try_{ntry}/ynorm.pt')
        vnorm = torch.load(f'../Graph/log/try_{ntry}/vnorm.pt')

        run=0
        acc_list = torch.load(f'graphs_data/graphs_particles_{datum}/acc_list_{run}.pt')
        acc_list[:, :, 0, :] = acc_list[:, :, 0, :] / ynorm[4]
        acc_list[:, :, 1, :] = acc_list[:, :, 1, :] / ynorm[5]

        edges = torch.load(f'../Graph/graphs_data/graphs_particles_{datum}/edge_{run}_0.pt')


        print(f"ntry :{ntry}")
        print(f"run :{run}")
        dataset = data.Data(x=GT0.cuda(), edge_index=edges.cuda())
        print('num_nodes: ', GT0.shape[0])
        nparticles = GT0.shape[0]
        print('dataset.num_node_features: ', dataset.num_node_features)

        x = GT0
        x = torch.permute(x, (2, 0, 1))
        x = torch.reshape(x, (nparticles * ntypes, 5))

        GT0 = x.clone()

        for it in tqdm(range(200)):

            x0 = torch.squeeze (GT[it+1])
            x0 = torch.permute(x0, (2, 0, 1))
            x0 = torch.reshape(x0, (nparticles * ntypes, 5))

            distance = torch.sum((x[:, None, 0:2] - x[None, :, 0:2]) ** 2, axis=2)
            t = torch.Tensor([radius ** 2])  # threshold
            adj_t = (distance < radius ** 2).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                y = model(dataset)  # acceleration estimation

            # y = torch.squeeze(acc_list[0, :, :, :])
            # y = torch.permute(y, (2, 0, 1))
            # y = torch.reshape(y, (nparticles * ntypes, 2))

            y[:, 0] = y[:, 0] * ynorm[4]
            y[:, 1] = y[:, 1] * ynorm[5]
            x[:, 2:4] = x[:, 2:4] + y  # speed update
            x[:, 0:2] = x[:, 0:2] + x[:, 2:4]  # position update

            fig = plt.figure(figsize=(25, 16))
            # plt.ion()
            ax = fig.add_subplot(2, 3, 1)
            for k in range(ntypes):
                plt.scatter(GT0[k*nparticles:(k+1)*nparticles, 0].detach().cpu(), GT0[k*nparticles:(k+1)*nparticles, 1].detach().cpu(), s=3)

            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, 'Distribution at t0 is 1.0x1.0')

            ax = fig.add_subplot(2, 3, 2)
            for k in range(ntypes):
                plt.scatter(x0[k*nparticles:(k+1)*nparticles, 0].detach().cpu(), x0[k*nparticles:(k+1)*nparticles, 1].detach().cpu(), s=3)
            ax = plt.gca()
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, 'True', fontsize=30)
            # plt.text(-0.25, 1.38, f'Frame: {min(niter,it)}')
            # plt.text(-0.25, 1.33, f'Physics simulation', fontsize=10)

            ax = fig.add_subplot(2, 3, 4)
            pos = dict(enumerate(x[:, 0:2].detach().cpu().numpy(), 0))
            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            nx.draw(vis, pos=pos, ax=ax, node_size=10, linewidths=0)
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, f'Frame: {it}')
            plt.text(-0.25, 1.33, f'Graph: {x.shape[0]} nodes {edge_index.shape[1]} edges ', fontsize=10)

            ax = fig.add_subplot(2, 3, 5)
            for k in range(ntypes):
                plt.scatter(x[k*nparticles:(k+1)*nparticles, 0].detach().cpu(), x[k*nparticles:(k+1)*nparticles, 1].detach().cpu(), s=3)
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            plt.xlim([-0.3, 1.3])
            plt.ylim([-0.3, 1.3])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            plt.text(-0.25, 1.38, 'Model', fontsize=30)

            plt.savefig(f"../Graph/ReconsGraph/Fig_{it}.tif")
            plt.close()










