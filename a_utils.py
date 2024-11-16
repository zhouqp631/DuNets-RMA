import os
import shelve

import cvxpy as cp
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio
from eit_circle_simulation import show_tripcolor

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
device = torch.device('cpu')

def compute_aplpha(G):
    P = G.T @ G
    epsilon = 1e-3
    while np.linalg.det(P)<1e-6:
        P += epsilon*np.eye(P.shape[0])
    m,n = G.shape
    x = cp.Variable(n)
    A = np.ones(n)
    b = 1
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)),[A @ x == b])
    prob.solve()
    # assert x.value.round(2).sum()==1.0
    return x.value.round(2)

def get_neighbor(element):
    neighbors = []
    for i, e1 in enumerate(element):
        temp = []
        for j, e2 in enumerate(element):
            if j != i and len(set(e1).intersection(e2)) > 0:
                temp.append(j)
        neighbors.append(temp)
    # hypers['neighbors'] = neighbors
    return neighbors

def compute_metrics(sigmas,sigma_preds,vs,hypers):
    fwd = hypers['fun']['fwd']
    ex_mat = hypers['data']['ex_mat']
    step = hypers['data']['step']
    neighbors = hypers['neighbors']
    metrics = []
    data_range = np.max(gpu2numpy(sigmas[0]))-np.min(gpu2numpy(sigmas[0]))
    c1 = np.max(gpu2numpy(sigmas[0]))*0.01**2
    c2 = np.max(gpu2numpy(sigmas[0]))*0.03**2
    for sigma,sigma_pred,v in zip(sigmas,sigma_preds,vs):
        sigma = gpu2numpy(sigma)
        sigma_pred = gpu2numpy(sigma_pred)
        MSE = np.mean((sigma-sigma_pred)**2)

        v = gpu2numpy(v)
        v_pred = fwd.solve_eit(ex_mat, step, perm=sigma_pred[0], parser="std")
        RE_sigma = np.linalg.norm(sigma-sigma_pred,1)/np.linalg.norm(sigma)
        RE_v = np.linalg.norm(v[0]-v_pred.v,1)/np.linalg.norm(v)
        DR =  (np.max(sigma_pred)-np.min(sigma_pred))/(np.max(sigma)-np.min(sigma))*100
        PSNR = peak_signal_noise_ratio(sigma, sigma_pred,data_range=data_range)

        # compute SSIM (ref: Learning Nonlinear Electrical Impedance Tomography)
        ssim = []
        for nb in neighbors:
            gt_n = sigma[0][nb]
            pred_n = sigma_pred[0][nb]

            mu_gt,mu_pred = gt_n.mean(),pred_n.mean()
            std_gt,std_pred = gt_n.std(),pred_n.std()

            std_gt_star = np.mean((gt_n-mu_gt)*(pred_n-mu_pred))

            temp1 = (2*mu_gt*mu_pred+c1)*(2*std_gt_star+c2)
            temp2 = (mu_gt**2+mu_pred**2+c1)*(std_gt**2+std_pred**2+c2)
            ssim_i = temp1/temp2
            if ssim_i>1:
                print('xx')
            ssim.append(ssim_i)
        SSIM = np.mean(ssim)
        metrics.append([MSE,RE_sigma,RE_v,DR,PSNR,SSIM])

    return np.array(metrics).mean(axis=0).tolist(),np.array(metrics).std(axis=0).tolist()

def plot_preds(pred,ground_truth,pred_gn,title,d):
    mesh_obj = d['data']['mesh_obj']

    n_col = ground_truth.shape[0]
    vmax = ground_truth[0].max().item()*1.1
    fig, axes = plt.subplots(3, n_col, figsize=(n_col * 4, 3 * 4))
    for i, (p, gt, pg) in enumerate(zip(pred, ground_truth, pred_gn)):
        is_set_colorbar = False
        mesh_obj['perm'] = p[0].cpu().numpy()
        show_tripcolor(mesh_obj, ax=axes[0, i], is_set_colorbar=is_set_colorbar,vmax=vmax)

        mesh_obj['perm'] = gt[0].cpu().numpy()
        show_tripcolor(mesh_obj, ax=axes[1, i], is_set_colorbar=is_set_colorbar,vmax=vmax)

        mesh_obj['perm'] = pg[0].cpu().numpy()
        im = show_tripcolor(mesh_obj, ax=axes[2, i], is_set_colorbar=is_set_colorbar,vmax=vmax)
        if i >= (n_col - 1): break
    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    return fig

def forward_and_jac(perms,hypers):
    fwd = hypers['fun']['fwd']
    ex_mat = hypers['data']['ex_mat']
    step = hypers['data']['step']

    perms = perms.cpu().detach().numpy()
    f1s, jacs = [],[]
    for perm in perms:
        f1 = fwd.solve_eit(ex_mat, step, perm=perm[0], parser="std")
        f1s.append(f1.v)
        jacs.append(f1.jac)
    Kf = torch.from_numpy(np.array(f1s)).float()
    jacs = torch.from_numpy(np.array(jacs)).float()
    return Kf.unsqueeze(axis=1).to(device),jacs.to(device)

class EITdataset(Dataset):
    def __init__(self, data_file_list):
        # pprint(data_file_list)
        xs_all,xs_gn_all, ys_all = None,None,None
        for data_file in data_file_list:
            if os.path.exists(data_file):
                d = np.load(data_file)
                xs_all = d['xs'] if (xs_all is None) else np.r_[xs_all,d['xs']]
                xs_gn_all = d['xs_gn'] if (xs_gn_all is None) else np.r_[xs_gn_all,d['xs_gn']]
                ys_all = d['ys'] if (ys_all is None) else np.r_[ys_all,d['ys']]
        self.xs = torch.from_numpy(xs_all).float().unsqueeze(axis=1)
        self.xs_gn = torch.from_numpy(xs_gn_all).float().unsqueeze(axis=1)
        self.ys = torch.from_numpy(ys_all).float().unsqueeze(axis=1)

    def __getitem__(self, index):
        x = self.xs[index]
        x_gn = self.xs_gn[index]
        y = self.ys[index]
        return x,x_gn,y

    def __len__(self):
        return len(self.xs)

class EITDataModule(pl.LightningDataModule):
    def __init__(self, train_data_file_list,
                 val_data_file_list,
                 batch_size= 4,
                 batch_size_val = 2,
                 num_data_loader_workers=0):

        super().__init__()
        self.train_data_file_list= train_data_file_list
        self.val_data_file_list= val_data_file_list
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.num_data_loader_workers = num_data_loader_workers

    def train_dataloader(self):
        dataset = EITdataset(self.train_data_file_list)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_data_loader_workers,
                                shuffle=True, pin_memory=True,drop_last=True)
        return dataloader

    def val_dataloader(self):
        dataset = EITdataset(self.val_data_file_list)
        dataloader =  DataLoader(dataset,
                                 batch_size=dataset.__len__(),
                                 num_workers=self.num_data_loader_workers,
                                 shuffle=False, pin_memory=True,drop_last=True)
        return dataloader

def gpu2numpy(x):
    return x.cpu().numpy()

def TVLoss(x):
    return torch.sum(torch.abs(x[:,:,1:]-x[:,:,:-1]))

def read_hypers(data_dir):
    is_read_hypers = False
    cnt_read = 0
    while not is_read_hypers:
        try:
            d = shelve.open(os.path.join(os.getcwd(),data_dir,'hypers.db'))
            is_read_hypers = True
            print(f'read hyperparameters!')
        except:
            cnt_read += 1
            print(f'read hyperparameters again:{cnt_read}')
    return d