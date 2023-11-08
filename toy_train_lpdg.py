import argparse
import glob
import os
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
import shelve
import time

from tqdm import tqdm
import sys
import platform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append('../')
sys.path.append('../../')
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from pprint import pprint
import numpy as np

from toy_proximalgradient import LearnedProximal

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# %%
class EITdataset(Dataset):
    """
    return x-perm, y-电压
    """
    def __init__(self, data_file_list):
        # pprint(data_file_list)
        xs_all, xs_inv_all, ys_all = None, None, None
        for data_file in data_file_list:
            if os.path.exists(data_file):
                d = np.load(data_file)
                xs_all = d['xs'] if (xs_all is None) else np.r_[xs_all, d['xs']]
                xs_inv_all = d['xs_inv'] if (xs_inv_all is None) else np.r_[xs_inv_all, d['xs_inv']]
                ys_all = d['ys'] if (ys_all is None) else np.r_[ys_all, d['ys']]
        self.xs = torch.from_numpy(xs_all).float().unsqueeze(axis=1).to(device)
        self.xs_inv = torch.from_numpy(xs_inv_all).float().unsqueeze(axis=1).to(device)
        self.ys = torch.from_numpy(ys_all).float().unsqueeze(axis=1).to(device)
        # print('-'*50)
        # print(f'   xs:{self.xs.shape}\nxs_inv:{self.xs_inv.shape}\n   ys:{self.ys.shape}')
        # print('-'*50)

    def __getitem__(self, index):
        x = self.xs[index]
        x_inv = self.xs_inv[index]
        y = self.ys[index]
        return x, x_inv, y

    def __len__(self):
        return len(self.xs)

class EITDataModule(pl.LightningDataModule):
    def __init__(self, train_data_file_list,
                 val_data_file_list,
                 batch_size=4,
                 batch_size_val=2,
                 num_data_loader_workers=0):
        super().__init__()
        self.train_data_file_list = train_data_file_list
        self.val_data_file_list = val_data_file_list
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val

    def train_dataloader(self):
        dataset = EITdataset(self.train_data_file_list)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=0,
                                shuffle=True, drop_last=True)
        print(f'TRAINING: \ndata length: {dataset.__len__()}\n batch_size: {dataloader.batch_size} \n iters/epoch: {len(dataloader)}\n')
        return dataloader

    def val_dataloader(self):
        dataset = EITdataset(self.val_data_file_list)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=0,
                                shuffle=False)
        print(f' VALIDATION: \n data length: {dataset.__len__()}\n batch_size: {dataloader.batch_size} \n iters/epoch: {len(dataloader)}\n')
        return dataloader

def gpu2numpy(x):
    return x.cpu().numpy()

def plot_preds(pred, ground_truth, pred_gn, title):
    n_col, n = ground_truth.shape[0], ground_truth.shape[-1]
    fig, axes = plt.subplots(4, 1, figsize=(n_col * 4, 3 * 4))
    for i, (p, gt, pg) in enumerate(zip(pred, ground_truth, pred_gn)):
        axes[i].step(range(n), gt[0], 'k-', label='gt', lw=2)
        axes[i].step(range(n), p[0], 'r-', label='pred')
        if i > 1:
            axes[i].step(range(n), pg[0], 'r--', label='init value')
        axes[i].set_xlim([n // 4, n // 4 * 2])
        if i >= (n_col - 1): break
    fig.suptitle(title, fontsize=20)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    return fig

def forward_and_jac(xs,args):
    """
    Args:
        xs:  [b,1,dim_x]
        args:

    Returns:
        ys: [b,1,dim_y]
        grads: [b,dim_x,dim_y]
    """
    W1 = torch.tensor(args['W1']).float().to(device)
    W2 = torch.tensor(args['W2']).float().to(device)
    steps = args['steps']
    x_dim = args['x_dim']
    y_dim = args['y_dim']
    a = torch.tensor(args['a']).to(device)
    s = args['size']
    w = s//2

    b, _, n = xs.shape
    Kf = torch.zeros(b,1,y_dim)
    jacs = torch.zeros(b,y_dim,x_dim)
    for ind,x in enumerate(xs):
        x = x[0]
        y = torch.zeros(y_dim)
        grads = torch.zeros(y_dim, x_dim)
        for i, s in enumerate(steps):
            x_temp = x[s - w:s + w + 1]

            y_temp = a * x_temp.T @ W1 @ x_temp + W2.T @ x_temp
            y[i] = y_temp.item()

            grads[i, s - w:s + w + 1] = 2*a*W1@x_temp.T+W2.T

        Kf[ind,0,:] = y
        jacs[ind,:,:] = grads
    return Kf.detach().to(device), jacs.detach().to(device)

# %
class LearnedPGD(pl.LightningModule):
    def __init__(self, shape_primal, grad_type, hypers):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.grad_type = grad_type
        self.hypers = hypers
        self.model = LearnedProximal(forward_and_jac, shape_primal, grad_type, hypers)
        self.compute_metrics_for_gn = True

    def forward(self, x_inv, y):
        return self.model(x_inv, y)

    def training_step(self, batch, batch_idx):
        ground_truth, x_inv, y = batch  #

        output = self.forward(x_inv, y)
        loss = self.mse_loss(output, ground_truth)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ground_truth, x_inv, y = batch  #
        output = self.forward(x_inv, y)
        loss = self.mse_loss(output, ground_truth)

        # checkpoint the model and log the loss
        self.log('val_loss', loss)
        self.last_batch = batch

        # idx_time = '_'.join([str(i) for i in time.localtime(time.time())[:5]])
        # print(f"time: {idx_time}")
        return loss

    def configure_optimizers(self):
        """
        Setup the optimizer. Currently, the ADAM optimizer is used.
        Returns
        -------
        optimizer : torch optimizer
            The Pytorch optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.001,
                                     betas=(0.9,0.99))
        # optimizer = torch.optim.SGD(self.parameters(),
        #                             lr=0.1,
        #                             weight_decay=0.01)
        reduce_on_plateu = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1000001) #xxx
        schedulers = {
            'scheduler': reduce_on_plateu,
            'monitor': 'train_loss',
            'interval': 'epoch',  # xxx step or epoch
            'frequency': 1}
        return [optimizer], [schedulers]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lpd')
    parser.add_argument('--n_epoch', type=int, default=50, help='the number of epoches')  # xxx 实验参数
    parser.add_argument('--batchsize', type=int, default=256, help='train batch size')  # xxx 实验参数
    parser.add_argument('--a', type=float, default=2.0, help='power of x')
    parser.add_argument('--train_ratio', type=float, default=1, help='limit_train_batches')
    parser.add_argument('--idx_grad', type=int, default=1, help='index of grad_type')  # xxx 寻优参数
    parser.add_argument('--layer', type=int, default=2, help='the number of layers in RNN')  # xxx 寻优参数
    parser.add_argument('--hidden', type=int, default=50, help='the number of hidden cells in RNN')  # xxx 寻优参数
    parser.add_argument('--n_test', type=int, default=0, help='number of test')  # xxx 实验参数
    parser.add_argument('--share_weight', type=int, default=0, help='1-share weight; 0-not share weight')
    args = parser.parse_args()
    print(args)
    pl.seed_everything(args.n_test)
    #
    data_type = f'a{args.a:.1f}'
    data_dir = os.path.join('datasets-toy-nonlinear', data_type)
    print(data_dir)
    #
    time.sleep(np.random.rand() * 3)
    hypers = np.load(os.path.join(data_dir, 'hypers.npz'))
    hypers = dict(hypers)
    hypers['layer_rnn'] = args.layer
    hypers['hidden_rnn'] = args.hidden
    hypers['share_weight'] = args.share_weight
    #
    data_file_list = [os.path.join(data_dir, f'{i}.npz') for i in range(22)]
    train_data_file_list = data_file_list
    val_data_file_list = data_file_list[-2:]
    dataset = EITDataModule(train_data_file_list=train_data_file_list,
                            val_data_file_list=val_data_file_list,
                            batch_size=args.batchsize)

    grad_type = ['baseline', 'momentum', 'lstm', 'gru'][args.idx_grad]
    method = f'gt_{grad_type}_test_{args.n_test}_l_{args.layer}_hidden_{args.hidden}'  # xxx
    experiments = 'exp-toy-lposw' if args.share_weight else 'exp-toy-lpo'
    log_dir = os.path.join(*['exp-toy', data_type, f'train_ratio{args.train_ratio:.1f}', experiments, method])
    #
    checkpoint_callback = ModelCheckpoint(save_top_k=1, verbose=True,monitor='val_loss', mode='min', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval=None)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)
    trainer_args = {'default_root_dir': log_dir
                , 'callbacks': [lr_monitor, checkpoint_callback]
                , 'num_sanity_val_steps': 0
                , 'benchmark': False
                , 'fast_dev_run': False
                , 'limit_train_batches': args.train_ratio/100.0
                , 'limit_val_batches': 1.0
                , 'gradient_clip_val': 1.0
                , 'logger': tb_logger
                , 'log_every_n_steps': 10
                , 'enable_progress_bar': True
                }
    pprint(trainer_args)
    #
    shape_primal = int(hypers['x_dim'])
    model = LearnedPGD(shape_primal, grad_type, hypers)
    trainer = pl.Trainer(max_epochs=args.n_epoch, **trainer_args)
    trainer.fit(model, datamodule=dataset)
