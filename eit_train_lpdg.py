import argparse
import glob
import os
import shelve

os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"

import sys
import platform
sys.path.append('../')
sys.path.append('../../')
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from pprint import pprint
from proximalgradient import LearnedProximal
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
from a_utils import compute_metrics, plot_preds, forward_and_jac, EITDataModule, TVLoss, get_neighbor, read_hypers


#%%
class LearnedPOEIT(pl.LightningModule):
    def __init__(self, shape_primal,grad_type,hypers_eit,hypers_model):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.grad_type = grad_type
        self.hypers_eit = hypers_eit
        self.hypers_model = hypers_model
        self.model = LearnedProximal(forward_and_jac, shape_primal,grad_type,hypers_eit,hypers_model)
        self.compute_metrics_for_gn = True

    def forward(self,x_gn,y):
        return self.model(x_gn,y)

    def training_step(self, batch, batch_idx):
        """
        Pytorch Lightning training step. Should be independent of forward()
        according to the documentation. The loss value is logged.
        Parameters
        ----------
        batch : tuple of tensor
            Batch of measurement y and ground truth reconstruction gt.
        batch_idx : int
            Index of the batch.
        Returns
        -------
        result : TYPE
            Result of the training step.
        """
        ground_truth, x_gn, y = batch  #

        output = self.forward(x_gn, y)
        loss_mse = self.mse_loss(output, ground_truth)
        loss_tv = TVLoss(output)
        loss = loss_mse + self.hypers_model['reg'] * loss_tv

        # Log the training loss
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Pytorch Lightning validation step. Should be independent of forward()
        according to the documentation. The loss value is logged and the
        best model according to the loss (lowest) checkpointed.
        Parameters
        ----------
        batch : tuple of tensor
            Batch of measurement y and ground truth reconstruction gt.
        batch_idx : int
            Index of the batch.
        Returns
        -------
        result : TYPE
            Result of the validation step.
        """
        ground_truth, x_gn, y = batch  #
        output = torch.zeros_like(ground_truth)
        for i in range(len(x_gn)):
            output[i] = self.forward(x_gn[[i],...],y[[i],...])
        loss = self.mse_loss(output, ground_truth)

        # checkpoint the model and log the loss
        self.log('val_loss', loss)
        self.last_batch = batch
        return loss

    def validation_epoch_end(self, result):
        """
        tensorboard --logdir=version_10
        no logging of histogram. Checkpoint gets big
        for name,params in self.named_parameters():
             self.logger.experiment.add_histogram(name, params, self.current_epoch)
        """
        tensorboard = self.logger.experiment
        ground_truth,x_gn,y = self.last_batch   # xxx (b,1,1342)  (b,1,208)

        if self.compute_metrics_for_gn:
            # pred_gn = torch.zeros_like(ground_truth)
            # for i, yi in enumerate(x_gn):
            #     dsi, _ = eit.gn(yi[0].cpu().numpy(), lamb_decay=0.1, lamb_min=1e-5, maxiter=10, verbose=False)
            #     pred_gn[i] = torch.from_numpy(dsi).unsqueeze(dim=0)
            pred_gn = x_gn
            (MSE, RE_sigma, RE_v, DR, PSNR, SSIM), (MSE_std, RE_sigma_std, RE_v_std, DR_std, PSNR_std, SSIM_std) = \
                compute_metrics(ground_truth, pred_gn, y,self.hypers_eit)
            title_gn = f' GN: MSE{MSE:.5f}+-{MSE_std:.5f}  PSNR{PSNR:.2f} +-{PSNR_std:.2f}  SSIM{SSIM:.2f}+-{SSIM_std:.2f}'
            self.pred_gn = pred_gn
            self.title_gn = title_gn
            print('compute metrics for GN','*'*30)
            print(title_gn)
            self.compute_metrics_for_gn = False

        with torch.no_grad():
            pred = torch.zeros_like(ground_truth)
            for i in range(len(y)):
                pred[i] = self.forward(x_gn[[i], ...], y[[i], ...])
            (MSE, RE_sigma,RE_v,DR, PSNR, SSIM),(MSE_std, RE_sigma_std,RE_v_std,DR_std, PSNR_std, SSIM_std) = \
                compute_metrics(ground_truth, pred, y,self.hypers_eit)
            title_pred = f'LPD: MSE{MSE:.5f}+-{MSE_std:.5f}  PSNR{PSNR:.2f} +-{PSNR_std:.2f}  SSIM{SSIM:.2f}+-{SSIM_std:.2f}'
            title = title_pred+'\n'+self.title_gn
            print(title_pred)
            result_dict = {'val_mse':MSE,'val_RE_sigma':RE_sigma,'val_RE_v':RE_v,'val_DR':DR,'val_PSNR':PSNR,'val_SSIM':SSIM}
            result_std_dict = {'val_mse_std': MSE_std, 'val_RE_sigma_std': RE_sigma_std, 'val_RE_v_std': RE_v_std, 'val_DR_std': DR_std, 'val_PSNR_std': PSNR_std, 'val_SSIM_std': SSIM_std}
            self.log_dict(result_dict)
            self.log_dict(result_std_dict)

            try:
                ground_truth = ground_truth[:4, ...]
                pred = pred[:4, ...]
                pred_gn = self.pred_gn[:4, ...]
            except:
                print('val_batch_size<5')
            tensorboard.add_figure(f'val_image_gt_{self.grad_type}',
                                   plot_preds(pred,ground_truth,pred_gn,title,self.hypers_eit),
                                   global_step=self.current_epoch)

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
        reduce_on_plateu = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10001)
        schedulers = {
            'scheduler': reduce_on_plateu,
            'monitor': 'train_loss',
            'interval': 'epoch',  # step or epoch
            'frequency': 1}

        return [optimizer], [schedulers]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lpo')
    parser.add_argument('--n_inclusion', type=int, default=4, help='number of inclusion')        #xxx 场景参数
    parser.add_argument('--h0', type=float, default=0.07, help='initial mesh size')              # 场景参数

    parser.add_argument('--share_weight', type=int, default=0, help='1-share weight; 0-not share weight')       # 寻优参数 【parser用bool有点麻烦】https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser.add_argument('--n_train', type=int, default=50, help='the number of training data')   #xxx 实验参数
    parser.add_argument('--n_epoch', type=int, default=20, help='the number of epoches')         # 实验参数
    parser.add_argument('--n_test', type=int, default=0, help='number of test')                  # 实验参数

    parser.add_argument('--layer', type=int, default=1, help='the number of layers in RNN')      #xxx 寻优参数
    parser.add_argument('--reg', type=float, default=0.0, help='reg of tv loss')                 # 寻优参数
    parser.add_argument('--idx_grad', type=int, default=3, help='index of grad_type')            # 寻优参数
    args = parser.parse_args()
    pl.seed_everything(args.n_test)
    print(args)

    num_inclusion = args.n_inclusion
    h0 = args.h0
    # data_type = f'circle{num_inclusion}_h0{h0}_{platform.system()}'
    data_type = f'circle{num_inclusion}_h0{h0}_Linux'
    data_dir = os.path.join('datasets-600',data_type)  #xxx

    hypers_eit = read_hypers(data_dir)
    mesh_obj = hypers_eit['data']['mesh_obj']
    eit      = hypers_eit['fun']['eit']
    fwd      = hypers_eit['fun']['fwd']
    ex_mat   = hypers_eit['data']['ex_mat']
    step     = hypers_eit['data']['step']

    hypers_model = dict()
    hypers_model['layer_rnn'] = args.layer
    hypers_model['reg'] = args.reg
    hypers_model['share_weight'] = args.share_weight
    hypers_model['neighbors'] = get_neighbor(mesh_obj['element'])

    data_file_list = [os.path.join(data_dir, f'{i}.npz') for i in range(52)]
    train_data_file_list = data_file_list[:-2]
    val_data_file_list = data_file_list[-2:]
    dataset = EITDataModule(train_data_file_list=train_data_file_list,
                            val_data_file_list=val_data_file_list,
                            batch_size=1)

    grad_type = ['baseline', 'momentum', 'lstm', 'lstm_low', 'gru','gru_low'][args.idx_grad]
    method = f'gt_{grad_type}_l_{args.layer}_reg_{args.reg}_test_{args.n_test}'      #xxx
    print(method)


    if args.n_inclusion==2:
        experiments = 'experiments-lpo' if args.share_weight else 'experiments-lpo-notshareweight'
    if args.n_inclusion==4:
        experiments = 'exp-lposw' if args.share_weight else 'exp-lpo'
    log_dir = os.path.join(*[experiments, f"{args.n_train}_"+data_type,method])
    # ----------load trained model and rerun------------------
    chkp_path = None
    # version = 'version_0'    #xxx
    # chkp_path_reg = os.path.join(*[log_dir,'*',version,'checkpoints','last.ckpt'])  # default, lightning_logs
    # try:
    #     print('*' * 60, f'\nload model from {chkp_path_reg}')
    #     chkp_path = glob.glob(chkp_path_reg)[0]
    #     print('*' * 60, f'\nload model from {chkp_path}\n', '*' * 60)
    # except:
    #     print('*' * 60, f'\nno trained model!!!!\n', '*' * 60)
    #     chkp_path = None
    # -----------load trained model and rerun--------------------
    checkpoint_callback = ModelCheckpoint(save_top_k=1,verbose=True,
                                          monitor='val_loss',mode='min',save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval=None)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)
    limit_train_batches = args.n_train/500
    limit_val_batches = 1.0
    trainer_args = { 'default_root_dir': log_dir
                     ,'callbacks': [lr_monitor, checkpoint_callback]
                     ,'num_sanity_val_steps': 0
                     ,'benchmark': False
                     ,'fast_dev_run': False
                     ,'limit_train_batches': limit_train_batches  #训练数据集的使用比例，用来测试和debug。
                     ,'limit_val_batches': limit_val_batches
                     ,'gradient_clip_val': 1.0
                     ,'logger': tb_logger
                     ,'log_every_n_steps': 20
                     # ,'enable_progress_bar': True
                     ,'enable_progress_bar': False
                     }
    pprint(trainer_args)
    #%
    shape_primal, shape_dual = 686, 208

    model = LearnedPOEIT(shape_primal,grad_type,hypers_eit,hypers_model)
    print(model)
    print('-'*60)
    print(sum(param.numel() for param in model.parameters()))
    print('-'*60)

    trainer = pl.Trainer(max_epochs=args.n_epoch, **trainer_args)
    trainer.fit(model, datamodule=dataset,ckpt_path=chkp_path)








