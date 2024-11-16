import glob
import os
os.environ["OMP_NUM_THREADS"]        = "4"
os.environ["OPENBLAS_NUM_THREADS"]   = "4"
os.environ["MKL_NUM_THREADS"]        = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"]    = "4"
import shelve
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pytorch_lightning as pl
from a_utils import compute_metrics,EITDataModule,get_neighbor

from eit_train_lpd import LearnedPDEIT
from eit_train_lpdg import LearnedPOEIT
# %% load data
num_inclusion = 2
h0 = 0.07
data_type = f'circle{num_inclusion}_h0{h0}_Linux'
data_dir = os.path.join('datasets-600', data_type)
data_file_list = [os.path.join(data_dir, f'{i}.npz') for i in range(52)]
train_data_file_list = data_file_list[:-2]
val_data_file_list = data_file_list[-2:]
dataset = EITDataModule(train_data_file_list=train_data_file_list,
                        val_data_file_list=val_data_file_list,
                        batch_size=1)

hypers = shelve.open(os.path.join(data_dir, 'hypers.db'))
mesh_obj = hypers['data']['mesh_obj']
eit = hypers['fun']['eit']
fwd = hypers['fun']['fwd']
ex_mat = hypers['data']['ex_mat']
step = hypers['data']['step']
hypers['neighbors'] = get_neighbor(mesh_obj['element'])
metrics = ['MSE', 'RE_sigma', 'RE_v', 'DR', 'PSNR', 'SSIM']
#%%
for ground_truth, x_gn, y in dataset.val_dataloader():
    pred_gn = x_gn
    (MSE, RE_sigma, RE_v, DR, PSNR, SSIM), (MSE_std, RE_sigma_std, RE_v_std, DR_std, PSNR_std, SSIM_std) = \
        compute_metrics(ground_truth, pred_gn, y, hypers)
    title_gn = f' GN: MSE{MSE:.5f}+-{MSE_std:.5f}  PSNR{PSNR:.3f} +-{PSNR_std:.3f}  SSIM{SSIM:.3f}+-{SSIM_std:.3f}'
    print(title_gn)
#%%
start = time.time()
# run_one_test = True
run_one_test = False
result_file = os.path.join('result_reconstruct', f'1105-final-{data_type}.csv')
for n_train in [50, 200, 400]:
    for experiments in ['experiments-lpo','experiments-lpo-notshareweight','experiments-lpd']:
        hypers['share_weight'] = 0 if experiments == 'experiments-lpo-notshareweight' else 1
        chkp_dir = os.path.join(experiments,f'{n_train}_{data_type}','*','*','*','checkpoints','*.ckpt')
        chkp_paths = glob.glob(chkp_dir)

        shape_primal, shape_dual = 686, 208
        # % load trained model
        methods = []
        tests = []
        versions = []
        metrics_values = []
        epoch_best = []
        for chkp_path in chkp_paths:
            # chkp_path = chkp_paths[4]
            print(f"TIME-{(time.time()-start)/3600:.2f} {n_train:5d}: {chkp_path}")

            epoch_name = chkp_path.split(os.sep)[-1].split('.')[0]
            if epoch_name=='last':
                continue

            epoch_best.append(int(epoch_name.split('-')[0].split('=')[1]))
            versions.append(int(chkp_path.split(os.sep)[-3].split('_')[-1]))

            method_str = chkp_path.split(os.sep)[2]
            methods.append(method_str.split('_test_')[0])
            tests.append(int(method_str.split('_test_')[1]))
            pl.seed_everything(int(method_str.split('_test_')[1]))

            grad_type = method_str.split('_l_')[0].split('gt_')[-1]
            hypers['layer'] = int(method_str.split('_l_')[1].split('_')[0])
            if experiments=='experiments-lpd':
                lpd = LearnedPDEIT.load_from_checkpoint(chkp_path, strict=True,
                                    shape_primal=shape_primal, shape_dual=shape_dual,grad_type=grad_type, hypers=hypers)
            else:
                lpd = LearnedPOEIT.load_from_checkpoint(chkp_path, strict=True,
                                    shape_primal=shape_primal, grad_type=grad_type, hypers=hypers)
            # %%
            for ground_truth, x_gn, y in dataset.val_dataloader():
                with torch.no_grad():
                    pred = torch.zeros_like(ground_truth)
                    for i in (range(len(y))):
                        pred[i] = lpd.model(x_gn[[i], ...], y[[i], ...])
                    (MSE, RE_sigma, RE_v, DR, PSNR, SSIM), (MSE_std, RE_sigma_std, RE_v_std, DR_std, PSNR_std, SSIM_std) = \
                        compute_metrics(ground_truth, pred, y, hypers)
                    title_pred = f'LPD: MSE{MSE:.5f}+-{MSE_std:.5f}  PSNR{PSNR:.3f} +-{PSNR_std:.3f}  SSIM{SSIM:.3f}+-{SSIM_std:.3f}'
                    title = title_gn + '\n' + title_pred
            metrics_values.append([MSE, RE_sigma, RE_v, DR, PSNR, SSIM])

            if run_one_test:
                break

        a = pd.DataFrame(methods,columns=['method'])
        a['experiments'] = experiments
        a['n_train'] = n_train
        b = pd.DataFrame(tests,columns=['test'])
        c = pd.DataFrame(metrics_values,columns=metrics)
        d = pd.DataFrame(versions,columns=['version'])
        e = pd.DataFrame(epoch_best,columns=['epoch_best'])
        result = pd.concat([a,d,b,c,e],axis=1)

        if os.path.exists(result_file):
            result.to_csv(result_file, mode='a', index=False, header=False)
        else:
            result.to_csv(result_file, index=False)
