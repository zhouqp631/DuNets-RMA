# DuNets-RMA 
Code for the paper [Deep Unrolling Networks with Recurrent Momentum Acceleration for Nonlinear Inverse Problems](https://iopscience.iop.org/article/10.1088/1361-6420/ad35e3).


#  Requirements
In order to run the code, you will need the following:
- `PyTorch` (>= 1.10.0)
- `Python` (>=3.7.xx)
- `PyEIT` ([link](https://github.com/eitcom/pyEIT))
- `pytorch_lightning`



# Structure
## A nonlinear deconvolution problem
1. simulate data: `toy_dataset_simulation.py`
2. train: LPD-RMA `toy_train_lpd.py`; LPGD-RMA `toy_train_lpgd.py`



##  Electrical impedance tomography
1. simulate data: `eit_circle_simulation.py`
2. train: LPD-RMA `eit_train_lpd.py`; LPGD-RMA `eit_train_lpgd.py`
3. compute the quantitative metrics:  `eit_reconstruct_2inclusion.py` and `eit_reconstruct_4inclusion.py`

