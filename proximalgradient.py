"""
ref:
Mardani, Morteza, Qingyun Sun, Shreyas Vasawanala, Vardan Papyan, Hatef Monajemi,
John Pauly, David Donoho. Neural Proximal Gradient Descent for Compressive Imaging.
ArXiv:1806.03963 [Cs]. http://arxiv.org/abs/1806.03963.

Meinhardt, Tim, Michael Moeller, Caner Hazirbas, Daniel Cremers.
Learning Proximal Operators: Using Denoising Networks for Regularizing Inverse Imaging Problems.
 2017 IEEE International Conference on Computer Vision (ICCV), 1799–1808. Venice: IEEE, 2017.
 https://doi.org/10.1109/ICCV.2017.198.
"""
import os
import torch
import torch.nn as nn
from torchsummary import summary
#%
class ConcatenateLayer(nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()

    def forward(self, *x):
        return torch.cat(list(x), dim=1)

class GRUnet(nn.Module):
    def __init__(self, H_in=1342, H_cell=256, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.H_in = H_in
        self.H_cell = H_cell
        self.H_out = H_cell

        self.rnn = nn.GRU(input_size=H_in, hidden_size=H_cell, num_layers=num_layers, batch_first=True)
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.fc = nn.Linear(num_layers*H_cell, H_in) #  hidden

    def forward(self, OpAdj_hs_history, hidden):
        batch_size = OpAdj_hs_history.shape[0]
        output, hn = self.rnn(OpAdj_hs_history, hidden)
        hn_flatten = hn.permute(1, 0, 2).contiguous().view(-1, self.num_layers * self.H_cell)
        OpAdj_h_new = self.fc(hn_flatten).view(-1, 1, self.H_in)
        return OpAdj_h_new,hidden

    def init_hc(self):
        hidden = torch.randn(self.num_layers, 1, self.H_out).requires_grad_(False)  # batch_size=1
        return hidden,None

    def _parameter(self):
        return sum(param.numel() for param in self.parameters())//1000

class LSTMnet(nn.Module):
    def __init__(self, H_in=1342, H_cell=256, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.H_in = H_in
        self.H_cell = H_cell
        self.H_out = H_cell

        self.rnn = nn.LSTM(input_size=H_in, hidden_size=H_cell, num_layers=num_layers, batch_first=True)
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        # self.fc = nn.Linear(H_cell, H_in)  # output xxx
        self.fc = nn.Linear(num_layers*H_cell, H_in) #  hidden

    def forward(self, OpAdj_hs_history, hidden, cell):
        output, (hidden, cell) = self.rnn(OpAdj_hs_history, (hidden, cell))
        hn_flatten = hidden.permute(1, 0, 2).contiguous().view(-1, self.num_layers * self.H_cell)
        OpAdj_h_new = self.fc(hn_flatten).view(-1, 1, self.H_in)
        return OpAdj_h_new,hidden, cell

    def init_hc(self):
        hidden = torch.randn(self.num_layers, 1, self.H_out).requires_grad_(False)  # batch_size=1
        cell = torch.randn(self.num_layers, 1, self.H_cell).requires_grad_(False)
        return hidden,cell

    def _parameter(self):
        return sum(param.numel() for param in self.parameters())//1000


class PrimalNet(nn.Module):
    """
    model = PrimalNet(5)
    summary(model.cuda(), [(5, 686), (1, 686)])
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
      ConcatenateLayer-1               [-1, 6, 686]               0
                Conv1d-2              [-1, 32, 686]             608
                 PReLU-3              [-1, 32, 686]               1
                Conv1d-4              [-1, 32, 686]           3,104
                 PReLU-5              [-1, 32, 686]               1
                Conv1d-6               [-1, 5, 686]             485
    ================================================================
    Total params: 4,199
    Trainable params: 4,199
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 8.98
    Forward/backward pass size (MB): 0.73
    Params size (MB): 0.02
    Estimated Total Size (MB): 9.72
    ----------------------------------------------------------------
    """
    def __init__(self, n_primal,n_OpAdj_hs, grad_type, rnn_net=None):
        """
        Args:
            n_primal:  int
            n_OpAdj_hs:  list
            grad_type:  string
        """
        super(PrimalNet, self).__init__()
        self.grad_type = grad_type
        self.n_primal = n_primal
        self.n_channels = n_primal + n_OpAdj_hs # +1 #xxx +1 for gn
        self.rnn_primal = rnn_net
        self.input_concat_layer = ConcatenateLayer()
        layers = [
            nn.Conv1d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(32, self.n_primal, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, f, OpAdj_hs, hidden=None,cell=None):
        if self.grad_type in ['baseline','momentum']:  # len(OpAdj_hs)==1 if baseline else historical grads
            inputs = [f] + OpAdj_hs
        elif self.grad_type in ['lstm_low']:
            assert (hidden is not None) and (cell is not None)
            OpAdj_hs_history = torch.cat(OpAdj_hs,dim=1)
            OpAdj_hs_hist= nn.functional.interpolate(OpAdj_hs_history, scale_factor=0.5,recompute_scale_factor=True)
            OpAdj_hs_hist_ouput,hidden,cell = self.rnn_primal(OpAdj_hs_hist,hidden,cell)
            OpAdj_h_new = nn.functional.interpolate(OpAdj_hs_hist_ouput, scale_factor=2.0)
            inputs = [f,OpAdj_h_new]
        elif self.grad_type in ['gru_low']:
            assert (hidden is not None) and (cell is None)
            OpAdj_hs_history = torch.cat(OpAdj_hs,dim=1)
            OpAdj_hs_hist= nn.functional.interpolate(OpAdj_hs_history, scale_factor=0.5,recompute_scale_factor=True)
            OpAdj_hs_hist_ouput,hidden = self.rnn_primal(OpAdj_hs_hist,hidden)
            OpAdj_h_new = nn.functional.interpolate(OpAdj_hs_hist_ouput, scale_factor=2.0)
            inputs = [f,OpAdj_h_new]
        elif self.grad_type in ['lstm']:  #   ['lstm','gru']
            assert (hidden is not None) and (cell is not None)
            OpAdj_hs_history = torch.cat(OpAdj_hs,dim=1)
            OpAdj_h_new,hidden,cell = self.rnn_primal(OpAdj_hs_history,hidden,cell)
            inputs = [f,OpAdj_h_new]
        elif self.grad_type in ['gru']:
            assert (hidden is not None) and (cell is None)
            OpAdj_hs_history = torch.cat(OpAdj_hs,dim=1)
            OpAdj_h_new,hidden = self.rnn_primal(OpAdj_hs_history,hidden)
            inputs = [f,OpAdj_h_new]
        else:
            raise NotImplementedError("grad_type is not implemented.")

        x = self.input_concat_layer(*inputs)
        x = f + self.block(x)
        if self.grad_type in ['baseline','momentum']:
            return x
        else:
            return x,hidden,cell

class LearnedProximal(nn.Module):
    def __init__(self,
                 forward_and_jac,
                 shape_primal,
                 grad_type,
                 hypers_eit,
                 hypers_model,
                 primal_architecture = PrimalNet,
                 n_iter = 20,
                 n_primal = 1):
        super(LearnedProximal, self).__init__()
        self.forward_and_jac = forward_and_jac
        self.primal_architecture = primal_architecture
        self.shape_primal = shape_primal
        self.hypers_eit = hypers_eit
        self.hypers_model = hypers_model
        self.eit = hypers_eit['fun']['eit']
        self.n_iter = n_iter
        self.n_primal = n_primal
        self.grad_type = grad_type
        self.share_weight = hypers_model['share_weight']

        self.primal_shape = (n_primal,) + (shape_primal,)

        self.concatenate_layer = ConcatenateLayer()
        num_layers = hypers_model['layer_rnn']
        print(f'num_layers:{num_layers}')
        if grad_type in ['lstm_low']:
            self.rnn_primal = LSTMnet(H_in=shape_primal//2,H_cell=64,num_layers=num_layers)
        if grad_type in ['lstm']:
            self.rnn_primal = LSTMnet(H_in=shape_primal,H_cell=128,num_layers=num_layers)

        if grad_type in ['gru_low']:
            self.rnn_primal = GRUnet(H_in=shape_primal//2, H_cell=64, num_layers=num_layers)
        if grad_type in ['gru']:
            self.rnn_primal = GRUnet(H_in=shape_primal, H_cell=128, num_layers=num_layers)
        if self.share_weight:
            if grad_type in ['lstm', 'lstm_low', 'gru', 'gru_low']:
                self.primal_net = primal_architecture(n_primal,1,self.grad_type,rnn_net=self.rnn_primal)
            else:
                self.primal_net = primal_architecture(n_primal, 1, self.grad_type)
        else:
            self.primal_nets = nn.ModuleList()
            for i in range(n_iter):
                if grad_type in ['lstm', 'lstm_low', 'gru', 'gru_low']:
                    self.primal_nets.append(
                        primal_architecture(n_primal, 1, self.grad_type, rnn_net=self.rnn_primal)
                    )
                else:
                    self.primal_nets.append(
                        primal_architecture(n_primal, 1, self.grad_type)
                    )
    def forward(self,x_gn, g,intermediate_values = False):
        if True:
            f = x_gn
        else:
            f = torch.rand(g.shape[0:1] + (self.primal_shape))  #xxx 全部是0，求解出现不可逆错误

        Kf, _ = self.forward_and_jac(f, self.hypers_eit)
        h  = g-Kf
        if intermediate_values:
            h_values = []
            f_values = []

        OpAdj_hs  =[]
        if self.grad_type in ['momentum']:
            vt = torch.zeros_like(f).requires_grad_(False)   # initialized velocity of momentum
        if self.grad_type in ['lstm', 'lstm_low', 'gru', 'gru_low']:
            hidden, cell = self.rnn_primal.init_hc()

        for i in range(self.n_iter):                                # [5, 1, 208]
            if intermediate_values:
                h_values.append(h)
            if self.grad_type in ['lstm', 'gru','lstm_low','gru_low']:
                Kf, grad_K = self.forward_and_jac(f,self.hypers_eit)    # (208, 2822)
                OpAdj_h = torch.bmm(g-Kf, grad_K)         # [5, 1, 1342]
                OpAdj_hs.append(OpAdj_h)
                if self.share_weight:
                    f,hidden,cell = self.primal_net(f,OpAdj_hs,hidden,cell)
                else:
                    f,hidden,cell = self.primal_nets[i](f,OpAdj_hs,hidden,cell)
            elif self.grad_type=='baseline':
                Kf, grad_K = self.forward_and_jac(f,self.hypers_eit)   # (208, 2822)
                OpAdj_h = torch.bmm(g-Kf, grad_K)        # [5, 1, 1342]
                if self.share_weight:
                    f = self.primal_net(f, [OpAdj_h])
                else:
                    f = self.primal_nets[i](f, [OpAdj_h])
            elif self.grad_type=='momentum': # (1) v_t <- gamma*v_{t-1}+eta*g_t  (2) theta_t <- theta_{t-1}-v_t
                Kf, grad_K = self.forward_and_jac(f,self.hypers_eit)   # (208, 2822)
                OpAdj_h = torch.bmm(g-Kf, grad_K)        # [5, 1, 1342]
                vt = 0.8*vt+1e-3*OpAdj_h                #
                if self.share_weight:
                    f = self.primal_net(f, [vt])
                else:
                    f = self.primal_nets[i](f, [vt])
            else:
                raise NotImplementedError(f"grad_type:{self.grad_type} is not implemented.")

            if intermediate_values:
                h_values.append(g-Kf)
                f_values.append(f)

        if intermediate_values:
            return f, f_values, h_values
        return f