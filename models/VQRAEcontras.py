import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from torch.autograd import Variable
from units.forget_mult import ForgetMult
from torch.optim import Adam, lr_scheduler
import time


import torch.nn.init as init

class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 2),
            nn.LeakyReLU(0.2, True), #1
            nn.Linear(1000, 2),
            nn.LeakyReLU(0.2, True), #2
            # nn.Linear(1000, 1000),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(1000, 1000),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(1000, 1000),
            # nn.LeakyReLU(0.2, True),
            #nn.Linear(1000, 2),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class PlanarTransform(nn.Module):
    def __init__(self, latent_dim=20):
        super(PlanarTransform, self).__init__()
        self.latent_dim = latent_dim
        self.u = nn.Parameter(torch.randn(1, self.latent_dim) * 0.01)
        self.w = nn.Parameter(torch.randn(1, self.latent_dim) * 0.01)
        self.b = nn.Parameter(torch.randn(()) * 0.01)
    def m(self, x):
        return -1 + torch.log(1 + torch.exp(x))
    def h(self, x):
        return torch.tanh(x)
    def h_prime(self, x):
        return 1 - torch.tanh(x) ** 2
    def forward(self, z, logdet=False):
        # z.size() = batch x dim
        u_dot_w = (self.u @ self.w.t()).view(())
        w_hat = self.w / torch.norm(self.w, p=2) # Unit vector in the direction of w
        u_hat = (self.m(u_dot_w) - u_dot_w) * (w_hat) + self.u # 1 x dim
        affine = z @ self.w.t() + self.b
        z_next = z + u_hat * self.h(affine) # batch x dim
        if logdet:
            psi = self.h_prime(affine) * self.w # batch x dim
            LDJ = -torch.log(torch.abs(psi @ u_hat.t() + 1) + 1e-8) # batch x 1
            return z_next, LDJ
        return z_next

class PlanarFlow(nn.Module):
    def __init__(self, latent_dim=20, K=16):
        super(PlanarFlow, self).__init__()
        self.latent_dim = latent_dim
        self.transforms = nn.ModuleList([PlanarTransform(self.latent_dim) for k in range(K)])

    def forward(self, z, logdet=False):
        zK = z
        SLDJ = 0.
        for transform in self.transforms:
            out = transform(zK, logdet=logdet)
            if logdet:
                SLDJ += out[1]
                zK = out[0]
            else:
                zK = out

        if logdet:
            return zK, SLDJ
        return zK

class QRNNLayer(nn.Module):
    r"""Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.
        use_cuda: If True, uses fast custom CUDA kernel. If False, uses naive for loop. Default: True.
    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.
    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size,device,hidden_size=None, save_prev_x=False, zoneout=0, window=1, output_gate=True):
        super(QRNNLayer, self).__init__()

        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate
        self.device = device

        # One large matmul with concat is faster than N small matmuls and no concat
        self.linear = nn.Linear(self.window * self.input_size, 3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()

        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            # Construct the x_{t-1} tensor with optional x_{-1}, otherwise a zeroed  out value for x_{-1}
            Xm1 = []
            Xm1.append(self.prevX if self.prevX is not None else X[:1, :, :] * 0)
            # Note: in case of len(X) == 1, X[:-1, :, :] results in slicing of empty tensor == bad
            if len(X) > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            # Convert two (seq_len, batch_size, hidden) tensors to (seq_len, batch_size, 2 * hidden)
            source = torch.cat([X, Xm1], 2)

        # Matrix multiplication for the three outputs: Z, F, O
        Y = self.linear(source)
        # Convert the tensor back to (batch, seq_len, len([Z, F, O]) * hidden_size)
        if self.output_gate:
            Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            Y = Y.view(seq_len, batch_size, 2 * self.hidden_size)
            Z, F = Y.chunk(2, dim=2)
        ###
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron keeps the old value
        if self.zoneout:
            if self.training:
                mask = Variable(F.data.new(*F.size()).bernoulli_(1 - self.zoneout), requires_grad=False)
                F = F * mask
            else:
                F *= 1 - self.zoneout

        # Ensure the memory is laid out as expected for the CUDA kernel
        # This is a null op if the tensor is already contiguous
        Z = Z.contiguous()
        F = F.contiguous()
        # The O gate doesn't need to be contiguous as it isn't used in the CUDA kernel

        # Forget Mult
        # For testing QRNN without ForgetMult CUDA kernel, C = Z * F may be useful
        C = ForgetMult()(F.to(torch.device("cuda:0")), Z.to(torch.device("cuda:0")), hidden.to(torch.device("cuda:0")))

        # Apply (potentially optional) output gate
        if self.output_gate:
            H = torch.sigmoid(O) * C.to(self.device)
        else:
            H = C

        # In an optimal world we may want to backprop to x_{t-1} but ...
        if self.window > 1 and self.save_prev_x:
            self.prevX = Variable(X[-1:, :, :].data, requires_grad=False)

        return H, C[-1:, :, :]

class VQRAEcontrast(nn.Module):
    def __init__(self,  args, device,x_dim):
        super(VQRAEcontrast, self).__init__()

        # 6.model_mics


        # 1.VQRAE_data

        self.rolling_size = args.rolling_size
        #self.pid = args.pid


        # 2.VQRAE_model
        self.x_dim = x_dim
        self.h_dim = args.h_dim
        self.dense_dim = args.h_dim
        self.z_dim = args.z_dim
        ## 2.1 optimization∂
        self.loss_function = args.loss_function
        ## 2.2 dropout
        ## 2.3 layers
        self.rnn_layers = args.rnn_layers
        self.use_PNF = args.use_PNF
        self.PNF_layers = args.PNF_layers
        self.use_bidirection = args.use_bidirection
        self.robust_coeff = args.robust_coeff
        self.device = device


        # 3.VQRAE_save_and_load

        # 4.VQRAE_metrics

        # 5.VQRAE_display


        # file info
        if self.use_bidirection:
            pass
        else:
            # encoder  x/u to z, input to latent variable, inference model
            self.phi_enc = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())
            self.enc_mean = nn.Sequential(
                nn.Linear(self.h_dim, self.z_dim),
                nn.Sigmoid()).to(self.device)
            self.enc_std = nn.Sequential(
                nn.Linear(self.h_dim, self.z_dim),
                nn.Softplus()).to(self.device)

            # prior transition of zt-1 to zt
            self.phi_prior = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU()).to(self.device)
            self.prior_mean = nn.Sequential(
                nn.Linear(self.h_dim, self.z_dim),
                nn.Sigmoid()).to(self.device)
            self.prior_std = nn.Sequential(
                nn.Linear(self.h_dim, self.z_dim),
                nn.Softplus()).to(self.device)

            # decoder
            self.phi_dec = nn.Sequential(
                nn.Linear(self.h_dim + self.z_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU()).to(self.device)
            self.dec_std = nn.Sequential(
                nn.Linear(self.h_dim, self.x_dim),
                nn.Softplus()).to(self.device)
            self.dec_mean = nn.Sequential(
                nn.Linear(self.h_dim, self.x_dim),
                nn.Sigmoid()).to(self.device)

            self.hidden_state_qrnn = torch.nn.ModuleList(
                [QRNNLayer(input_size=self.x_dim if l == 0 else self.h_dim,device=self.device, hidden_size=self.h_dim)
                 for l in range(self.rnn_layers)]).to(self.device)
            self.qrnn = torch.nn.ModuleList(
                [QRNNLayer(input_size=self.x_dim + self.h_dim if l == 0 else self.h_dim,device=self.device, hidden_size=self.h_dim)
                 for l in range(self.rnn_layers)]).to(self.device)

        if self.use_PNF:
            # self.PNF = PlanarFlowSequence(sequence_length=self.rolling_size, latent_dim=self.z_dim, K=self.PNF_layers).to(self.device)
            self.PNF = PlanarFlow(latent_dim=self.z_dim, K=self.PNF_layers).to(self.device)

    def forward(self, x, y,hidden=None):
        if self.use_bidirection:
            fh_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(self.device)
            bh_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(self.device)
            fh, fh_t = self.forward_hidden_state_qrnn[0](x, fh_0)
            bh, bh_t = self.backward_hidden_state_qrnn[0](torch.flip(x, [0]), bh_0)

            # reversing hidden state list
            reversed_fh = torch.flip(fh, dims=[0])
            reversed_bh = torch.flip(bh, dims=[0])

            # reversing y_t list
            reversed_y = torch.flip(y, dims=[0])
            original_y = y

            # concat reverse h with reverse x_t
            concat_fh_ry = torch.cat([reversed_y, reversed_fh], dim=2)
            concat_bh_oy = torch.cat([original_y, reversed_bh], dim=2)

            # compute reverse a_t
            fa_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(self.device)
            ba_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(self.device)
            fa, fa_t = self.forward_qrnn[0](concat_fh_ry, fa_0)
            ba, ba_t = self.backward_qrnn[0](concat_bh_oy, ba_0)

            reversed_fa = torch.flip(fa, dims=[0])
            reversed_ba = torch.flip(ba, dims=[0])

            enc = self.phi_enc(torch.cat([reversed_fa, reversed_ba], dim=2).permute(1, 0, 2)).unsqueeze(-2)
            enc_mean = self.enc_mean(enc).squeeze(2)
            enc_std = self.enc_std(enc).squeeze(2)
            z_0 = self.reparameterized_sample(enc_mean, enc_std)

            if self.use_PNF:
                z_k, logdet = self.PNF(z_0, True)

            prior = self.phi_prior(torch.cat([fh, bh], dim=2).permute(1, 0, 2)).unsqueeze(-2)
            prior_mean = self.prior_mean(prior).squeeze(2)
            prior_std = self.prior_std(prior).squeeze(2)

            if self.use_PNF:
                dec = self.phi_dec(torch.cat([z_k, fh.permute(1, 0, 2), bh.permute(1, 0, 2)], dim=2)).unsqueeze(-2)
            else:
                dec = self.phi_dec(torch.cat([z_0, fh.permute(1, 0, 2), bh.permute(1, 0, 2)], dim=2)).unsqueeze(-2)

            dec_mean = self.dec_mean(dec).squeeze(2)
            dec_std = self.dec_std(dec).squeeze(2)

            if self.use_PNF:
                kld_loss = self.kld_gaussian_w_logdet(mean_1=enc_mean, std_1=enc_std, mean_2=prior_mean, std_2=prior_std, z_0=z_0, z_k=z_k, SLDJ=logdet)
            else:
                kld_loss = self.kld_gaussian(mean_1=enc_mean, std_1=enc_std, mean_2=prior_mean, std_2=prior_std)
                # kld_loss = self.kld_gaussian(mean_1=enc_mean, std_1=enc_std, mean_2=None, std_2=None)
            if self.loss_function == 'nll':
                nll_loss = self.nll_gaussian_1(mean=dec_mean, std=dec_std, x=y.permute(1, 0, 2))
            elif self.loss_function == 'mse':
                nll_loss = self.mse(mean=dec_mean, x=y.permute(1, 0, 2))
            elif self.loss_function == 'beta_1':
                nll_loss = self.beta_gaussian_1(mean=dec_mean, x=y.permute(1, 0, 2), beta=self.robust_coeff)
            elif self.loss_function == 'beta_2':
                nll_loss = self.beta_gaussian_2(mean=dec_mean, std=dec_std, x=y.permute(1, 0, 2), beta=self.robust_coeff)


            # return nll_loss, kld_loss, torch.zeros(enc.squeeze(2).shape).to(device), enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(device), dec_mean, dec_std
            if self.use_PNF:
                return nll_loss, kld_loss, z_k, enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(self.device), dec_mean, dec_std
            else:
                return nll_loss, kld_loss, z_0, enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(self.device), dec_mean, dec_std

        else:

            # 1. Encoder
            # print("input_x:",x.shape) [32, 64, 18]  [W,B,F] because of RNN
            # print("input_y:",y.shape) [32, 64, 18]
            # computing hidden state in list and x_t & y_t in list outside the loop
            h_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(self.device)
            h, h_t = self.hidden_state_qrnn[0](x, h_0)


            # reversing hidden state list
            reversed_h = torch.flip(h, dims=[0])

            # reversing y_t list
            reversed_y = torch.flip(y, dims=[0])

            # concat reverse h with reverse x_t
            concat_h_y = torch.cat([reversed_y, reversed_h], dim=2)

            # compute reverse a_t

            a_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(self.device)
            a, a_t = self.qrnn[0](concat_h_y, a_0)
            reversed_a = torch.flip(a, dims=[0])

            #print("reversed_a shape:",reversed_a.shape)
            enc = self.phi_enc(reversed_a.permute(1, 0, 2)).unsqueeze(-2)
            enc_mean = self.enc_mean(enc).squeeze(2)
            enc_std = self.enc_std(enc).squeeze(2)
            z_0 = self.reparameterized_sample(enc_mean, enc_std)

            if self.use_PNF:
                z_k, logdet = self.PNF(z_0, True)

            # Prior
            #print("output_h:", h.permute(1, 0, 2).shape)
            ''' Prior
            input:
                h [64, 32, 32] (B, W, h_dim)
            output:
                mean [64, 32, 16] (B, W, z_dim)
                std  [64, 32, 16] (B, W, z_dim)
            '''
            prior = self.phi_prior(h.permute(1, 0, 2)).unsqueeze(-2)
            prior_mean = self.prior_mean(prior).squeeze(2)
            prior_std = self.prior_std(prior).squeeze(2)


            if self.use_PNF:
                dec = self.phi_dec(torch.cat([z_k, h.permute(1, 0, 2)], dim=2)).unsqueeze(-2)
            else:
                # print("dec:", torch.cat([z_0, h.permute(1, 0, 2)], dim=2).shape)
                # assert False
                dec = self.phi_dec(torch.cat([z_0, h.permute(1, 0, 2)], dim=2)).unsqueeze(-2)

            # Decoder

            dec_mean = self.dec_mean(dec).squeeze(2)
            dec_std = self.dec_std(dec).squeeze(2)

            if self.use_PNF:
                kld_loss = self.kld_gaussian_w_logdet(mean_1=enc_mean, std_1=enc_std, mean_2=prior_mean, std_2=prior_std, z_0=z_0, z_k=z_k, SLDJ=logdet)
            else:
                kld_loss = self.kld_gaussian(mean_1=enc_mean, std_1=enc_std, mean_2=prior_mean, std_2=prior_std)
                # kld_loss = self.kld_gaussian(mean_1=enc_mean, std_1=enc_std, mean_2=None, std_2=None)
            if self.loss_function == 'nll':
                nll_loss = self.nll_gaussian_1(mean=dec_mean, std=dec_std, x=y.permute(1, 0, 2))
            elif self.loss_function == 'mse':
                nll_loss = self.mse(mean=dec_mean, x=y.permute(1, 0, 2))
            elif self.loss_function == 'beta_1':
                nll_loss = self.beta_gaussian_1(mean=dec_mean, x=y.permute(1, 0, 2), beta=self.robust_coeff)
            elif self.loss_function == 'beta_2':
                nll_loss = self.beta_gaussian_2(mean=dec_mean, std=dec_std, x=y.permute(1, 0, 2), beta=self.robust_coeff)


            # contrast part

            # return nll_loss, kld_loss, torch.zeros(enc.squeeze(2).shape).to(device), enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(device), dec_mean, dec_std
            if self.use_PNF:
                return nll_loss, kld_loss, z_k, enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(self.device), dec_mean, dec_std
            else:
                return nll_loss, kld_loss, z_0, enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(self.device), dec_mean, dec_std

    def nll_gaussian_1(self, mean, std, x):

        return 0.5 * (torch.sum(std) + torch.sum(((x - mean) / std.mul(0.5).exp_()) ** 2))  # Owned definition
    def beta_gaussian_1(self, mean, x, beta, sigma=0.5):
        D = mean.shape[1]
        term1 = -((1 + beta) / beta)
        K1 = 1 / pow((2 * math.pi * (sigma ** 2)), (beta * D / 2))
        term2 = torch.nn.functional.mse_loss(input=mean, target=x, reduction='mean')
        term3 = torch.exp(-(beta / (2 * (sigma ** 2))) * term2)
        loss = torch.sum(term1 * (K1 * term3 - 1))
        return loss

    def beta_gaussian_2(self, mean, std, x, beta, sigma=0.5):
        D = mean.shape[1]
        term1 = -((1 + beta) / beta)
        K1 = 1 / pow((2 * math.pi * (sigma ** 2)), (beta * D / 2))
        term2 = 0.5 * (torch.sum(std) + torch.sum(((x - mean) / std.mul(0.5).exp_()) ** 2))
        # term2 = torch.nn.functional.mse_loss(input=mean, target=x, reduction='sum')
        term3 = torch.exp(-(beta / (2 * (sigma ** 2))) * term2)
        loss = torch.sum(term1 * (K1 * term3 - 1))
        return loss


    def mse(self, mean, x):

        REC = torch.nn.functional.mse_loss(input=mean, target=x, reduction='mean')


        #rec = torch.nn.functional.mse_loss(input=mean, target=x, reduction='mean').div(x.shape[0])
        return REC
    def kld_gaussian(self, mean_1, std_1, mean_2, std_2):
        if mean_2 is not None and std_2 is not None:
            kl_loss = 0.5 * torch.sum(
                2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(
                    2) - 1)
        else:
            kl_loss = -0.5 * torch.sum(1 + std_1 - mean_1.pow(2) - std_1.exp())
            #kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
        return kl_loss
    def reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return eps.mul(std).add_(mean)
    def kld_gaussian_w_logdet(self, mean_1, std_1, mean_2, std_2, z_0, z_k, SLDJ):
        if mean_2 is not None and std_2 is not None:
            # q0 = torch.distributions.normal.Normal(mean_1, (0.5 * std_1).exp())
            q0 = torch.distributions.normal.Normal(mean_1, std_1)
            prior = torch.distributions.normal.Normal(mean_2, std_2)
            log_prior_zK = prior.log_prob(z_k).sum(-1)
            log_q0_z0 = q0.log_prob(z_0).sum(-1)
            log_q0_zK = log_q0_z0 + SLDJ.sum(-1)
            kld = (log_q0_zK - log_prior_zK).sum()
            return kld
        else:
            # q0 = torch.distributions.normal.Normal(mean_1, (0.5 * std_1).exp())
            q0 = torch.distributions.normal.Normal(mean_1, std_1)
            prior = torch.distributions.normal.Normal(0., 1.)
            log_prior_zK = prior.log_prob(z_k).sum(-1)
            log_q0_z0 = q0.log_prob(z_0).sum(-1)
            log_q0_zK = log_q0_z0 + SLDJ.sum(-1)
            kld = (log_q0_zK - log_prior_zK).sum()
            return kld




