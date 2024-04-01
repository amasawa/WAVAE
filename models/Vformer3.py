import torch
import math
import torch.nn as nn
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

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

class Vformer(nn.Module):
    def __init__(self, args):
        super(Vformer, self).__init__()
        self.configs = args
        self.seq_len = self.configs.seq_len
        self.label_len = self.configs.label_len
        self.pred_len = self.configs.pred_len
        self.output_attention = self.configs.output_attention

        # Decomp
        kernel_size = self.configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.configs.enc_in, self.configs.d_model, self.configs.embed, self.configs.freq,
                                                  self.configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.configs.dec_in, self.configs.d_model, self.configs.embed, self.configs.freq,
                                                  self.configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.configs.factor, attention_dropout=self.configs.dropout,
                                        output_attention=self.configs.output_attention),
                        self.configs.d_model, self.configs.n_heads),
                    self.configs.d_model,
                    self.configs.d_ff,
                    moving_avg=self.configs.moving_avg,
                    dropout=self.configs.dropout,
                    activation=self.configs.activation
                ) for l in range(self.configs.e_layers)
            ],
            norm_layer=my_Layernorm(self.configs.d_model)
        )
        # Pred_Decoder
        self.formerdecoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, self.configs.factor, attention_dropout=self.configs.dropout,
                                        output_attention=False),
                        self.configs.d_model, self.configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.configs.factor, attention_dropout=self.configs.dropout,
                                        output_attention=False),
                        self.configs.d_model, self.configs.n_heads),
                    self.configs.d_model,
                    self.configs.c_out,
                    self.configs.d_ff,
                    moving_avg=self.configs.moving_avg,
                    dropout=self.configs.dropout,
                    activation=self.configs.activation,
                )
                for l in range(self.configs.d_layers)
            ],
            norm_layer=my_Layernorm(self.configs.d_model),
            projection=nn.Linear(self.configs.d_model, self.configs.c_out, bias=True)
        )

        # # prior
        # self.phi_prior = nn.Sequential(
        #     nn.Linear(self.configs.d_model, self.configs.h_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.configs.h_dim, self.configs.h_dim),
        #     nn.ReLU()).cuda()
        # self.prior_mean = nn.Sequential(
        #     nn.Linear(self.configs.h_dim, self.configs.z_dim),
        #     nn.Sigmoid()).cuda()
        # self.prior_std = nn.Sequential(
        #     nn.Linear(self.configs.h_dim, self.configs.z_dim),
        #     nn.Softplus()).cuda()

        # encoder
        self.phi_enc = nn.Sequential(
            nn.Linear(self.configs.pred_len*self.configs.d_model, self.configs.h_dim),
            nn.ReLU(),
            nn.Linear(self.configs.h_dim, self.configs.h_dim),
            nn.ReLU())
        self.enc_mean = nn.Sequential(
            nn.Linear(self.configs.h_dim, self.configs.z_dim),
            nn.Sigmoid()).cuda()
        self.enc_std = nn.Sequential(
            nn.Linear(self.configs.h_dim, self.configs.z_dim),
            nn.Softplus()).cuda()

        # decoder
        self.phi_dec = nn.Sequential(
            nn.Linear(self.configs.z_dim, self.configs.h_dim),
            nn.ReLU(),
            nn.Linear(self.configs.h_dim, self.configs.h_dim),
            nn.ReLU()).cuda()
        self.dec_std = nn.Sequential(
            nn.Linear(self.configs.h_dim, self.pred_len*self.configs.x_dim),
            nn.Softplus()).cuda()
        self.dec_mean = nn.Sequential(
            nn.Linear(self.configs.h_dim, self.pred_len*self.configs.x_dim),
            nn.Sigmoid()).cuda()
        if self.configs.use_PNF:
            # self.PNF = PlanarFlowSequence(sequence_length=self.rolling_size, latent_dim=self.z_dim, K=self.PNF_layers).to(self.device)
            self.PNF = PlanarFlow(latent_dim=self.z_dim, K=self.PNF_layers).to(self.device)


    def forward(self,x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        '''
        FormerEncoder
            input:
                enc_emd (embedding from x_enc [64, 96, 18] and x_mark_enc [64, 96, 3])
            output:
                enc_out [64, 96, 512]
        FormerDecoder
            input:
                dec_emd (embedding from seasonal_init [64, 96, 18] and x_mark_dec [64, 96, 3])
                        (seasonal_init [64, 96, 18] is embedded from x_enc [64, 96, 18])
                enc_out
            output:
                dec_out (trend_part [64, 144, 18] + seasonal_part [64, 144, 18])
        return
            outputs [64, 96, 18]


        Encoder: q(z_{t}|x_{t})
            input:
                enc_out [64, 96, 512]
            output:
                enc_mean     [64, 16]
                enc_std      [64, 16]
                z_0          [64, 16]
        Decoder: P(z_{t}|x_{t})
            input:
                z_{0}/z_{k} [64, 16]
            output:
                dec_mean  [64, 96, 18]
                dec_std   [64, 96, 18]
        # prior: P(z_{t}|z_{t-1})
        #     input:
        #         enc_out [64, 96, 512]
        #     output:
        #         prior_mean [64, 96, 16]
        #         prior_std [64, 96, 16]
        '''

        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)  # trend_init is after the avgPooling and seasonal_init is x-poooling


        # informerdecoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)


        # Encoder
        enc_emd = self.enc_embedding(x_enc, x_mark_enc) # [64, 96, 512]


        enc_out, attns = self.encoder(enc_emd, attn_mask=enc_self_mask) # [64, 96, 512]

        # formerDecoder
        dec_emd = self.dec_embedding(seasonal_init, x_mark_dec) #[64, 144, 512]


        seasonal_part, trend_part = self.formerdecoder(dec_emd, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init) #[64, 144, 18],[64, 144, 18]

        dec_out = trend_part + seasonal_part
        if self.configs.output_attention:
            outputs = dec_out[:, -self.configs.pred_len:, :], attns
        else:
            outputs = dec_out[:, -self.configs.pred_len:, :]  # [B, L, D]


        # Prior
        # prior = self.phi_prior(enc_out).unsqueeze(-2)
        # prior_mean = self.prior_mean(prior).squeeze(2)  # [64, 96, 16]
        # prior_std = self.prior_std(prior).squeeze(2)    # [64, 96, 16]


        # encoder

        enc = self.phi_enc(enc_out.reshape(-1,self.configs.pred_len*self.configs.d_model))
        enc_mean = self.enc_mean(enc)#[64, 16]
        enc_std = self.enc_std(enc)   #[64, 16]
        z_0 = self.reparameterized_sample(enc_mean, enc_std)  #[64, 16]


        if self.configs.use_PNF:
            z_k, logdet = self.PNF(z_0, True)

        # decoder
        if self.configs.use_PNF:
            dec = self.phi_dec(z_k)
        else:
            dec = self.phi_dec(z_0)
        dec_mean = self.dec_mean(dec).reshape(-1,x_enc.shape[1],x_enc.shape[2])# [64, 96*18]
        dec_std = self.dec_std(dec).reshape(-1,x_enc.shape[1],x_enc.shape[2])   # [64, 96*18]

        if self.configs.use_PNF:
            kld_loss = self.kld_gaussian_w_logdet(mean_1=enc_mean, std_1=enc_std, mean_2=None, std_2=None,
                                                  z_0=z_0, z_k=z_k, SLDJ=logdet)
        else:
            kld_loss = self.kld_gaussian(mean_1=enc_mean, std_1=enc_std, mean_2=None, std_2=None)


        if self.configs.loss_function == 'nll':
            nll_loss = self.nll_gaussian_1(mean=dec_mean, std=dec_std, x = x_enc)
        elif self.configs.loss_function == 'mse':
            nll_loss = self.mse(mean=dec_mean, x = x_enc)
        elif self.configs.loss_function == 'beta_1':
            nll_loss = self.beta_gaussian_1(mean=dec_mean, x=x_enc, beta=self.configs.robust_coeff)
        elif self.configs.loss_function == 'beta_2':
            nll_loss = self.beta_gaussian_2(mean=dec_mean, std=dec_std, x=x_enc, beta=self.configs.robust_coeff)



        if self.configs.use_PNF:
            return nll_loss, kld_loss, z_k, enc_mean, enc_std, None, dec_mean, dec_std,outputs
        else:
            return nll_loss, kld_loss, z_0, enc_mean, enc_std, None, dec_mean, dec_std,outputs



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

    def reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        return eps.mul(std).add_(mean)

    def kld_gaussian(self, mean_1, std_1, mean_2, std_2):
        if mean_2 is not None and std_2 is not None:
            kl_loss = 0.5 * torch.sum(
                2 * torch.log(std_2) - 2 * torch.log(std_1+(1e-10)) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / (std_2).pow(
                    2) - 1)
        else:
            kl_loss = -0.5*(1+std_1-mean_1**2-std_1.exp()).sum(1).mean()
        return kl_loss
    def nll_gaussian_1(self, mean, std, x):
        meana = mean.reshape(-1,x.shape[1],x.shape[2])
        return 0.5 * (torch.sum(std) + torch.sum(((x - meana) / std.mul(0.5).exp_()) ** 2))  # Owned definition
    def mse(self, mean, x):
        return torch.nn.functional.mse_loss(mean, x, size_average=False).div(x.shape[0])
        #return torch.nn.functional.mse_loss(input=mean.reshape(-1,x.shape[1],x.shape[2]), target=x, reduction='mean')
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
        loss = torch.sum(term1 * (K1 * term3 - 1)).div(x.shape[0])
        return loss
