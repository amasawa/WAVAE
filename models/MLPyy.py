import torch
import math
import torch.nn as nn
import torch.nn.init as init
from mlp_mixer_pytorch import MLPMixer




class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.configs = args

        # Encoder
        self.enc = nn.Sequential(
            MLPMixer(
            image_size=(90, self.configs.x_dim),
            channels=1,
            patch_size=self.configs.x_dim,
            dim=512,
            depth=12,
            num_classes = self.configs.z_dim*2
        ),
            nn.Softplus(),
        ).cuda()



        self.dec = nn.Sequential(
            nn.Linear(self.configs.z_dim, self.configs.h_dim),
            nn.ReLU(),
            nn.Linear(self.configs.h_dim, self.configs.h_dim),
            nn.ReLU()).cuda()
        self.dec_std = nn.Sequential(
            nn.Linear(self.configs.h_dim, 90 * self.configs.x_dim),
            nn.Softplus()).cuda()
        self.dec_mean = nn.Sequential(
            nn.Linear(self.configs.h_dim, 90 * self.configs.x_dim),
            nn.ReLU()).cuda()

    def forward(self,x, y):
        # decomp init

        '''
        p(y|y) -KL(q(z|y)||p(z))
        Input
            x: input       [64, 90, 18]
            y: prediction  [64, 126, 18]
        Encoder
            input
                y [64, 1, 126, 18]
            output
                y_z [64， 10]
        Decoder
            input
                y_z [64, 10]
            output
                y_hat [64, 126, 18]
        prior
            input
                x [64, 90, 18]
            output
                x_z [64, 10]
        '''

        # Encoder

        enc = self.enc(y.unsqueeze(1))
        enc_mean = enc[:,:self.configs.z_dim]
        enc_std = enc[:,-self.configs.z_dim:]


        z  = self.reparameterized_sample(enc_mean, enc_std)

        # Decoder
        dec = self.dec(z)

        dec_mean = self.dec_mean(dec).view(y.shape)
        dec_std =  self.dec_std(dec).view(y.shape)

        kld_loss = self.kld_gaussian(mean_1=enc_mean, std_1=enc_std, mean_2=None, std_2=None)




        if self.configs.loss_function == 'nll':
            nll_loss = self.nll_gaussian_1(mean=dec_mean, std=dec_std, x = y)
        elif self.configs.loss_function == 'mse':
            nll_loss = self.mse(mean=dec_mean, x = y)
        elif self.configs.loss_function == 'beta_1':
            nll_loss = self.beta_gaussian_1(mean=dec_mean, x=y, beta=self.configs.robust_coeff)
        elif self.configs.loss_function == 'beta_2':
            nll_loss = self.beta_gaussian_2(mean=dec_mean, std=dec_std, x=y, beta=self.configs.robust_coeff)
        elif self.configs.loss_function == 'bce':
            nll_loss = self.bce(dec_mean,y)

        return nll_loss, kld_loss, dec_mean, dec_std


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
                2 * torch.log(std_2/std_1)  + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / (std_2).pow(
                    2) - 1).mean()
        else:
            kl_loss = -0.5*(1+std_1-mean_1**2-(std_1).exp()).sum(1).mean()
        return kl_loss
    def nll_gaussian_1(self, mean, std, x):
        meana = mean.reshape(-1,x.shape[1],x.shape[2])
        return 0.5 * (torch.sum(std) + torch.sum(((x - meana) / std.mul(0.5).exp_()) ** 2))  # Owned definition

    def bce(self,mean, x):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(mean, x, size_average=False).div(x.shape[0]*x.shape[1])
        return loss
    def mse(self, mean, x):
        REC = torch.nn.functional.mse_loss(input=mean, target=x, reduction='mean')
        # print(REC.item())
        # assert False
        return REC
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
        loss = torch.sum(term1 * (K1 * term3 - 1))
        return loss