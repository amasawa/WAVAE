import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
# import dgl
# from dgl.nn import GATConv
# from torch.nn import TransformerEncoder
# from torch.nn import TransformerDecoder
# from src.dlutils import *
# from src.constants import *
import math

## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
	def __init__(self, feats):
		super(LSTM_Univariate, self).__init__()
		self.name = 'LSTM_Univariate'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 1
		self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

	def forward(self, x):
		hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64),
			torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
		outputs = []
		for i, g in enumerate(x):
			multivariate_output = []
			for j in range(self.n_feats):
				univariate_input = g.view(-1)[j].view(1, 1, -1)
				out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
				multivariate_output.append(2 * out.view(-1))
			output = torch.cat(multivariate_output)
			outputs.append(output)
		return torch.stack(outputs)

## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
	def __init__(self, feats):
		super(Attention, self).__init__()
		self.name = 'Attention'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = 5 # MHA w_size = 5
		self.n = self.n_feats * self.n_window
		self.atts = [ nn.Sequential( nn.Linear(self.n, feats * feats),
				nn.ReLU(True))	for i in range(1)]
		self.atts = nn.ModuleList(self.atts)

	def forward(self, g):
		for at in self.atts:
			ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
			g = torch.matmul(g, ats)
		return g, ats

## LSTM_AD Model
class LSTM_AD(nn.Module):
	def __init__(self, feats):
		super(LSTM_AD, self).__init__()
		self.name = 'LSTM_AD'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 64
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.lstm2 = nn.LSTM(feats, self.n_feats)
		self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

	def forward(self, x):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		hidden2 = (torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
		outputs = []
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
			out = self.fcn(out.view(-1))
			outputs.append(2 * out.view(-1))
		return torch.stack(outputs)

## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
	def __init__(self, feats):
		super(DAGMM, self).__init__()
		self.name = 'DAGMM'
		self.lr = 0.0001
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 8
		self.n_window = 5 # DAGMM w_size = 5
		self.n = self.n_feats * self.n_window
		self.n_gmm = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.estimate = nn.Sequential(
			nn.Linear(self.n_latent+2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
			nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
		)

	def compute_reconstruction(self, x, x_hat):
		relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
		cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
		return relative_euclidean_distance, cosine_similarity

	def forward(self, x):
		## Encode Decoder
		x = x.view(1, -1)
		z_c = self.encoder(x)
		x_hat = self.decoder(z_c)
		## Compute Reconstructoin
		rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
		z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
		## Estimate
		gamma = self.estimate(z)
		return z_c, x_hat.view(-1), z, gamma.view(-1)

## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
	def __init__(self, IS,BS,SL,lf):
		super(OmniAnomaly, self).__init__()
		self.name = 'OmniAnomaly'
		self.lr = 0.002
		self.beta = 0.01
		self.BS = BS
		self.SL = SL
		self.n_IS = IS
		self.n_hidden = 32
		self.n_latent = 8
		self.loss_function = lf
		self.lstm = nn.GRU(IS, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden),nn.SELU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.SELU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.SELU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.SELU(),
			#nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent),
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.SELU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.SELU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.SELU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.SELU(),
			nn.Linear(self.n_hidden, self.n_IS), nn.SELU(),
		)

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
		# rec = torch.nn.functional.mse_loss(input=mean, target=x, reduction='mean').div(x.shape[0])
		return REC

	def kld_gaussian(self, mean_1, std_1, mean_2, std_2):

		if mean_2 is not None and std_2 is not None:
			kl_loss = 0.5 * torch.sum(
				2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(
					2) - 1)
		else:
			kl_loss = -0.5 * torch.sum(1 + std_1 - mean_1.pow(2) - std_1.exp())
			#kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
		# kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
		return kl_loss

	def forward(self, x, hidden = None):
		# print(x.shape)
		# print(x.permute(1,0,2).shape)

		# x shape [64, 32, 18]  [BS, SL, IS]
		hidden = torch.rand(self.BS, self.SL, self.IS, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x.permute(1,0,2), hidden)

		## Encode
		latent = self.encoder(out)
		mu, logvar = torch.split(latent, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		z = mu + eps*std
		## Decoder
		out = self.decoder(z)

		kld_loss = self.kld_gaussian(mean_1=mu.permute(1, 0, 2), std_1=logvar.permute(1, 0, 2), mean_2=None, std_2=None)

		if self.loss_function == 'nll':
			nll_loss = self.nll_gaussian_1(mean=out, std=None, x=x.permute(1, 0, 2))
		elif self.loss_function == 'mse':
			nll_loss = self.mse(mean=out.permute(1, 0, 2), x=x)
		elif self.loss_function == 'beta_1':
			nll_loss = self.beta_gaussian_1(out.permute(1, 0, 2), x , beta=0.5)
		elif self.loss_function == 'beta_2':
			nll_loss = self.beta_gaussian_2(mean=out, std=logvar, x=x.permute(1, 0, 2), beta=0.005)

		return out, mu, logvar, hidden, nll_loss, kld_loss

## OmniAnomaly Model (KDD 19)
class OmniAnomaly2(nn.Module):
	def __init__(self, features, bz, seq_len,loss):
		super(OmniAnomaly2, self).__init__()
		self.name = 'OmniAnomaly'
		self.lr = 0.006  # 0.001 0.999 T4
		self.beta = 0.01
		self.features = features
		self.bz = bz
		self.seq_len = seq_len
		self.loss_function = loss
		self.n_feats = self.seq_len * self.bz * self.features
		self.n_hidden = 32
		self.n_latent = 8
		self.lstm = nn.GRU(self.n_feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2 * self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

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
		# rec = torch.nn.functional.mse_loss(input=mean, target=x, reduction='mean').div(x.shape[0])
		return REC

	def kld_gaussian(self, mean_1, std_1, mean_2, std_2):

		if mean_2 is not None and std_2 is not None:
			kl_loss = 0.5 * torch.sum(
				2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(
					2) - 1)
		else:
			kl_loss = -0.5 * torch.sum(1 + std_1 - mean_1.pow(2) - std_1.exp())
			#kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
		# kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
		return kl_loss

	def forward(self, x, hidden = None):
		# print(x.shape)
		# print(x.permute(1,0,2).shape)

		# x shape [64, 32, 18]  [BS, SL, IS]
		# hidden [2 ,1, 32], input must[seqLen=1,1,feats=] , gru must [feats ,32, 1]
		hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		# print(x.view(1,1,-1).shape)
		out, hidden = self.lstm(x.view(1,1,-1), hidden)

		## Encode
		latent = self.encoder(out)
		mu, logvar = torch.split(latent, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		z = mu + eps*std
		## Decoder
		out = self.decoder(z)


		kld_loss = self.kld_gaussian(mean_1=mu.view(1, 1, -1), std_1=logvar.view(1, 1, -1), mean_2=None, std_2=None)

		# print(kld_loss.item())

		if self.loss_function == 'nll':
			nll_loss = self.nll_gaussian_1(mean=out, std=None, x=x.permute(1, 0, 2))
		elif self.loss_function == 'mse':
			nll_loss = self.mse(mean=out.view(self.bz, self.seq_len,self.features), x=x)
		elif self.loss_function == 'beta_1':
			nll_loss = self.beta_gaussian_1(mean=out, x=x.permute(1, 0, 2), beta=self.robust_coeff)
		elif self.loss_function == 'beta_2':
			nll_loss = self.beta_gaussian_2(mean=out, std=logvar, x=x.permute(1, 0, 2), beta=0.005)


		return out.view(self.bz, self.seq_len,self.features), mu, logvar, hidden, nll_loss, kld_loss



## USAD Model (KDD 20)
class USAD(nn.Module):
	def __init__(self, feats):
		super(USAD, self).__init__()
		self.name = 'USAD'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 5
		self.n_window = 5 # USAD w_size = 5
		self.n = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)


	def forward(self, g):
		## Encode
		z = self.encoder(g.view(1,-1))
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)

## MSCRED Model (AAAI 19)
# class MSCRED(nn.Module):
# 	def __init__(self, feats):
# 		super(MSCRED, self).__init__()
# 		self.name = 'MSCRED'
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.n_window = feats
# 		self.encoder = nn.ModuleList([
# 			ConvLSTM(1, 32, (3, 3), 1, True, True, False),
# 			ConvLSTM(32, 64, (3, 3), 1, True, True, False),
# 			ConvLSTM(64, 128, (3, 3), 1, True, True, False),
# 			]
# 		)
# 		self.decoder = nn.Sequential(
# 			nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
# 			nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
# 			nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
# 		)
#
# 	def forward(self, g):
# 		## Encode
# 		z = g.view(1, 1, self.n_feats, self.n_window)
# 		for cell in self.encoder:
# 			_, z = cell(z.view(1, *z.shape))
# 			z = z[0][0]
# 		## Decode
# 		x = self.decoder(z)
# 		return x.view(-1)

## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
	def __init__(self, feats):
		super(CAE_M, self).__init__()
		self.name = 'CAE_M'
		self.lr = 0.001
		self.n_feats = feats
		self.n_window = feats
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = g.view(1, 1, self.n_feats, self.n_window)
		z = self.encoder(z)
		## Decode
		x = self.decoder(z)
		return x.view(-1)

# ## MTAD_GAT Model (ICDM 20)
# class MTAD_GAT(nn.Module):
# 	def __init__(self, feats):
# 		super(MTAD_GAT, self).__init__()
# 		self.name = 'MTAD_GAT'
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.n_window = feats
# 		self.n_hidden = feats * feats
# 		self.g = dgl.graph((torch.tensor(list(range(1, feats+1))), torch.tensor([0]*feats)))
# 		self.g = dgl.add_self_loop(self.g)
# 		self.feature_gat = GATConv(feats, 1, feats)
# 		self.time_gat = GATConv(feats, 1, feats)
# 		self.gru = nn.GRU((feats+1)*feats*3, feats*feats, 1)
#
# 	def forward(self, data, hidden):
# 		hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
# 		data = data.view(self.n_window, self.n_feats)
# 		data_r = torch.cat((torch.zeros(1, self.n_feats), data))
# 		feat_r = self.feature_gat(self.g, data_r)
# 		data_t = torch.cat((torch.zeros(1, self.n_feats), data.t()))
# 		time_r = self.time_gat(self.g, data_t)
# 		data = torch.cat((torch.zeros(1, self.n_feats), data))
# 		data = data.view(self.n_window+1, self.n_feats, 1)
# 		x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
# 		x, h = self.gru(x, hidden)
# 		return x.view(-1), h

# ## GDN Model (AAAI 21)
# class GDN(nn.Module):
# 	def __init__(self, feats):
# 		super(GDN, self).__init__()
# 		self.name = 'GDN'
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.n_window = 5
# 		self.n_hidden = 16
# 		self.n = self.n_window * self.n_feats
# 		src_ids = np.repeat(np.array(list(range(feats))), feats)
# 		dst_ids = np.array(list(range(feats))*feats)
# 		self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
# 		self.g = dgl.add_self_loop(self.g)
# 		self.feature_gat = GATConv(1, 1, feats)
# 		self.attention = nn.Sequential(
# 			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
# 		)
# 		self.fcn = nn.Sequential(
# 			nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
# 		)
#
# 	def forward(self, data):
# 		# Bahdanau style attention
# 		att_score = self.attention(data).view(self.n_window, 1)
# 		data = data.view(self.n_window, self.n_feats)
# 		data_r = torch.matmul(data.permute(1, 0), att_score)
# 		# GAT convolution on complete graph
# 		feat_r = self.feature_gat(self.g, data_r)
# 		feat_r = feat_r.view(self.n_feats, self.n_feats)
# 		# Pass through a FCN
# 		x = self.fcn(feat_r)
# 		return x.view(-1)

# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
	def __init__(self, feats):
		super(MAD_GAN, self).__init__()
		self.name = 'MAD_GAN'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_window = 5 # MAD_GAN w_size = 5
		self.n = self.n_feats * self.n_window
		self.generator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.discriminator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Generate
		z = self.generator(g.view(1,-1))
		## Discriminator
		real_score = self.discriminator(g.view(1,-1))
		fake_score = self.discriminator(z.view(1,-1))
		return z.view(-1), real_score.view(-1), fake_score.view(-1)

# # Proposed Model (VLDB 22)
# class TranAD_Basic(nn.Module):
# 	def __init__(self, feats):
# 		super(TranAD_Basic, self).__init__()
# 		self.name = 'TranAD_Basic'
# 		self.lr = lr
# 		self.batch = 128
# 		self.n_feats = feats
# 		self.n_window = 10
# 		self.n = self.n_feats * self.n_window
# 		self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
# 		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
# 		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
# 		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
# 		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
# 		self.fcn = nn.Sigmoid()
#
# 	def forward(self, src, tgt):
# 		src = src * math.sqrt(self.n_feats)
# 		src = self.pos_encoder(src)
# 		memory = self.transformer_encoder(src)
# 		x = self.transformer_decoder(tgt, memory)
# 		x = self.fcn(x)
# 		return x

# Proposed Model (FCN) + Self Conditioning + Adversarial + MAML (VLDB 22)
# class TranAD_Transformer(nn.Module):
# 	def __init__(self, feats):
# 		super(TranAD_Transformer, self).__init__()
# 		self.name = 'TranAD_Transformer'
# 		self.lr = lr
# 		self.batch = 128
# 		self.n_feats = feats
# 		self.n_hidden = 8
# 		self.n_window = 10
# 		self.n = 2 * self.n_feats * self.n_window
# 		self.transformer_encoder = nn.Sequential(
# 			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
# 			nn.Linear(self.n_hidden, self.n), nn.ReLU(True))
# 		self.transformer_decoder1 = nn.Sequential(
# 			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
# 			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
# 		self.transformer_decoder2 = nn.Sequential(
# 			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
# 			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
# 		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
#
# 	def encode(self, src, c, tgt):
# 		src = torch.cat((src, c), dim=2)
# 		src = src.permute(1, 0, 2).flatten(start_dim=1)
# 		tgt = self.transformer_encoder(src)
# 		return tgt
#
# 	def forward(self, src, tgt):
# 		# Phase 1 - Without anomaly scores
# 		c = torch.zeros_like(src)
# 		x1 = self.transformer_decoder1(self.encode(src, c, tgt))
# 		x1 = x1.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
# 		x1 = self.fcn(x1)
# 		# Phase 2 - With anomaly scores
# 		c = (x1 - src) ** 2
# 		x2 = self.transformer_decoder2(self.encode(src, c, tgt))
# 		x2 = x2.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
# 		x2 = self.fcn(x2)
# 		return x1, x2

# # Proposed Model + Self Conditioning + MAML (VLDB 22)
# class TranAD_Adversarial(nn.Module):
# 	def __init__(self, feats):
# 		super(TranAD_Adversarial, self).__init__()
# 		self.name = 'TranAD_Adversarial'
# 		self.lr = lr
# 		self.batch = 128
# 		self.n_feats = feats
# 		self.n_window = 10
# 		self.n = self.n_feats * self.n_window
# 		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
# 		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
# 		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
# 		decoder_layers = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
# 		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
# 		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
#
# 	def encode_decode(self, src, c, tgt):
# 		src = torch.cat((src, c), dim=2)
# 		src = src * math.sqrt(self.n_feats)
# 		src = self.pos_encoder(src)
# 		memory = self.transformer_encoder(src)
# 		tgt = tgt.repeat(1, 1, 2)
# 		x = self.transformer_decoder(tgt, memory)
# 		x = self.fcn(x)
# 		return x
#
# 	def forward(self, src, tgt):
# 		# Phase 1 - Without anomaly scores
# 		c = torch.zeros_like(src)
# 		x = self.encode_decode(src, c, tgt)
# 		# Phase 2 - With anomaly scores
# 		c = (x - src) ** 2
# 		x = self.encode_decode(src, c, tgt)
# 		return x
#
# # Proposed Model + Adversarial + MAML (VLDB 22)
# class TranAD_SelfConditioning(nn.Module):
# 	def __init__(self, feats):
# 		super(TranAD_SelfConditioning, self).__init__()
# 		self.name = 'TranAD_SelfConditioning'
# 		self.lr = lr
# 		self.batch = 128
# 		self.n_feats = feats
# 		self.n_window = 10
# 		self.n = self.n_feats * self.n_window
# 		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
# 		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
# 		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
# 		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
# 		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
# 		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
# 		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
# 		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
#
# 	def encode(self, src, c, tgt):
# 		src = torch.cat((src, c), dim=2)
# 		src = src * math.sqrt(self.n_feats)
# 		src = self.pos_encoder(src)
# 		memory = self.transformer_encoder(src)
# 		tgt = tgt.repeat(1, 1, 2)
# 		return tgt, memory
#
# 	def forward(self, src, tgt):
# 		# Phase 1 - Without anomaly scores
# 		c = torch.zeros_like(src)
# 		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
# 		# Phase 2 - With anomaly scores
# 		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
# 		return x1, x2
#
# # Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
# class TranAD(nn.Module):
# 	def __init__(self, feats):
# 		super(TranAD, self).__init__()
# 		self.name = 'TranAD'
# 		self.lr = lr
# 		self.batch = 128
# 		self.n_feats = feats
# 		self.n_window = 10
# 		self.n = self.n_feats * self.n_window
# 		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
# 		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
# 		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
# 		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
# 		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
# 		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
# 		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
# 		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
#
# 	def encode(self, src, c, tgt):
# 		src = torch.cat((src, c), dim=2)
# 		src = src * math.sqrt(self.n_feats)
# 		src = self.pos_encoder(src)
# 		memory = self.transformer_encoder(src)
# 		tgt = tgt.repeat(1, 1, 2)
# 		return tgt, memory
#
# 	def forward(self, src, tgt):
# 		# Phase 1 - Without anomaly scores
# 		c = torch.zeros_like(src)
# 		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
# 		# Phase 2 - With anomaly scores
# 		c = (x1 - src) ** 2
# 		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
#		return x1, x2