import numpy as np
import torch
import random
from utils.utils import str2bool
import argparse

from solver import Solver


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # new seed
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def main(args):

    setup_seed(args.seed)
    TSAD = Solver(args)
    if args.train:
        TSAD.train()
    else:
        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    ## 0.mics
    parser.add_argument('--train', action='store_true', help='True when in shh, otherwise False')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_name', type=str, default="VQRAE")
    parser.add_argument('--model_name',metavar='-m',type=str,required=False,default='VQRAE||autoformer||MLP|KDD19',help="model name")


    ## 1.data
    ## 1.1 dataset
    ## 1.2 sequenceProcessing

    parser.add_argument('--dataset', '-data', type=str, default="VQRAE")
    parser.add_argument('--preprocessing', '-pre', action='store_true')
    parser.add_argument('--rolling_size', '-rs', type=int, default=32)

    ## 1.3 former_processing
    parser.add_argument('--formerdata', type=str, default='custom', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly, u:microsecondly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--root_path', type=str, default='/home/zhanwu/Data/dataset/GD/data/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Genesis_AnomalyLabels.csv', help='data file')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=32, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=18, help='start token length')
    parser.add_argument('--pred_len', type=int, default=32, help='prediction sequence length')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    ## fomer model
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--enc_in', type=int, default=18, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=18, help='decoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--activation', type=str, default='relu', help='activation')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--c_out', type=int, default=18, help='output size')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')



    ## 2.model
    parser.add_argument('--x_dim', '-xd', type=int, default=18)
    parser.add_argument('--h_dim', '-hd', type=int, default=32)
    parser.add_argument('--z_dim', '-zd', type=int, default=16)
    ## 2.1 optimization
    parser.add_argument('--epochs', '-ep', type=int, default=100)
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-8) #-6
    parser.add_argument('--loss_function', type=str, default='mse')
    parser.add_argument('--gradient_clip_norm', type=float, default=10)
    parser.add_argument('--use_clip_norm', action='store_true')
    ## 2.3 layers
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--use_PNF', '-up', type=str2bool, default=False)
    parser.add_argument('--PNF_layers', type=int, default=10)
    parser.add_argument('--use_bidirection', '-ub', action='store_true')
    parser.add_argument('--robust_coeff', type=float, default=0.005)
    ## 2.4 VAE
    parser.add_argument('--annealing', type=str2bool, default=False)
    parser.add_argument('--lmbda', type=float, default=0.0001)
    ## 2.4 MTS


    ## 3.save and load
    parser.add_argument('--save_model', action='store_true')  # save model
    parser.add_argument('--pid', type=int, default=0)



    # contrast

    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument("--latent_clr_weight", type=float, default=0.3, help="weight for latent clr loss")
    ## 4.metrics

    ## 5.display
    parser.add_argument('--eval_epoch', '-de', type=int, default=3)
    parser.add_argument('--display_epoch', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--discriminator', action='store_true', help='True when in shh, otherwise False')





    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    # print(unknown)
    # print(args)
    main(args)

