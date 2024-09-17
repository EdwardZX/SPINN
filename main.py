# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
# os.system('pip install fbm')
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob

from TrainJumpSDEsTransform import *
from trajutils import generate_x
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default= 100, #50 75
                    help='Default number of training epochs (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=20, help= '(default: %(default)s)')
parser.add_argument('--lr', type=float, default=1e-3, help= '(default: %(default)s)')
parser.add_argument('--seed', type=int, default=0, help= '(default: %(default)s)')
parser.add_argument('--clip_grad', type=float, default=5.0, help= '(default: %(default)s)')
parser.add_argument('--default_enc_nn', type=tuple, default=((128, 'gelu'), (128, 'gelu')),
                    help= '(default: %(default)s)') # 128
parser.add_argument('--hidden_size', type=int, default=128, # 128
                    help='Default number of RNN hidden size (default: %(default)s)')
parser.add_argument('--layers', type=int, default=2,
                    help='Default number of LSTM layers (default: %(default)s)')
parser.add_argument('--h_dim', type=int, default=64, # 64
                    help='Default number of embedding size (default: %(default)s)')
parser.add_argument('--d_model', type=int, default=128, # 128
                    help='Transformer hidden dimension (default: %(default)s)')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Default number of attention head (default: %(default)s)')
parser.add_argument('--enc_layers', type=int, default=3,
                    help='Default number of attention head (default: %(default)s)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='Default number of dropout (default: %(default)s)')
parser.add_argument('--scale', type=float, default= 0.375, #0.375, 0.25
                    help='schedual learning rates scaling (default: %(default)s)')
parser.add_argument('--in_dim', type=int, default=2,
                    help='Default number of input time series dimension (default: %(default)s)')
parser.add_argument('--out_dim', type=int, default=2,
                    help='Default number of input time series dimension (default: %(default)s)')
parser.add_argument('--encoder_length', type=int, default=50,
                    help='Default encoder length of input time series (default: %(default)s)')
parser.add_argument('--warmup_steps', type=int, default=2000, #400, 2000
                    help='Training warmup steps (default: %(default)s)')
parser.add_argument('--sample_nums', type=int, default=5,
                    help='evaluation figures samples (default: %(default)s)')
parser.add_argument('--dload', type=str, default='./model_dir',
                    help='save_dir (default: %(default)s)')
parser.add_argument('--beta_ac', type=float, default=0.5, #[0.2-0.3], for sample 0.1,7.5 is OK,5
                    help='The autocorrelation ratio (default: %(default)s)')
parser.add_argument('--beta_observe', type=float, default=0.0,#5e-1
                    help='The observation delete paramters (default: %(default)s)')
parser.add_argument('--beta_weight', type=float, default=0.5,#5e-3, 0.57 [0.5-0.6] 0.725, 0.675 no
                    help='The PINN loss bias (default: %(default)s)')
parser.add_argument('--beta_conti', type=float, default=0.,
                    help='The continuous of time series (default: %(default)s)')
parser.add_argument('--RESUME', type=bool, default=False,
                    help='The checkpoint is resume train (default: %(default)s)')
parser.add_argument('--noise_adding', type=float, default=0.25,
                    help='The external noise is adding noise level (default: %(default)s)')
parser.add_argument('--filename', type=str, default='andi',
                    help='The filename in data folder')

config = parser.parse_args()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    is_train = True
    alpha = [1.0] #[0,2.0-5000 OK]
    # alpha = [0.1,0.75,1.0]  # [0,2.0-5000 OK]
    # # alpha = np.linspace(0.05,2,2).tolist()
    Dt = [1]
    filename_tags = ['var','vel']
    # save_name = 'noise-var-test-noise-1-2'
    # save_name = 'alpha' + '-'.join(list(map(str, alpha))) + '_Dt' + '-'.join(list(map(str, Dt)))
    # config.filename = None
    # num = 5000
    # T = 100
    # x, _ = generate_x(alpha=alpha, Dt=Dt, time_span=num * T, seed=config.seed)
    # X = x.reshape(num * len(alpha) * len(Dt), T, 2)
    # X_1 = (1 * np.random.randn(num * len(alpha) * len(Dt), T, 2)) ** 2
    # X_2 = (0.5 * np.random.randn(num * len(alpha) * len(Dt), T, 2))**2
    # X = np.concatenate([X_1[:2000//2], X_2[2000//2:]], axis=0)
    # config.beta_obs = 0

    if config.filename is not None:
        # searching training number
        saved_name = '-'.join([config.filename,'noise',str(config.noise_adding)])
        file_sets = glob.glob(os.path.join(config.dload, saved_name +"*.pth"))
        cnt_minus = 0
        for tag_f in filename_tags:
            for f in file_sets:
                if tag_f in f:

                    cnt_minus +=1
        cnt = max(len(file_sets)  - cnt_minus,0)
        if is_train and (not config.RESUME):
            save_name = '-'.join([saved_name,str(cnt+1)])
        else:
            save_name = '-'.join([saved_name, str(cnt)])


    config.save_name = save_name
    config.model_name = save_name + '.pth'
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train_dataset, test_dataset = generate_dataset_torch(DatasetTracks(X,config=config,sample_rate=1.))
    # train_dataset, test_dataset = generate_dataset_torch(TensorDataset(torch.from_numpy(X).type(dtype=torch.float)))
    train_dataset, test_dataset = generate_dataset_torch(
                                    merge_datasets(filepath='./data',
                                                filenames=[config.filename + '_annotation.txt'],
                                                config=config,sample_rate=1.),seed=config.seed)

    if is_train:
        model = train(train_datasets=train_dataset, config=config, test_datasets=test_dataset)
        # model = train(train_datasets=train_dataset, config=config, test_datasets=test_dataset, L2=True)

    else:
        model = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                    h_dim=config.h_dim, default_enc_nn=config.default_enc_nn,
                                    hidden_dim=config.hidden_size, layers=config.layers,
                                    d_model=config.d_model, n_head=config.n_heads, layers_enc=config.enc_layers,
                                    beta_obs=config.beta_observe, beta_conti=config.beta_conti,
                                    beta_weight= config.beta_weight,
                                    device=config.device).to(config.device)
        load(model, config.dload, config.model_name)

    # del x
    # use memmap to open large file

    # y0, y1, sigma_run = transform(train_dataset, model, config) #LxBxD
    # _, _, sigma_run = transformToMemmap(train_dataset, model, config)

    # evaluate_draw(train_dataset,model, config)
    # evaluate_draw_tracks(train_dataset,model, config)
    #######
    # traing of Dt
    #######
    # is_train = True
    config.epochs = 50
    # config.beta_observe = 5e-2
    config.beta_ac = 0.5
    config.beta_weight = 0.5
    # _, s1 = train_sigma(DataX, model, config, is_train=is_train, beta_obs=2e0)
    s_train_dataset, model_s = train_sigma(train_dataset, model, config, beta_obs=0.0,is_train=is_train)

    # evaluate_draw(s_train_dataset, model_s, config)
    # evaluate_draw_tracks(s_train_dataset, model_s, config)
    # print(s1.shape)

    #######
    # traing of velocity
    #######
    v_train_dataset, model_v = train_velocity(train_dataset, model, config,is_train=is_train)
    # evaluate_draw(v_train_dataset, model_v, config)
    ## drawing total in here, not load in memory


    ## get the generated trajectories
    traj_multipe_transformer_memmap(test_dataset, model,config,num=50)

    clear_clash(config)

