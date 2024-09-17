from model.JumpSDEsTrans import JumpSDEsTransformer, RawGRU
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset,Dataset,ConcatDataset
import torch.optim as optim
from sklearn.model_selection  import train_test_split
import os
import logging
import argparse
from trajutils import generate_x, reject_outliers
import time, datetime
from torch.autograd import Variable
from model.sam import SAM
import pandas as pd
from robust_loss_pytorch import adaptive
from scipy import signal
from itertools import cycle
import re

class CustomSchedule(object):
    def __init__(self, d_model, warmup_steps = 400, optimizer = None, scale=0.1):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.d_model = float(d_model)
        self.warmup_steps = warmup_steps
        self.steps = 1.
        self.optimizer = optimizer
        self.scale = scale

    def step(self):
        arg1 = self.steps** -0.5
        arg2 = self.steps * (self.warmup_steps ** -1.5)
        self.steps += 1.
        lr = self.scale*(self.d_model ** -0.5) * min(arg1,arg2)
        if self.steps > self.warmup_steps:
            lr = max(lr,5e-5)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

# class AutocorrelationSchedule(object):
#     def __init__(self, lo=5e-2, hi = 2.0, warmup_steps = 400,eps = 5e-2,delta_ac = 0.1):
#         super(AutocorrelationSchedule, self).__init__()
#
#         self.warmup_steps = warmup_steps
#         self.steps = 1.
#         self.lo = max(lo,eps)
#         self.hi = max(hi,eps)
#         self.eps = eps
#         self.dt = delta_ac
#
#     def step(self):
#         arg1 = -self.steps** -0.5
#         arg2 = -self.steps * (self.warmup_steps ** -1.5)
#         x0 = self.warmup_steps ** -0.5
#         # scale = (self.hi-self.lo) / x0
#         scale = (self.hi - self.eps) / x0
#         self.steps += 1.
#         ac_weight = scale * (max(arg1,arg2) + x0) + self.eps
#         # ac_weight = max(ac_weight, self.eps)
#         if ac_weight < self.lo:
#             ac_weight = self.lo
#         if ac_weight + self.dt > self.hi:
#             ac_weight = self.hi
#
#         # ac_weight = min(ac_weight+self.dt, self.hi)
#         return ac_weight


class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MyLogger(object, metaclass=SingletonType):
    _logger = None

    def __init__(self,filename,verbosity=1):
        self._logger = logging.getLogger("crumbs")
        self._logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s \t [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s')

        now = datetime.datetime.now()

        fileHandler = logging.FileHandler(filename,"a")

        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        self._logger.addHandler(fileHandler)
        self._logger.addHandler(streamHandler)

        print("Generate new instance")

    def get_logger(self):
        return self._logger

class DatasetTracks(Dataset):
    def __init__(self, X, config,sample_rate = 1.):
        self.config = config
        self.r_noise = config.noise_adding
        # self.gaussian_kernel = self.gaussian_filter1d(sigma=1.5)
        # self.gaussian_kernel = np.repeat(self.gaussian_kernel[:, np.newaxis], self.config.in_dim, axis=1)
        if isinstance(X, np.ndarray):
            # self.datasets = X
            samples = np.random.choice(X.shape[0], int(sample_rate * X.shape[0]), replace=False)
            samples = np.sort(samples)
            self.datasets = X[samples]
            self.label = 'from_workspace'


        elif isinstance(X, str):
            tmp = np.load(X)[:]
            samples = np.random.choice(tmp.shape[0],int(sample_rate * tmp.shape[0]),replace=False)
            samples = np.sort(samples)
            self.datasets = tmp[samples]
            # self.label = ('.').join(X.split("/")[-1].split(".")[:-1])  # filename
            self.label = ('.').join(X.split("_")[-1].split(".")[:-1])
        else:
            raise ValueError("Unsupported data type")


    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        tracks = self.datasets[idx]
        if bool(re.match(r'^[-+]?[0-9]*\.?[0-9]+$', self.label)):
            label = float(self.label)
        else:
            label = self.label

        # raw methods with additional tracks
        x = torch.from_numpy(tracks).type(dtype=torch.float)
        x = x - x[self.config.encoder_length - 1, :].reshape(1, self.config.in_dim)
        torch.manual_seed(idx)
        x += self.r_noise * torch.randn_like(x)

        return x, label

class DatasetTracksAlpha(Dataset):
    def __init__(self, X, config,sample_rate = 1.):
        self.config = config
        self.r_noise = config.noise_adding
        # self.gaussian_kernel = self.gaussian_filter1d(sigma=1.5)
        # self.gaussian_kernel = np.repeat(self.gaussian_kernel[:, np.newaxis], self.config.in_dim, axis=1)
        if isinstance(X, tuple):
            # self.datasets = X
            x = X[0]
            y = X[1]
            samples = np.random.choice(x.shape[0], int(sample_rate * x.shape[0]), replace=False)
            samples = np.sort(samples)
            self.datasets = x[samples]
            self.label = y[samples]


        elif isinstance(X, str):
            tmp = np.load(X)[:]
            samples = np.random.choice(tmp.shape[0],int(sample_rate * tmp.shape[0]),replace=False)
            samples = np.sort(samples)
            self.datasets = tmp[samples]
            self.label = ('.').join(X.split("_")[-1].split(".")[:-1])  # filename
        else:
            raise ValueError("Unsupported data type")


    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        tracks = self.datasets[idx]
        label = self.label

        # raw methods with additional tracks
        x = torch.from_numpy(tracks).type(dtype=torch.float)
        x = x - x[self.config.encoder_length - 1, :].reshape(1, self.config.in_dim)
        torch.manual_seed(idx)
        x += self.r_noise * torch.randn_like(x)

        return x, label

    # def gaussian_filter1d(self, sigma=3.):
    #     filter_range = np.linspace(-int(self.config.encoder_length / 2), int(self.config.encoder_length / 2), self.config.encoder_length)
    #     gaussian_filter = [1 / (sigma * np.sqrt(2 * np.pi))
    #                        * np.exp(-x ** 2 / (2 * sigma ** 2)) for x in filter_range]
    #     return np.asarray(gaussian_filter)


# class DatasetFilterTracks(Dataset):
#     def __init__(self,x,y1,y_shape,config):
#         self.config = config
#         self.y_shape = y_shape
#         self.raw = x
#         if isinstance(y1, str):
#             self.Y1 = np.memmap(y1,mode='r',dtype='float32',shape=y_shape)
#         else:
#             raise ValueError("Unsupported data type")
#
#
#     def __len__(self):
#         return self.y_shape[0]
#
#     def __getitem__(self, idx):
#         y1 = torch.from_numpy(np.asarray(self.Y1[idx]).copy()).type(dtype=torch.float)
#         x_raw = self.raw[idx][0]
#         # y1 concatenate x header, has delete mean val
#         x_input = torch.concatenate([x_raw[:self.config.encoder_length],y1],dim=0)
#         # b = x_raw[self.config.encoder_length:]
#         return x_input,x_raw[self.config.encoder_length:]

class DatasetLargeSigma(Dataset):
    def __init__(self,S,S0,s_shape,config):
        self.s_shape = s_shape
        self.config = config

        if isinstance(S, str):
            self.S = np.memmap(S,mode='r',dtype='float32',shape=s_shape)
            self.S0 = np.memmap(S0, mode='r',dtype='float32',shape=s_shape)
        else:
            raise ValueError("Unsupported data type")



    def __len__(self):
        return self.s_shape[0]

    def __getitem__(self, idx):
        s1 = np.asarray(self.S[idx]).copy()
        data0 = np.asarray(self.S0[idx]).copy()

        # s1 = torch.from_numpy(self.S[idx]).type(dtype=torch.float)
        # data0 = torch.from_numpy(self.S0[idx]).type(dtype=torch.float)
        s1 = torch.from_numpy(s1).type(dtype=torch.float)
        data0 = torch.from_numpy(data0).type(dtype=torch.float)
        s0 = torch.zeros(self.config.encoder_length, s1.shape[1])
        minlen_sz = min([self.config.encoder_length, s1.shape[0]])
        s0[:minlen_sz] = torch.flip(data0,dims=[0])[:minlen_sz]
        x = torch.concatenate([s0, s1], dim=0)**2
        return x,-1




class DatasetLargeVelocity(Dataset):
    def __init__(self,V,V0,v_shape,config):
        self.v_shape = v_shape
        self.config = config

        if isinstance(V, str):
            self.V = np.memmap(V,mode='r',dtype='float32',shape=v_shape)
            self.V0 = np.memmap(V0, mode='r',dtype='float32',shape=v_shape)
        else:
            raise ValueError("Unsupported data type")



    def __len__(self):
        return self.v_shape[0]

    def __getitem__(self, idx):
        v1 = np.asarray(self.V[idx]).copy()
        data0 = np.asarray(self.V0[idx]).copy()

        # s1 = torch.from_numpy(self.S[idx]).type(dtype=torch.float)
        # data0 = torch.from_numpy(self.S0[idx]).type(dtype=torch.float)
        v1 = torch.from_numpy(v1).type(dtype=torch.float)
        data0 = torch.from_numpy(data0).type(dtype=torch.float)
        v0 = torch.zeros(self.config.encoder_length, v1.shape[1])
        minlen_sz = min([self.config.encoder_length, v1.shape[0]])
        v0[:minlen_sz] = torch.flip(data0,dims=[0])[:minlen_sz]
        x = torch.concatenate([v0, v1], dim=0)
        return x,-1


class DatasetLargeSigmaPredict(Dataset):
    def __init__(self,S,s_shape):
        self.s_shape = s_shape

        if isinstance(S, str):
            self.S = np.memmap(S,mode='r',dtype='float32',shape=s_shape)
        else:
            raise ValueError("Unsupported data type")



    def __len__(self):
        return self.s_shape[0]

    def __getitem__(self, idx):
        s = np.asarray(self.S[idx]).copy()
        x = torch.sqrt(torch.abs(torch.from_numpy(s).type(dtype=torch.float))+1e-6)
        return x,-1

class DatasetFilterLargeSigma(Dataset):
    def __init__(self,Y0,Y1,Y0_reverse,Y1_reverse,s_shape,config):
        self.s_shape = s_shape
        self.config = config

        if isinstance(Y0, str) and isinstance(Y1, str) and isinstance(Y0_reverse, str) and isinstance(Y1_reverse, str):
            self.Y0 = np.memmap(Y0,mode='r',dtype='float32',shape=s_shape)
            self.Y0_reverse = np.memmap(Y0_reverse, mode='r',dtype='float32',shape=s_shape)
            self.Y1= np.memmap(Y1, mode='r', dtype='float32', shape=s_shape)
            self.Y1_reverse = np.memmap(Y1_reverse, mode='r', dtype='float32', shape=s_shape)
        else:
            raise ValueError("Unsupported data type")



    def __len__(self):
        return self.s_shape[0]

    def __getitem__(self, idx):
        y1 = np.asarray(self.Y1[idx]).copy()
        y0 = np.asarray(self.Y0[idx]).copy()
        s1 = y0[config.encoder_length + 1:] - y1[:-1]
        s1 = torch.from_numpy(s1).type(dtype=torch.float)
        s1 = torch.concatenate([s1, torch.zeros((1, s1.shape[1]))], dim=0) # L x D

        y1_reverse = np.asarray(self.Y1_reverse[idx]).copy()
        y0_reverse = np.asarray(self.Y0_reverse[idx]).copy()
        s0_reverse = y0_reverse[config.encoder_length + 1:] - y1_reverse[:-1]
        data0 = torch.from_numpy(s0_reverse).type(dtype=torch.float)
        data0 = torch.concatenate([data0, torch.zeros((1, s1.shape[1]))], dim=0)  # L x D


        s0 = torch.zeros(self.config.encoder_length, s1.shape[1])
        minlen_sz = min([self.config.encoder_length, s1.shape[0]])
        s0[:minlen_sz] = torch.flip(data0,dims=[0])[:minlen_sz]
        x = torch.concatenate([s0, s1], dim=0)**2
        return x,-1

## merge dataloader
def merge_datasets(config, filepath = '',filenames = ['annotation.txt'],sample_rate = 1.):
    dataloaders = []
    for filename in filenames:
        df = pd.read_csv(os.path.join(filepath,filename),delimiter=',',header=None)
        # (filelocation,)
        for i in range(df.shape[0]):
            dataloaders.append(DatasetTracks(df.iloc[i,0],config,sample_rate))
    return ConcatDataset(dataloaders)

def generate_dataset_torch(full_dataset,train_prob = 0.75, seed = 0):
    train_sz = int(train_prob * len(full_dataset))
    test_sz = len(full_dataset) - train_sz
    torch.manual_seed(seed=seed)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_sz, test_sz])
    return train_dataset, test_dataset

def get_datasets_shape(datasets,config):
    dataloader = DataLoader(dataset=datasets,
                            batch_size=config.batch_size,
                            shuffle=False)  # Don't sh
    for t, X in enumerate(dataloader):
        L,_,D = X[0].permute(1, 0, 2).shape
        L -= config.encoder_length
        if t >= 1:
            break

    return L,len(datasets),D

def save(model,dload,file_name):
    PATH = dload + '/' + file_name
    if os.path.exists(dload):
        pass
    else:
        os.mkdir(dload)
    torch.save(model.state_dict(), PATH)

def save_checkpoint(ck,dload,file_name):
    ckp_PATH = dload + '/' + file_name
    PATH = ckp_PATH + '/checkpoint_ckpt.pth'
    if os.path.exists(ckp_PATH):
        pass
    else:
        os.mkdir(ckp_PATH)
    torch.save(ck, PATH)

def load(model,dload,file_name):
    PATH = dload + '/' + file_name
    model.load_state_dict(torch.load(PATH))


def generate_data_sets(X, config):
    # def my_func(x):
    #     t = np.linspace(0, 1, config.encoder_length)
    #     ys = csaps(t, x, t, smooth=0.98)
    #     return ys

    # lp_smooth = np.apply_along_axis(my_func, 1, X[:, :config.encoder_length, :])
    # data = X - np.mean(X[:, :config.encoder_length, :], axis=1).reshape(-1, 1, config.in_dim)
    data = X - X[:, config.encoder_length-1, :].reshape(-1, 1, config.in_dim)
    # data = X - lp_smooth[:, -1, :].reshape(-1, 1, config.in_dim)
    X_train, X_val, y_train, y_val = train_test_split(data, np.zeros_like(data),
                                                      test_size=0.3, random_state=config.seed)
    return X_train, X_val, data

def generate_data_sets_support(X, config, S):
    # def my_func(x):
    #     t = np.linspace(0, 1, config.encoder_length)
    #     ys = csaps(t, x, t, smooth=0.98)
    #     return ys

    # lp_smooth = np.apply_along_axis(my_func, 1, X[:, :config.encoder_length, :])
    # data = X - np.mean(X[:, :config.encoder_length, :], axis=1).reshape(-1, 1, config.in_dim)
    # supporting Test 0.02
    data = X - X[:, config.encoder_length-1, :].reshape(-1, 1, config.in_dim)
    data[:, config.encoder_length-1, :] = S
    # data = X - lp_smooth[:, -1, :].reshape(-1, 1, config.in_dim)
    X_train, X_val, y_train, y_val = train_test_split(data, np.zeros_like(data),
                                                      test_size=0.4, random_state=config.seed)
    return X_train, X_val, data

def train(train_datasets, config, test_datasets = None,L2=None, shuffle = True):
    #data: BxLxD
    filename_set = config.save_name
    model_name = config.model_name
    logger_name = config.dload + '/'+ config.save_name + '.log'
    model = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                h_dim=config.h_dim, default_enc_nn = config.default_enc_nn,
                                hidden_dim=config.hidden_size, layers=config.layers,
                                d_model=config.d_model, n_head=config.n_heads, layers_enc = config.enc_layers,
                                dropout=config.dropout,
                                beta_obs = config.beta_observe, beta_ac = config.beta_ac, beta_conti = config.beta_conti,
                                beta_weight=config.beta_weight,
                                device = config.device).to(config.device)
    # print(list(model.parameters()))
    # optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.)
    # optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-2)
    base_optimizer = torch.optim.AdamW  # define an optimizer for the "sharpness-aware" update
    # optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay=1e-2)
    # base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=config.batch_size,
                              shuffle=shuffle)

    L,_,D = get_datasets_shape(train_datasets,config=config)

    loss_fn = adaptive.AdaptiveLossFunction(num_dims=D * (L - 1), float_dtype=np.float32, device=config.device)
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = SAM(params, base_optimizer, lr=config.lr, weight_decay=5e-2)
    # optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay = 1e-2 )
    lr_scheduler = CustomSchedule(config.d_model, optimizer=optimizer, warmup_steps= config.warmup_steps, scale= config.scale)
    # ac_scheduler = AutocorrelationSchedule(lo=config.beta_ac_min,
    #                                             hi = config.beta_ac,
    #                                             warmup_steps = max(int(0.06*config.epochs) * len(train_loader),
    #                                                                config.warmup_steps))
    # train_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))


    logger = MyLogger.__call__(logger_name).get_logger()
    logger.info('starting training with config:')
    logger.info(config)
    epochs = config.epochs
    start_epoch = -1

    path_checkpoint = config.dload + '/' + config.save_name + '/checkpoint_ckpt.pth'
    if config.RESUME and os.path.exists(path_checkpoint):
        # path_checkpoint = config.dload + '/' + config.save_name + '/checkpoint_ckpt.pth'
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        loss_fn.load_state_dict(checkpoint['loss_fn'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.steps = (start_epoch+1) * len(train_loader)
        # ac_scheduler.steps = (start_epoch+1) * len(train_loader)

    for i in range(start_epoch+1,epochs):
        model.train()
        epoch_loss = 0
        epoch_pred = 0
        epoch_rec = 0
        epoch_ac = 0
        for t, X in enumerate(train_loader):
            x = X[0].permute(1, 0, 2).to(config.device)
            # optimizer.zero_grad()
            y0, y1= model(x)

            loss, pre, rec, ac = model.cal_loss(y0, y1, x[config.encoder_length:],
                                                loss_fn=loss_fn, L2=L2)
            epoch_loss += loss.item()
            epoch_pred += pre.item()
            epoch_rec += rec.item()
            epoch_ac += ac.item()

            loss.backward()
            optimizer.first_step(zero_grad=True)

            y02, y12 = model(x)
            model.cal_loss(y02, y12, x[config.encoder_length:],
                           loss_fn=loss_fn, L2=L2)[0].backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            lr_scheduler.step()
            # model.beta_ac = ac_scheduler.step()
            # optimizer.step()

            if (t+1) % 1000 == 0:
                print('epoch {}: {:d}/{:d}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
                  .format(i,t, len(train_loader),epoch_loss / (t+1), epoch_pred / (t+1), epoch_rec / (t+1), epoch_ac / (t+1)))

        print('learning rates: {:4f}, auto correlation weight: {:4f}, steps: {:1f}'
                      .format(lr_scheduler.get_lr(), model.beta_ac, lr_scheduler.steps))
        logger.info('epoch {}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
              .format(i, epoch_loss / len(train_loader), epoch_pred / len(train_loader), epoch_rec / len(train_loader),epoch_ac / len(train_loader)))


        if (i+1) % 10 == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": i,
                'loss_fn': loss_fn.state_dict(),
            }
            if test_datasets is not None:
                logger.info(evaluate(test_datasets,model,config))
            save_checkpoint(checkpoint, config.dload, config.save_name)
            print('checkpoint {} saved'.format(i))



    save(model, config.dload, model_name)
    return model

def train_filter(train_datasets, config, test_datasets = None):
    #data: BxLxD
    filename_set = config.save_name
    model_name = config.model_name
    logger_name = config.dload + '/'+ config.save_name + '.log'
    model = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                h_dim=config.h_dim, default_enc_nn = config.default_enc_nn,
                                hidden_dim=config.hidden_size, layers=config.layers,
                                d_model=config.d_model, n_head=config.n_heads, layers_enc = config.enc_layers,
                                dropout=config.dropout,
                                beta_obs = config.beta_observe, beta_ac = config.beta_ac, beta_conti = config.beta_conti,
                                beta_weight=config.beta_weight,
                                device = config.device).to(config.device)
    # print(list(model.parameters()))
    # optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.)
    # optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-2)
    base_optimizer = torch.optim.AdamW  # define an optimizer for the "sharpness-aware" update
    # optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay=1e-2)
    # base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
    optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay = 5e-2 )
    lr_scheduler = CustomSchedule(config.d_model, optimizer=optimizer, warmup_steps= config.warmup_steps, scale= config.scale)

    # train_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))
    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=config.batch_size,
                              shuffle=False)

    logger = MyLogger.__call__(logger_name).get_logger()
    logger.info('starting training with config:')
    logger.info(config)
    epochs = config.epochs
    start_epoch = -1

    path_checkpoint = config.dload + '/' + config.save_name + '/checkpoint_ckpt.pth'
    if config.RESUME and os.path.exists(path_checkpoint):
        # path_checkpoint = config.dload + '/' + config.save_name + '/checkpoint_ckpt.pth'
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.steps = (start_epoch+1) * len(train_loader)


    for i in range(start_epoch+1,epochs):
        model.train()
        epoch_loss = 0
        epoch_pred = 0
        epoch_rec = 0
        epoch_ac = 0
        for t, X in enumerate(train_loader):
            x = X[0].permute(1, 0, 2).to(config.device)
            y_val = X[1].permute(1, 0, 2).to(config.device)
            # optimizer.zero_grad()
            y0, y1= model(x)

            loss, pre, rec, ac = model.cal_loss_filter(y0, y1,y_val)
            epoch_loss += loss.item()
            epoch_pred += pre.item()
            epoch_rec += rec.item()
            epoch_ac += ac.item()

            loss.backward()
            optimizer.first_step(zero_grad=True)

            y02, y12 = model(x)
            model.cal_loss(y02, y12, x[config.encoder_length:])[0].backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            lr_scheduler.step()
            # optimizer.step()

            if (t+1) % 1000 == 0:
                print('epoch {}: {:d}/{:d}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
                  .format(i,t, len(train_loader),epoch_loss / (t+1), epoch_pred / (t+1), epoch_rec / (t+1), epoch_ac / (t+1)))

        print('learning rates: {:4f}, steps: {:1f}'
                      .format(lr_scheduler.get_lr(), lr_scheduler.steps))
        logger.info('epoch {}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
              .format(i, epoch_loss / len(train_loader), epoch_pred / len(train_loader), epoch_rec / len(train_loader),epoch_ac / len(train_loader)))


        if (i+1) % 10 == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": i
            }
            if test_datasets is not None:
                logger.info(evaluate(test_datasets,model,config))
            save_checkpoint(checkpoint, config.dload, config.save_name)
            print('checkpoint {} saved'.format(i))



    save(model, config.dload, model_name)
    return model

def train_v(X, Y, config):
    #data: BxLxD
    filename_set = config.save_name
    model_name = config.model_name
    logger_name = config.dload + '/'+ config.save_name + '.log'
    model = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                h_dim=config.h_dim, default_enc_nn = config.default_enc_nn,
                                hidden_dim=config.hidden_size, layers=config.layers,
                                d_model=config.d_model, n_head=config.n_heads, layers_enc = config.enc_layers, 
                                beta_obs = config.beta_observe, beta_ac = config.beta_ac,  beta_conti = config.beta_conti,
                                beta_weight=config.beta_weight,
                                device = config.device).to(config.device)
    # print(list(model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = CustomSchedule(config.d_model, optimizer=optimizer, warmup_steps= config.warmup_steps, scale= config.scale)


    train_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float), torch.from_numpy(Y).type(dtype=torch.float))
    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=config.batch_size,
                              shuffle=True)

    logger = MyLogger.__call__(logger_name).get_logger()
    logger.info('starting training')
    epochs = config.epochs

    for i in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_pred = 0
        epoch_rec = 0
        epoch_ac = 0
        for t, X in enumerate(train_loader):
            x = X[0].permute(1, 0, 2).to(config.device)
            y = X[1].permute(1, 0, 2).to(config.device)
            optimizer.zero_grad()
            y0,y1 = model(x)
            loss, pre, rec, ac = model.cal_loss(y0, y1, y)

            epoch_loss += loss.item()
            epoch_pred += pre.item()
            epoch_rec += rec.item()
            epoch_ac += ac.item()
            loss.backward()

            lr_scheduler.step()
            optimizer.step()

            if (t+1) % 200 == 0:
                print('epoch {}: {:d}/{:d}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
                  .format(i,t, len(train_loader),epoch_loss, epoch_pred, epoch_rec, epoch_ac))

        print('learning rates: {:4f}, steps: {:1f}'
                      .format(lr_scheduler.get_lr(), lr_scheduler.steps))
        logger.info('epoch {}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
              .format(i, epoch_loss, epoch_pred, epoch_rec,epoch_ac))

    save(model, config.dload, model_name)
    return model


def train_Dt(train_datasets, config, test_datasets = None,L2=None, shuffle = True):
    #data: BxLxD
    filename_set = config.save_name
    model_name = config.model_name
    logger_name = config.dload + '/'+ config.save_name + '.log'
    model = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                h_dim=config.h_dim, default_enc_nn = config.default_enc_nn,
                                hidden_dim=config.hidden_size, layers=config.layers,
                                d_model=config.d_model, n_head=config.n_heads, layers_enc = config.enc_layers,
                                dropout=config.dropout,
                                beta_obs = config.beta_observe, beta_ac = config.beta_ac, beta_conti = config.beta_conti,
                                beta_weight=config.beta_weight,
                                device = config.device).to(config.device)
    # print(list(model.parameters()))
    # optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.)
    # optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-2)
    base_optimizer = torch.optim.AdamW  # define an optimizer for the "sharpness-aware" update
    # optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay=1e-2)
    # base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=config.batch_size,
                              shuffle=shuffle)

    L,_,D = get_datasets_shape(train_datasets,config=config)

    loss_fn = adaptive.AdaptiveLossFunction(num_dims=D * (L - 1), float_dtype=np.float32, device=config.device)
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = SAM(params, base_optimizer, lr=config.lr, weight_decay=5e-2)
    # optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay = 1e-2 )
    lr_scheduler = CustomSchedule(config.d_model, optimizer=optimizer, warmup_steps= config.warmup_steps, scale= config.scale)
    # ac_scheduler = AutocorrelationSchedule(lo=config.beta_ac_min,
    #                                             hi = config.beta_ac,
    #                                             warmup_steps = max(int(0.06*config.epochs) * len(train_loader),
    #                                                                config.warmup_steps))
    # train_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))


    logger = MyLogger.__call__(logger_name).get_logger()
    logger.info('starting Dt training with config, beta NLL Gaussian loss:')
    logger.info(config)
    epochs = config.epochs
    start_epoch = -1

    path_checkpoint = config.dload + '/' + config.save_name + '/checkpoint_ckpt.pth'
    if config.RESUME and os.path.exists(path_checkpoint):
        # path_checkpoint = config.dload + '/' + config.save_name + '/checkpoint_ckpt.pth'
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        loss_fn.load_state_dict(checkpoint['loss_fn'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.steps = (start_epoch+1) * len(train_loader)
        # ac_scheduler.steps = (start_epoch+1) * len(train_loader)

    for i in range(start_epoch+1,epochs):
        model.train()
        epoch_loss = 0
        epoch_pred = 0
        epoch_rec = 0
        epoch_ac = 0
        for t, X in enumerate(train_loader):
            x = X[0].permute(1, 0, 2).to(config.device)
            # optimizer.zero_grad()
            y0, y1= model(x)

            # loss, pre, rec, ac = model.cal_loss(y0, y1, x[config.encoder_length:],
            #                                     loss_fn=loss_fn, L2=L2)

            loss, pre, rec, ac = model.cal_loss_sigma(y0, y1, x[config.encoder_length:],
                                                      )

            epoch_loss += loss.item()
            epoch_pred += pre.item()
            epoch_rec += rec.item()
            epoch_ac += ac.item()

            loss.backward()
            optimizer.first_step(zero_grad=True)

            y02, y12 = model(x)
            # model.cal_loss(y02, y12, x[config.encoder_length:],
            #                loss_fn=loss_fn, L2=L2)[0].backward()  # make sure to do a full forward pass

            model.cal_loss_sigma(y02, y12, x[config.encoder_length:],
                                 )[0].backward() # make sure to do a full forward pass

            optimizer.second_step(zero_grad=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            lr_scheduler.step()
            # model.beta_ac = ac_scheduler.step()
            # optimizer.step()

            if (t+1) % 1000 == 0:
                print('epoch {}: {:d}/{:d}: loss {:4f}, pred: {:4f}, recon: {:4f}, NLLGaussian: {:4f}'
                  .format(i,t, len(train_loader),epoch_loss / (t+1), epoch_pred / (t+1), epoch_rec / (t+1), epoch_ac / (t+1)))

        print('learning rates: {:4f}, auto correlation weight: {:4f}, steps: {:1f}'
                      .format(lr_scheduler.get_lr(), model.beta_ac, lr_scheduler.steps))
        logger.info('epoch {}: loss {:4f}, pred: {:4f}, recon: {:4f}, NLLGaussian: {:4f}'
              .format(i, epoch_loss / len(train_loader), epoch_pred / len(train_loader), epoch_rec / len(train_loader),epoch_ac / len(train_loader)))


        if (i+1) % 10 == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": i,
                'loss_fn': loss_fn.state_dict(),
            }
            if test_datasets is not None:
                logger.info(evaluate(test_datasets,model,config))
            save_checkpoint(checkpoint, config.dload, config.save_name)
            print('checkpoint {} saved'.format(i))



    save(model, config.dload, model_name)
    return model


def evaluate_draw(X,model,config):
    if isinstance(X,np.ndarray):
        train_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))
    else:
        train_datasets = X

    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=config.batch_size,
                              shuffle=False)

    for t, X in enumerate(train_loader):
        if t > min(config.sample_nums-1,len(train_loader)-2): # drop_last batch
            break
        x = X[0].permute(1, 0, 2).to(config.device)
        _, output = model(x)
        x = x[config.encoder_length:]
        plt.plot(x[1:, 1, 0].data.cpu().numpy())
        plt.plot(output[:, 1, 0].data.cpu().numpy())
        plt.show()
        ##v##
        # plt.plot(output[:-1, 1, 0].data.cpu().numpy() - x[:-1, 1, 0].data.cpu().numpy())
        plt.plot(x[1:, 1, 0].data.cpu().numpy() - x[:-1, 1, 0].data.cpu().numpy())
        plt.plot(output[:-1, 1, 0].data.cpu().numpy() - x[:-1, 1, 0].data.cpu().numpy())
        plt.show()
        ##v~sigma##
        # v = output[1:, 1, 0].data.cpu().numpy() - x[1:, 1, 0].data.cpu().numpy()
        # sig = x[1:, 1, 0].data.cpu().numpy() - output[:-1, 1, 0].data.cpu().numpy()
        # plt.plot(v, sig)
        # # plt.plot(x[1:, 1, 0].data.cpu().numpy() - x[:-1, 1, 0].data.cpu().numpy())
        # plt.show()

    # sigma = np.mean(x[1:,:,0].data.cpu().numpy()
    #                 -output[:-1,:,0].data.cpu().numpy(),axis=1)
    #     a = model.cal_autocorrelation(x[1:, :, :]-output[:-1, :, :]).data.cpu().numpy()

        # raw
        sigma_raw = (x[1:, :, 0].data.cpu().numpy() - x[:-1, :, 0].data.cpu().numpy())
        ac = []
        for i in range(sigma_raw.shape[1]):
            ac.append(np.correlate(sigma_raw[:, i], sigma_raw[:, i], mode='full'))
        # ac_raw = np.correlate(sigma, sigma, mode='full')
        ac_raw = np.mean(np.array(ac), axis=0)
        print(ac_raw[ac_raw.size // 2:ac_raw.size // 2 + 5])
        plt.plot(ac_raw[ac_raw.size // 2:ac_raw.size // 2 + 50])

        # velocity
        v_raw = (output[:-1, :, 0].data.cpu().numpy() - x[:-1, :, 0].data.cpu().numpy())
        ac = []
        for i in range(v_raw.shape[1]):
            ac.append(np.correlate(v_raw[:, i], v_raw[:, i], mode='full'))
        # ac_raw = np.correlate(sigma, sigma, mode='full')
        ac_raw = np.mean(np.array(ac), axis=0)
        print(ac_raw[ac_raw.size // 2:ac_raw.size // 2 + 5])
        plt.plot(ac_raw[ac_raw.size // 2:ac_raw.size // 2 + 50])

        # Dt
        sigma = (x[1:, :, 0].data.cpu().numpy() - output[:-1, :, 0].data.cpu().numpy())
        # v = (output[:-1,:,0].data.cpu().numpy()-x[:-1,:,0].data.cpu().numpy())
        ac = []
        for i in range(sigma.shape[1]):
            ac.append(np.correlate(sigma[:, i], sigma[:, i], mode='full'))
        # ac_raw = np.correlate(sigma, sigma, mode='full')
        ac_raw = np.mean(np.array(ac), axis=0)
        print(ac_raw[ac_raw.size // 2:ac_raw.size // 2 + 5])
        plt.plot(ac_raw[ac_raw.size // 2:ac_raw.size // 2 + 50])

        plt.show()

def evaluate_draw_tracks(X,model,config, is_gt = None):
    if isinstance(X,np.ndarray):
        train_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))
    else:
        train_datasets = X

    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=config.batch_size,
                              shuffle=False)

    for t, X in enumerate(train_loader):
        if t > config.sample_nums:
            break
        x = X[0].permute(1, 0, 2).to(config.device)

        y0, y1 = model(x)
        x = x[config.encoder_length:]
        plt.plot(x[1:, 1, 0].data.cpu().numpy())
        plt.plot(y0[:-1, 1, 0].data.cpu().numpy())
        plt.plot(y1[:-1, 1, 0].data.cpu().numpy())
        if is_gt is not None:
            y_raw = X[1].permute(1, 0, 2).to(config.device)
            plt.plot(y_raw[:-1, 1, 0].data.cpu().numpy())
        plt.show()

def evaluate(test_datasets,model,config):
    # test_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))
    test_loader = DataLoader(dataset=test_datasets,
                              batch_size=config.batch_size,
                              shuffle=True)
    for i in range(1):
        model.eval()
        epoch_loss = 0
        epoch_pred = 0
        epoch_rec = 0
        epoch_ac = 0
        for t, X in enumerate(test_loader):

            x = X[0].permute(1, 0, 2).to(config.device)
            y0, y1 = model(x)
            loss, pre, rec, ac = model.cal_loss(y0, y1, x[config.encoder_length:])

            epoch_loss += loss.item()
            epoch_pred += pre.item()
            epoch_rec += rec.item()
            epoch_ac += ac.item()


            if (t + 1) % 1000 == 0:
                print('evaluation {}: {:d}/{:d}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
                      .format(i, t, len(test_loader), epoch_loss / (t + 1), epoch_pred / (t + 1), epoch_rec / (t + 1),
                              epoch_ac / (t + 1)))

        # print('evaluation {}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
        #             .format(i, epoch_loss / len(test_loader), epoch_pred / len(test_loader),
        #                     epoch_rec / len(test_loader), epoch_ac / len(test_loader)))
        eval_acc = ('evaluation {}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
                    .format(i, epoch_loss / len(test_loader), epoch_pred / len(test_loader),
                            epoch_rec / len(test_loader), epoch_ac / len(test_loader)))
        return eval_acc

def batch_reconstruct(x, model,config):
    x = Variable(x.type(torch.float), requires_grad=False)
    with torch.no_grad():
        y0, y1 = model(x)
    sigma = x[config.encoder_length+1:] - y1[:-1]
    sigma = sigma.cpu().data.numpy()
    s = np.concatenate([sigma, np.zeros((1, sigma.shape[1], sigma.shape[2]))], axis=0)
    return y0.cpu().data.numpy(),y1.cpu().data.numpy(), s


def transform(test_datasets,model,config):
    '''
    Given the ordered input, return y0, y1, sigma L x B x D
    '''
    model.eval()
    # test_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))
    test_loader = DataLoader(dataset=test_datasets,
                              batch_size=config.batch_size,
                              shuffle=False) # Don't shuffle for test_loader

    x_run, y0_run,y1_run,sigma_run = [],[],[],[]
    for t,X in enumerate(test_loader):
        x = X[0].permute(1, 0, 2).to(config.device)
        y0, y1, sigma = batch_reconstruct(x,model,config)
        x_run.append(x.cpu().data.numpy())
        y0_run.append(y0)
        y1_run.append(y1)
        sigma_run.append(sigma)


    return np.concatenate(y0_run, axis=1), np.concatenate(y1_run, axis=1), np.concatenate(sigma_run, axis=1), np.concatenate(x_run, axis=1)

def getlabel(test_datasets,config):
    '''
    Given the ordered input, return y0, y1, sigma L x B x D
    '''
    # test_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))
    test_loader = DataLoader(dataset=test_datasets,
                              batch_size=config.batch_size,
                              shuffle=False) # Don't shuffle for test_loader

    label_run = []
    for t,X in enumerate(test_loader):
        label_run.append(X[1].cpu().data.numpy())

    return np.concatenate(label_run, axis=0)

def transformToMemmap(datasets,model,config,tags_s='',is_train = True, model_clash_pos = None):
    '''
    Given the ordered input, return y0, y1, sigma L x B x D
    '''
    model.eval()
    dataloader = DataLoader(dataset=datasets,
                              batch_size=config.batch_size,
                              shuffle=False) # Don't shuffle for test_loader
    L,_,D = get_datasets_shape(datasets,config)

    if model_clash_pos is not None:
        model_clash_pos = model_clash_pos
    else:
        model_clash_pos = config.save_name

    if len(tags_s)>0 :
        tags_s = '_' + tags_s

    fp_y0_run_path = os.path.join(config.dload, model_clash_pos, 'memmap_clash', 'y0_run'+tags_s+'.npy')
    fp_y1_run_path = os.path.join(config.dload, model_clash_pos, 'memmap_clash', 'y1_run'+ tags_s + '.npy')
    fp_sigma_run_path = os.path.join(config.dload, model_clash_pos, 'memmap_clash', 'sigma_run' + tags_s+ '.npy')
    # fp_x_run_path = os.path.join(config.dload, config.save_name, 'memmap_clash', 'x_run' + tags_s + '.npy')
    #
    if is_train:
        if os.path.exists(os.path.join(config.dload, model_clash_pos, 'memmap_clash')):
            pass
        else:
            os.makedirs(os.path.join(config.dload, model_clash_pos, 'memmap_clash'))
        #
        fp_y0_run = np.memmap(fp_y0_run_path, dtype='float32',mode='w+',shape= (len(datasets),L, D) )
        fp_y1_run = np.memmap(fp_y1_run_path, dtype='float32', mode='w+', shape=(len(datasets) ,L, D))

        fp_sigma_run = np.memmap(fp_sigma_run_path, dtype='float32', mode='w+', shape=(len(datasets) ,L,  D))
        # fp_x_run = np.memmap(fp_x_run_path, dtype='float32', mode='w+', shape=(len(datasets),L + config.encoder_length, D))
        # np.save(shape_run_path,np.array([L,len(datasets),D]))

        cnt = 0

        for t,X in enumerate(dataloader):
            x = X[0].permute(1, 0, 2).to(config.device) # L x B x D
            if tags_s == 'reverse':
                x = torch.flip(x,dims = [0])

            cnt_add = x.shape[1]
            y0, y1, sigma = batch_reconstruct(x,model,config)
            # fp_x_run[cnt:cnt + cnt_add] = x.cpu().data.numpy().transpose(1, 0, 2)
            fp_y0_run[cnt:cnt+cnt_add] = y0.transpose(1, 0, 2)
            fp_y1_run[cnt:cnt+cnt_add] = y1.transpose(1, 0, 2)
            fp_sigma_run[cnt:cnt+cnt_add] = sigma.transpose(1, 0, 2)
            cnt += cnt_add


    # del fp_y0_run, fp_y1_run, fp_sigma_run
    return fp_y0_run_path, fp_y1_run_path, fp_sigma_run_path

def transformVelocityToMemmap(datasets,model,config,tags_s='',is_train = True):
    '''
    Given the ordered input, return y0, y1, sigma L x B x D
    '''
    model.eval()
    dataloader = DataLoader(dataset=datasets,
                              batch_size=config.batch_size,
                              shuffle=False) # Don't shuffle for test_loader
    L,_,D = get_datasets_shape(datasets,config)
    if len(tags_s)>0 :
        tags_s = '_' + tags_s

    fp_velocity_run_path = os.path.join(config.dload, config.save_name, 'memmap_clash', 'v_run' + tags_s+ '.npy')
    #
    if is_train:
        if os.path.exists(os.path.join(config.dload, config.save_name, 'memmap_clash')):
            pass
        else:
            os.makedirs(os.path.join(config.dload, config.save_name, 'memmap_clash'))
        #

        fp_velocity_run = np.memmap(fp_velocity_run_path, dtype='float32', mode='w+', shape=(len(datasets) ,L,  D))


        cnt = 0

        for t,X in enumerate(dataloader):
            x = X[0].permute(1, 0, 2).to(config.device) # L x B x D
            if tags_s == 'reverse':
                x = -torch.flip(x,dims = [0])

            cnt_add = x.shape[1] # batch
            _, y1, _ = batch_reconstruct(x,model,config)
            v = y1 - x[config.encoder_length:].cpu().data.numpy()
            # fp_x_run[cnt:cnt + cnt_add] = x.cpu().data.numpy().transpose(1, 0, 2)
            fp_velocity_run[cnt:cnt+cnt_add] = v.transpose(1, 0, 2)

            cnt += cnt_add

    return fp_velocity_run_path

def batch_raw_gen(h,s,my_gen_model, adding_jumping = None):
    my_gen_model.eval()
    with torch.no_grad():
        y0, y1, x0 = my_gen_model(h, s, adding_jumping)
      
    return y0.cpu().data.numpy(), y1.cpu().data.numpy(), x0.cpu().data.numpy()

def generate_sigma_sets(S0, S, config):
    
    data = S #reject_outliers(S,m=5)
    # sc = np.std(data, axis=1)[:,:,np.newaxis].repeat(config.encoder_length,axis=2).transpose(0,2,1)
    # data = np.concatenate([sc * np.random.randn(data.shape[0],config.encoder_length, data.shape[2]), data], axis=1)
    data0 = np.zeros((data.shape[0],config.encoder_length, data.shape[2]))
    ## left alignment
    minlen_sz = min([config.encoder_length, S0.shape[1]])
    data0[:,-minlen_sz:,:] = S0[:,-minlen_sz:,:]
    # data = data **2
    data = np.concatenate([data0, data], axis=1)
    data = data **2
    X_train, X_val, y_train, y_val = train_test_split(data, np.zeros_like(data),
                                                      test_size=0.25, random_state=config.seed)
    return X_train, X_val, data

def generate_sigma_sets_torch(train_dataset, model,config, is_train = True):
    L,B,D = get_datasets_shape(train_dataset, config=config)
    # if is_train:
    # else:
    y0_run_path, y1_run_path, sigma_run_path = transformToMemmap(train_dataset, model, config,
                                                                 is_train= is_train) # padding the header, also X
    y0_run_path_reverse, y1_run_path_reverse, sigma_run_path_reverse = transformToMemmap(train_dataset, model, config,
                                                                                         tags_s='reverse',is_train= is_train)  # padding the header, also X
    s = DatasetLargeSigma(S = sigma_run_path,
                           S0 = sigma_run_path_reverse,
                          s_shape = (B,L,D), config=config
                           )

    # if is_filtering is not None:
    #     s_filter = DatasetFilterLargeSigma(Y0=y0_run_path, Y0_reverse=y0_run_path_reverse,
    #                           Y1=y1_run_path, Y1_reverse= y1_run_path_reverse,
    #                           s_shape=(B, L, D), config=config
    #                           )
    #     full_dataset = ConcatDataset[s,s_filter]
    #     train_sz = int(0.5*len(full_dataset))
    #     test_sz = len(full_dataset)-train_sz
    #     s, _ = torch.utils.data.random_split(full_dataset, [train_sz, test_sz])

    return s

def generate_velocity_sets_torch(train_dataset, model,config, is_train = True):
    L,B,D = get_datasets_shape(train_dataset, config=config)
    velocity_run_path = transformVelocityToMemmap(train_dataset, model, config,
                                                                 is_train= is_train) # padding the header, also X
    velocity_run_path_reverse = transformVelocityToMemmap(train_dataset, model, config,
                                                        tags_s='reverse',is_train= is_train)  # padding the header, also X
    v = DatasetLargeVelocity(V = velocity_run_path,
                           V0 = velocity_run_path_reverse,
                          v_shape = (B,L,D), config=config
                           )

    return v

def generate_sigma_predicted_torch(gen_dataset,model,config):
    L,B,D = get_datasets_shape(gen_dataset, config=config)

    _,sigma_predict_path,_ = transformToMemmap(gen_dataset,
                                           model,
                                           config,
                                           is_train= True,
                                           model_clash_pos=config.save_name + '-gen') # reading the saved path


    s = DatasetLargeSigmaPredict(S = sigma_predict_path,s_shape = (B,L,D))

    return s


def eval_sigma_sets_torch(train_dataset, model,config, is_filtering = None):
    L,B,D = get_datasets_shape(train_dataset, config=config)
    y0_run_path, y1_run_path, sigma_run_path = transformToMemmap(train_dataset, model, config) # padding the header, also X
    y0_run_path_reverse, y1_run_path_reverse, sigma_run_path_reverse = transformToMemmap(train_dataset, model, config,tags_s='reverse')  # padding the header, also X
    s = DatasetLargeSigma(S = sigma_run_path,
                           S0 = sigma_run_path_reverse,
                          s_shape = (B,L,D), config=config
                           )
    if is_filtering is not None:
        s = DatasetFilterLargeSigma(Y0=y0_run_path, Y0_reverse=y0_run_path_reverse,
                              Y1=y1_run_path, Y1_reverse= y1_run_path_reverse,
                              s_shape=(B, L, D), config=config
                              )

    return s

def train_sigma(train_dataset,model,config, is_train=False, beta_obs = 1e-6,test_dataset = None):
    # y0, y1, s = transform(X, model, config)
    # _, _, s0 = transform(np.flip(X,axis=1).copy(), model, config)
    strain = generate_sigma_sets_torch(train_dataset,model,config)
    if test_dataset is not None:
        sval = generate_sigma_sets_torch(test_dataset, model, config)
    # s = s.transpose(1, 0, 2)
    # s0 = s0[::-1].transpose(1, 0, 2)
    # strain, _, s = generate_sigma_sets(s0, s, config)
    name_raw = config.save_name
    model_name_raw = config.model_name
    beta_observe_raw = config.beta_observe
    config.save_name = config.save_name+ '-var'
    config.model_name = config.save_name + '.pth'
    config.beta_observe = beta_obs
    # strain, _, s = generate_sigma_sets(s0, s, config)
    if is_train:
        # model_s = train(train_datasets=strain, config=config, L2 = True, shuffle=False)
        model_s = train_Dt(train_datasets=strain, config=config, L2=True, shuffle=False)
    else:
        model_s = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                    h_dim=config.h_dim, default_enc_nn=config.default_enc_nn,
                                    hidden_dim=config.hidden_size, layers=config.layers,
                                    d_model=config.d_model, n_head=config.n_heads, layers_enc=config.enc_layers, 
                                    beta_obs = config.beta_observe, beta_ac = config.beta_ac, beta_conti = config.beta_conti,
                                    beta_weight=config.beta_weight,
                                    device=config.device).to(config.device)
        load(model_s, config.dload, config.model_name)

    # s0_path, s1_path, _ = transform(s, model_s, config)
    transformToMemmap(strain, model_s, config,is_train= is_train)

    config.save_name_var = config.save_name
    config.model_name_var = config.model_name
    config.save_name = name_raw
    config.model_name = model_name_raw
    config.beta_observe = beta_observe_raw
    # return np.sqrt(np.abs(s[:,config.encoder_length:,:])+1e-16), np.sqrt(np.abs(s1)+1e-16).transpose(1,0,2)

    return strain, model_s


def train_velocity(train_dataset,model,config, is_train=False,test_dataset = None):

    vtrain = generate_velocity_sets_torch(train_dataset,model,config)
    if test_dataset is not None:
        vval = generate_velocity_sets_torch(test_dataset, model, config)

    name_raw = config.save_name
    model_name_raw = config.model_name
    beta_observe_raw = config.beta_observe
    config.save_name = config.save_name+ '-vel'
    config.model_name = config.save_name + '.pth'
    # strain, _, s = generate_sigma_sets(s0, s, config)
    if is_train:
        model_v = train(train_datasets=vtrain, config=config, L2 = True, shuffle=False)
    else:
        model_v = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                    h_dim=config.h_dim, default_enc_nn=config.default_enc_nn,
                                    hidden_dim=config.hidden_size, layers=config.layers,
                                    d_model=config.d_model, n_head=config.n_heads, layers_enc=config.enc_layers,
                                    beta_obs = config.beta_observe, beta_ac = config.beta_ac, beta_conti = config.beta_conti,
                                    beta_weight=config.beta_weight,
                                    device=config.device).to(config.device)
        load(model_v, config.dload, config.model_name)

    transformToMemmap(vtrain, model_v, config, is_train=is_train)

    config.save_name_vel = config.save_name
    config.model_name_vel = config.model_name
    config.save_name = name_raw
    config.model_name = model_name_raw
    config.beta_observe = beta_observe_raw

    return vtrain, model_v

# def train_y_jump(train_dataset,model,config,is_train=False):
#     L, B, D = get_datasets_shape(train_dataset, config=config)
#     _, y1, _ = transformToMemmap(train_dataset, model, config)
#     y_jump_train = DatasetFilterTracks(x = train_dataset, y1=y1,y_shape=(B, L, D), config=config)
#     name_raw = config.save_name
#     model_name_raw = config.model_name
#     config.save_name =config.save_name+ '_jump'
#     config.model_name = config.save_name + '.pth'
#     # strain, _, s = generate_sigma_sets(s, config)
#     # config.beta_observe
#     if is_train:
#         model_y_jump = train_filter(train_datasets=y_jump_train, config=config)
#     else:
#         model_y_jump = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
#                                     h_dim=config.h_dim, default_enc_nn=config.default_enc_nn,
#                                     hidden_dim=config.hidden_size, layers=config.layers,
#                                     d_model=config.d_model, n_head=config.n_heads, layers_enc=config.enc_layers,
#                                     beta_obs = config.beta_observe, beta_ac = config.beta_ac, beta_conti = config.beta_conti,
#                                     beta_weight=config.beta_weight,
#                                     device=config.device).to(config.device)
#         load(model_y_jump, config.dload, config.model_name)
#     # _, y0_filter_path_run, _ = transform(X,model_y_jump, config)
#     ## recover
#     config.save_name = name_raw
#     config.model_name = model_name_raw
#     return y_jump_train, model_y_jump


def traj_generate(X, model, config):
    train_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))
    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=config.batch_size,
                              shuffle=False)
    my_gen = RawGRU(model, hidden_dim=64, layers=2, h_dim=20, generator_num=50).to(config.device)
    for t, X in enumerate(train_loader):
        if t > config.sample_nums:
            break
        x = X[0].permute(1, 0, 2).to(config.device)
        _, output = model(x)
        x_tail = x[config.encoder_length:]

        sigma = x_tail[1:, :, :]- output[:-1, :, :]
        sigma = torch.std(sigma, dim=0).squeeze().repeat(config.batch_size,1,1).permute(1, 0, 2)

        y0, y1, x0 = my_gen(x, sigma)
        plt.plot(x0[1:, 1, 0].data.cpu().numpy())
        plt.plot(y0[1:, 1, 0].data.cpu().numpy())
        plt.plot(y1[:, 1, 0].data.cpu().numpy())
        plt.show()


        #### gen velocity
        v = (y1[:-1, :, 0].data.cpu().numpy() - x0[:-1, :, 0].data.cpu().numpy())
        ac = []
        for i in range(v.shape[1]):
            ac.append(np.correlate(v[:, i], v[:, i], mode='full'))
        # ac_raw = np.correlate(sigma, sigma, mode='full')
        ac_raw = np.mean(np.array(ac), axis=0)
        print(ac_raw[ac_raw.size // 2:ac_raw.size // 2 + 5])
        plt.plot(ac_raw[ac_raw.size // 2:ac_raw.size // 2 + 50])

        v = output[1:, :, 0].data.cpu().numpy() - x_tail[1:, :, 0].data.cpu().numpy()
        ac = []
        for i in range(v.shape[1]):
            ac.append(np.correlate(v[:, i], v[:, i], mode='full'))
        # ac_raw = np.correlate(sigma, sigma, mode='full')
        ac_raw = np.mean(np.array(ac), axis=0)
        print(ac_raw[ac_raw.size // 2:ac_raw.size // 2 + 5])
        plt.plot(ac_raw[ac_raw.size // 2:ac_raw.size // 2 + 50])
        plt.show()


def traj_multipe_transformer(X,S,model,config, num=1):
    model.eval()
    min_sz = min(X.shape[0],S.shape[0])
    y0_num, y1_num, x0_num = [], [], []
    test_datasets = TensorDataset(torch.from_numpy(X[:min_sz]).type(dtype=torch.float),
                                      torch.from_numpy(S[:min_sz]).type(dtype=torch.float))
    test_loader = DataLoader(dataset=test_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=False)
    my_gen = RawGRU(model, 
                    hidden_dim=config.hidden_size, 
                    layers=config.layers, h_dim=config.h_dim, 
                    generator_num=config.encoder_length).to(config.device)

    for i in range(num):
        y0_run, y1_run, x0_run = [], [], []
        for t, x in enumerate(test_loader):
            h = x[0].permute(1, 0, 2).to(config.device)
            s = x[1].permute(1, 0, 2).to(config.device)

            y0, y1, x0 = batch_raw_gen(h, s, my_gen)
            y0_run.append(y0)
            y1_run.append(y1)
            x0_run.append(x0)


        y0_num.append(np.concatenate(y0_run, axis=1)[:,:,:,np.newaxis])
        y1_num.append(np.concatenate(y1_run, axis=1)[:,:,:,np.newaxis])
        x0_num.append(np.concatenate(x0_run, axis=1)[:,:,:,np.newaxis])

    return (np.concatenate(y0_num, axis=-1), np.concatenate(y1_num, axis=-1) ,np.concatenate(x0_num, axis=-1))


def traj_multipe_transformer_memmap(datasets,model,config, num=10, gen_name = None):

    logger_name = config.dload + '/' + config.save_name + '.log'
    logger = MyLogger.__call__(logger_name).get_logger()
    logger.info('starting regenerating with config:')
    logger.info(config)

    model.eval()
    s_train, model_s = train_sigma(datasets, model, config, beta_obs=0.0, is_train=False)
    s_dataset = generate_sigma_predicted_torch(s_train,model=model_s,config = config)
    outliers_const = 1.96
    max_integer_const = 2147483647
    x_dataloader = DataLoader(dataset=datasets,
                                batch_size=config.batch_size,
                                  shuffle=False)
    s_dataloader = DataLoader(dataset=s_dataset,
                                batch_size=config.batch_size,
                                  shuffle=False)

    my_gen = RawGRU(model,
                    hidden_dim=config.hidden_size,
                    layers=config.layers, h_dim=config.h_dim,
                    generator_num=config.encoder_length).to(config.device)

    # saving path
    L, B, D = get_datasets_shape(datasets, config)

    if gen_name is not None:
        gen_save_name = gen_name
    else:
        gen_save_name = config.save_name
    logger.info(gen_save_name + ' the file regenerate')

    fp_x0_gen_path = os.path.join(config.dload, gen_save_name + '-gen', 'memmap_gen', 'x0_gen.npy')
    fp_y0_gen_path = os.path.join(config.dload, gen_save_name + '-gen', 'memmap_gen', 'y0_gen.npy')
    fp_y1_gen_path = os.path.join(config.dload, gen_save_name + '-gen', 'memmap_gen', 'y1_gen.npy')

    if os.path.exists(os.path.join(config.dload, gen_save_name + '-gen', 'memmap_gen')):
        pass
    else:
        os.makedirs(os.path.join(config.dload, gen_save_name + '-gen', 'memmap_gen'))
    #
    fp_x0_gen = np.memmap(fp_x0_gen_path, dtype='float32', mode='w+', shape=(len(datasets), L, D, num))
    fp_y0_gen = np.memmap(fp_y0_gen_path, dtype='float32', mode='w+', shape=(len(datasets), L, D, num))

    fp_y1_gen = np.memmap(fp_y1_gen_path, dtype='float32', mode='w+', shape=(len(datasets), L, D, num))

    np.save(os.path.join(config.dload, gen_save_name + '-gen', 'memmap_gen','shape_gen.npy'),
            np.asarray([len(datasets), L, D, num]))
    seeds_set = torch.randint(max_integer_const,(len(x_dataloader),num))
    for i in range(num):
        cnt = 0
        cnt_tracks = 0
        for item_x, item_s in zip(x_dataloader, cycle(s_dataloader)):

            x = item_x[0].permute(1, 0, 2).to(config.device)
            s = item_s[0].permute(1, 0, 2).to(config.device)

            # const 1.96 of the normal, if y(1,t) - y(1,t-1) > const * sigma_std

            _, y1, = model(x)
            v = x[config.encoder_length:] - y1
            idv = ((torch.abs(v) - outliers_const*abs(s)) > 1e-6) # outliers
            # v1^2 + v2^2 = ... sigma1^2 + sigma2^2 + ... +
            s[idv] = torch.abs(v[idv])
            ## jump region padding normal distribution
            # s[idv] = 0.
            # baseline_v = torch.zeros_like(s)
            # baseline_v[idv] = v[idv]
            ## velocity adding jump point baseline of jump points x+Dt+b
            ## saving, memmap, and (1) s_save
            # y0_gen, y1_gen, x0_gen = batch_raw_gen(x, s, my_gen, adding_jumping=baseline_v)
            # torch.manual_seed(cnt + i)
            ## note: every loop the manual seed with equal the same
            # a = torch.randint(max_integer_const,(1,))
            # torch.manual_seed(torch.randint(max_integer_const,(1,)))

            torch.manual_seed(seeds_set[cnt_tracks,i])
            y0_gen, y1_gen, x0_gen = batch_raw_gen(x, s, my_gen)

            cnt_add = x.shape[1] # batch

            fp_x0_gen[cnt:cnt + cnt_add,:,:,i] = x0_gen.transpose(1, 0, 2)
            fp_y0_gen[cnt:cnt + cnt_add,:,:,i] = y0_gen.transpose(1, 0, 2)
            fp_y1_gen[cnt:cnt + cnt_add,:,:,i] = y1_gen.transpose(1, 0, 2)

            cnt += cnt_add
            cnt_tracks += 1

        logger.info('group generated: {:d}/{:d}'
              .format(i, num))
        # print('group generated: {:d}/{:d}'
        #       .format(i, num))


    return fp_x0_gen_path, fp_y0_gen_path ,fp_y1_gen_path


def clear_clash(config):
    def clear_single_folder(file_path, file_name, tags = 'memmap_clash'):
        file_sets_raw = os.listdir(os.path.join(file_path,file_name,tags))
        for f in file_sets_raw:
            filename = os.path.join(file_path,
                                    file_name,
                                    tags, f)
            os.remove(filename)
    if os.path.exists(os.path.join(config.dload, config.save_name)):
        clear_single_folder(config.dload, config.save_name)
    if os.path.exists(os.path.join(config.dload, config.save_name + '-var')):
        clear_single_folder(config.dload, config.save_name + '-var')
    if os.path.exists(os.path.join(config.dload, config.save_name + '-vel')):
        clear_single_folder(config.dload, config.save_name + '-vel')
    if os.path.exists(os.path.join(config.dload, config.save_name + '-gen')):
        clear_single_folder(config.dload, config.save_name + '-gen')





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,
                        help='Default number of training epochs')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--default_enc_nn', type=tuple, default=((50, 'gelu'), (50, 'gelu')))
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Default number of RNN hidden size')
    parser.add_argument('--layers', type=int, default=2,
                        help='Default number of LSTM layers')
    parser.add_argument('--h_dim', type=int, default=20,
                        help='Default number of embedding size')
    parser.add_argument('--d_model', type=int, default=64,
                        help='Transformer hidden dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Default number of attention head')
    parser.add_argument('--enc_layers', type=int, default=3,
                        help='Default number of attention head')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Default number of dropout')
    parser.add_argument('--scale', type=float, default=0.1,
                        help='schedual learning rates scaling')
    parser.add_argument('--in_dim', type=int, default=2,
                        help='Default number of input time series dimension')
    parser.add_argument('--out_dim', type=int, default=2,
                        help='Default number of input time series dimension')
    parser.add_argument('--encoder_length', type=int, default=50,
                        help='Default encoder length of input time series')
    parser.add_argument('--warmup_steps', type=int, default=400,
                        help='Training warmup steps')
    parser.add_argument('--sample_nums', type=int, default=5,
                        help='evaluation figures samples')
    parser.add_argument('--dload', type=str, default='./model_dir',
                        help='save_dir')
    parser.add_argument('--beta_observe', type=float, default=1e-5,
                        help='The observation noise delete')


    config = parser.parse_args()


    is_train = False
    dload = './model_dir'
    alpha = [1e-3]
    Dt = [1]
    save_name = 'alpha' + '-'.join(list(map(str, alpha))) + '_Dt' + '-'.join(list(map(str, Dt)))
    # x, x_idx = generate_x(alpha=alpha,Dt=Dt,time_span=time_span,seed=0)
    config.save_name = save_name
    config.model_name = save_name + '.pth'
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x, _ = generate_x(alpha=alpha, Dt=Dt, time_span=1000 * 100, seed=0)
    X = x.reshape(1000 * len(alpha) * len(Dt), 100, 2)
    X, _,_ = generate_data_sets(X,config)

    if is_train:
        model = train(X=X,config=config)
    else:
        model = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                    h_dim=config.h_dim, default_enc_nn=config.default_enc_nn,
                                    hidden_dim=config.hidden_size, layers=config.layers,
                                    d_model=config.d_model, n_head=config.n_heads, layers_enc=config.enc_layers,
                                    beta_weight=config.beta_weight,
                                    device=config.device).to(config.device)
        load(model,dload,config.model_name)

    y0,y1 = train_y_jump(X,model,config,is_train=False)
    # s0, s1 = train_sigma(X, model, config, is_train=is_train)
    # a = traj_multipe_transformer(X,s1, model,config,num=1)
    # output = a[1]
    plt.plot(X[:, 0, 0])
    plt.plot(y0[:,0, 0])
    plt.plot(y1[:, 0, 0])
    plt.show()












