from model.JumpSDEsTrans import JumpSDEsTransformer, RawGRU
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
import logging
import time, datetime
from torch.autograd import Variable
from model.sam import SAM
import pandas as pd
from robust_loss_pytorch import adaptive
from itertools import cycle
import re


class CustomSchedule(object):
    """
    A custom learning rate scheduler that adjusts the learning rate based on the number of training steps.
    This scheduler uses a warm-up phase followed by a decay phase.

    Args:
        d_model (int): The dimensionality of the model.
        warmup_steps (int, optional): The number of warm-up steps. Defaults to 400.
        optimizer (torch.optim.Optimizer, optional): The optimizer to adjust the learning rate for. Defaults to None.
        scale (float, optional): A scaling factor for the learning rate. Defaults to 0.1.
    """

    def __init__(self, d_model, warmup_steps=400, optimizer=None, scale=0.1):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.d_model = float(d_model)
        self.warmup_steps = warmup_steps
        self.steps = 1.
        self.optimizer = optimizer
        self.scale = scale

    def step(self):
        arg1 = self.steps ** -0.5
        arg2 = self.steps * (self.warmup_steps ** -1.5)
        self.steps += 1.
        lr = self.scale * (self.d_model ** -0.5) * min(arg1, arg2)
        if self.steps > self.warmup_steps:
            lr = max(lr, 5e-5)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']


class SingletonType(type):
    """
    Metaclass implementing the Singleton pattern.
    Ensures a class has only one instance and provides a global point of access.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MyLogger(object, metaclass=SingletonType):
    """Singleton logger class that writes to both file and console"""
    _logger = None

    def __init__(self, filename, verbosity=1):
        # Initialize logger with name "crumbs"
        self._logger = logging.getLogger("crumbs")
        self._logger.setLevel(logging.DEBUG)

        # Define log message format
        formatter = logging.Formatter('%(asctime)s \t [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s')
        now = datetime.datetime.now()
        fileHandler = logging.FileHandler(filename, "a")
        streamHandler = logging.StreamHandler()

        # Apply formatter to both handlers
        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        # Add both handlers to logger
        self._logger.addHandler(fileHandler)
        self._logger.addHandler(streamHandler)

        print("Generate new instance")

    def get_logger(self):
        return self._logger


class DatasetTracks(Dataset):
    """Dataset class for track data that supports both numpy arrays and file paths"""

    def __init__(self, X, config, sample_rate=1.):
        self.config = config
        self.r_noise = config.noise_adding

        # Handle numpy array input
        if isinstance(X, np.ndarray):
            # self.datasets = X
            samples = np.random.choice(X.shape[0], int(sample_rate * X.shape[0]), replace=False)
            samples = np.sort(samples)
            self.datasets = X[samples]
            self.label = 'from_workspace'

        # Handle file path input
        elif isinstance(X, str):
            tmp = np.load(X)[:]
            samples = np.random.choice(tmp.shape[0], int(sample_rate * tmp.shape[0]), replace=False)
            samples = np.sort(samples)
            self.datasets = tmp[samples]
            # self.label = ('.').join(X.split("/")[-1].split(".")[:-1])  # filename
            self.label = ('.').join(X.split("_")[-1].split(".")[:-1])
        else:
            raise ValueError("Unsupported data type")

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        # Get a sample with optional noise and preprocessing
        tracks = self.datasets[idx]

        # Convert label to float if numeric, else -1
        if bool(re.match(r'^[-+]?[0-9]*\.?[0-9]+$', self.label)):
            label = float(self.label)
        else:
            label = -1  # self.label

        # Preprocess tracks: convert to tensor, normalize, add noise
        x = torch.from_numpy(tracks).type(dtype=torch.float)
        x = x - x[self.config.encoder_length - 1, :].reshape(1, self.config.in_dim)
        torch.manual_seed(idx)
        x += self.r_noise * torch.randn_like(x)

        return x, label


class DatasetTracksAlpha(Dataset):
    """Dataset class handling tuple pairs (data, labels) or file paths"""

    def __init__(self, X, config, sample_rate=1.):
        self.config = config
        self.r_noise = config.noise_adding

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
            samples = np.random.choice(tmp.shape[0], int(sample_rate * tmp.shape[0]), replace=False)
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


class DatasetLargeSigma(Dataset):
    """Dataset class for handling large memory-mapped sigma_run to diffusion predicted data"""

    def __init__(self, S, S0, s_shape, config):
        """
        Initialize the dataset with memory-mapped files for large data.

        Args:
            S (str): Path to the current order sigma_run data file.
            S0 (str): Path to the flip order sigma_run data file.
            s_shape (tuple): Shape of the data.
            config: Configuration object with parameters.
        """
        self.s_shape = s_shape
        self.config = config

        # Initialize memory-mapped arrays for large data handling
        if isinstance(S, str):
            self.S = np.memmap(S, mode='r', dtype='float32', shape=s_shape)  # current
            self.S0 = np.memmap(S0, mode='r', dtype='float32', shape=s_shape)  # head
        else:
            raise ValueError("Unsupported data type")

    def __len__(self):
        return self.s_shape[0]

    def __getitem__(self, idx):
        # memory-map to tensors
        s1 = np.asarray(self.S[idx]).copy()
        data0 = np.asarray(self.S0[idx]).copy()
        s1 = torch.from_numpy(s1).type(dtype=torch.float)
        data0 = torch.from_numpy(data0).type(dtype=torch.float)

        # Create padded head sigma_run
        s0 = torch.zeros(self.config.encoder_length, s1.shape[1])
        minlen_sz = min([self.config.encoder_length, s1.shape[0]])
        s0[:minlen_sz] = torch.flip(data0, dims=[0])[:minlen_sz]

        # square of the sigma_run term
        x = torch.concatenate([s0, s1], dim=0) ** 2
        return x, -1


class DatasetLargeVelocity(Dataset):
    """
    Dataset class for handling large memory-mapped velocity data efficiently.
    """

    def __init__(self, V, V0, v_shape, config):
        """
        Initialize the dataset with memory-mapped velocity data.

        Args:
            V (str): Path to the current order velocity data file.
            V0 (str): Path to the flip order velocity data file.
            v_shape (tuple): Shape of the data.
            config: Configuration object with parameters.
        """
        self.v_shape = v_shape
        self.config = config

        # Load memory-mapped arrays for large data handling
        if isinstance(V, str):
            self.V = np.memmap(V, mode='r', dtype='float32', shape=v_shape)
            self.V0 = np.memmap(V0, mode='r', dtype='float32', shape=v_shape)
        else:
            raise ValueError("Unsupported data type")

    def __len__(self):
        return self.v_shape[0]

    def __getitem__(self, idx):
        # Load data for the current and head velocity from memory-mapped arrays
        v1 = np.asarray(self.V[idx]).copy()
        data0 = np.asarray(self.V0[idx]).copy()

        # Convert data to PyTorch tensors
        # s1 = torch.from_numpy(self.S[idx]).type(dtype=torch.float)
        # data0 = torch.from_numpy(self.S0[idx]).type(dtype=torch.float)
        v1 = torch.from_numpy(v1).type(dtype=torch.float)
        data0 = torch.from_numpy(data0).type(dtype=torch.float)
        v0 = torch.zeros(self.config.encoder_length, v1.shape[1])

        # for different length of encoder sequence and prediction sequence
        minlen_sz = min([self.config.encoder_length, v1.shape[0]])
        v0[:minlen_sz] = torch.flip(data0, dims=[0])[:minlen_sz]

        # Concatenate head (v0) and current (v1) velocity data
        x = torch.concatenate([v0, v1], dim=0)
        return x, -1


class DatasetLargeSigmaPredict(Dataset):
    """
    Dataset class for prediction using memory-mapped sigma_run data.
    """

    def __init__(self, S, s_shape):
        self.s_shape = s_shape

        # Initialize memory-mapped array for large data
        if isinstance(S, str):
            self.S = np.memmap(S, mode='r', dtype='float32', shape=s_shape)
        else:
            raise ValueError("Unsupported data type")

    def __len__(self):
        return self.s_shape[0]

    def __getitem__(self, idx):
        s = np.asarray(self.S[idx]).copy()

        # non negtive value
        x = torch.sqrt(torch.abs(torch.from_numpy(s).type(dtype=torch.float)) + 1e-6)
        return x, -1


class DatasetFilterLargeSigma(Dataset):
    """
    Dataset class for filtering and processing large memory-mapped y data to sigma_run.
    Combines forward and reverse sigma data with padding and transformations.
    """

    def __init__(self, Y0, Y1, Y0_reverse, Y1_reverse, s_shape, config):
        """
        Initialize the dataset with memory-mapped sigma data.

        Args:
            Y0 (str): Path to forward y reconstructed data (Y0).
            Y1 (str): Path to forward y predicted data (Y1).
            Y0_reverse (str): Path to reverse y reconstructed data (Y0_reverse).
            Y1_reverse (str): Path to reverse y predicted data (Y1_reverse).
            s_shape (tuple): Shape of the data.
            config: Configuration object with parameters`.
        """
        self.s_shape = s_shape
        self.config = config

        if isinstance(Y0, str) and isinstance(Y1, str) and isinstance(Y0_reverse, str) and isinstance(Y1_reverse, str):
            self.Y0 = np.memmap(Y0, mode='r', dtype='float32', shape=s_shape)
            self.Y0_reverse = np.memmap(Y0_reverse, mode='r', dtype='float32', shape=s_shape)
            self.Y1 = np.memmap(Y1, mode='r', dtype='float32', shape=s_shape)
            self.Y1_reverse = np.memmap(Y1_reverse, mode='r', dtype='float32', shape=s_shape)
        else:
            raise ValueError("Unsupported data type")

    def __len__(self):
        return self.s_shape[0]

    def __getitem__(self, idx):
        # Load forward y0, y1, sigma_run data for the current index
        y1 = np.asarray(self.Y1[idx]).copy()
        y0 = np.asarray(self.Y0[idx]).copy()
        s1 = y0[self.config.encoder_length + 1:] - y1[:-1]
        s1 = torch.from_numpy(s1).type(dtype=torch.float)
        s1 = torch.concatenate([s1, torch.zeros((1, s1.shape[1]))], dim=0)  # L x D

        # Load reverse y0, y1, sigma_run data for the current index
        y1_reverse = np.asarray(self.Y1_reverse[idx]).copy()
        y0_reverse = np.asarray(self.Y0_reverse[idx]).copy()
        s0_reverse = y0_reverse[self.config.encoder_length + 1:] - y1_reverse[:-1]
        data0 = torch.from_numpy(s0_reverse).type(dtype=torch.float)
        data0 = torch.concatenate([data0, torch.zeros((1, s1.shape[1]))], dim=0)  # L x D

        # Compute reverse sigma difference (s0_reverse) and pad it
        s0 = torch.zeros(self.config.encoder_length, s1.shape[1])
        minlen_sz = min([self.config.encoder_length, s1.shape[0]])
        s0[:minlen_sz] = torch.flip(data0, dims=[0])[:minlen_sz]
        x = torch.concatenate([s0, s1], dim=0) ** 2
        return x, -1


def merge_datasets(config, filepath='', filenames=['annotation.txt'], sample_rate=1.):
    """
    Merge multiple datasets into a single concatenated dataset.

    Args:
        config: Configuration object containing dataset-specific parameters.
        filepath (str): Path to the directory containing the annotation files.
        filenames (list of str): List of annotation file names to process.
        sample_rate (float): Fraction of data to sample from each dataset (default: 1.0).

    Returns:
        ConcatDataset: A concatenated dataset containing all tracks from the given annotation files.
    """
    dataloaders = []
    for filename in filenames:
        df = pd.read_csv(os.path.join(filepath, filename), delimiter=',', header=None)
        for i in range(df.shape[0]):
            dataloaders.append(DatasetTracks(df.iloc[i, 0], config, sample_rate))
    return ConcatDataset(dataloaders)


def generate_dataset_torch(full_dataset, train_prob=0.75, seed=0):
    """
    Split a dataset into training and testing subsets with an optional deterministic seed.
    Optionally, reorder the test dataset indices in ascending order if `train_prob` is very small.

    Args:
        full_dataset (Dataset): The complete dataset to be split.
        train_prob (float): Probability (or fraction) of data to assign to the training set.
                            The remaining data will go to the test set. Default is 0.75.
        seed (int): Random seed for reproducibility. Default is 0.

    Returns:
        tuple: A tuple containing the training dataset (train_dataset) and
               the testing dataset (test_dataset).
    """
    train_sz = int(train_prob * len(full_dataset))
    test_sz = len(full_dataset) - train_sz
    torch.manual_seed(seed=seed)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_sz, test_sz])

    # ordered the original
    if train_prob <= 1e-6:
        # Sort the indices in ascending order
        sorted_indices = sorted(test_dataset.indices)

        # Use the sorted indices to reorder the test dataset
        test_dataset = torch.utils.data.Subset(test_dataset.dataset, sorted_indices)
    return train_dataset, test_dataset


def get_datasets_shape(datasets, config):
    """
    Compute the shape of the dataset for use in model configuration.

    Args:
        datasets (Dataset): The dataset to analyze.
        config: Configuration object containing relevant parameters like `batch_size`
                and `encoder_length`.

    Returns:
        tuple: A tuple containing:
            - L (int): The adjusted sequence length (original length minus encoder length).
            - len(datasets) (int): Total number of samples in the dataset.
            - D (int): The dimensionality of the dataset features.
    """
    dataloader = DataLoader(dataset=datasets,
                            batch_size=config.batch_size,
                            shuffle=False)  # Don't sh

    # Iterate over the DataLoader to extract the shape of its first batch
    for t, X in enumerate(dataloader):
        L, _, D = X[0].permute(1, 0, 2).shape
        L -= config.encoder_length
        if t >= 1:
            break

    return L, len(datasets), D


def save(model, dload, file_name):
    """
    Save a PyTorch model's state dictionary to a specified file.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        dload (str): Directory path where the model file will be saved.
        file_name (str): Name of the file to save the model state dictionary.

    Returns:
        None
    """
    PATH = dload + '/' + file_name
    if os.path.exists(dload):
        pass
    else:
        os.mkdir(dload)
    torch.save(model.state_dict(), PATH)


def save_checkpoint(ck, dload, file_name):
    """
    Save a checkpoint dictionary to a specified file.

    Args:
        ck (dict): The checkpoint dictionary to save (e.g., model state, optimizer state, etc.).
        dload (str): Directory path where the checkpoint will be saved.
        file_name (str): Name of the folder where the checkpoint file will be stored.

    Returns:
        None
    """
    ckp_PATH = dload + '/' + file_name
    PATH = ckp_PATH + '/checkpoint_ckpt.pth'
    if os.path.exists(ckp_PATH):
        pass
    else:
        os.mkdir(ckp_PATH)
    torch.save(ck, PATH)


def load(model, dload, file_name):
    """
    Load a PyTorch model's state dictionary from a specified file.

    Args:
        model (torch.nn.Module): The PyTorch model into which the state dictionary will be loaded.
        dload (str): Directory path where the model file is located.
        file_name (str): Name of the file containing the saved state dictionary.

    Returns:
        torch.nn.Module: The model with the loaded state dictionary.
    """
    PATH = dload + '/' + file_name
    model.load_state_dict(torch.load(PATH))


def generate_data_sets(X, config):
    """
    Generate training and validation datasets

    Args:
        X: Input data array of shape (batch_size, sequence_length, input_dimension)
        config: Configuration object containing model parameters

    Returns:
        X_train: Training data subset (70% of normalized data)
        X_val: Validation data subset (30% of normalized data)
        data: Full normalized data array
    """
    data = X - X[:, config.encoder_length - 1, :].reshape(-1, 1, config.in_dim)
    X_train, X_val, y_train, y_val = train_test_split(data, np.zeros_like(data),
                                                      test_size=0.3, random_state=config.seed)
    return X_train, X_val, data


def generate_data_sets_support(X, config, S):
    """
    Generate training and validation datasets with support value adjustment

    Args:
        X: Input data array of shape (batch_size, sequence_length, input_dimension)
        config: Configuration object containing model parameters
        S: Support values to inject at encoder endpoint

    Returns:
        X_train: Training data subset (60% of normalized data)
        X_val: Validation data subset (40% of normalized data)
        data: Full normalized data array with support values
    """
    data = X - X[:, config.encoder_length - 1, :].reshape(-1, 1, config.in_dim)
    data[:, config.encoder_length - 1, :] = S
    X_train, X_val, y_train, y_val = train_test_split(data, np.zeros_like(data),
                                                      test_size=0.4, random_state=config.seed)
    return X_train, X_val, data


def train(train_datasets, config, test_datasets=None, L2=None, shuffle=True):
    """
    Train the JumpSDEsTransformer model

    Args:
        train_datasets: Training dataset
        config: Configuration object containing model parameters
        test_datasets: Optional test dataset for evaluation
        L2: L2 regularization parameter
        shuffle: Whether to shuffle training data

    Returns:
        model: Trained JumpSDEsTransformer model
    """
    # Set up file paths for model saving and logging, and data: BxLxD
    filename_set = config.save_name
    model_name = config.model_name
    logger_name = config.dload + '/' + config.save_name + '.log'

    # Initialize model with configuration parameters
    model = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                h_dim=config.h_dim, default_enc_nn=config.default_enc_nn,
                                hidden_dim=config.hidden_size, layers=config.layers,
                                d_model=config.d_model, n_head=config.n_heads, layers_enc=config.enc_layers,
                                dropout=config.dropout,
                                beta_obs=config.beta_observe, beta_ac=config.beta_ac, beta_conti=config.beta_conti,
                                beta_weight=config.beta_weight,
                                device=config.device).to(config.device)
    # optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-2)
    base_optimizer = torch.optim.AdamW  # define an optimizer for the "sharpness-aware" update
    # optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay=1e-2)
    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=config.batch_size,
                              shuffle=shuffle)

    # Initialize adaptive loss function for less outliers
    L, _, D = get_datasets_shape(train_datasets, config=config)
    loss_fn = adaptive.AdaptiveLossFunction(num_dims=D * (L - 1), float_dtype=np.float32, device=config.device)

    # Setup optimizer with SAM (Sharpness-Aware Minimization)
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = SAM(params, base_optimizer, lr=config.lr, weight_decay=5e-2)

    # Initialize learning rate scheduler
    lr_scheduler = CustomSchedule(config.d_model, optimizer=optimizer, warmup_steps=config.warmup_steps,
                                  scale=config.scale)

    # Setup logging
    logger = MyLogger.__call__(logger_name).get_logger()
    logger.info('starting training with config:')
    logger.info(config)

    # Initialize training variables
    epochs = config.epochs
    start_epoch = -1

    # Load checkpoint if resuming training
    path_checkpoint = config.dload + '/' + config.save_name + '/checkpoint_ckpt.pth'
    if config.RESUME and os.path.exists(path_checkpoint):
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        loss_fn.load_state_dict(checkpoint['loss_fn'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.steps = (start_epoch + 1) * len(train_loader)

    for i in range(start_epoch + 1, epochs):
        model.train()
        epoch_loss = 0
        epoch_pred = 0
        epoch_rec = 0
        epoch_ac = 0
        for t, X in enumerate(train_loader):
            x = X[0].permute(1, 0, 2).to(config.device)

            # Forward pass with SAM first step
            y0, y1 = model(x)

            loss, pre, rec, ac = model.cal_loss(y0, y1, x[config.encoder_length:],
                                                loss_fn=loss_fn, L2=L2)
            epoch_loss += loss.item()
            epoch_pred += pre.item()
            epoch_rec += rec.item()
            epoch_ac += ac.item()

            # SAM optimization steps
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # SAM second forward-backward pass
            y02, y12 = model(x)
            model.cal_loss(y02, y12, x[config.encoder_length:],
                           loss_fn=loss_fn, L2=L2)[0].backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)

            # Gradient clipping and scheduler step
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            lr_scheduler.step()

            if (t + 1) % 1000 == 0:
                print('epoch {}: {:d}/{:d}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
                      .format(i, t, len(train_loader), epoch_loss / (t + 1), epoch_pred / (t + 1), epoch_rec / (t + 1),
                              epoch_ac / (t + 1)))

        # Log epoch results
        print('learning rates: {:4f}, auto correlation weight: {:4f}, steps: {:1f}'
              .format(lr_scheduler.get_lr(), model.beta_ac, lr_scheduler.steps))
        logger.info('epoch {}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
                    .format(i, epoch_loss / len(train_loader), epoch_pred / len(train_loader),
                            epoch_rec / len(train_loader), epoch_ac / len(train_loader)))

        # Save checkpoint every 10 epochs
        if (i + 1) % 10 == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": i,
                'loss_fn': loss_fn.state_dict(),
            }
            if test_datasets is not None:
                logger.info(evaluate(test_datasets, model, config))
            save_checkpoint(checkpoint, config.dload, config.save_name)
            print('checkpoint {} saved'.format(i))

    # Save final model
    save(model, config.dload, model_name)
    return model


def train_v(X, Y, config):
    """
    The old version of training JumpSDEsTransformer model with validation data

    Args:
        X: Input sequences array
        Y: Target sequences array
        config: Configuration object containing model parameters

    Returns:
        model: Trained JumpSDEsTransformer model
    """
    # data: BxLxD
    filename_set = config.save_name
    model_name = config.model_name
    logger_name = config.dload + '/' + config.save_name + '.log'
    model = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                h_dim=config.h_dim, default_enc_nn=config.default_enc_nn,
                                hidden_dim=config.hidden_size, layers=config.layers,
                                d_model=config.d_model, n_head=config.n_heads, layers_enc=config.enc_layers,
                                beta_obs=config.beta_observe, beta_ac=config.beta_ac, beta_conti=config.beta_conti,
                                beta_weight=config.beta_weight,
                                device=config.device).to(config.device)
    # print(list(model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = CustomSchedule(config.d_model, optimizer=optimizer, warmup_steps=config.warmup_steps,
                                  scale=config.scale)

    train_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float),
                                   torch.from_numpy(Y).type(dtype=torch.float))
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
            y0, y1 = model(x)
            loss, pre, rec, ac = model.cal_loss(y0, y1, y)

            epoch_loss += loss.item()
            epoch_pred += pre.item()
            epoch_rec += rec.item()
            epoch_ac += ac.item()
            loss.backward()

            lr_scheduler.step()
            optimizer.step()

            if (t + 1) % 200 == 0:
                print('epoch {}: {:d}/{:d}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
                      .format(i, t, len(train_loader), epoch_loss, epoch_pred, epoch_rec, epoch_ac))

        print('learning rates: {:4f}, steps: {:1f}'
              .format(lr_scheduler.get_lr(), lr_scheduler.steps))
        logger.info('epoch {}: loss {:4f}, pred: {:4f}, recon: {:4f}, autocorrelation: {:4f}'
                    .format(i, epoch_loss, epoch_pred, epoch_rec, epoch_ac))

    save(model, config.dload, model_name)
    return model


def train_Dt(train_datasets, config, test_datasets=None, L2=None, shuffle=True, is_var_head=True):
    """
    Train JumpSDEsTransformer model with optional variance head for uncertainty estimation

    Args:
        train_datasets: Training dataset
        config: Configuration object containing model parameters
        test_datasets: Optional test dataset for evaluation
        L2: L2 regularization parameter
        shuffle: Whether to shuffle training data
        is_var_head: Whether to use variance head, can be overridden by config

    Returns:
        model: Trained JumpSDEsTransformer model
    """
    # data: BxLxD
    filename_set = config.save_name
    model_name = config.model_name
    logger_name = config.dload + '/' + config.save_name + '.log'

    # Check if variance head setting exists in config
    if hasattr(config, 'is_var_head'):
        is_var_head = config.is_var_head

    # Initialize model
    model = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                h_dim=config.h_dim, default_enc_nn=config.default_enc_nn,
                                hidden_dim=config.hidden_size, layers=config.layers,
                                d_model=config.d_model, n_head=config.n_heads, layers_enc=config.enc_layers,
                                dropout=config.dropout,
                                beta_obs=config.beta_observe, beta_ac=config.beta_ac, beta_conti=config.beta_conti,
                                beta_weight=config.beta_weight,
                                device=config.device,
                                is_var_head=is_var_head).to(config.device)

    base_optimizer = torch.optim.AdamW  # define an optimizer for the "sharpness-aware" update
    # optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay=1e-2)
    # base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=config.batch_size,
                              shuffle=shuffle)

    L, _, D = get_datasets_shape(train_datasets, config=config)

    # Initialize adaptive loss function and optimizer
    loss_fn = adaptive.AdaptiveLossFunction(num_dims=D * (L - 1), float_dtype=np.float32, device=config.device)
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = SAM(params, base_optimizer, lr=config.lr, weight_decay=5e-2)
    # optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay = 1e-2 )
    lr_scheduler = CustomSchedule(config.d_model, optimizer=optimizer, warmup_steps=config.warmup_steps,
                                  scale=config.scale)

    # Setup logging
    logger = MyLogger.__call__(logger_name).get_logger()
    logger.info('starting Dt training with config, beta NLL Gaussian loss:')
    logger.info(config)
    epochs = config.epochs
    start_epoch = -1

    # Load checkpoint if resuming training
    path_checkpoint = config.dload + '/' + config.save_name + '/checkpoint_ckpt.pth'
    if config.RESUME and os.path.exists(path_checkpoint):
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        loss_fn.load_state_dict(checkpoint['loss_fn'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.steps = (start_epoch + 1) * len(train_loader)

    for i in range(start_epoch + 1, epochs):
        model.train()
        epoch_loss = 0
        epoch_pred = 0
        epoch_rec = 0
        epoch_ac = 0
        for t, X in enumerate(train_loader):
            x = X[0].permute(1, 0, 2).to(config.device)

            # SAM optimization step 1
            y0, y1 = model(x)

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

            # SAM optimization step 2
            y02, y12 = model(x)
            # model.cal_loss(y02, y12, x[config.encoder_length:],
            #                loss_fn=loss_fn, L2=L2)[0].backward()  # make sure to do a full forward pass

            model.cal_loss_sigma(y02, y12, x[config.encoder_length:],
                                 )[0].backward()  # make sure to do a full forward pass

            optimizer.second_step(zero_grad=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            lr_scheduler.step()

            if (t + 1) % 1000 == 0:
                print('epoch {}: {:d}/{:d}: loss {:4f}, pred: {:4f}, recon: {:4f}, NLLGaussian: {:4f}'
                      .format(i, t, len(train_loader), epoch_loss / (t + 1), epoch_pred / (t + 1), epoch_rec / (t + 1),
                              epoch_ac / (t + 1)))

        print('learning rates: {:4f}, auto correlation weight: {:4f}, steps: {:1f}'
              .format(lr_scheduler.get_lr(), model.beta_ac, lr_scheduler.steps))
        logger.info('epoch {}: loss {:4f}, pred: {:4f}, recon: {:4f}, NLLGaussian: {:4f}'
                    .format(i, epoch_loss / len(train_loader), epoch_pred / len(train_loader),
                            epoch_rec / len(train_loader), epoch_ac / len(train_loader)))

        if (i + 1) % 10 == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": i,
                'loss_fn': loss_fn.state_dict(),
            }
            if test_datasets is not None:
                logger.info(evaluate(test_datasets, model, config))
            save_checkpoint(checkpoint, config.dload, config.save_name)
            print('checkpoint {} saved'.format(i))

    save(model, config.dload, model_name)
    return model


def evaluate_draw(X, model, config):
    """
    Evaluate model predictions and plot comparison graphs

    Args:
        X: Input data (numpy array or tensor dataset)
        model: Trained model
        config: Configuration object containing model parameters

    Plots:
        1. Original vs predicted trajectories
        2. Raw differences vs predicted differences
        3. Autocorrelation plots for raw data, velocities and diffusion term
    """
    if isinstance(X, np.ndarray):
        train_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))
    else:
        train_datasets = X

    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=config.batch_size,
                              shuffle=False)

    for t, X in enumerate(train_loader):
        if t > min(config.sample_nums - 1, len(train_loader) - 2):  # drop_last batch
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
        plt.plot(ac_raw[ac_raw.size // 2:])

        # Dt
        sigma = (x[1:, :, 0].data.cpu().numpy() - output[:-1, :, 0].data.cpu().numpy())
        # v = (output[:-1,:,0].data.cpu().numpy()-x[:-1,:,0].data.cpu().numpy())
        ac = []
        for i in range(sigma.shape[1]):
            ac.append(np.correlate(sigma[:, i], sigma[:, i], mode='full'))
        # ac_raw = np.correlate(sigma, sigma, mode='full')
        ac_raw = np.mean(np.array(ac), axis=0)
        plt.plot(ac_raw[ac_raw.size // 2:])
        plt.show()


def evaluate_draw_tracks(X, model, config, is_gt=None):
    """
    Evaluate and visualize model predictions including multiple trajectory estimates

    Args:
        X: Input data (numpy array or tensor dataset)
        model: Trained model
        config: Configuration object containing model parameters
        is_gt: Ground truth data if available

    Plots:
        Original trajectory vs multiple predicted trajectories (y0, y1)
        and ground truth if provided
    """
    if isinstance(X, np.ndarray):
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


def evaluate(test_datasets, model, config):
    """
    Evaluate model performance on test dataset

    Args:
        test_datasets: Test dataset
        model: Trained model to evaluate
        config: Configuration object containing model parameters

    Returns:
        eval_acc: String containing evaluation metrics (loss, prediction error,
                 reconstruction error, autocorrelation)
    """
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


def evaluate_draw_savefig(X_sets, model_sets, config, y_labels):
    """
    Evaluate and visualize multiple models with corresponding datasets and save plots

    Args:
        X_sets: List of input datasets
        model_sets: List of trained models
        config: Configuration object containing parameters
        y_labels: Labels for y-axis of plots

    Saves:
        Plot file 'spinn_visualized_sample.png' showing:
        - Original vs predicted trajectories
        - Autocorrelation analysis for raw data, drift and diffusion
    """

    ## plot of models set [coordinate, drift and diffusion]
    rows = len(X_sets)
    fig, axes = plt.subplots(rows, 2, figsize=(15, 20))  # Adjusted figsize for 3x2 layout
    # Create a figure with 3x2 subplots
    for idx, (X, model) in enumerate(zip(X_sets, model_sets)):
        if isinstance(X, np.ndarray):
            train_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))
        else:
            train_datasets = X

        train_loader = DataLoader(dataset=train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=False)

        for t, X in enumerate(train_loader):
            if t > min(config.sample_nums - 1, len(train_loader) - 2):  # drop_last batch
                break
            x = X[0].permute(1, 0, 2).to(config.device)
            _, output = model(x)
            x = x[config.encoder_length:]

            axes[idx, 0].plot(x[1:, 1, 0].data.cpu().numpy(), label=f'raw{t}')
            axes[idx, 0].plot(output[:, 1, 0].data.cpu().numpy(), label=f'pred{t}')
            axes[idx, 0].set_title(f'{y_labels[idx]} value')
            axes[idx, 0].set_ylabel(y_labels[idx])
            axes[idx, 0].set_xlabel('npoints')
            axes[idx, 0].legend()

            # autocorrelation
            sigma_raw = (x[1:, :, 0].data.cpu().numpy() - x[:-1, :, 0].data.cpu().numpy())
            ac = []
            for i in range(sigma_raw.shape[1]):
                ac.append(np.correlate(sigma_raw[:, i], sigma_raw[:, i], mode='full'))
            # ac_raw = np.correlate(sigma, sigma, mode='full')
            ac_raw = np.mean(np.array(ac), axis=0)

            ac_init0 = ac_raw[ac_raw.size // 2]  # the initial value in zero lags to normalized
            axes[idx, 1].plot(ac_raw[ac_raw.size // 2:] / ac_init0, label=f'raw{t}')

            # velocity
            v_raw = (output[:-1, :, 0].data.cpu().numpy() - x[:-1, :, 0].data.cpu().numpy())
            ac = []
            for i in range(v_raw.shape[1]):
                ac.append(np.correlate(v_raw[:, i], v_raw[:, i], mode='full'))
            # ac_raw = np.correlate(sigma, sigma, mode='full')
            ac_raw = np.mean(np.array(ac), axis=0)
            axes[idx, 1].plot(ac_raw[ac_raw.size // 2:] / ac_init0, label=f'drift{t}')

            # Dt
            sigma = (x[1:, :, 0].data.cpu().numpy() - output[:-1, :, 0].data.cpu().numpy())

            ac = []
            for i in range(sigma.shape[1]):
                ac.append(np.correlate(sigma[:, i], sigma[:, i], mode='full'))
            # ac_raw = np.correlate(sigma, sigma, mode='full')

            ac_raw = np.mean(np.array(ac), axis=0)
            axes[idx, 1].plot(ac_raw[ac_raw.size // 2:] / ac_init0, label=f'diffusion{t}')
            axes[idx, 1].set_ylabel('R')
            axes[idx, 1].set_xlabel('Lag')
            axes[idx, 1].legend()
    plt.tight_layout()
    plt.savefig(f'spinn_visualized_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization for samples saved as 'spinn_visualized_sample.png'.")


def batch_reconstruct(x, model, config):
    """
    Reconstruct predictions and calculate residuals for a batch of data

    Args:
        x: Input tensor
        model: Trained model
        config: Configuration object containing parameters

    Returns:
        y0: First prediction trajectory (numpy array)
        y1: Second prediction trajectory (numpy array)
        s: Residuals/sigma values (numpy array)
    """
    x = Variable(x.type(torch.float), requires_grad=False)
    with torch.no_grad():
        y0, y1 = model(x)
    sigma = x[config.encoder_length + 1:] - y1[:-1]
    sigma = sigma.cpu().data.numpy()
    s = np.concatenate([sigma, np.zeros((1, sigma.shape[1], sigma.shape[2]))], axis=0)
    return y0.cpu().data.numpy(), y1.cpu().data.numpy(), s


def transform(test_datasets, model, config):
    """
    Transform test dataset through model and return predictions and residuals

    Args:
        test_datasets: Test dataset
        model: Trained model
        config: Configuration object containing parameters

    Returns:
        Concatenated arrays across all batches:
        - y0: First prediction trajectories (L x B x D)
        - y1: Second prediction trajectories (L x B x D)
        - sigma: Residuals (L x B x D)
        - x: Original input data (L x B x D)

    Note: L = sequence length, B = batch size, D = feature dimension
    """
    model.eval()
    # test_datasets = TensorDataset(torch.from_numpy(X).type(dtype=torch.float))
    test_loader = DataLoader(dataset=test_datasets,
                             batch_size=config.batch_size,
                             shuffle=False)  # Don't shuffle for test_loader

    x_run, y0_run, y1_run, sigma_run = [], [], [], []
    for t, X in enumerate(test_loader):
        x = X[0].permute(1, 0, 2).to(config.device)
        y0, y1, sigma = batch_reconstruct(x, model, config)
        x_run.append(x.cpu().data.numpy())
        y0_run.append(y0)
        y1_run.append(y1)
        sigma_run.append(sigma)

    return np.concatenate(y0_run, axis=1), np.concatenate(y1_run, axis=1), np.concatenate(sigma_run,
                                                                                          axis=1), np.concatenate(x_run,
                                                                                                                  axis=1)


def getlabel(test_datasets, config):
    """
    Extract labels from test dataset in order

    Args:
        test_datasets: Test dataset containing input and labels
        config: Configuration object containing parameters

    Returns:
        labels: Concatenated label array across all batches

    Note: Maintains original data order by not shuffling the DataLoader
    """
    test_loader = DataLoader(dataset=test_datasets,
                             batch_size=config.batch_size,
                             shuffle=False)  # Don't shuffle for test_loader

    label_run = []
    for t, X in enumerate(test_loader):
        label_run.append(X[1].cpu().data.numpy())

    return np.concatenate(label_run, axis=0)


def transformToMemmap(datasets, model, config, tags_s='', is_train=True, model_clash_pos=None):
    """
    Transform dataset predictions to memory-mapped files

    Args:
        datasets: Input datasets
        model: Trained model
        config: Configuration object
        tags_s: String tag for filenames
        is_train: Boolean indicating if in training mode
        model_clash_pos: Optional model name override

    Returns:
        Tuple of paths to memory-mapped files for y0, y1, and sigma predictions
    """
    model.eval()
    dataloader = DataLoader(dataset=datasets,
                            batch_size=config.batch_size,
                            shuffle=False)  # Don't shuffle for test_loader
    L, _, D = get_datasets_shape(datasets, config)

    if model_clash_pos is not None:
        model_clash_pos = model_clash_pos
    else:
        model_clash_pos = config.save_name

    if len(tags_s) > 0:
        tags_s = '_' + tags_s

    fp_y0_run_path = os.path.join(config.dload, model_clash_pos, 'memmap_clash', 'y0_run' + tags_s + '.npy')
    fp_y1_run_path = os.path.join(config.dload, model_clash_pos, 'memmap_clash', 'y1_run' + tags_s + '.npy')
    fp_sigma_run_path = os.path.join(config.dload, model_clash_pos, 'memmap_clash', 'sigma_run' + tags_s + '.npy')
    # fp_x_run_path = os.path.join(config.dload, config.save_name, 'memmap_clash', 'x_run' + tags_s + '.npy')
    #
    if is_train:
        if os.path.exists(os.path.join(config.dload, model_clash_pos, 'memmap_clash')):
            pass
        else:
            os.makedirs(os.path.join(config.dload, model_clash_pos, 'memmap_clash'))
        #
        fp_y0_run = np.memmap(fp_y0_run_path, dtype='float32', mode='w+', shape=(len(datasets), L, D))
        fp_y1_run = np.memmap(fp_y1_run_path, dtype='float32', mode='w+', shape=(len(datasets), L, D))

        fp_sigma_run = np.memmap(fp_sigma_run_path, dtype='float32', mode='w+', shape=(len(datasets), L, D))
        # fp_x_run = np.memmap(fp_x_run_path, dtype='float32', mode='w+', shape=(len(datasets),L + config.encoder_length, D))
        # np.save(shape_run_path,np.array([L,len(datasets),D]))

        cnt = 0

        for t, X in enumerate(dataloader):
            x = X[0].permute(1, 0, 2).to(config.device)  # L x B x D
            if tags_s == 'reverse':
                x = torch.flip(x, dims=[0])

            cnt_add = x.shape[1]
            y0, y1, sigma = batch_reconstruct(x, model, config)
            # fp_x_run[cnt:cnt + cnt_add] = x.cpu().data.numpy().transpose(1, 0, 2)
            fp_y0_run[cnt:cnt + cnt_add] = y0.transpose(1, 0, 2)
            fp_y1_run[cnt:cnt + cnt_add] = y1.transpose(1, 0, 2)
            fp_sigma_run[cnt:cnt + cnt_add] = sigma.transpose(1, 0, 2)
            cnt += cnt_add

    # del fp_y0_run, fp_y1_run, fp_sigma_run
    return fp_y0_run_path, fp_y1_run_path, fp_sigma_run_path


def transformVelocityToMemmap(datasets, model, config, tags_s='', is_train=True):
    """
    Transform dataset through model and save results as memory-mapped files

    Args:
        datasets: Input dataset
        model: Trained model
        config: Configuration object
        tags_s: Optional tag string to append to filenames
        is_train: Boolean indicating if in training mode
        model_clash_pos: Optional model name for file paths

    Returns:
        Tuple of paths to saved memory-mapped files:
        (y0_path, y1_path, sigma_path)
    """
    model.eval()
    dataloader = DataLoader(dataset=datasets,
                            batch_size=config.batch_size,
                            shuffle=False)  # Don't shuffle for test_loader
    L, _, D = get_datasets_shape(datasets, config)
    if len(tags_s) > 0:
        tags_s = '_' + tags_s

    fp_velocity_run_path = os.path.join(config.dload, config.save_name, 'memmap_clash', 'v_run' + tags_s + '.npy')
    #
    if is_train:
        if os.path.exists(os.path.join(config.dload, config.save_name, 'memmap_clash')):
            pass
        else:
            os.makedirs(os.path.join(config.dload, config.save_name, 'memmap_clash'))
        #

        fp_velocity_run = np.memmap(fp_velocity_run_path, dtype='float32', mode='w+', shape=(len(datasets), L, D))

        cnt = 0

        for t, X in enumerate(dataloader):
            x = X[0].permute(1, 0, 2).to(config.device)  # L x B x D
            if tags_s == 'reverse':
                x = -torch.flip(x, dims=[0])

            cnt_add = x.shape[1]  # batch
            _, y1, _ = batch_reconstruct(x, model, config)
            v = y1 - x[config.encoder_length:].cpu().data.numpy()
            # fp_x_run[cnt:cnt + cnt_add] = x.cpu().data.numpy().transpose(1, 0, 2)
            fp_velocity_run[cnt:cnt + cnt_add] = v.transpose(1, 0, 2)

            cnt += cnt_add

    return fp_velocity_run_path


def batch_raw_gen(h, s, my_gen_model, adding_jumping=None):
    """
    Generate tracks using a pre-trained model for a batch of inputs

    Args:
        h: Hidden state/latent representation
        s: Noise/residual input
        my_gen_model: Generative model
        adding_jumping: Optional parameter for jump conditions

    Returns:
        Tuple of numpy arrays:
        - y0: First prediction trajectory
        - y1: Second prediction trajectory
        - x0: Generated samples
    """
    my_gen_model.eval()
    with torch.no_grad():
        y0, y1, x0 = my_gen_model(h, s, adding_jumping)

    return y0.cpu().data.numpy(), y1.cpu().data.numpy(), x0.cpu().data.numpy()


def generate_sigma_sets(S0, S, config):
    """
    Generate squared sigma/residual datasets with train/val split

    Args:
        S0: head sigma values
        S: regular sigma values
        config: Configuration object containing parameters

    Returns:
        X_train: Training dataset
        X_val: Validation dataset
        data: Complete processed dataset
    """
    data = S  # reject_outliers(S,m=5)
    # sc = np.std(data, axis=1)[:,:,np.newaxis].repeat(config.encoder_length,axis=2).transpose(0,2,1)
    # data = np.concatenate([sc * np.random.randn(data.shape[0],config.encoder_length, data.shape[2]), data], axis=1)
    data0 = np.zeros((data.shape[0], config.encoder_length, data.shape[2]))
    ## left alignment
    minlen_sz = min([config.encoder_length, S0.shape[1]])
    data0[:, -minlen_sz:, :] = S0[:, -minlen_sz:, :]
    # data = data **2
    data = np.concatenate([data0, data], axis=1)
    data = data ** 2
    X_train, X_val, y_train, y_val = train_test_split(data, np.zeros_like(data),
                                                      test_size=0.25, random_state=config.seed)
    return X_train, X_val, data


def generate_sigma_sets_torch(train_dataset, model, config, is_train=True):
    """
    Generate sigma datasets using memory-mapped files for forward and reverse predictions

    Args:
        train_dataset: Training dataset
        model: Trained model
        config: Configuration object
        is_train: Boolean indicating if in training mode

    Returns:
        DatasetLargeSigma object containing forward and reverse sigma predictions
    """
    L, B, D = get_datasets_shape(train_dataset, config=config)
    # if is_train:
    # else:
    y0_run_path, y1_run_path, sigma_run_path = transformToMemmap(train_dataset, model, config,
                                                                 is_train=is_train)  # padding the header, also X
    y0_run_path_reverse, y1_run_path_reverse, sigma_run_path_reverse = transformToMemmap(train_dataset, model, config,
                                                                                         tags_s='reverse',
                                                                                         is_train=is_train)  # padding the header, also X
    s = DatasetLargeSigma(S=sigma_run_path,
                          S0=sigma_run_path_reverse,
                          s_shape=(B, L, D), config=config
                          )
    return s


def generate_velocity_sets_torch(train_dataset, model, config, is_train=True):
    """
    Generate velocity datasets using memory-mapped files for forward and reverse predictions

    Args:
        train_dataset: Training dataset
        model: Trained model
        config: Configuration object
        is_train: Boolean indicating if in training mode

    Returns:
        DatasetLargeVelocity object containing forward and reverse velocity predictions
    """
    L, B, D = get_datasets_shape(train_dataset, config=config)
    velocity_run_path = transformVelocityToMemmap(train_dataset, model, config,
                                                  is_train=is_train)  # padding the header, also X
    velocity_run_path_reverse = transformVelocityToMemmap(train_dataset, model, config,
                                                          tags_s='reverse',
                                                          is_train=is_train)  # padding the header, also X
    v = DatasetLargeVelocity(V=velocity_run_path,
                             V0=velocity_run_path_reverse,
                             v_shape=(B, L, D), config=config
                             )

    return v


def generate_sigma_predicted_torch(gen_dataset, model, config):
    """
    Generate predicted sigma dataset using memory-mapped files

    Args:
        gen_dataset: Generator dataset
        model: Trained model
        config: Configuration object

    Returns:
        DatasetLargeSigmaPredict object containing predicted sigma values
    """
    L, B, D = get_datasets_shape(gen_dataset, config=config)

    _, sigma_predict_path, _ = transformToMemmap(gen_dataset,
                                                 model,
                                                 config,
                                                 is_train=True,
                                                 model_clash_pos=config.save_name + '-gen')  # reading the saved path

    s = DatasetLargeSigmaPredict(S=sigma_predict_path, s_shape=(B, L, D))

    return s


def eval_sigma_sets_torch(train_dataset, model, config, is_filtering=None):
    """
    Generate and evaluate sigma datasets with optional filtering

    Args:
        train_dataset: Training dataset
        model: Trained model
        config: Configuration object
        is_filtering: Boolean indicating if filtering should be applied

    Returns:
        Dataset object containing sigma values and optionally filtered predictions
    """
    L, B, D = get_datasets_shape(train_dataset, config=config)
    y0_run_path, y1_run_path, sigma_run_path = transformToMemmap(train_dataset, model,
                                                                 config)  # padding the header, also X
    y0_run_path_reverse, y1_run_path_reverse, sigma_run_path_reverse = transformToMemmap(train_dataset, model, config,
                                                                                         tags_s='reverse')  # padding the header, also X
    s = DatasetLargeSigma(S=sigma_run_path,
                          S0=sigma_run_path_reverse,
                          s_shape=(B, L, D), config=config
                          )
    if is_filtering is not None:
        s = DatasetFilterLargeSigma(Y0=y0_run_path, Y0_reverse=y0_run_path_reverse,
                                    Y1=y1_run_path, Y1_reverse=y1_run_path_reverse,
                                    s_shape=(B, L, D), config=config
                                    )

    return s


def train_sigma(train_dataset, model, config, is_train=False, beta_obs=0., test_dataset=None, is_var_head=True):
    """
    Train or load a sigma prediction model

    Args:
        train_dataset: Training dataset
        model: Base model
        config: Configuration object
        is_train: Boolean indicating if model should be trained
        beta_obs: Beta observation parameter
        test_dataset: Optional test dataset
        is_var_head: Boolean indicating if using variance head

    Returns:
        Tuple of (sigma dataset, sigma prediction model)
    """
    # y0, y1, s = transform(X, model, config)
    # _, _, s0 = transform(np.flip(X,axis=1).copy(), model, config)
    strain = generate_sigma_sets_torch(train_dataset, model, config)
    if test_dataset is not None:
        sval = generate_sigma_sets_torch(test_dataset, model, config)

    name_raw = config.save_name
    model_name_raw = config.model_name
    beta_observe_raw = config.beta_observe
    config.save_name = config.save_name + '-var'
    config.model_name = config.save_name + '.pth'
    config.beta_observe = beta_obs
    # strain, _, s = generate_sigma_sets(s0, s, config)

    if hasattr(config, 'is_var_head'):
        is_var_head = config.is_var_head

    if is_train:
        # model_s = train(train_datasets=strain, config=config, L2 = True, shuffle=False)
        model_s = train_Dt(train_datasets=strain, config=config, L2=True, shuffle=False, is_var_head=is_var_head)
    else:
        model_s = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                      h_dim=config.h_dim, default_enc_nn=config.default_enc_nn,
                                      hidden_dim=config.hidden_size, layers=config.layers,
                                      d_model=config.d_model, n_head=config.n_heads, layers_enc=config.enc_layers,
                                      beta_obs=config.beta_observe, beta_ac=config.beta_ac,
                                      beta_conti=config.beta_conti,
                                      beta_weight=config.beta_weight,
                                      device=config.device,
                                      is_var_head=is_var_head).to(config.device)
        load(model_s, config.dload, config.model_name)

    # s0_path, s1_path, _ = transform(s, model_s, config)
    transformToMemmap(strain, model_s, config, is_train=is_train)

    config.save_name_var = config.save_name
    config.model_name_var = config.model_name
    config.save_name = name_raw
    config.model_name = model_name_raw
    config.beta_observe = beta_observe_raw
    # return np.sqrt(np.abs(s[:,config.encoder_length:,:])+1e-16), np.sqrt(np.abs(s1)+1e-16).transpose(1,0,2)

    return strain, model_s


def train_velocity(train_dataset, model, config, is_train=False, test_dataset=None):
    """
    Train or load velocity prediction model

    Args:
        train_dataset: Training dataset
        model: Base model
        config: Configuration object
        is_train: Boolean indicating if model should be trained
        test_dataset: Optional test dataset

    Returns:
        Tuple of (velocity dataset, velocity prediction model)
    """
    vtrain = generate_velocity_sets_torch(train_dataset, model, config)
    if test_dataset is not None:
        vval = generate_velocity_sets_torch(test_dataset, model, config)

    name_raw = config.save_name
    model_name_raw = config.model_name
    beta_observe_raw = config.beta_observe
    config.save_name = config.save_name + '-vel'
    config.model_name = config.save_name + '.pth'
    # strain, _, s = generate_sigma_sets(s0, s, config)
    if is_train:
        model_v = train(train_datasets=vtrain, config=config, L2=True, shuffle=False)
    else:
        model_v = JumpSDEsTransformer(in_dim=config.in_dim, out_dim=config.in_dim, encoding_len=config.encoder_length,
                                      h_dim=config.h_dim, default_enc_nn=config.default_enc_nn,
                                      hidden_dim=config.hidden_size, layers=config.layers,
                                      d_model=config.d_model, n_head=config.n_heads, layers_enc=config.enc_layers,
                                      beta_obs=config.beta_observe, beta_ac=config.beta_ac,
                                      beta_conti=config.beta_conti,
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


def traj_generate(X, model, config):
    """
     Generate and visualize trajectories from the model generate new tracks.

     Args:
         X: Input data
         model: Trained model
         config: Configuration object
     """
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

        sigma = x_tail[1:, :, :] - output[:-1, :, :]
        sigma = torch.std(sigma, dim=0).squeeze().repeat(config.batch_size, 1, 1).permute(1, 0, 2)

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


def traj_multipe_transformer(X, S, model, config, num=1):
    """
    Generate multiple trajectory predictions using transformer model

    Args:
        X: Input data
        S: Sigma data
        model: Trained model
        config: Configuration object
        num: Number of trajectories to generate

    Returns:
        Tuple of (y0, y1, x0) predictions stacked across multiple runs
    """
    model.eval()
    min_sz = min(X.shape[0], S.shape[0])
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

        y0_num.append(np.concatenate(y0_run, axis=1)[:, :, :, np.newaxis])
        y1_num.append(np.concatenate(y1_run, axis=1)[:, :, :, np.newaxis])
        x0_num.append(np.concatenate(x0_run, axis=1)[:, :, :, np.newaxis])

    return (np.concatenate(y0_num, axis=-1), np.concatenate(y1_num, axis=-1), np.concatenate(x0_num, axis=-1))


def traj_multipe_transformer_memmap(datasets, model, config, num=10, gen_name=None):
    """
    Generate multiple trajectories and save to memory-mapped files

    Args:
        datasets: Input datasets
        model: Trained model
        config: Configuration object
        num: Number of trajectories to generate
        gen_name: Optional name for generated files

    Returns:
        Tuple of file paths for generated x0, y0, y1 trajectories
    """
    logger_name = config.dload + '/' + config.save_name + '.log'
    logger = MyLogger.__call__(logger_name).get_logger()
    logger.info('starting regenerating with config:')
    logger.info(config)

    model.eval()
    s_train, model_s = train_sigma(datasets, model, config, beta_obs=0.0, is_train=False)
    s_dataset = generate_sigma_predicted_torch(s_train, model=model_s, config=config)
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

    np.save(os.path.join(config.dload, gen_save_name + '-gen', 'memmap_gen', 'shape_gen.npy'),
            np.asarray([len(datasets), L, D, num]))
    seeds_set = torch.randint(max_integer_const, (len(x_dataloader), num))
    for i in range(num):
        cnt = 0
        cnt_tracks = 0
        for item_x, item_s in zip(x_dataloader, cycle(s_dataloader)):
            x = item_x[0].permute(1, 0, 2).to(config.device)
            s = item_s[0].permute(1, 0, 2).to(config.device)

            # const 1.96 of the normal, if y(1,t) - y(1,t-1) > const * sigma_std

            _, y1, = model(x)
            v = x[config.encoder_length:] - y1
            idv = ((torch.abs(v) - outliers_const * abs(s)) > 1e-6)  # outliers
            # v1^2 + v2^2 = ... sigma1^2 + sigma2^2 + ... +
            s[idv] = torch.abs(v[idv])  # for outliers jumping
            torch.manual_seed(seeds_set[cnt_tracks, i])
            y0_gen, y1_gen, x0_gen = batch_raw_gen(x, s, my_gen)

            cnt_add = x.shape[1]  # batch

            fp_x0_gen[cnt:cnt + cnt_add, :, :, i] = x0_gen.transpose(1, 0, 2)
            fp_y0_gen[cnt:cnt + cnt_add, :, :, i] = y0_gen.transpose(1, 0, 2)
            fp_y1_gen[cnt:cnt + cnt_add, :, :, i] = y1_gen.transpose(1, 0, 2)

            cnt += cnt_add
            cnt_tracks += 1

        logger.info('group generated: {:d}/{:d}'
                    .format(i, num))

    return fp_x0_gen_path, fp_y0_gen_path, fp_y1_gen_path


def clear_clash(config):
    """
    Clear memory-mapped files from various folders

    Args:
        config: Configuration object containing save paths and names
    """
    def clear_single_folder(file_path, file_name, tags='memmap_clash'):
        """
        Remove all files from a specific folder

        Args:
            file_path: Base path
            file_name: Name of folder
            tags: Subfolder name, defaults to 'memmap_clash'
        """
        file_sets_raw = os.listdir(os.path.join(file_path, file_name, tags))
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
