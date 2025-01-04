"""
SPINN - Stochastic Particle-Informed Neural Network
Authors: Yongyu Zhang, et al.

This script trains a neural network and enables the decomposition of
particle trajectories into deterministic drift and stochastic diffusion components
using self-supervised learning.
"""

import argparse, glob
from TrainJumpSDEsTransform import *
from utils.csv2npy import process_and_save_tracks, process_and_save_tracks_short
from utils.export_results import *

# Define command-line arguments
parser = argparse.ArgumentParser()

# Training and model parameters
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs (default: %(default)s)')
parser.add_argument('--epochs_VDt', type=int, default=50,
                    help='Number of training epochs for drift and variance (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=20,
                    help='Batch size for training (default: %(default)s)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Init learning rate for the optimizer (default: %(default)s)')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed (default: %(default)s)')
parser.add_argument('--clip_grad', type=float, default=5.0,
                    help='Maximum gradient norm for gradient clipping (default: %(default)s)')
parser.add_argument('--default_enc_nn', type=tuple, default=((128, 'gelu'), (128, 'gelu')),
                    help='Tuple of Res-FFNN blocks, each block defined as (units, activation) (default: %(default)s)')  # 128
parser.add_argument('--hidden_size', type=int, default=128,
                    help='Number of RNN hidden size (default: %(default)s)')
parser.add_argument('--layers', type=int, default=2,
                    help='Number of GRU layers (default: %(default)s)')
parser.add_argument('--h_dim', type=int, default=64,
                    help='Number of embedding size (default: %(default)s)')
parser.add_argument('--d_model', type=int, default=128,
                    help='Hidden dimension of the Transformer (default: %(default)s)')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention heads in the Transformer (default: %(default)s)')
parser.add_argument('--enc_layers', type=int, default=3,
                    help='Number of encoder layers in the Transformer (default: %(default)s)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='Dropout rate (default: %(default)s)')
parser.add_argument('--disable_var_head',
                    action='store_false', dest='is_var_head',
                    help="Disable the softplus head for predicting variance time series")
parser.add_argument('--in_dim', type=int, default=2,
                    help='Input dimension of the time series (default: %(default)s)')
parser.add_argument('--out_dim', type=int, default=2,
                    help='Output dimension of the time series (default: %(default)s)')
parser.add_argument('--pred_length', type=int, default=50,
                    help='Prediction length of input time series (default: %(default)s)')
parser.add_argument('--encoder_length', type=int, default=50,
                    help='Encoded header length of the input time series, '
                         'should be less than the prediction length (default: %(default)s)')
parser.add_argument('--scale', type=float, default=0.375,  # for lr 0.375, 0.25
                    help='Scaling factor for learning rate schedule (default: %(default)s)')
parser.add_argument('--warmup_steps', type=int, default=2000,
                    help='Number of warmup steps for learning rate schedule (default: %(default)s)')
parser.add_argument('--sample_nums', type=int, default=0,
                    help='Number of samples for drawing evaluated figures (default: %(default)s)')
parser.add_argument('--sample_rates', type=float, default=1.0,
                    help='Sampling rate for large datasets, range from 0 to 1 (default: %(default)s)')
parser.add_argument('--beta_ac', type=float, default=0.5,
                    help='Autocorrelation weight (default: %(default)s)')
parser.add_argument('--beta_weight', type=float, default=0.5,
                    help='The jump term weight in SDEs loss (default: %(default)s)')
parser.add_argument('--beta_observe', type=float, default=0.0,
                    help='Total Variation (TV) regularization (default: %(default)s)')
parser.add_argument('--beta_conti', type=float, default=0.,
                    help='Continuity parameter of the time series (default: %(default)s)')
parser.add_argument('--RESUME', type=bool, default=False,
                    help='Resume training from a checkpoint (default: %(default)s)')

# Dataset loading and saving flags
parser.add_argument('--noise_adding', type=float, default=0.0,
                    help='Additional level of noise added to the dataset (default: %(default)s)')
parser.add_argument('--data_format', type=str, default='csv', choices=['npy', 'csv', 'csv_short'],
                    help='File format of the dataset in the data folder. Choices: [npy, csv, csv_short]. '
                         'The "csv_short" format is a datapool approach '
                         'designed for handling short tracks (default: %(default)s)')
parser.add_argument('--scale_units', type=float, default=1.,
                    help='Scale units for spatial coordinates for visualization (default: %(default)s)')
parser.add_argument('--dload', type=str, default='./model_dir',
                    help='Directory to save model checkpoints and other outputs (default: %(default)s)')
parser.add_argument('--save_micro', type=str,
                    choices=['sample', 'all', 'no-saving'],
                    default='all',
                    help='The mode for saving microdynamics data, choices: [sample, all, no-saving] (default: %(default)s)')
parser.add_argument('--save_csv', action='store_true',
                    help='Save csv files for microdynamics data')
parser.add_argument('--gen_tracks', action='store_true',
                    help='Generate tracks from learned data')
parser.add_argument('--gen_tracks_num', type=int, default=50,
                    help='Number of generated tracks (default: %(default)s)')
parser.add_argument('--filename', type=str, default='inputData',
                    help='Name of the folder containing the tracks (default: %(default)s)')
parser.add_argument('--disable_training', action='store_false',
                    dest='is_train',
                    help='Disable the training process.')

# Parse the arguments
configs = parser.parse_args()

if __name__ == '__main__':
    # Get training mode from configuration
    is_train = configs.is_train

    # Define tags that should be excluded when counting model versions
    filename_tags = ['var', 'vel']
    if configs.filename is not None:
        # Searching for the next training number
        saved_name = '-'.join([configs.filename, 'noise', str(configs.noise_adding)])
        file_sets = glob.glob(os.path.join(configs.dload, saved_name + "*.pth"))
        cnt_minus = 0
        for tag_f in filename_tags:
            for f in file_sets:
                if tag_f in f:
                    cnt_minus += 1
        cnt = max(len(file_sets) - cnt_minus, 0)
        if is_train and (not configs.RESUME):
            save_name = '-'.join([saved_name, str(cnt + 1)])
        else:
            save_name = '-'.join([saved_name, str(cnt)])
    configs.save_name = save_name
    configs.model_name = save_name + '.pth'
    configs.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # When using csv, process and save tracks based on the specified data format
    csv_save_short = None
    if configs.data_format == 'csv':
        process_and_save_tracks(
            path=os.path.join('./data/', configs.filename),
            save_path=os.path.join('./data/', configs.filename, 'Npy'),
            seq_encoder=configs.pred_length,
            scale_units=configs.scale_units,
        )
    elif configs.data_format == 'csv_short':
        process_and_save_tracks_short(
            path=os.path.join('./data/', configs.filename),
            save_path=os.path.join('./data/', configs.filename, 'Npy'),
            seq_encoder=configs.pred_length,
            scale_units=configs.scale_units,
        )
        csv_save_short = (os.path.join('./data/', configs.filename), configs.scale_units)

    # Generate training and test datasets
    train_dataset, test_dataset = generate_dataset_torch(
        merge_datasets(filepath='./data',
                       filenames=[configs.filename + '_annotation.txt'],
                       config=configs, sample_rate=configs.sample_rates), seed=configs.seed)

    # Train the model
    if is_train:
        model = train(train_datasets=train_dataset, config=configs, test_datasets=test_dataset)

        # Update configurations for variance and drift training
        configs.epochs = configs.epochs_VDt
        configs.beta_ac = 0.5
        configs.beta_weight = 0.5

        # Train the diffusion component
        train_sigma(train_dataset,
                    model,
                    configs,
                    is_train=is_train,
                    )

        # Train the drift velocity component
        train_velocity(train_dataset,
                       model,
                       configs,
                       is_train=is_train)

    # Load trained model if not in training mode
    else:
        model = JumpSDEsTransformer(in_dim=configs.in_dim, out_dim=configs.in_dim, encoding_len=configs.encoder_length,
                                    h_dim=configs.h_dim, default_enc_nn=configs.default_enc_nn,
                                    hidden_dim=configs.hidden_size, layers=configs.layers,
                                    d_model=configs.d_model, n_head=configs.n_heads, layers_enc=configs.enc_layers,
                                    beta_obs=configs.beta_observe, beta_conti=configs.beta_conti,
                                    beta_weight=configs.beta_weight,
                                    device=configs.device).to(configs.device)
        load(model, configs.dload, configs.model_name)

    # Save microdynamics data
    is_train = False
    if configs.save_micro == 'all':
        # Load full annotation file
        full_filenames = configs.filename + '_annotation_visual.txt'
        df = pd.read_csv(os.path.join('./data', full_filenames), delimiter=',', header=None)

        # Create directory for saving files
        os.makedirs(os.path.join(configs.dload, configs.filename), exist_ok=True)

        # Process each row in the dataset
        for i in range(df.shape[0]):
            file_i_name = df.iloc[i, 0].split('/')[-1].split('.')[0]

            # Generate test dataset
            _, test_dataset = generate_dataset_torch(
                DatasetTracks(df.iloc[i, 0], configs, sample_rate=1.0),
                seed=configs.seed, train_prob=1e-6)

            # Extract and save dynamics
            y0, y1, sigma_run, x_raw = transform(test_dataset, model, configs)

            # Diffusion
            s_test_dataset, model_s = train_sigma(test_dataset,
                                                  model,
                                                  configs,
                                                  is_train=is_train)
            _, s1, _, s_raw = transform(s_test_dataset, model_s, configs)

            # Velocity
            v_test_dataset, model_v = train_velocity(test_dataset,
                                                     model,
                                                     configs,
                                                     is_train=is_train)
            _, v1, _, v_raw = transform(v_test_dataset, model_v, configs)

            # Save microdynamics data to a compressed file
            micro_save_file = os.path.join(configs.dload,
                                           configs.filename,
                                           configs.save_name + '_' + file_i_name + '_micro.npz')
            np.savez_compressed(micro_save_file,
                                x_raw=x_raw, s_raw=s_raw, v_raw=v_raw,
                                y0=y0, y1=y1, sigma_run=sigma_run,
                                s1=s1, v1=v1, sample_indices=test_dataset.indices)
            print(f'evaluate {file_i_name} all finished')

    elif configs.save_micro == 'sample':
        # Transform and save dynamics for a sample
        y0, y1, sigma_run, x_raw = transform(test_dataset, model, configs)
        xlabel = getlabel(test_dataset, configs)

        # Diffusion
        s_test_dataset, model_s = train_sigma(test_dataset, model, configs,
                                              is_train=is_train)
        _, s1, _, s_raw = transform(s_test_dataset, model_s, configs)

        # Velocity
        v_test_dataset, model_v = train_velocity(test_dataset, model, configs, is_train=is_train)
        _, v1, _, v_raw = transform(v_test_dataset, model_v, configs)

        # Save microdynamics data to a compressed file
        micro_save_file = os.path.join(configs.dload,
                                       configs.save_name + '_micro.npz')
        np.savez_compressed(micro_save_file,
                            x_raw=x_raw, s_raw=s_raw, v_raw=v_raw,
                            y0=y0, y1=y1, sigma_run=sigma_run,
                            s1=s1, v1=v1,
                            xlabel=xlabel,
                            sample_indices=test_dataset.indices)
        print(f'evaluate {configs.save_name} sample finished')

    # Save microdynamics results to CSV if required
    if configs.save_csv and configs.save_micro == 'all':
        # Load annotation file and concate for head-tracks and tracks
        full_filenames = configs.filename + '_annotation.txt'
        df = pd.read_csv(os.path.join('./data', full_filenames), delimiter=',', header=None)

        # Process each row in the dataset
        for i in range(df.shape[0]):
            file_i_name = df.iloc[i, 0].split('/')[-1].split('.')[0].rstrip('-tracks')
            folder_normalized = os.path.join(('/').join(df.iloc[i, 0].split('/')[:-1]), 'normalized')
            rslt = load_microdynamics_tracks(tracks_each_name=file_i_name,
                                             model_name=configs.save_name,
                                             model_dir=os.path.join(configs.dload, configs.filename),
                                             folder_normalized=folder_normalized,
                                             seq_len=configs.pred_length, )

            # Determine tracks name for CSV
            if csv_save_short is not None:
                tracks_each_name = file_i_name.rstrip('-tracks-preprocessed-tracks')
            else:
                tracks_each_name = file_i_name.rstrip('-tracks')

            # Save microdynamics results to CSV
            microdynamics_results_to_csv(rslt,
                                         tracks_each_name=tracks_each_name,
                                         save_path=os.path.join(configs.dload, configs.filename),
                                         short=csv_save_short)

    # Generate tracks if required and RECOMMENDED due to high time & disk space usage
    if configs.gen_tracks and configs.save_micro == 'all':
        # Load full annotation file
        full_filenames = configs.filename + '_annotation_visual.txt'
        df = pd.read_csv(os.path.join('./data', full_filenames), delimiter=',', header=None)

        # Process each row in the dataset
        for i in range(df.shape[0]):
            file_i_name = df.iloc[i, 0].split('/')[-1].split('.')[0]

            # Generate test dataset
            _, test_dataset = generate_dataset_torch(
                DatasetTracks(df.iloc[i, 0], configs, sample_rate=1.0),
                seed=configs.seed, train_prob=1e-6)

            # Generate and save multiple transformer memmaps
            traj_multipe_transformer_memmap(test_dataset,
                                            model,
                                            configs,
                                            num=configs.gen_tracks_num,
                                            gen_name=os.path.join(configs.filename,
                                                                  configs.save_name + '_' + file_i_name))

    elif configs.gen_tracks and configs.save_micro == 'sample':
        # Generate and save multiple transformer memmaps
        traj_multipe_transformer_memmap(test_dataset,
                                        model,
                                        configs,
                                        num=configs.gen_tracks_num)

    # visualization
    if configs.sample_nums > 0:
        _, test_dataset = generate_dataset_torch(
            merge_datasets(filepath='./data',
                           filenames=[configs.filename + '_annotation.txt'],
                           config=configs, sample_rate=configs.sample_rates), seed=configs.seed)

        # Velocity
        v_test_dataset, model_v = train_velocity(test_dataset,
                                                 model,
                                                 configs,
                                                 is_train=is_train)

        # Diffusion
        s_test_dataset, model_s = train_sigma(test_dataset,
                                              model,
                                              configs,
                                              is_train=is_train)

        # plot evaluate samples for x dim
        evaluate_draw_savefig([test_dataset, v_test_dataset,s_test_dataset ],
                              [model, model_v, model_s],
                              configs, ['x', 'v', 'var'])

    # Clear any memory clashes when using memory mapping for large files.
    clear_clash(configs)

    # The SPINN algorithm execution is now complete.

