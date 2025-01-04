import argparse
import numpy as np
import pandas as pd
import os
import numpy.lib.stride_tricks as nps


def prepare_trajectory_segments(filename, scale_units=1., window_size=100, steps=100, dim=2, is_head=False):
    """
    Load and preprocess trajectory data from a CSV file with sliding window sampling.

    Args:
        filename (str): Path to CSV file containing trajectory data
        scale_units (float): Scaling factor for spatial coordinates (default: 1.0), where the unit = 5.86 / 40 for 40x objective
        window_size (int): Size of sliding window for trajectory segments (default: 100)
        steps (int): Step size between sampled windows for non-overlap computing, better equals to window_size (default: 100)
        dim (int): tracks dimensions (default: 2)
        is_head (bool): If True, pad beginning of trajectories with initial position (default: False)

    Returns:
        tuple: (trajectory_coords, timestamps)
            - trajectory_coords: Array of shape (B, T, D) containing spatial coordinates dimension
            - timestamps: Array of shape (B, T) containing corresponding timestamps
    """

    def sliding_window(X):
        x = X
        x = (nps.sliding_window_view(x, window_size, axis=0).
             transpose(0, 2, 1)[::steps])
        return x

    dataset = pd.read_csv(filename).values
    ## id t, x, y which have no the same shape of seq_len
    id_unique = np.unique(dataset[:, 0])
    tracks = []  # t, x, y
    for i in id_unique:
        tmp = dataset[dataset[:, 0] == i, 1:]
        tracks.append(tmp)

    if is_head:
        # keep the same length of head
        tracks_out = [sliding_window(np.concatenate([np.ones([steps // 2, dim + 1]) * tracks[i][0],
                                                     tracks[i][:-steps // 2]], axis=0)) for i in
                      range(len(tracks))]
    else:
        tracks_out = [sliding_window(tracks[i]) for i in range(len(tracks))]

    tracks_out = np.concatenate(tracks_out, axis=0)  # B x T x dim

    # tracks normalized um?, :
    tracks_out[:, :, -dim:] = tracks_out[:, :, -dim:] * scale_units
    return tracks_out[:, :, -dim:], tracks_out[:, :, 0]


def v_normalize_batch_encoder(x, encoder_len=50):
    """
    Normalize the trajectory data batch-wise.

    Parameters:
    - x (numpy.ndarray): Input trajectory data, shape (B, L, D).
    - encoder_len (int): Length of the encoder sequence. Default is 50.

    Returns:
    - x_normalized (numpy.ndarray): Normalized trajectory data, shape (B, L, D).
    - v_norm_batch (numpy.ndarray): Batch-wise normalization factors, shape (D,).
    """
    # B x L x D
    axis_dim = 1
    v0 = np.diff(x, axis=axis_dim)
    v = v0[:, encoder_len:, :]
    v_norm_batch = np.sqrt(np.mean(np.mean(v ** 2, axis=axis_dim), axis=0) + 1e-6)
    v_norm = v0 / v_norm_batch
    x_0 = np.array([0, 0]).reshape(-1, x.shape[2]).repeat(x.shape[0], axis=0).reshape(-1, 1, x.shape[2])
    return np.cumsum(np.concatenate([x_0, v_norm], axis=1), axis=1), v_norm_batch


def process_and_save_tracks(path, save_path, seq_encoder=50, scale_units=1.):
    """
    Process all CSV files in the specified directory and save the results as NPY and NPZ files.

    Parameters:
    - path (str): Path to the input data directory.
    - save_path (str): Path to the output data directory.
    - seq_encoder (int): Sequence length for the encoder. Default is 50.
    - scale_units (float): Sequence length for the encoder. Default is 50.
    """
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'normalized'), exist_ok=True)

    files = os.listdir(path)

    full_annotations = []

    for f in files:
        if f.endswith('.csv'):
            # Process trajectory data
            tracks_out, t = prepare_trajectory_segments(os.path.join(path, f),
                                                        window_size=2 * seq_encoder,
                                                        steps=2 * seq_encoder,
                                                        scale_units=scale_units)
            tracks_init = tracks_out[:, seq_encoder - 1]
            tracks_out, tracks_out_norm = v_normalize_batch_encoder(tracks_out)

            # Save raw data and normalized scale
            save_name = ('').join(f.split('.')[:-1])
            np.save(os.path.join(save_path, save_name + '-tracks.npy'), tracks_out)
            np.savez_compressed(os.path.join(save_path, 'normalized', save_name + '-tracks-Normalize-scale.npz'),
                                time=t,
                                tracks_init=tracks_init,
                                tracks_norm=tracks_out_norm)

            # Process head part of the trajectory data
            tracks_out_head, t_head = prepare_trajectory_segments(os.path.join(path, f),
                                                                  window_size=2 * seq_encoder,
                                                                  steps=2 * seq_encoder,
                                                                  scale_units=scale_units,
                                                                  is_head=True)
            tracks_init_head = tracks_out_head[:, seq_encoder - 1]
            tracks_out_head, tracks_out_head_norm = v_normalize_batch_encoder(tracks_out_head)

            # Save raw head data and normalized scale
            save_name_head = save_name + '-head'
            np.save(os.path.join(save_path, save_name_head + '-tracks.npy'), tracks_out_head)
            np.savez_compressed(os.path.join(save_path, 'normalized', save_name_head + '-tracks-Normalize-scale.npz'),
                                time=t_head,
                                tracks_init=tracks_init_head,
                                tracks_norm=tracks_out_head_norm)

            print(f'The file {f}.npy is saved at {save_path} with fix tags of -tracks')

            full_annotations.append(os.path.join(save_path, save_name + '-tracks.npy'))
            full_annotations.append(os.path.join(save_path, save_name_head + '-tracks.npy'))

    # write the annotation
    annotation_name = path.split('/')[-1] or path.split('/')[-2]
    write_annotation(full_annotations, annotation_name)


def concate_tracks_pool(trackersFolder, f):
    """
    Process tracking data from CSV file and save preprocessed arrays

    Args:
        trackersFolder: Folder containing tracking CSV files
        f: CSV filename containing tracking data with columns: id, t, x, y
    """

    # Read CSV file containing tracking data with columns: id, t, x, y
    df = pd.read_csv(os.path.join(trackersFolder, f))  # id, t, x, y
    # T x y with different length
    ids_tracks = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    dT = np.hstack([np.array(-1), np.diff(ids_tracks)])
    dT_h = dT != 0
    v = np.vstack([np.array([0, 0]).reshape(1, -1), np.diff(X[:, 1:], axis=0)])
    v[dT_h] = 0
    X_single = np.cumsum(v, axis=0)
    # the raw tracks split pos
    TX = np.hstack([np.arange(X_single.shape[0]).reshape(-1, 1), X_single])
    X0 = np.zeros_like(X)  # adding id, t , x, y
    X0[dT_h, 1:] = X[dT_h, 1:]
    X0[:, 0] = X[:, 0]
    val = np.array([0., 0.])
    for i in range(X0.shape[0]):
        if np.sum(np.abs(X0[i, 1:])) > 1e-3:
            val = X0[i, 1:]
        else:
            X0[i, 1:] = val

    filename = ('.').join(f.split('.')[:-1])
    save_name = os.path.join(trackersFolder, filename + '-tracks-preprocessed' + '.npy')
    save_name_X0 = os.path.join(trackersFolder, filename + '-X0-preprocessed' + '.npy')
    save_name_id = os.path.join(trackersFolder, filename + '-id-preprocessed' + '.npy')
    np.save(save_name, TX)
    np.save(save_name_X0, X0)
    np.save(save_name_id, ids_tracks)


def prepare_trajectory_segments_short(filename, scale_units=1., window_size=100, steps=100, dim=2, is_head=False):
    """
    Load and process track data from a file using a sliding window approach.

    Parameters:
    - filename: str, the path to the file containing the track data.
    - scale_units: float, the scaling factor to apply to the track data, where the unit = 5.86 / 40 for 40x objective
    - window_size: int, the size of the sliding window.
    - steps: int, the step size for the sliding window.
    - dim: int, the number of dimensions to keep from the track data.
    - is_head: bool, whether to process the data as a "head" track.

    Returns:
    - tracks_out_scaled: numpy array, the processed track data scaled by `scale_units`.
    - tracks_out_time: numpy array, the time component of the processed track data.
    """

    def sliding_window(X):
        x = X
        x = (nps.sliding_window_view(x, window_size, axis=0).
             transpose(0, 2, 1)[::steps])
        return x

    tracks = np.load(filename)
    if is_head:
        # keep the same length of head
        tracks_out = sliding_window(np.concatenate([np.ones([steps // 2, dim + 1]) * tracks[0, :],
                                                    tracks[:-steps // 2, :]], axis=0))
    else:
        tracks_out = sliding_window(tracks)

    return tracks_out[:, :, -dim:] * scale_units, tracks_out[:, :, 0]


def process_and_save_tracks_short(path, save_path, seq_encoder=50, scale_units=1.):
    """
    Process all CSV files in the specified directory and save the results as NPY and NPZ files.

    Parameters:
    - path (str): Path to the input data directory.
    - save_path (str): Path to the output data directory.
    - seq_encoder (int): Sequence length for the encoder. Default is 50.
    - scale_units (float): Sequence length for the encoder. Default is 50.
    """
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'normalized'), exist_ok=True)

    ## processing short tracks
    files = os.listdir(path)
    for f in files:
        if f.endswith('.csv'):
            concate_tracks_pool(trackersFolder=path, f=f)

    files = os.listdir(path)
    full_annotations = []
    for f in files:
        if not (f.endswith('-tracks-preprocessed.npy')):
            continue
            # Process trajectory data
        tracks_out, t = prepare_trajectory_segments_short(os.path.join(path, f),
                                                          window_size=2 * seq_encoder,
                                                          steps=2 * seq_encoder,
                                                          scale_units=scale_units)
        tracks_init = tracks_out[:, seq_encoder - 1]
        tracks_out, tracks_out_norm = v_normalize_batch_encoder(tracks_out)

        # Save raw data and normalized scale
        save_name = ('').join(f.split('.')[:-1])
        np.save(os.path.join(save_path, save_name + '-tracks.npy'), tracks_out)
        np.savez_compressed(os.path.join(save_path, 'normalized', save_name + '-tracks-Normalize-scale.npz'),
                            time=t,
                            tracks_init=tracks_init,
                            tracks_norm=tracks_out_norm, )

        # Process head part of the trajectory data
        tracks_out_head, t_head = prepare_trajectory_segments_short(os.path.join(path, f),
                                                                    window_size=2 * seq_encoder,
                                                                    steps=2 * seq_encoder,
                                                                    scale_units=scale_units,
                                                                    is_head=True)
        tracks_init_head = tracks_out_head[:, seq_encoder - 1]
        tracks_out_head, tracks_out_head_norm = v_normalize_batch_encoder(tracks_out_head)

        # Save raw head data and normalized scale
        save_name_head = save_name + '-head'
        np.save(os.path.join(save_path, save_name_head + '-tracks.npy'), tracks_out_head)
        np.savez_compressed(os.path.join(save_path, 'normalized', save_name_head + '-tracks-Normalize-scale.npz'),
                            time=t_head,
                            tracks_init=tracks_init_head,
                            tracks_norm=tracks_out_head_norm,
                            )

        print(f'The file {f} is saved at {save_path} with fix tags of -tracks')

        full_annotations.append(os.path.join(save_path, save_name + '-tracks.npy'))
        full_annotations.append(os.path.join(save_path, save_name_head + '-tracks.npy'))

        # write the annotation
    annotation_name = path.split('/')[-1] or path.split('/')[-2]
    write_annotation(full_annotations, annotation_name)


def write_annotation(full_annotations, annotation_name, path='./data'):
    """
    Write annotations to two separate text files

    Args:
        full_annotations: List of annotation strings to write
        annotation_name: Base name for output files
        path: Directory path for output files (default: './data')
    """
    with open(os.path.join(path, annotation_name + '_annotation.txt'), 'w') as f:
        for anno in full_annotations:
            if '-head-tracks' not in anno:  # only write if it doesn't contain '-head-tracks'
                f.write(f'{anno}\n')

    with open(os.path.join(path, annotation_name + '_annotation_visual.txt'), 'w') as f:
        for anno in full_annotations:
            f.write('{}\n'.format(anno))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process trajectory data from CSV files.')

    # Required arguments
    parser.add_argument('--input_path', type=str,
                        default='./data_generator/input_data',
                        help='Input directory containing CSV files')
    parser.add_argument('--output_path', type=str,
                        default='./data',
                        help='Output directory for processed files')

    # Optional arguments
    parser.add_argument('--mode', type=str,
                        help='using short processing mode: "short" or "long"')
    parser.add_argument('--seq_encoder', type=int, default=50,
                        help='Sequence length for encoder (default: 50)')
    parser.add_argument('--scale_units', type=float, default=1.,
                        help='Scale units for spatial coordinates (default: 1)')

    args = parser.parse_args()

    if args.mode == 'short':
        process_and_save_tracks_short(
            path=args.input_path,
            save_path=args.output_path,
            seq_encoder=args.seq_encoder,
            scale_units=args.scale_units
        )
    else:
        process_and_save_tracks(
            path=args.input_path,
            save_path=args.output_path,
            seq_encoder=args.seq_encoder,
            scale_units=args.scale_units
        )
