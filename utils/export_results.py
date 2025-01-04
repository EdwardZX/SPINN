import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def tracks_restore(X, x_0, v_norm):
    """
    Restore tracks by computing cumulative positions from velocities and initial position

    Args:
        X: Array of positions along tracks
        x_0: Initial position array
        v_norm: Velocity normalization factors

    Returns:
        Array of restored track positions
    """
    v = np.diff(X, axis=1)
    x_0 = x_0.reshape(-1, 1, 2)
    dx = v * v_norm.reshape(1, 1, 2)
    return np.cumsum(np.concatenate([x_0, dx], axis=1), axis=1)


def get_Ixy_restore_initial(X, v_norm, c=50):
    """
    Calculate initial velocity at center point after restoring track positions

    Args:
        X: Array of track positions
        v_norm: Velocity normalization factors
        c: Center point index (default: 50)

    Returns:
        Initial velocity at center point of restored tracks
    """
    v = np.diff(X, axis=1)
    dx = v * v_norm.reshape(1, 1, 2)  # keeping center equal zeros
    x0 = np.array([0, 0]).reshape(-1, X.shape[2]).repeat(X.shape[0], axis=0).reshape(-1, 1, X.shape[2])
    x_r = np.cumsum(np.concatenate([x0, dx], axis=1), axis=1)
    # y = x_r - (x_r[:,c-1]).reshape(-1,1,2) # keep center zeros
    return (x_r[:, c]) - (x_r[:, c - 1])


def y1_align_raw(X, Y1):
    """
    Align predicted trajectories (Y1) with input trajectories (X) by centering

    Args:
        X: Input trajectory positions
        Y1: Predicted trajectory positions

    Returns:
        Aligned predicted trajectories with same center as input
    """
    x_c = np.mean(X, axis=1)
    y_c = np.mean(Y1, axis=1)
    delta = (y_c - x_c).reshape(-1, 1, 2)
    return Y1 - delta


def resort_func(x, idy):
    """
    Reorder array elements according to given indices

    Args:
        x: Input array to be reordered
        idy: Array of indices for reordering

    Returns:
        New array with elements reordered according to idy
    """
    new_x = np.zeros_like(x)
    new_x[idy] = x
    return new_x


def load_microdynamics_tracks(tracks_each_name, model_name,
                              model_dir='./model_dir/input_data/',
                              folder_normalized='./data/normalized/',
                              normalized_tags='tracks-Normalize-scale',
                              seq_len=50, tags='tracks'):
    """
    Load and process microdynamics tracks data including both body and head tracks

    Args:
        tracks_each_name: Base name for track files
        model_name: Name of model
        model_dir: Directory containing model input data
        folder_normalized: Directory containing normalized data
        normalized_tags: Tags for normalized files
        seq_len: Sequence length to use
        tags: Additional tags for filenames

    Returns:
        Dictionary containing processed track data
    """
    filename = model_name + '_' + tracks_each_name + '-' + tags

    filename_normalized = os.path.join(folder_normalized, tracks_each_name + '-' + normalized_tags + '.npz')
    data_set_normalized = np.load(filename_normalized)
    dataset_t = data_set_normalized['time'][:, seq_len:]
    x0_init = data_set_normalized['tracks_init']
    track_norm = data_set_normalized['tracks_norm']

    dataset = np.load(os.path.join(model_dir, filename + '_micro.npz'))
    x_raw = dataset['x_raw'].transpose(1, 0, 2)[:, seq_len:]  # L x B x D
    y1 = dataset['y1'].transpose(1, 0, 2)[:, -seq_len:]
    sigma_run = dataset['sigma_run'].transpose(1, 0, 2)[:, -seq_len:]
    idy = dataset['sample_indices']

    ##
    s_raw = dataset['s_raw'].transpose(1, 0, 2)[:, seq_len:]
    s1 = dataset['s1'].transpose(1, 0, 2)[:, -seq_len:]

    ##
    v_raw = dataset['v_raw'].transpose(1, 0, 2)[:, seq_len:]
    v1 = dataset['v1'].transpose(1, 0, 2)[:, -seq_len:]

    #####

    ## resort val
    x_raw, y1, sigma_run = resort_func(x_raw, idy=idy), resort_func(y1, idy=idy), resort_func(sigma_run, idy=idy)
    s_raw, s1 = resort_func(s_raw, idy=idy), resort_func(s1, idy=idy)
    v_raw, v1 = resort_func(v_raw, idy=idy), resort_func(v1, idy=idy)

    ## baseline of raw trajectories 1, inital; 2, normalized val
    x_raw = tracks_restore(x_raw, x_0=x0_init, v_norm=track_norm)
    y1 = tracks_restore(y1, x_0=x0_init, v_norm=track_norm)
    y1 = y1_align_raw(x_raw, y1)

    sigma_run *= track_norm
    v_raw *= track_norm
    v1 *= track_norm
    s_raw *= track_norm ** 2
    s1 *= track_norm ** 2

    ##### for head
    filename_head = model_name + '_' + tracks_each_name + '-head-' + tags
    filename_normalized_head = os.path.join(folder_normalized, tracks_each_name + '-head-' + normalized_tags + '.npz')
    # filename_init = './extracted_data/'+id_tracks + '-tracks.npy'
    data_set_normalized_head = np.load(filename_normalized_head)

    dataset_t_head = data_set_normalized_head['time'][:, seq_len:]
    x0_init_head = data_set_normalized_head['tracks_init']
    track_norm_head = data_set_normalized_head['tracks_norm']

    dataset_head = np.load(os.path.join(model_dir, filename_head + '_micro.npz'))

    x_raw_head = dataset_head['x_raw'].transpose(1, 0, 2)[:, seq_len:]  # L x B x D
    y1_head = dataset_head['y1'].transpose(1, 0, 2)[:, -seq_len:]
    sigma_run_head = dataset_head['sigma_run'].transpose(1, 0, 2)[:, -seq_len:]
    idy_head = dataset_head['sample_indices']

    ##
    s_raw_head = dataset_head['s_raw'].transpose(1, 0, 2)[:, seq_len:]
    s1_head = dataset_head['s1'].transpose(1, 0, 2)[:, -seq_len:]

    ##
    v_raw_head = dataset_head['v_raw'].transpose(1, 0, 2)[:, seq_len:]
    v1_head = dataset_head['v1'].transpose(1, 0, 2)[:, -seq_len:]
    #####
    #
    # note: some error in shape_gen (B x L x D x num)
    ###

    ## resort val
    x_raw_head, y1_head, sigma_run_head = resort_func(x_raw_head, idy=idy_head), resort_func(y1_head,
                                                                                             idy=idy_head), resort_func(
        sigma_run_head, idy=idy_head)
    s_raw_head, s1_head = resort_func(s_raw_head, idy=idy_head), resort_func(s1_head, idy=idy_head)
    v_raw_head, v1_head = resort_func(v_raw_head, idy=idy_head), resort_func(v1_head, idy=idy_head)

    ## baseline of raw trajectories 1, inital; 2, normalized val
    x_raw_head = tracks_restore(x_raw_head, x_0=x0_init_head, v_norm=track_norm_head)
    y1_head = tracks_restore(y1_head, x_0=x0_init_head, v_norm=track_norm_head)
    y1_head = y1_align_raw(x_raw_head, y1_head)

    sigma_run_head *= track_norm_head

    v_raw_head *= track_norm_head
    v1_head *= track_norm_head
    s_raw_head *= track_norm_head ** 2
    s1_head *= track_norm_head ** 2

    # concatecate
    x_raw, y1 = np.concatenate([x_raw_head, x_raw], axis=1), np.concatenate([y1_head, y1], axis=1)
    v_raw, v1 = np.concatenate([v_raw_head, v_raw], axis=1), np.concatenate([v1_head, v1], axis=1)
    s_raw, s1 = np.concatenate([s_raw_head, s_raw], axis=1), np.concatenate([s1_head, s1], axis=1)
    sigma_run = np.concatenate([sigma_run_head, sigma_run], axis=1)
    dataset_t = np.concatenate([dataset_t_head, dataset_t], axis=1)

    ## getting the ids
    tracks_id = get_tracks_from_T_batch(dataset_t, x_raw)
    results = {
        'x_raw': x_raw,
        'y1': y1,
        'v_raw': v_raw,
        'v1': v1,
        's_raw': s_raw,
        's1': s1,
        'sigma_run': sigma_run,
        't': dataset_t,
        'tracks_id': tracks_id,
    }
    return results


def load_microdynamics_sample(filename,
                              model_dir='./model_dir/input_data/',
                              seq_len=50
                              ):
    """
    Load and process single microdynamics sample data

    Args:
        filename: Base filename for sample data
        model_dir: Directory containing model input data
        seq_len: Sequence length to use

    Returns:
        Dictionary containing processed sample data
    """
    dataset = np.load(os.path.join(model_dir, filename + '_micro.npz'))

    x_raw = dataset['x_raw'].transpose(1, 0, 2)[:, seq_len:]  # L x B x D
    y1 = dataset['y1'].transpose(1, 0, 2)[:, -seq_len:]
    sigma_run = dataset['sigma_run'].transpose(1, 0, 2)[:, -seq_len:]
    idy = dataset['sample_indices']

    ##
    s_raw = dataset['s_raw'].transpose(1, 0, 2)[:, seq_len:]
    s1 = dataset['s1'].transpose(1, 0, 2)[:, -seq_len:]

    ##
    v_raw = dataset['v_raw'].transpose(1, 0, 2)[:, seq_len:]
    v1 = dataset['v1'].transpose(1, 0, 2)[:, -seq_len:]

    #####
    ## resort val
    x_raw, y1, sigma_run = resort_func(x_raw, idy=idy), resort_func(y1, idy=idy), resort_func(sigma_run, idy=idy)
    s_raw, s1 = resort_func(s_raw, idy=idy), resort_func(s1, idy=idy)
    v_raw, v1 = resort_func(v_raw, idy=idy), resort_func(v1, idy=idy)

    results = {
        'x_raw': x_raw,
        'y1': y1,
        'v_raw': v_raw,
        'v1': v1,
        's_raw': s_raw,
        's1': s1,
        'sigma_run': sigma_run,
    }
    return results


def load_generated_tracks(filename,
                          model_dir='./model_dir/input_data/',
                          seq_len=50
                          ):
    """
    Load and process generated track data including raw and generated trajectories

    Args:
        filename: Base filename for track data
        model_dir: Directory containing model input data
        seq_len: Sequence length to use

    Returns:
        Dictionary containing processed raw and generated track data
    """
    dataset = np.load(os.path.join(model_dir, filename + '_micro.npz'))
    x_raw = dataset['x_raw'].transpose(1, 0, 2)[:, seq_len:]  # L x B x D
    y1 = dataset['y1'].transpose(1, 0, 2)[:, -seq_len:]

    filename_gen = os.path.join(model_dir, filename)
    shape_gen = np.load(filename_gen + '-gen/memmap_gen/shape_gen.npy')  # before has error
    x_gen = np.memmap(filename_gen + '-gen/memmap_gen/x0_gen.npy',
                      mode='r', dtype='float32', shape=(shape_gen[0], shape_gen[1], shape_gen[2], shape_gen[3]))
    y1_gen = np.memmap(filename_gen + '-gen/memmap_gen/y1_gen.npy',
                       mode='r', dtype='float32', shape=(shape_gen[0], shape_gen[1], shape_gen[2], shape_gen[3]))

    idy = dataset['sample_indices']

    ## resort val
    x_raw, y1, = resort_func(x_raw, idy=idy), resort_func(y1, idy=idy)
    x_gen, y1_gen = resort_func(np.array(x_gen), idy=idy), resort_func(np.array(y1_gen), idy=idy)

    results = {
        'x_raw': x_raw,
        'y1': y1,
        'x_gen': x_gen,
        'y1_gen': y1_gen,
    }
    return results


def get_tracks_from_T_batch(T_batch, X_batch, max_displacement=30.0):
    """
    Generate track IDs by finding breaks in trajectories based on time and displacement

    Args:
        T_batch: Time values for trajectories (B x L)
        X_batch: Position values for trajectories (B x L x D)
        max_displacement: Maximum allowed displacement between consecutive points

    Returns:
        Array of track IDs for each point
    """
    # T is B x L, X is B x L x D
    T = T_batch.reshape(-1)
    X = X_batch.reshape(-1, X_batch.shape[2])

    # Time difference
    dT = np.hstack([np.array([-1]), np.diff(T)])

    # Spatial displacement
    dX = np.sqrt(np.sum((X[1:] - X[:-1]) ** 2, axis=1))
    dX = np.hstack([np.array([0]), dX])

    track_breaks = (dT != 1) | (dX > max_displacement)

    # Generate track IDs
    tracks_id = np.cumsum(track_breaks) - 1
    return tracks_id


def microdynamics_results_to_csv(rslt, tracks_each_name,
                                 save_path='./model_dir',
                                 short=None):
    """
    Convert microdynamics results to CSV format with optional short track alignment

    Args:
        rslt: Dictionary containing results data
        tracks_each_name: Base name for track files
        save_path: Directory to save output CSV
        short: Tuple of (short_folder, scale_units) for short track alignment

    Returns:
        None - saves results to CSV file
    """
    D = rslt['x_raw'].shape[2]
    # y1, v1 ,s1
    V, S = rslt['v1'].reshape(-1, D), rslt['s1'].reshape(-1, D)
    X, Y = rslt['x_raw'].reshape(-1, D), rslt['y1'].reshape(-1, D)
    T, tracks_id = rslt['t'].reshape(-1, 1), rslt['tracks_id'].reshape(-1, 1)

    ## save the df
    save_name_SPINN = os.path.join(save_path, tracks_each_name + '-SPINNResults.csv')
    save_table = np.concatenate([tracks_id, T, X, Y, V, S], axis=1)
    df = pd.DataFrame(save_table,
                      columns=['id_tracks', 'frame', 'X', 'Y', 'Y1X', 'Y1Y', 'VX', 'VY', 'VarX', 'VarY'])

    if short is not None:
        short_folder = short[0]
        scale_units = short[1]
        TX0 = np.load(os.path.join(short_folder, tracks_each_name + '-X0-preprocessed.npy'))
        tracks_id = np.load(os.path.join(short_folder, tracks_each_name + '-id-preprocessed.npy'))  # id, t, x, y
        df_align = pd.read_csv(os.path.join(short_folder, tracks_each_name + '.csv'))

        TX0[:, 1:] = TX0[:, 1:] * scale_units
        df_align.iloc[:, 2:] = df_align.iloc[:, 2:] * scale_units

        ## to change the name
        # y1, v1 ,s1
        X0 = TX0[:Y.shape[0], :]
        tracks_id = tracks_id[:Y.shape[0]][:, None]
        df_align = df_align.iloc[:Y.shape[0], :]
        # offset
        Y = Y + X0[:, 1:]
        X = X + X0[:, 1:]

        save_table = np.concatenate([tracks_id, X0[:, 0].reshape(-1, 1), X, Y, V, S], axis=1)
        df = pd.DataFrame(save_table,
                          columns=['id_tracks', 'frame', 'X', 'Y', 'Y1X', 'Y1Y', 'VX', 'VY', 'VarX', 'VarY'])
        grouped = df.groupby('id_tracks')
        grouped_align = df_align.groupby(df_align.columns[0])

        x_cols, y_cols = df_align.columns[2], df_align.columns[3]
        for name, group in tqdm(grouped, desc=f'Aligned {tracks_each_name} short tracks groups'):
            align_group = grouped_align.get_group(name)
            x_init, y_init = group['X'].iloc[0], group['Y'].iloc[0]
            x_offset = - x_init + align_group[x_cols].iloc[0]
            y_offset = - y_init + align_group[y_cols].iloc[0]

            ##
            # Get indices for this group
            mask = (df['id_tracks'] == name)  # or whatever column you grouped by

            # Update original DataFrame
            df.loc[mask, 'X'] = group['X'] + x_offset
            df.loc[mask, 'Y'] = group['Y'] + y_offset
            df.loc[mask, 'Y1X'] = group['Y1X'] + x_offset
            df.loc[mask, 'Y1Y'] = group['Y1Y'] + y_offset

    df.to_csv(save_name_SPINN, index=False)
    print(f'{tracks_each_name}-SPINNResults.csv have saved!')
