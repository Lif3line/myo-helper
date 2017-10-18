"""Utility functions for working with Myo Armband Data."""

import os
import errno
import pickle
import numpy as np
import scipy.io as sio
from itertools import combinations, chain
from sklearn.preprocessing import StandardScaler


def import_subj_myo(folder_path, subject, expts=1):
    """Get data from Myo experiment for v1 testing."""
    nb_expts = np.size(expts)
    if nb_expts < 1:
        raise ValueError("Experiment ID(s) argument, 'expts', must have atleast 1 value.")

    expts = np.asarray(expts)  # In case scalar
    file_name = "s{}_ex{}.mat".format(subject, expts.item(0))
    file_path = os.path.join(folder_path, file_name)

    data = sio.loadmat(file_path)

    data['move'] = np.squeeze(data['move'])
    data['rep'] = np.squeeze(data['rep'])
    data['emg_time'] = np.squeeze(data['emg_time'])

    if nb_expts > 1:
        for expt in expts[1:]:
            file_name = "s{}_ex{}.mat".format(subject, expt)
            file_path = os.path.join(folder_path, file_name)

            data_tmp = sio.loadmat(file_path)

            # Label experiments as later repetitions
            cur_max_rep = np.max(data['rep'])
            data_tmp['rep'][data_tmp['rep'] != -1] += cur_max_rep

            data['move'] = np.concatenate((data['move'], np.squeeze(data_tmp['move'])))
            data['rep'] = np.concatenate((data['rep'], np.squeeze(data_tmp['rep'])))
            data['emg_time'] = np.concatenate((data['emg_time'], np.squeeze(data_tmp['emg_time'])))

            data['emg'] = np.concatenate((data['emg'], data_tmp['emg']))
            data['gyro'] = np.concatenate((data['gyro'], data_tmp['gyro']))
            data['orient'] = np.concatenate((data['orient'], data_tmp['orient']))
            data['acc'] = np.concatenate((data['acc'], data_tmp['acc']))

    return data


def import_subj_delsys(folder_path, subject, expts=1):
    """Get data from Myo experiment for v1 testing."""
    nb_expts = np.size(expts)
    if nb_expts < 1:
        raise ValueError("Experiment ID(s) argument, 'expts', must have atleast 1 value.")

    expts = np.asarray(expts)  # In case scalar
    file_name = "s{}_ex{}_delsys.mat".format(subject, expts.item(0))
    file_path = os.path.join(folder_path, file_name)

    data = sio.loadmat(file_path)

    data['move'] = np.squeeze(data['move'])
    data['rep'] = np.squeeze(data['rep'])
    data['emg_time'] = np.squeeze(data['emg_time'])

    if nb_expts > 1:
        for expt in expts[1:]:
            file_name = "s{}_ex{}_delsys.mat".format(subject, expt)
            file_path = os.path.join(folder_path, file_name)

            data_tmp = sio.loadmat(file_path)

            # Label experiments as later repetitions
            cur_max_rep = np.max(data['rep'])
            data_tmp['rep'][data_tmp['rep'] != -1] += cur_max_rep

            data['move'] = np.concatenate((data['move'], np.squeeze(data_tmp['move'])))
            data['rep'] = np.concatenate((data['rep'], np.squeeze(data_tmp['rep'])))
            data['emg_time'] = np.concatenate((data['emg_time'], np.squeeze(data_tmp['emg_time'])))

            data['acc'] = np.concatenate((data['acc'], data_tmp['acc']))

    return data


def import_supplemental(file_path):
    """Get data from a supplemental file"""
    data = sio.loadmat(file_path)

    data['move'] = np.squeeze(data['move'])
    data['rep'] = np.squeeze(data['rep'])
    data['emg_time'] = np.squeeze(data['emg_time'])

    return data


def gen_split_balanced(rep_ids, nb_test, base=None):
    """Create a balanced split for training and testing based on repetitions (all reps equally tested + trained on) .

    Args:
        rep_ids (array): Repetition identifiers to split
        nb_test (int): The number of repetitions to be used for testing in each each split
        base (array, optional): A specific test set to use (must be of length nb_test)

    Returns:
        Arrays: Training repetitions and corresponding test repetitions as 2D arrays [[set 1], [set 2] ..]
    """
    nb_reps = rep_ids.shape[0]
    nb_splits = nb_reps

    train_reps = np.zeros((nb_splits, nb_reps - nb_test,), dtype=int)
    test_reps = np.zeros((nb_splits, nb_test), dtype=int)

    # Generate all possible combinations
    all_combos = combinations(rep_ids, nb_test)
    all_combos = np.fromiter(chain.from_iterable(all_combos), int)
    all_combos = all_combos.reshape(-1, nb_test)

    if base is not None:
        test_reps[0, :] = base
        all_combos = np.delete(all_combos, np.where(np.all(all_combos == base, axis=1))[0][0], axis=0)
        cur_split = 1
    else:
        cur_split = 0

    all_combos_copy = all_combos
    reset_counter = 0
    while cur_split < (nb_splits):
        if reset_counter >= 10 or all_combos.shape[0] == 0:
            all_combos = all_combos_copy
            test_reps = np.zeros((nb_splits, nb_test), dtype=int)
            if base is not None:
                test_reps[0, :] = base
                cur_split = 1
            else:
                cur_split = 0

            reset_counter = 0

        randomIndex = np.random.randint(0, all_combos.shape[0])
        test_reps[cur_split, :] = all_combos[randomIndex, :]
        all_combos = np.delete(all_combos, randomIndex, axis=0)

        _, counts = np.unique(test_reps[:cur_split + 1], return_counts=True)

        if max(counts) > nb_test:
            test_reps[cur_split, :] = np.zeros((1, nb_test), dtype=int)
            reset_counter += 1
            continue
        else:
            cur_split += 1
            reset_counter = 0

    for i in range(nb_splits):
        train_reps[i, :] = np.setdiff1d(rep_ids, test_reps[i, :])

    return train_reps, test_reps


def gen_ttv_balanced(rep_ids, nb_test, nb_val, split_multiplier=1, base=None):
    """Create a balanced split for training, testing and validation based on repetitions.

    Args:
        rep_ids (array): Repetition identifiers to split
        nb_test (int): The number of repetitions to be used for testing in each each split
        nb_val (int): The number of repetitions to be used for validation in each each split
        split_multiplier (int, optional): Multiplier for the number of splits generated
        base (array, optional): A specific test set to use (must be of length nb_test)

    Returns:
        Arrays: Training repetitions and corresponding test repetitions as 2D arrays [[set 1], [set 2] ..]

    Notes:
        Somewhat inelegant - will run forever if split_multiplier too high of if there is no solution, may also simply
        be slow
    """
    nb_reps = rep_ids.shape[0]
    nb_splits = nb_reps * split_multiplier

    train_val_pool_reps = np.zeros((nb_splits, nb_reps - nb_test,), dtype=int)
    train_reps = np.zeros((nb_splits, nb_reps - nb_test - nb_val,), dtype=int)
    test_reps = np.zeros((nb_splits, nb_test), dtype=int)
    val_reps = np.zeros((nb_splits, nb_val), dtype=int)

    # Select test combinations
    all_combos = combinations(rep_ids, nb_test)
    all_combos = np.fromiter(chain.from_iterable(all_combos), int)
    all_combos = all_combos.reshape(-1, nb_test)

    if base is not None:
        test_reps[0, :] = base
        all_combos = np.delete(all_combos, np.where(np.all(all_combos == base, axis=1))[0][0], axis=0)
        cur_split = 1
    else:
        cur_split = 0

    all_combos_copy = all_combos
    reset_counter = 0
    while cur_split < (nb_splits):
        if reset_counter >= 10 or all_combos.shape[0] == 0:
            all_combos = all_combos_copy
            test_reps = np.zeros((nb_splits, nb_test), dtype=int)
            if base is not None:
                test_reps[0, :] = base
                cur_split = 1
            else:
                cur_split = 0

            reset_counter = 0

        randomIndex = np.random.randint(0, all_combos.shape[0])
        test_reps[cur_split, :] = all_combos[randomIndex, :]
        all_combos = np.delete(all_combos, randomIndex, axis=0)

        _, counts = np.unique(test_reps[:cur_split + 1], return_counts=True)
        if max(counts) > nb_test * split_multiplier:
            test_reps[cur_split, :] = np.zeros((1, nb_test), dtype=int)
            reset_counter += 1
            continue
        else:
            cur_split += 1
            reset_counter = 0

    for i in range(nb_splits):
        train_val_pool_reps[i, :] = np.setdiff1d(rep_ids, test_reps[i, :])

    # Select Validation Combinations
    cur_split = 0
    reset_counter = 0
    while cur_split < nb_splits:
        if reset_counter >= 10:
            val_reps = np.zeros((nb_splits, nb_val), dtype=int)
            reset_counter = 0
            cur_split = 0

        val_reps[cur_split, :] = np.random.permutation(train_val_pool_reps[cur_split, :])[:nb_val]

        _, counts = np.unique(val_reps[:cur_split + 1], return_counts=True)

        if max(counts) > nb_val * split_multiplier:
            val_reps[cur_split, :] = np.zeros((1, nb_val), dtype=int)
            reset_counter += 1
            continue
        else:
            cur_split += 1
            reset_counter = 0

    for i in range(nb_splits):
        train_reps[i, :] = np.setdiff1d(train_val_pool_reps[i, :], val_reps[i, :])

    return train_reps, test_reps, val_reps


def normalise_by_rep(emg, rep, train_reps):
    """Preprocess train+test data to mean 0, std 1 based on training data only."""
    # Locate valid window end indices (window must be window_len long and window_inc away from last)
    train_idx = np.where(np.in1d(rep, train_reps))

    scaler = StandardScaler(with_mean=True,
                            with_std=True,
                            copy=False).fit(emg[train_idx])

    return scaler.transform(emg), scaler


def window_emg(window_len, window_inc, emg, move, rep, which_moves=None, which_reps=None,
               emg_dtype=np.float32, y_dtype=np.int8, r_dtype=np.int8):
    """Window the EMG data explicitly.

    If using which_moves then y_data will be indices into which_moves rather than the original movement label; this is
    to fix issues in some machine learning libraries such as Tensorflow which have issues using sparse categorical
    cross-entropy and non-sequential labels."""
    nb_obs = emg.shape[0]
    nb_channels = emg.shape[1]

    # Locate valid window end indices (window must be window_len long and window_inc away from last)
    targets = np.array(range(window_len - 1, nb_obs, window_inc))

    # Reduce targets by allowed movements
    if which_moves is not None:
        targets = targets[np.in1d(move[targets], which_moves)]

    # Reduce targets by allowed repetitions
    if which_reps is not None:
        targets = targets[np.in1d(rep[targets], which_reps)]

    x_data = np.zeros([targets.shape[0], window_len, nb_channels, 1], dtype=emg_dtype)
    y_data = np.zeros([targets.shape[0], ], dtype=y_dtype)
    r_data = np.zeros([targets.shape[0], ], dtype=r_dtype)

    for i, win_end in enumerate(targets):
        win_start = win_end - (window_len - 1)
        x_data[i, :, :, 0] = emg[win_start:win_end + 1, :]  # Include end
        y_data[i] = move[win_end]
        r_data[i] = rep[win_end]

    if which_moves is not None:
        y_data_tmp = np.copy(y_data)
        for i, label in enumerate(which_moves):
            y_data[np.where(y_data_tmp == label)] = i

    return x_data, y_data, r_data


def save_object(obj, filename):
    """Simple function for saving an object with pickle to a file. Create dir + file if neccessary."""
    filename = os.path.normpath(filename)

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(filename, 'wb+') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
