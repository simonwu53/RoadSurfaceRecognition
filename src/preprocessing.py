'''
Paper Title: Road Surface Recognition Based on DeepSense Neural Network using Accelerometer Data
Created by ITS Lab, Institute of Computer Science, University of Tartu
'''

# Libraries
import numpy as np
from functools import reduce
import math
import multiprocessing as mp
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import utils
import logging

logging.basicConfig(level=logging.ERROR)

"""
Feature Extraction
"""


def extract_features_deepsense(raw_data, segment=11, step=10, T=10, test_split=0.2,
                               balance=True, cores=mp.cpu_count()):
    """
    extract features from raw data based on tsfresh
    https://tsfresh.readthedocs.io/en/latest/

    :param raw_data: raw data loaded from utils
    :param segment: segment length aka. window size
    :param step: steps (length) to move to next segment
    :param T: window number for each sample, number of windows
    :param test_split: size for the test set, pass 0.0 to ignore
    :param balance: balance number of categories
    :param cores: specify cores to use
    :return: training dataset and testing dataset
    """
    # get total tasks & distribute to workers
    total_keys = list(raw_data.keys())
    avg = len(total_keys) / cores
    ranges = list((int(i * avg), int((i + 1) * avg)) for i in range(cores))
    queue = mp.Queue()

    def job(worker_id):
        i_start, i_end = ranges[worker_id]
        worker_collection_feat = []
        worker_collection_label = []

        # iterating each file
        for i in range(i_start, i_end):
            cache = raw_data[total_keys[i]]

            windows = utils.window_slice(cache.shape[0], segment=segment, step=step)
            # check all windows has the same length
            if not windows[-1][-1] - windows[-1][0] + 1 == segment:
                print('Key: %s \nwindow can not be fully divided! last window dropped!' % total_keys[i], windows[-1])
                windows = windows[:-1]

            # create features & labels
            features = np.zeros((int(len(windows)/T), 3, 2*segment, T), np.float64)
            # features = np.zeros((int(len(windows)/T), T, 3, 2*segment), np.float64)
            labels = np.full(int(len(windows)/T), cache[0]['road'], np.int8)
            for j, w in enumerate(windows):
                f_x = np.fft.fft(cache[w]['x'])
                f_y = np.fft.fft(cache[w]['y'])
                f_z = np.fft.fft(cache[w]['z'])
                features[int(j/T), 0, :, j%T] = np.concatenate(list(zip(f_x.real, f_x.imag)))
                features[int(j/T), 1, :, j%T] = np.concatenate(list(zip(f_y.real, f_y.imag)))
                features[int(j/T), 2, :, j%T] = np.concatenate(list(zip(f_z.real, f_z.imag)))
                # features[int(j/T), j%T, 0, :] = np.concatenate(list(zip(f_x.real, f_x.imag)))
                # features[int(j/T), j%T, 1, :] = np.concatenate(list(zip(f_y.real, f_y.imag)))
                # features[int(j/T), j%T, 2, :] = np.concatenate(list(zip(f_z.real, f_z.imag)))

            # add to collection
            worker_collection_feat.append(features)
            worker_collection_label.append(labels)

        # add to queue
        worker_collection_feat = np.concatenate(worker_collection_feat)
        worker_collection_label = np.concatenate(worker_collection_label)
        queue.put((worker_id, worker_collection_feat, worker_collection_label))
        return

    # create processes and execute
    processes = [mp.Process(target=job, args=(x,)) for x in range(cores)]

    # start processes
    for p in processes:
        p.start()

    # get output from workers
    results = [queue.get() for _ in processes]

    # wait all processes to stop
    for p in processes:
        p.join()

    # sort result
    results.sort(key=lambda x:x[0])
    total_features = np.concatenate([x[1] for x in results])
    total_labels = np.concatenate([x[2] for x in results])

    if balance:
        train_features, train_labels, test_features, test_labels = \
            split_and_balance_category(total_features, total_labels,
                                       test_split, T=1, use_idx=False)
    else:
        test_size = int(total_labels.shape[0] * test_split)
        test_pick = np.random.choice(total_labels.shape[0], test_size)
        test_mask = np.zeros(total_labels.shape[0], np.bool)
        test_mask[test_pick] = 1
        train_features = total_features[np.invert(test_mask)]
        train_labels = total_labels[np.invert(test_mask)]
        test_features = total_features[test_mask]
        test_labels = total_labels[test_mask]
    return train_features, train_labels, test_features, test_labels


def extract_features_tsfresh(raw_data, segment=51, step=50, T=2, normalize=False,
                             n_comps=30, test_split=0.2, standardize=False, no_ds_format=False,
                             cores=mp.cpu_count(), return_tsfresh=False, return_ae=False):
    """
    extract features from raw data based on tsfresh
    https://tsfresh.readthedocs.io/en/latest/

    :param raw_data: raw data loaded from utils
    :param segment: segment length aka. window size
    :param step: steps (length) to move to next segment
    :param T: window number for each sample, number of windows
    :param n_comps: number of features to preserve after PCA
    :param return_ae: if False, perform PCA after extraction of massive features
                      if True, output shape will be  -> (samples, T, n_features), for autoencoders
    :param no_ds_format: if true, return the data with reduced features (after PCA), not applying ds format
    :param test_split: size for the test set, pass 0.0 to ignore
    :param cores: specify cores to use
    :param return_tsfresh: if True, return the results right after extractio of massive features
    :return: training dataset and testing dataset
             e.g. training data set -> (samples, 1, n_features, T) and labels -> (samples,)
    """
    # split train test
    # every two samples will be merged into one input for deepsense,
    # so only select even index to split

    # check the data type of the raw input file
    train_features, train_labels, test_features, test_labels = [], [], [], []
    if isinstance(raw_data, list):
        for raw in raw_data:
            collection_features, collection_labels = \
                file_segmentation(raw, segment=segment, step=step, cores=cores)

            # train, test separation
            Xtrain, Ytrain, Xtest, Ytest = split_and_balance_category(collection_features, collection_labels,
                                                                      test_split=test_split, T=T, use_idx=True)

            if normalize:
                Xtrain, Xtest, _ = normalization(Xtrain, Xtest)
            if standardize:
                Xtrain, Xtest, _ = standardization(Xtrain, Xtest)

            train_features.append(Xtrain)
            train_labels.append(Ytrain)
            test_features.append(Xtest)
            test_labels.append(Ytest)
        # merge list
        train_features = utils.merge_raw(train_features)
        test_features = utils.merge_raw(test_features)
        train_labels = np.concatenate(train_labels)
        test_labels = np.concatenate(test_labels)
    else:
        collection_features, collection_labels = \
            file_segmentation(raw_data, segment=segment, step=step, cores=cores)

        train_features, train_labels, test_features, test_labels = \
            split_and_balance_category(collection_features, collection_labels,
                                       test_split=test_split, T=T, use_idx=True)

        if normalize:
            train_features, test_features, _ = normalization(train_features, test_features)
        if standardize:
            train_features, test_features, _ = standardization(train_features, test_features)

    # extract massive features
    train_features = extract_features(pd.DataFrame(train_features),
                                      column_id='idx', column_sort='ts',
                                      impute_function=impute)
    test_features = extract_features(pd.DataFrame(test_features),
                                     column_id='idx', column_sort='ts',
                                     impute_function=impute)

    # start sifting features for each category,
    # 'extract_features' only works for binary category or regression,
    # so we set labels as one to all category to sift features separately
    # Then merge sifted features for all categories
    selected_features = set()

    for l in np.unique(train_labels):
        train_labels_binary = train_labels == l
        train_features_filtered = select_features(train_features, train_labels_binary)
        print('Number of relevant features for class %d: %d/%d' %
              (l, train_features_filtered.shape[1], train_features.shape[1]))
        selected_features = selected_features.union(set(train_features_filtered.columns))

    print('Selected features: ', len(selected_features))
    train_features = train_features[list(selected_features)]
    test_features = test_features[list(selected_features)]

    if return_tsfresh:
        return train_features, train_labels, test_features, test_labels

    if return_ae:
        train_features, train_labels = adapt_to_ae(train_features.values, train_labels, T)
        test_features, test_labels = adapt_to_ae(test_features.values, test_labels, T)
        return train_features, train_labels, test_features, test_labels
    else:
        # perform PCA
        pca_train_features, pca_test_features = pca_reduction(train_features, test_features, n_comps)
        if no_ds_format:
            return pca_train_features, train_labels, pca_test_features, test_labels
        train_features, train_labels = adapt_to_ds(pca_train_features, train_labels, T)
        test_features, test_labels = adapt_to_ds(pca_test_features, test_labels, T)

    return train_features, train_labels, test_features, test_labels


def pca_reduction(Xtrain, Xtest, n_comps, svd_solver='full'):
    """
    perform PCA on trian, test datasets
    :param Xtrain: training data
    :param Xtest: testing data
    :param n_comps: number of components to keep
    :param svd_solver: check PCA documentation svd_solver
    :return: dimension reduced training, test dataset
    """
    pca = PCA(n_components=n_comps, svd_solver=svd_solver)
    pca.fit(Xtrain)
    pca_Xtrain = pca.transform(Xtrain)
    pca_Xtest = pca.transform(Xtest)
    return pca_Xtrain, pca_Xtest


def adapt_to_ds(X, y, T):
    """
    Transform from shape (samples, n_features) to (samples/T, 1, n_features, T)
    :param X: features data
    :param y: labels data
    :param T: window (timesteps)
    :return: transformed features & labels
    """
    origin_shape = X.shape
    feat = np.zeros((int(origin_shape[0] / T), 1, origin_shape[1], T), np.float64)
    label = np.zeros(int(origin_shape[0] / T), np.int8)
    for j in range(int(origin_shape[0] / T)):
        for k in range(T):
            feat[j, 0, :, k] = X[j * T + k, :]

        assert y[j * T] == y[(j+1)*T-1]
        label[j] = y[j * T]
    return feat, label


def adapt_to_ae(X, y, T):
    """
    Transform from shape (samples, n_features) to (samples/T, T, n_features)
    :param X: features data
    :param y: labels data
    :param T: window (timesteps)
    :return: transformed features & labels
    """
    n_features = X.shape[1]
    n_samples = X.shape[0]
    feat = np.zeros((int(n_samples/T), T, n_features), np.float64)
    label = np.zeros(int(n_samples/T), np.int8)
    for j in range(int(n_samples/T)):
        for k in range(T):
            feat[j, k, :] = X[j*T+k, :]

        assert y[j*T] == y[(j+1)*T-1]
        label[j] = y[j*T]
    return feat, label


def split_and_balance_category(features, labels, test_split, T=2, use_idx=True):
    """
    split the train test set and equalize every category quantity
    :param features: features for CNN
    :param labels: labels for CNN
    :param test_split: 0-1 float, the fraction for test set
    :param T: window size, default 2, means every two samples will be merged later for input to CNN
    :param use_idx: if True, features data is separated by index column
    :return: balanced features and labels
    """
    # sort features, labels
    aux_labels = np.zeros((labels.shape[0],2), np.int32)
    aux_labels[:,0] = np.arange(labels.shape[0])
    aux_labels[:,1] = labels
    aux_labels = np.array(sorted(aux_labels, key=lambda x: x[1]), dtype=np.int32)
    if use_idx:
        features = reorder_features(features, aux_labels[:,0], column='idx', mode='sequence')
    else:
        features = features[aux_labels[:,0]]
    labels = aux_labels[:, 1]
    aux_labels[:,0] = np.arange(labels.shape[0])

    # get count for each class
    cu, ci, cc = np.unique(labels, return_counts=True, return_index=True)
    # get the size for each class must can be divided by T,
    # later will merge them
    if min(cc) % T == 0:
        acceptable_size = min(cc) / T
    else:
        count = 0
        while count * T < min(cc):
            count += 1
        acceptable_size = count-1
    test_size_each_class = int(acceptable_size*test_split)
    train_size_each_class = int(acceptable_size - test_size_each_class)
    # create masks for train test set
    test_mask = np.zeros(labels.shape[0], np.bool)
    train_mask = np.zeros(labels.shape[0], np.bool)

    # loop each class to pick up selections
    for cls in cu:  # [0,1,2]
        selection_range = np.arange(ci[cls], ci[cls]+cc[cls], T)
        pickups = np.random.choice(selection_range,
                                   (test_size_each_class+train_size_each_class),
                                   replace=False)
        train_pickups = pickups[:train_size_each_class]
        test_pickups = pickups[-test_size_each_class:]
        assert train_pickups.shape[0]+test_pickups.shape[0] == pickups.shape[0]

        # update masks
        test_mask[test_pickups] = 1
        train_mask[train_pickups] = 1
        # for each pick up, apply to all window T
        for t in range(1, T):
            test_mask[test_pickups+t] = 1
            train_mask[train_pickups+t] = 1

    if use_idx:
        train_features = reorder_features(features, train_mask, column='idx', mode='mask')
        test_features = reorder_features(features, test_mask, column='idx', mode='mask')
    else:
        train_features = features[train_mask]
        test_features = features[test_mask]
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]
    return train_features, train_labels, test_features, test_labels


def file_segmentation(raw, segment=51, step=50, cores=mp.cpu_count()):

    total_keys = list(raw.keys())
    avg = len(total_keys) / cores
    ranges = list((int(i * avg), int((i + 1) * avg)) for i in range(cores))
    queue = mp.Queue()

    def job(worker_id, data):
        i_start, i_end = ranges[worker_id]
        worker_collection_feat = []
        worker_collection_label = []
        idx = 0

        # iterating each file
        for i in range(i_start, i_end):
            cache = data[total_keys[i]]

            windows = utils.window_slice(cache.shape[0], segment=segment, step=step)
            # check all windows has the same length
            if not windows[-1][-1] - windows[-1][0] + 1 == segment:
                print('Key: %s \nwindow can not be fully divided! last window dropped!' % total_keys[i])
                windows = windows[:-1]

            # get label for this type of road
            road_type = cache[0]['road']

            # only keep necessary ts&xyz columns, then create final data with idx for tsfresh
            cache = cache[['ts', 'x', 'y', 'z']]
            cache_extended = np.zeros(len(windows)*segment,
                                      dtype=np.dtype([('ts', np.float64), ('x', np.float64),
                                                      ('y', np.float64), ('z', np.float64),
                                                      ('idx', np.int32)]))
            cache_labels = np.full(len(windows), road_type, dtype=np.int8)
            cache_pivot = 0

            # for each window, create sample
            for win in windows:
                cache_extended[cache_pivot:cache_pivot+segment][['ts', 'x', 'y', 'z']] = cache[win]
                cache_extended[cache_pivot:cache_pivot+segment]['idx'] = idx
                cache_pivot += segment
                idx += 1

            assert cache_extended[cache_extended['ts']==0].shape[0] == 0
            worker_collection_feat.append(cache_extended)
            worker_collection_label.append(cache_labels)

        # add to queue
        worker_collection_feat = np.concatenate(worker_collection_feat)
        worker_collection_label = np.concatenate(worker_collection_label)
        queue.put((worker_id, worker_collection_feat, worker_collection_label))
        return

    # create processes and execute
    processes = [mp.Process(target=job, args=(x, raw)) for x in range(cores)]

    # start processes
    for p in processes:
        p.start()

    # get output from workers
    results = [queue.get() for p in processes]

    # wait all processes to stop
    for p in processes:
        p.join()

    # sort result
    results.sort(key=lambda x: x[0])
    # resign index
    collection_features = [x[1] for x in results]
    collection_labels = np.concatenate([x[2] for x in results])
    collection_features = utils.merge_raw(collection_features)

    return collection_features, collection_labels


def reorder_features(features, order, column='idx', mode='sequence'):

    if mode == 'sequence':
        # reoder features data based on given index sequence
        reordered = np.concatenate([features[features[column]==idx] for idx in order])
        # reindex features data
        reordered = reindex_data(reordered, column=column)
        return reordered
    if mode == 'mask':
        # assume features already reindexed. np.unique(idx_column) == np.arange(0,max_idx+1)
        new_order = np.unique(features[column])[order]
        reordered = np.concatenate([features[features[column] == idx] for idx in new_order])
        reordered = reindex_data(reordered, column=column)
        return reordered
    return


def reindex_data(data, column='idx'):

    idx = 0
    previous = data[0][column]
    pivot = 0
    count = 0
    for i in range(data.shape[0]):
        if data[i][column] == previous:
            count += 1
        else:
            data[pivot:pivot+count][column] = idx
            pivot += count
            idx += 1
            count = 1
            previous = data[i][column]
    data[pivot:pivot+count][column] = idx

    return data


"""
Cross Validation
"""


def make_data_kfold(Xtrain, Ytrain, k, shuffle=True, random_state=53):
    """
    Create a list of folds
    :param Xtrain: features set
    :param Ytrain: labels set
    :param k: num of folds
    :param shuffle: shuffle before making k folds
    :param random_state: for reproduction
    :return: list of folds (each fold contains idx)
    """
    return list(StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state).split(Xtrain, Ytrain))


"""
EXPERIMENTAL
"""


def standardization(Xtrain, Xtest):
    scalar = StandardScaler()
    scalar.fit(Xtrain[['x','y','z']].tolist())
    standard_train = scalar.transform(Xtrain[['x','y','z']].tolist())
    standard_test = scalar.transform(Xtest[['x','y','z']].tolist())
    Xtrain['x'] = standard_train[:, 0]
    Xtrain['y'] = standard_train[:, 1]
    Xtrain['z'] = standard_train[:, 2]
    Xtest['x'] = standard_test[:, 0]
    Xtest['y'] = standard_test[:, 1]
    Xtest['z'] = standard_test[:, 2]
    return Xtrain, Xtest, scalar


def normalization(Xtrain, Xtest):
    normalizer = Normalizer()
    normalizer.fit(Xtrain[['x','y','z']].tolist())
    normalized_train = normalizer.transform(Xtrain[['x','y','z']].tolist())
    normalized_test = normalizer.transform(Xtest[['x','y','z']].tolist())
    Xtrain['x'] = normalized_train[:, 0]
    Xtrain['y'] = normalized_train[:, 1]
    Xtrain['z'] = normalized_train[:, 2]
    Xtest['x'] = normalized_test[:, 0]
    Xtest['y'] = normalized_test[:, 1]
    Xtest['z'] = normalized_test[:, 2]
    return Xtrain, Xtest, normalizer
