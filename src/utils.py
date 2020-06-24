'''
Paper Title: Road Surface Recognition Based on DeepSense Neural Network using Accelerometer Data
Created by ITS Lab, Institute of Computer Science, University of Tartu
'''

# Libraries
import numpy as np
from numpy.lib.recfunctions import repack_fields
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from os import listdir
from os.path import join, isfile, basename, dirname
import multiprocessing as mp
from scipy.signal import butter, lfilter, filtfilt


SAMPLE_DTYPE = np.dtype([('ts', np.float64), ('x_raw', np.float64),
                         ('y_raw', np.float64), ('z_raw', np.float64),
                         ('x', np.float64), ('y', np.float64),
                         ('z', np.float64), ('road', '<U8')])

SAMPLE_DTYPE2 = np.dtype([('ts', np.float64), ('x_raw', np.float64),
                          ('y_raw', np.float64), ('z_raw', np.float64),
                          ('x', np.float64), ('y', np.float64),
                          ('z', np.float64), ('road', np.int8)])

ROAD_TYPE = {'Smooth': 0, 'Bumpy': 1, 'Rough': 2}

ROAD_TYPE_REVERSE = {0: 'Smooth', 1: 'Bumpy', 2: 'Rough'}


"""
LOAD DATA AND REORGANIZE
"""


def load_data(path_to_dir='../data/sensor/RoadSurfaceDataCollector',
              cores=mp.cpu_count(), details=False):
    """
    read the csv sensor files in the folder recursively.

    :param path_to_dir: path to the desired directory
    :param cores: specify how many cores to use to speed up processing
    :param details: to show details outputs (info)
    :return: merged data in Numpy structured array
    """

    # read all files in the directory
    valid_files = [join(path_to_dir, f) for f in listdir(path_to_dir)
                   if isfile(join(path_to_dir, f)) and f.endswith('csv') and not f.endswith('cords.csv')]

    if details:
        print('Files found in the folder: \n'
              '-------------------------------------------------------------\n'
              '%s\n'
              '-------------------------------------------------------------' %
              '\n'.join(valid_files))

    avg = len(valid_files) / cores
    ranges = list((int(i * avg), int((i + 1) * avg)) for i in range(cores))
    queue = mp.Queue()

    def job(worker_id):
        collector = {}
        i_start, i_end = ranges[worker_id]

        # collect data
        for i in range(i_start, i_end):
            # load file into Numpy structured array
            data = np.genfromtxt(valid_files[i], skip_header=1, dtype=SAMPLE_DTYPE, delimiter=',')
            # convert UNIX milliseconds
            data['ts'] = data['ts'] / 1000
            # convert labels to efficient number representation
            for j in range(data.shape[0]):
                # convert road type from string to int representation
                data[j]['road'] = str(ROAD_TYPE[data[j]['road']])
            # change to optimised dtype
            data = data.astype(dtype=SAMPLE_DTYPE2)
            # add to collection
            collector[dirname(valid_files[i]).split('/')[-1]+'_'+basename(valid_files[i])] = data

        queue.put(collector)
        return

    # create processes and execute
    processes = [mp.Process(target=job, args=(x,)) for x in range(cores)]

    # start processes
    for p in processes:
        p.start()

    # get output from workers
    results = [queue.get() for p in processes]

    # wait all processes to stop
    for p in processes:
        p.join()

    # concatenate results -> dict
    total = {}
    for r in results:
        total = {**total, **r}

    return total


def load_folder_recursive(root_dir='../data/sensor/', confirm=False,
                          return_dup=False, clean=True):
    """
    load dataset from the root directory recursively
    :param root_dir: root directory
    :param confirm: whether to sift the directories in the root directory
    :param return_dup: if True, the duplicated data will be returned along with raw data
    :param clean: perform clean&filter process after loading
    :return: merged dataset with index
    """
    collector = {}
    duplicated = {}
    dup_idx = 0
    dup_key = []
    total_files = 0

    # find list of folders in the root folder
    list_dir = find_dir(root_dir, confirm=confirm)
    # load data from the folders
    list_raw = [load_data(folder) for folder in list_dir]
    # merge idx of raw data
    for r in list_raw:
        # record total files
        total_files += len(r.keys())
        # check duplicated keys
        for newkey in r:
            if newkey in collector:
                duplicated[newkey+'_'+str(dup_idx)] = r[newkey]
                dup_idx += 1
                dup_key.append(newkey)
        # remove duplicated
        if dup_idx > 0:
            for k in dup_key:
                del r[k]
        dup_key = []
        # merge dicts
        collector = {**collector, **r}
    print('Total file loaded: %d' % total_files)

    if dup_idx > 0:
        print('Found duplicated keys in results!')

    if return_dup:
        return collector, duplicated

    merged = {**collector, **duplicated}

    if clean:
        cleaned = reorganize(merged)
        return cleaned

    return merged


def find_dir(root_dir, confirm=False):
    """
    find list of dirs in root dir
    :param root_dir: root directory want to search
    :param confirm: let user to sift directories
    :return: list of desired dirs
    """
    list_dir = [f for i, f in enumerate(listdir(root_dir)) if not f.startswith('.')]
    list_dir.sort()
    list_dir = ['%03d' % i + '. ' + f for i, f in enumerate(list_dir)]
    print('Directories found in the folder: \n'
          '-------------------------------------------------------------\n'
          '%s\n'
          '-------------------------------------------------------------' %
          '\n'.join(list_dir))

    if confirm:
        c = input('\ninput the index of folder you do not want: \n(separate with comma, leave empty to keep all)\n')
        if not c == '':
            index = [int(idx) for idx in c.split(',')]
            list_dir = [join(root_dir, d[5:]) for i, d in enumerate(list_dir) if i not in index]
        else:
            list_dir = [join(root_dir, d[5:]) for d in list_dir]
    else:
        list_dir = [join(root_dir, d[5:]) for d in list_dir]

    return list_dir


def reorganize(raw, cores=mp.cpu_count()):
    """
    reorganize the data, eliminating outliers
    :param raw: raw data loaded from files
    :param cores: specify number of cores to use
    :return: the same dictionary format raw data
    """
    # create tasks for workers
    keys = list(raw.keys())
    avg = len(keys) / cores
    ranges = list((int(i * avg), int((i + 1) * avg)) for i in range(cores))
    queue = mp.Queue()

    # examine every data, clean & split into more keys
    def job(worker_id):
        output = {}
        i_start, i_end = ranges[worker_id]

        for i in range(i_start, i_end):
            # iterating each file in the task
            key = keys[i]
            file_cache = raw[key]
            # split file for each road type
            file_cache = separate_road_types(file_cache)

            if len(file_cache) > 1:
                print('>w%d: key %s has been splitted(more than 1 road type)' % (worker_id,key))

            # iterating each splited part
            for j, cache in enumerate(file_cache):
                # remove outliers
                status, cache = remove_standby_acc(cache)

                if len(status) > 0:
                    print('>w%d: key %s, seq %d has been sifted. window' % (worker_id, key,j), status)
                    print('-----------------------------------------------------------')

                if len(file_cache) > 1:
                    output[key+'_'+str(j)] = cache
                else:
                    output[key] = cache

        queue.put((worker_id, output))
        return

    # create processes and execute
    processes = [mp.Process(target=job, args=(x,)) for x in range(cores)]

    for p in processes:
        p.start()

    results = [queue.get() for p in processes]

    for p in processes:
        p.join()

    results.sort(key=lambda x:x[0])
    results = [x[1] for x in results]
    merged_output = {}
    for o in results:
        merged_output = {**merged_output, **o}
    return merged_output


"""
PLOTTING
"""


def plot_waveform(data, show_xyz=True, axis='z', freq=10, raw=False, save='', extra=None, color='r'):
    """
    plot sampled sensor data

    :param data: separated file from func 'sort_and_store'
    :param show_xyz: display 3 axis together in a figure
    :param freq: sampled frequency
    :param raw: if use non-structured 1-D numpy data
    :return: -
    """
    if extra is not None:
        key, road = extra

    if data.shape[0] < 1:
        print('Empty data?')
        return

    fig, ax = plt.subplots(figsize=(15, 5))

    dur_sec = data.shape[0] / freq
    time = np.linspace(0, dur_sec, data.shape[0])

    if raw:
        ax.plot(time, data, color='r', label='raw-data')
    elif show_xyz:
        ax.plot(time, data['x'], color='r', label='x-axis')
        ax.plot(time, data['y'], color='g', label='y-axis')
        ax.plot(time, data['z'], color='b', label='z-axis')
    else:
        ax.plot(time, data[axis], color=color, label=axis+'-axis')

    ax.set_xlim(0, time[-1])

    if extra is not None:
        ax.set_title('Accelerometer; Sampling Frequency: %dHz; Key: %s; Road: %d' %
                     (freq, key, road),
                     fontsize=23)
    else:
        ax.set_title('Accelerometer; Sampling Frequency: %dHz' % freq, fontsize=23)

    ax.set_xlabel('time [s]', fontsize=20)
    ax.set_ylabel('Acceleration Value', fontsize=20)
    plt.legend()

    if len(save) > 0:
        if extra is not None:
            fname = join(save, '%s_%d.png' % (key,road))
        else:
            fname = join(save, 'autosave.png')
        plt.savefig(fname)
    else:
        plt.show()
    return


def check_plots(data, key, func, args=None):
    """
    a function to check plots recursively
    :param data: data want to plot
    :param key: key in the data to separate plots
    :param func: plot function to plot data
    :param args: a dictionary to specify the args used in the function.
                 key:value pairs. leave 'data':None
                 example: {'data':None, 'show_xyz':True,'freq':10}
    :return: -
    """
    keys = np.unique(data[key])

    print('Total plots: %d' % keys.shape[0])
    num = input('Num of plots want to check (<=%d): ' % keys.shape[0])

    if int(num) > keys.shape[0]:
        print('Too less to plot.')

    for i in range(int(num)):
        sample = data[data[key]==keys[i]]

        if isinstance(args, dict):
            args['data'] = sample
            func(**args)
        else:
            func(sample)
    return


def plot_train_hist(hist_dict, val_include=True, savefig=False, name=None):
    savepath = './out/'+name+'_hist.png' if name is not None else './out/train_hist.png'
    figkwarg = {'figsize':(8,8), 'dpi':200}

    if val_include:
        f, axarr = plt.subplots(2, 2, **figkwarg)
    else:
        f, axarr = plt.subplots(1, 2, **figkwarg)
    axarr[0, 0].plot(hist_dict['loss'])
    axarr[0, 0].set_title('loss')
    axarr[0, 0].set(ylabel='loss value')
    axarr[0, 1].plot(hist_dict['acc'])
    axarr[0, 1].set_title('acc')
    axarr[0, 1].set(ylabel='accuracy')
    if val_include:
        axarr[1, 0].plot(hist_dict['val_loss'])
        axarr[1, 0].set_title('val_loss')
        axarr[1, 0].set(ylabel='val_loss value', xlabel='epochs')
        axarr[1, 1].plot(hist_dict['val_acc'])
        axarr[1, 1].set_title('val_acc')
        axarr[1, 1].set(ylabel='accuracy', xlabel='epochs')
    else:
        axarr[0, 0].set(xlabel='epochs')
        axarr[0, 1].set(xlabel='epochs')

    f.subplots_adjust(hspace=0.3, wspace=0.3)

    if savefig:
        plt.savefig(savepath, dpi=200)
    else:
        plt.show()
    return


def plot_data(data, figsize=(10,6), xlabel='epoch', ylabel='loss', title='Loss',
              xbase=100, ybase=0.1, xlim=[-10, 1210], ylim=None, show=True):
    plt.figure(figsize=figsize)
    ax = plt.subplot()
    ax.plot(data)
    loc = plticker.MultipleLocator(base=xbase)
    ax.xaxis.set_major_locator(loc)
    loc = plticker.MultipleLocator(base=ybase)
    ax.yaxis.set_major_locator(loc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(xlim)
    plt.ylim(ylim)

    if show:
        plt.show()
    return


"""
UTILITY
"""


def remove_standby_acc(data, ths=0.06):
    """
    simply remove idle collections which has extremely low deviation
    :param data: data loaded from each file
    :param ths: threshold for determine it's idle
    :return: status indicates the result & sifted data
    """
    status = []
    if data[0]['road'] == 0:
        return status, data

    # create windows and a mask for sift
    windows = window_slice(data.shape[0], segment=101, step=100)
    cache_mask = np.ones(data.shape[0], np.bool)
    for win in windows:
        std = np.std(data[win]['z'])
        if std < ths:
            cache_mask[win[:-1]] = 0
            status.append((std, win[0], win[-1]))

    # apply the mask
    data = data[cache_mask]
    return status, data


def merge_raw(list_data):
    """
    merge raw data read from load_data
    :param list_data: a list of raw data
    :return: merged raw data with unified idx
    """
    init_idx = max(np.unique(list_data[0]['idx'])) + 1

    for i in range(1, len(list_data)):
        new_idx = np.unique(list_data[i]['idx']).shape[0]
        list_data[i] = reset_index(list_data[i], init=init_idx)
        init_idx += new_idx
    return np.concatenate(list_data, axis=0)


def separate_road_types(np_data):
    """
    separate numpy raw data which has more than one road types
    :param np_data: raw numpy data read from 'load_data'
    :return: list of separated numpy raw data
    """
    cc = 0
    count = 0
    road = np_data[0]['road']
    output = []
    start_flag = False
    for i in range(np_data.shape[0]):
        if np_data[i]['road'] == road:
            if np_data[i]['lon'] != -1 and start_flag:
                output.append(np_data[cc:cc + count])
                cc += count
                count = 0
                road = np_data[i]['road']

            # flip switch if has coordinates
            if np_data[i]['lon'] != -1:
                start_flag = True
            else:
                start_flag = False

            count += 1
        else:
            output.append(np_data[cc:cc+count])
            cc += count
            count = 1
            road = np_data[i]['road']

    if count > 1:
        output.append(np_data[cc:cc+count])
    return output


def reset_index(data, init=0):
    """
    reset idx column in dataset
    :param data: data contains idx column
    :param init: initial value of the idx
    :return: data with reorganized idx
    """
    wave_idx = init
    cu, ci, cc = np.unique(data['idx'], return_index=True, return_counts=True)

    for i in range(cu.shape[0]):
        data[ci[i]:ci[i]+cc[i]]['idx'] = wave_idx
        wave_idx += 1
    return data


def check_road_type_in_file(data):
    """
    check road type in the data
    :param data: sensor data in Numpy structured array
    :return: -
    """
    road_types = np.unique(data['road'])
    print('The road types in the data: ')
    for road_type in road_types:
        print(ROAD_TYPE_REVERSE[road_type])
    return


def evaluate_accuracy(pred, truth, p=False):
    """
    evaluate the accuracy of prediction
    :param pred: predicted labels
    :param truth: truth labels
    :param p: print result
    :return: accuracy float number
    """
    if pred.shape[0] != truth.shape[0]:
        print('Labels shape are different!')
        return 0.0

    acc = sum([1 for i in range(pred.shape[0]) if pred[i] == truth[i]])/pred.shape[0]

    if p:
        print('Accuracy: %f' % acc)

    return acc


def window_slice(length, segment=20, step=10, return_range=False):
    """
    get sliced segments' index or segments range tuple
    :param length: data length
    :param segment: segment length
    :param step: window of points stride, if segment==step, means no overlapping window
    :param return_range: if True, return range tuple not index
    :return: list of index or list of range tuple
    """
    if return_range:
        windows = [(i,i+segment) if i+segment<=length else (i,length) for i in range(0, length, step)]
        ths = length
    else:
        windows = [np.arange(i,i+segment) if i+segment<=length else np.arange(i,length) for i in range(0, length, step)]
        ths = length - 1

    if len(windows) > 1 and windows[-2][-1] == ths:
        return windows[:-1]
    else:
        return windows


def get_idx(data, rtype, idx, axis=None):
    """
    get a waveform from data
    :param data: data loaded from 'load_data' or 'sort_and_store'
    :param rtype: road type want to extract
    :param idx: idx want to extract
    :param axis: str, extract axis also
    :return: extracted data
    """
    rtype = ROAD_TYPE[str.title(rtype)]
    extracted = data[(data['road']==rtype)*(data['idx']==idx)]

    if isinstance(axis, str):
        return extracted[axis]
    else:
        return extracted


def save_obj(name, obj, prefix='./out/', suffix='.pkl'):
    with open(join(prefix, name)+suffix, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, prefix='./out/', suffix='.pkl'):
    with open(join(prefix, name)+suffix, 'rb') as f:
        return pickle.load(f)


def show_classes(raw):
    classes, cc = np.unique([raw[k][0]['road'] for k in raw.keys()], return_counts=True)
    print('Classes: ', classes, ' Counts: ', cc)
    return


"""
EXPERIMENTAL
"""


def filter_data_freq_lowpass(data, cut=0.2, order=3, sf=10, plot=True):
    # Determine Nyquist frequency
    nyq = sf / 2
    low = cut / nyq

    b, a = butter(order, low, btype='lowpass', analog=False)
    low_passed = filtfilt(b, a, data)

    if plot:
        plt.plot(np.arange(data.shape[0]), data, 'b-', linewidth=1)
        plt.plot(np.arange(data.shape[0]), low_passed, 'r-', linewidth=3)
        plt.ylabel('Amplitude')
        plt.xlabel('Sequence')
        plt.legend(['raw', 'low_passed'])
        plt.title('Low Pass Filter Cutoff Frequency %.2f' % cut)
        plt.show()
    return low_passed


"""
Rectifying Dataset
"""


def update_data(path_to_dir='../data/sensor/RoadSurfaceDataCollector', cores=mp.cpu_count()):
    new_header = ['timestamp', 'X-raw', 'Y-raw', 'Z-raw', 'X-axis', 'Y-axis', 'Z-axis', 'road_type']
    # function to remove specific fields in a structured numpy array
    rmfield = lambda a, *f: a[[n for n in a.dtype.names if n not in f]]

    # read all files in the directory
    valid_files = [join(path_to_dir, f) for f in listdir(path_to_dir)
                   if isfile(join(path_to_dir, f)) and f.endswith('csv') and not f.endswith('cords.csv')]

    # distribute tasks
    avg = len(valid_files) / cores
    ranges = list((int(i * avg), int((i + 1) * avg)) for i in range(cores))

    def job(worker_id):
        i_start, i_end = ranges[worker_id]

        # collect data
        for i in range(i_start, i_end):
            # load file into Numpy structured array
            data = np.genfromtxt(valid_files[i], skip_header=1, dtype=SAMPLE_DTYPE, delimiter=',')

            # remove field
            data_cleaned = repack_fields(rmfield(data, 'lon', 'lat'))

            np.savetxt(valid_files[i], data_cleaned, delimiter=',', header=','.join(new_header),
                       fmt=','.join(['%d', '%f', '%f', '%f', '%f', '%f', '%f', '%s']))
        return

    # create processes and execute
    processes = [mp.Process(target=job, args=(x,)) for x in range(cores)]

    # start processes
    for p in processes:
        p.start()

    # wait all processes to stop
    for p in processes:
        p.join()

    return


def update_data_recursive(root_dir='../data/sensor/'):
    # find list of folders in the root folder
    list_dir = find_dir(root_dir, confirm=True)
    # load data from the folders
    for folder in list_dir:
        update_data(folder)
    return