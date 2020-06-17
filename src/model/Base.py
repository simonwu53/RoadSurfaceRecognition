'''
Paper Title: Road Surface Recognition Based on DeepSense Neural Network using Accelerometer Data
Created by ITS Lab, Institute of Computer Science, University of Tartu
'''

# Libraries
import numpy as np
from tensorflow import keras
import utils, preprocessing
import multiprocessing as mp
import datetime


# A base framework for keras models, need to implement the model by creating a subclass.
class BaseModel:
    def __init__(self, name):
        # default attributes
        self.name = name
        self.features = None
        self.labels = None
        self.labels_origin = None  # for cross validation
        self.val_labels_origin = None  # for cross validation
        self.val_features = None
        self.val_labels = None
        self.hist = None
        self.hist_dump = []
        self.model = None

        self.OPTIMIZER = None
        self.LOSS = None
        self.METRICS = None

        self.WIN_WIDTH = None
        self.WIN_SIZE = None
        self.N_COMPS = None

        return

    def build_model(self):

        return

    def pre_processing(self, root_dir='../data/sensor/', test_split=0.2,
                       confirm=True, cores=mp.cpu_count(), no_ds_format=True,
                       return_tsfresh=False, return_ae=False):
        """
        Preprocessing of the data
        :param root_dir: directory to the root of the data
        :param test_split: fraction of test data size [0,1]
        :param confirm: select folders before load them
        :param cores: specify number of cores to use
        :param no_ds_format: if True, the final shape will be (n_samples, n_features). otherwise, return ds format
        :param return_tsfresh: if True, the final shape will be (n_samples, n_features). original size of features
        :param return_ae: if True, the final shape will be (n_samples, timesteps, n_features)
        :return: -
        """
        cores = 8 if cores > 8 else cores
        raw = utils.load_folder_recursive(root_dir, confirm=confirm, clean=True)
        self.features, labels, self.val_features, val_labels = \
            preprocessing.extract_features_tsfresh(raw, segment=self.WIN_WIDTH, step=self.WIN_WIDTH - 1,
                                                   T=self.WIN_SIZE, n_comps=self.N_COMPS,
                                                   test_split=test_split, cores=cores, no_ds_format=no_ds_format,
                                                   return_tsfresh=return_tsfresh, return_ae=return_ae)
        self.labels_origin = labels
        self.labels = keras.utils.to_categorical(labels)
        self.val_labels_origin = val_labels
        self.val_labels = keras.utils.to_categorical(val_labels)
        print('Validation feature samples %d' % self.val_features.shape[0])
        print('Training feature samples %d' % self.features.shape[0])
        return

    def compile(self, **kwargs):
        """
        Model Compile
        """
        if len(kwargs.keys()) > 0:
            self.model.compile(optimizer=kwargs['optimizer'], loss=kwargs['loss'], metrics=kwargs['metrics'])
        else:
            self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        return

    def create_callbacks(self, checkpoint_callback=True, earlystop_callback=False,
                         csv_callback=False, patience=40, ckpt_period=1, log='',
                         monitor='val_acc', batch_size=32, plateau_callback=False,
                         plateau_monitor='val_loss', plateau_patience=50,
                         plateau_fractor=0.8, ckpt_path=''):
        if len(ckpt_path) == 0:
            ckpt_path = './out/checkpoint/%s_%s_ckpt.h5' % (datetime.datetime.now().strftime("%Y-%m-%d"),
                                                            self.name)

        callbacks = []
        print('Monitor metric: %s' % monitor)

        """
        CallBacks
        """
        # plateau callback
        if plateau_callback:
            plateau = keras.callbacks.ReduceLROnPlateau(monitor=plateau_monitor,
                                                        fractor=plateau_fractor,
                                                        patience=plateau_patience,
                                                        min_lr=0.0001, verbose=1)
            callbacks.append(plateau)

        # early stop callback
        if earlystop_callback:
            earlystop = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0,
                                                      patience=patience, mode='auto',
                                                      restore_best_weights=True)
            callbacks.append(earlystop)

        # check point callback
        if checkpoint_callback:
            checkpoint = keras.callbacks.ModelCheckpoint(
                ckpt_path,
                monitor=monitor, save_best_only=True, save_weights_only=False,
                mode='auto', period=ckpt_period, verbose=1)
            callbacks.append(checkpoint)

        # TensorBoard callback
        if len(log) > 0:
            tensorboard = keras.callbacks.TensorBoard(log_dir=log, batch_size=batch_size)
            callbacks.append(tensorboard)

        # csv logger callback
        if csv_callback:
            csv_logger = keras.callbacks.CSVLogger('./out/%s_log.csv' % self.name,
                                                   separator=',', append=False)
            callbacks.append(csv_logger)

        # check callbacks size
        if len(callbacks) == 0:
            return None
        return callbacks

    def train(self, checkpoint_callback=True, earlystop_callback=False,
              csv_callback=False, patience=40,
              validation_split=0.0, validation_data=False,
              ckpt_period=1, shuffle=False, log='', epoch=1000, batch_size=128,
              plateau_callback=False, plateau_patience=50,
              plateau_fractor=0.8, ckpt_path=''):
        """
        training the given model by using given feature set and label set
        :param checkpoint_callback: whether to use checkpoint callback
        :param earlystop_callback: whether to use earlystop callback
        :param csv_callback: whether to save training results into a csv file
        :param validation_split: fraction to split validation dataset, will valid after each epoch
        :param validation_data: if True, using validation data, validation_split will be ignored
        :param patience: patience value for early stop monitor
        :param ckpt_period: checkpoint period
        :param shuffle: shuffle the data before each epoch
        :param log: path to the log files, used for tensorboard
        :param epoch: epoches to train
        :param batch_size: batch size used in training, if give 0, full-batch will be used
        :param plateau_callback: whether to use reduce leaning rate on plateau
        :param plateau_patience: epoches to wait before reducing learnning rate
        :param plateau_fractor: actor by which the learning rate will be reduced. new_lr = lr * factor
        :param ckpt_path: path to save the checkpoint
        :return: -
        """
        if self.model is None or self.features is None or self.labels is None:
            print('You have to create features and labels first!')
            return

        # if no batch-size, use full-batch
        batch_size = batch_size if batch_size != 0 else self.features.shape[0]

        """
        Validation
        """
        if validation_data:
            if self.val_features is None or self.val_labels is None:
                print('You dont have validation data set prepared yet!')
                return
            validation_data = (self.val_features, self.val_labels)
        else:
            validation_data = None

        # callbacks
        if validation_split > 0.0 or validation_data is not None:
            monitor = 'val_acc'
            plateau_monitor = 'val_loss'
        else:
            monitor = 'loss'
            plateau_monitor = 'loss'

        ckpt_path = '' if len(ckpt_path) == 0 else ckpt_path

        callbacks = self.create_callbacks(checkpoint_callback=checkpoint_callback,
                                          earlystop_callback=earlystop_callback,
                                          csv_callback=csv_callback, log=log,
                                          patience=patience, ckpt_period=ckpt_period,
                                          monitor=monitor, batch_size=32,
                                          plateau_callback=plateau_callback,
                                          plateau_fractor=plateau_fractor,
                                          plateau_monitor=plateau_monitor,
                                          plateau_patience=plateau_patience,
                                          ckpt_path=ckpt_path)

        """
        Training
        """
        self.hist = self.model.fit(self.features, self.labels,
                                   batch_size=batch_size, epochs=epoch, callbacks=callbacks,
                                   validation_split=validation_split, validation_data=validation_data, shuffle=shuffle)

        """
        Save history
        """
        self.hist_dump.append(self.hist.history)
        return

    def load_dataset(self, features=None, labels=None, val_features=None, val_labels=None,
                     to_cat=True, npz_path=''):
        """
        load dataset from local
        :param features: features used in the training
        :param labels: labels used in the training
        :param val_features: validation data used in testing
        :param val_labels: validation labels used in testing
        :param to_cat: transform labels from class to one-hot category
        :param npz_path: load from local numpy compressed file
        :return: -
        """
        if len(npz_path) > 0:
            npz_file = np.load(npz_path)
            self.features = npz_file['features']
            self.labels = npz_file['labels']
            if 'val_features' in npz_file.keys():
                self.val_features = npz_file['val_features']
                self.val_labels = npz_file['val_labels']
            self.to_1dlabel()
            return

        if features is None or labels is None:
            print('No dataset given.')
            return

        if to_cat:
            labels = keras.utils.to_categorical(labels)
            if val_labels is not None:
                val_labels = keras.utils.to_categorical(val_labels)

        self.features = features
        self.labels = labels
        if val_features is not None and val_labels is not None:
            self.val_features = val_features
            self.val_labels = val_labels
        self.to_1dlabel()
        return

    def load_model(self, path):
        """
        load model from local
        :param path: path to the model
        :return: -
        """
        self.model = keras.models.load_model(path)
        return

    def plot_model(self, name=None, prefix='./out/'):
        """
        save model structure as a picture in local
        :param name: name for this test (filename)
        :param prefix: path to the storage place
        :return:
        """
        suffix = '_model.png'
        filename = prefix+name+suffix if name is not None else prefix+self.name+suffix
        keras.utils.plot_model(model=self.model, to_file=filename % self.name)
        print('Model saved to file: %s' % filename)
        return

    def save_hist(self, name=None, prefix='./out/'):
        """
        save training history
        :param name: name for this test (filename)
        :param prefix: path to the storage place
        :return: -
        """
        filename = name+'_hist' if name is not None else self.name+'_hist'
        utils.save_obj(filename, self.hist_dump, prefix=prefix)
        return

    def save_dataset(self, name=None, prefix='./out/'):
        """
        save all the dataset used in this test
        :param name: name for this test (filename)
        :param prefix: path to the storage place
        :return: -
        """
        filename = prefix+name+'_data.npz' if name is not None else prefix+self.name+'_data.npz'
        np.savez_compressed(filename, features=self.features, labels=self.labels,
                            val_features=self.val_features, val_labels=self.val_labels)
        return

    def save_model(self, name=None, prefix='./out/checkpoint/'):
        """
        save model's current state (all)
        :param name: model name to use
        :param prefix: path to the storage place
        :return: -
        """
        filename = prefix+name+'_model.h5' if name is not None else prefix+self.name+'_model.h5'
        self.model.save(filename)
        return

    def evaluate(self, validation_data=None, batch_size=None):
        """
        perform evaluate action for the model
        :param validation_data: (features, labels) tuple or use its own validation data
        :param batch_size: batch size to use while evaluating
        :return: -
        """
        val_features, val_labels = validation_data if validation_data is not None \
            else (self.val_features, self.val_labels)
        scores = self.model.evaluate(val_features, val_labels, batch_size=batch_size)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        # evaluate for each category
        cat1, cat2, cat3 = [], [], []
        cat = [cat1, cat2, cat3]
        for l in range(val_labels.shape[0]):
            cat[np.argmax(val_labels[l])].append(l)
        for i, c in enumerate(cat):
            cfeatures, clabels = val_features[c], val_labels[c]
            scores = self.model.evaluate(cfeatures, clabels, batch_size=batch_size)
            print('Category %d, Test loss: %.4f' % (i, scores[0]))
            print('Category %d, Test accuracy: %.4f' % (i, scores[1]))
        return

    def plot_hist(self, savefig=True):
        # sum up all histories
        hist_dict = {}
        for hist in self.hist_dump:
            hist_dict = {**hist_dict, **hist}
        val_include = True if self.val_features is not None else False
        utils.plot_train_hist(hist_dict, val_include=val_include,
                              savefig=savefig, name=self.name)
        return

    def to_1dlabel(self):
        if self.labels is not None:
            self.labels_origin = np.zeros(self.labels.shape[0], np.int8)
            for i in range(self.labels.shape[0]):
                self.labels_origin[i] = np.argmax(self.labels[i])

        if self.val_labels is not None:
            self.val_labels_origin = np.zeros(self.val_labels.shape[0], np.int8)
            for i in range(self.val_labels.shape[0]):
                self.val_labels_origin[i] = np.argmax(self.val_labels[i])
        return

    @staticmethod
    def clear_session():
        keras.backend.clear_session()
        return
