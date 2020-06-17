'''
Paper Title: Road Surface Recognition Based on DeepSense Neural Network using Accelerometer Data
Created by ITS Lab, Institute of Computer Science, University of Tartu
'''

# Libraries
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
import utils, preprocessing
from model.CustomLayers import TFGather, TFUnstack, TFExpandDim
from model.Base import BaseModel


class DeepSenseTS(BaseModel):

    def __init__(self, preprocess=False, name='DeepSenseTS', n_gpu=1, **kwargs):
        """
        initializing DeepSense configuration
        :param preprocess: start preprocessing data
        :param name: name of the this model
        :param n_gpu: specify number of gpus to use
        :param kwargs: keyword arguments, hyperparams for neural network
        """
        super(DeepSenseTS, self).__init__(name=name)
        self.ics_model = None
        self.mcs_model = None
        self.gru_model = None

        # MODEL hyperparameters

        ########################
        # Model Compile
        ########################
        self.LEARNING_RATE = kwargs['lr'] if 'lr' in kwargs.keys() else 0.001
        self.DECAY = kwargs['decay'] if 'decay' in kwargs.keys() else 0.0
        self.OPTIMIZER = kwargs['optimizer'] if 'optimizer' in kwargs.keys() else \
            keras.optimizers.RMSprop(lr=self.LEARNING_RATE, decay=self.DECAY)
        self.LOSS = kwargs['loss'] if 'loss' in kwargs.keys() else 'categorical_crossentropy'
        self.METRICS = kwargs['metrics'] if 'metrics' in kwargs.keys() else ['acc']

        ########################
        # Data Config
        ########################
        self.WIN_SIZE = kwargs['winsize'] if 'winsize' in kwargs.keys() else 2
        self.WIN_WIDTH = kwargs['winwidth'] if 'winwidth' in kwargs.keys() else 51
        # for window const to put win_width
        self.WIN_SHAPE = kwargs['winshape'] if 'winshape' in kwargs.keys() else [1, 1]
        # window width in seconds
        self.WIN_T = kwargs['winT'] if 'winT' in kwargs.keys() else 5
        self.K_SIZE = kwargs['Ksize'] if 'Ksize' in kwargs.keys() else 1
        self.N_COMPS = kwargs['ncomps'] if 'ncomps' in kwargs.keys() else 10

        ########################
        # CNN params
        ########################
        self.CONV1_KERNEL = kwargs['c1kernel'] if 'c1kernel' in kwargs.keys() else [1, 5]
        self.CONV1_STRIDE = kwargs['c1stride'] if 'c1stride' in kwargs.keys() else [1, 1]

        self.CONV2_KERNEL = kwargs['c2kernel'] if 'c2kernel' in kwargs.keys() else [1, 3]
        self.CONV2_STRIDE = kwargs['c2stride'] if 'c2stride' in kwargs.keys() else [1, 1]

        self.CONV3_KERNEL = kwargs['c3kernel'] if 'c3kernel' in kwargs.keys() else [1, 3]
        self.CONV3_STRIDE = kwargs['c3stride'] if 'c3stride' in kwargs.keys() else [1, 1]

        self.CONV4_KERNEL = kwargs['c4kernel'] if 'c4kernel' in kwargs.keys() else [self.K_SIZE, 5]
        self.CONV4_STRIDE = kwargs['c4stride'] if 'c4stride' in kwargs.keys() else [self.K_SIZE, 1]

        self.CONV5_KERNEL = kwargs['c5kernel'] if 'c5kernel' in kwargs.keys() else [1, 3]
        self.CONV5_STRIDE = kwargs['c5stride'] if 'c5stride' in kwargs.keys() else [1, 1]

        self.CONV6_KERNEL = kwargs['c6kernel'] if 'c6kernel' in kwargs.keys() else [1, 3]
        self.CONV6_STRIDE = kwargs['c6stride'] if 'c6stride' in kwargs.keys() else [1, 1]

        self.GRU_DROPOUT = kwargs['grudropout'] if 'grudropout' in kwargs.keys() else 0.5
        self.GRU_UNITS_1 = kwargs['gruunits1'] if 'gruunits1' in kwargs.keys() else 10
        self.GRU_UNITS_2 = kwargs['gruunits2'] if 'gruunits2' in kwargs.keys() else 3

        self.INPUT_SHAPE = (1, self.N_COMPS, self.WIN_SIZE)  # (d, n_features, T)
        self.CONV_FILTER = kwargs['cfilter'] if 'cfilter' in kwargs.keys() else 128

        ########################
        # Model Input Shapes
        ########################
        self.ICS_INPUT_SHAPE = (1, self.N_COMPS, 1)
        self.MCS_INPUT_SHAPE = (1, (self.N_COMPS - (self.CONV1_KERNEL[1] - 1) - (self.CONV2_KERNEL[1] - 1) -
                                    (self.CONV3_KERNEL[1] - 1)) * self.CONV_FILTER, 1)
        self.GRU_INPUT_SHAPE = (self.WIN_SIZE, (self.MCS_INPUT_SHAPE[1] - (self.CONV4_KERNEL[1] - 1) -
                                                (self.CONV5_KERNEL[1] - 1) - (self.CONV6_KERNEL[1] - 1)) *
                                self.CONV_FILTER + 1)

        if preprocess:
            self.pre_processing_ds()
            print('Model, features, labels are done.')

        # build model based on config
        if n_gpu == 1:
            self.cpu_model = None
            self.model = self.build_model()
        else:
            with tf.device('/cpu:0'):
                self.cpu_model = self.build_model()
            self.model = multi_gpu_model(self.cpu_model, gpus=n_gpu)
        return

    def build_model(self, summary=False, detail=False):
        """
        DeepSense framework implementation using keras API
        :param summary: print summary after building the model
        :param detail: print extra info
        :return: DeepSense model instance
        """
        if detail:
            print('Initializing DeepSense Model, it may take a while...')

        # create sub module models
        """
        Individual Convolutional Subnet
        """
        self.ics_model = [keras.models.Sequential([
            ##############
            # first conv layer, 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV1_KERNEL,
                                strides=self.CONV1_STRIDE, padding='valid',
                                input_shape=self.ICS_INPUT_SHAPE),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            ##############
            # second conv layer, with 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV2_KERNEL,
                                strides=self.CONV2_STRIDE, padding='valid'),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            ##############
            # third conv layer, with 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV3_KERNEL,
                                strides=self.CONV3_STRIDE, padding='valid'),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            # flatten results
            keras.layers.Reshape(target_shape=(1, int(self.MCS_INPUT_SHAPE[1])))
        ], name='Individual_Convolutional_Subnet_%d' % i) for i in range(self.K_SIZE)]

        """
        Merge Convolutional Subnet
        """
        self.mcs_model = keras.models.Sequential([
            ##############
            # first conv layer, 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV4_KERNEL,
                                strides=self.CONV4_STRIDE, padding='valid',
                                input_shape=self.MCS_INPUT_SHAPE),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            ##############
            # second conv layer, with 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV5_KERNEL,
                                strides=self.CONV5_STRIDE, padding='valid'),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            ##############
            # third conv layer, with 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV6_KERNEL,
                                strides=self.CONV6_STRIDE, padding='valid'),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            # flatten results
            keras.layers.Reshape(target_shape=(1, int(self.GRU_INPUT_SHAPE[1]-1))),
        ], name='Merge_Convolutional_Subnet')

        """
        GRU model
        """
        self.gru_model = keras.models.Sequential([
            ##############
            # Recurrent Layers
            # using GRU
            ##############
            keras.layers.GRU(units=self.GRU_UNITS_1, return_sequences=True,
                             unroll=True, activation='tanh', input_shape=self.GRU_INPUT_SHAPE),

            # keras.layers.BatchNormalization()(gru1)
            keras.layers.Dropout(rate=self.GRU_DROPOUT),

            keras.layers.GRU(units=self.GRU_UNITS_2, return_sequences=True,
                             unroll=True, activation='tanh')
        ], name='Stacked_GRU_Model')

        individual_subnet_output = []
        merge_subnet_output = []

        # create Input Layers: 1 for window t, k for all sensors
        t = keras.Input(shape=self.WIN_SHAPE)
        k_inputs = [keras.Input(shape=self.INPUT_SHAPE) for _ in range(self.K_SIZE)]
        k_inputs.append(t)

        config = {'trainable':False}
        expand_dim_layer = TFExpandDim(axis=-1, **config)
        unstack_layer = TFUnstack(output_dim=self.GRU_UNITS_2, **config)
        inputs_i_layer = TFGather(axis=3, **config)

        ##############
        # For each time-window, for each sensor data, apply 3 layers of convolution
        # Concat the output of each sensor, apply the 3 layer convolution again
        # Flatten the output of the last layer and concat with window t.
        # Then feed the (win_size, features) tensor to stacked GRU
        ##############
        for win in range(self.WIN_SIZE):
            for sensor_k in range(self.K_SIZE):

                # get specific model for this sensor
                model = self.ics_model[sensor_k]
                # use current sensor's input and window slice to get current window features
                # get one window of the input
                inputs_i_layer.set_window(win)
                inputs_i = inputs_i_layer(k_inputs[sensor_k])

                # get output of individual convolutional subnet for this sensor
                ics_out = model(inputs_i)

                individual_subnet_output.append(ics_out)

            # concat individual_subnet_output to form a new matrix
            if self.K_SIZE != 1:
                merged = keras.layers.concatenate(individual_subnet_output, axis=1)
            else:
                merged = individual_subnet_output[0]

            # prepare merged tensor for mcs model
            merged = expand_dim_layer(merged)
            individual_subnet_output = []

            # get output of merged convolutional subnet for concatenated sensors output
            msc_out = self.mcs_model(merged)

            # concatenate mcs output with window size t
            msc_out = keras.layers.concatenate(inputs=[msc_out, t], axis=2)
            merge_subnet_output.append(msc_out)

        # concatenate all windows outputs
        conv_output = keras.layers.concatenate(merge_subnet_output, axis=1)

        # get output from GRU model
        gru_out = self.gru_model(conv_output)

        ##############
        # Output Layers
        # Using averaging features over time -> final feature
        # Then feed it to softmax layer
        ##############
        unstacked = unstack_layer(gru_out)
        avg_f = keras.layers.average(inputs=unstacked)
        output = keras.layers.Softmax()(avg_f)

        # build the final model
        model = keras.Model(inputs=k_inputs, outputs=output)

        if summary:
            print(model.summary())
        if detail:
            print('DeepSense model created.')
        return model

    def pre_processing_ds(self, root_dir='../data/sensor/', confirm=True, separate=False,
                          test_split=0.2, normalize=False, standardize=False):
        """
        preprocessing raw data to feed into DeepSense network
        :param root_dir: root directory to store dataset files
        :param separate: read each directory separately
        :param confirm: sift directory before reading datasets
        :param test_split: size for the test set
        :param normalize: normalize data before feature extraction
        :param standardize: standardize data before feature extraction
        :return: (raw data), features dataset, labels dataset
        """
        if separate:
            dirs = utils.find_dir(root_dir, confirm=True)
            raw = [utils.load_data(d) for d in dirs]
            raw = [utils.reorganize(r) for r in raw]
            cores = 4
        else:
            raw = utils.load_folder_recursive(root_dir, confirm=confirm, clean=True)
            cores = 8

        self.features, labels, self.val_features, val_labels = \
            preprocessing.extract_features_tsfresh(raw, segment=self.WIN_WIDTH, step=self.WIN_WIDTH-1,
                                                   T=self.WIN_SIZE, normalize=normalize, standardize=standardize,
                                                   n_comps=self.N_COMPS, test_split=test_split, cores=cores)
        self.labels_origin = labels
        self.labels = keras.utils.to_categorical(labels)
        self.val_labels_origin = val_labels
        self.val_labels = keras.utils.to_categorical(val_labels)
        print('Validation feature samples %d' % self.val_features.shape[0])
        print('Training feature samples %d' % self.features.shape[0])

        return

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
        :param ckpt_period: number of epochs to save checkpoint
        :param shuffle: shuffle the data before each epoch
        :param log: path to the log files, used for tensorboard
        :param epoch: epoches to train
        :param batch_size: batch size used in training, if give 0, full-batch will be used
        :param plateau_callback: whether to use reduce leaning rate on plateau
        :param plateau_patience: epoches to wait before reducing learnning rate
        :param plateau_fractor: factor by which the learning rate will be reduced. new_lr = lr * factor
        :param ckpt_path: path to save the checkpoint
        :return: -
        """
        if type(self.model) == type(None) or type(self.features) == type(None) or type(self.labels) == type(None):
            print('You have to create features and labels first!')
            return

        # if no batch-size, use full-batch
        batch_size = batch_size if batch_size !=0 else self.features.shape[0]

        """
        Validation
        """
        if validation_data:
            if type(self.val_features) == type(None) or type(self.val_labels) == type(None):
                print('You dont have validation data set prepared yet!')
                return
            validation_data = ([self.val_features, np.full((self.val_features.shape[0],1,1),self.WIN_T)],
                               self.val_labels)
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
        self.hist = self.model.fit([self.features, np.full((self.features.shape[0], 1, 1), self.WIN_T)], self.labels,
                                   batch_size=batch_size, epochs=epoch, callbacks=callbacks,
                                   validation_split=validation_split, validation_data=validation_data, shuffle=shuffle)

        """
        Save history
        """
        self.hist_dump.append(self.hist.history)
        return

    def load_model(self, path):
        """
        load model from local
        :param path: path to the model
        :return: -
        """
        self.model = keras.models.load_model(path, custom_objects={'TFGather':TFGather,
                                                                   'TFExpandDim':TFExpandDim,
                                                                   'TFUnstack':TFUnstack})
        return

    def evaluate(self, validation_data=None, win_t=None, batch_size=None):
        """
        perform evaluate action for the model
        :param validation_data: (features, labels) tuple or use its own validation data
        :param win_t: time window used in input data, if None, use its own default value
        :param batch_size: batch size to use while evaluating
        :return: -
        """
        T = win_t if win_t is not None else self.WIN_T
        val_features, val_labels = validation_data if validation_data is not None \
            else (self.val_features, self.val_labels)
        scores = self.model.evaluate([val_features, np.full((val_features.shape[0],1,1),T)],val_labels,
                                     batch_size=batch_size)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        # evaluate for each category
        cat1, cat2, cat3 = [], [], []
        cat = [cat1, cat2, cat3]
        for l in range(val_labels.shape[0]):
            cat[np.argmax(val_labels[l])].append(l)
        for i, c in enumerate(cat):
            cfeatures, clabels = val_features[c], val_labels[c]
            scores = self.model.evaluate([cfeatures, np.full((clabels.shape[0], 1, 1), T)], clabels,
                                         batch_size=batch_size)
            print('Category %d, Test loss: %.4f' % (i, scores[0]))
            print('Category %d, Test accuracy: %.4f' % (i, scores[1]))
        return

    def reset_model(self):
        """
        reset instance
        """
        self.ics_model = None
        self.mcs_model = None
        self.gru_model = None
        self.clear_session()
        self.cpu_model = None
        self.model = self.build_model()
        self.compile(**{'optimizer': keras.optimizers.RMSprop(lr=self.LEARNING_RATE, decay=self.DECAY),
                        'loss': 'categorical_crossentropy',
                        'metrics': ['acc']})
        return


"""
Origin DeepSense implementation
"""


class DeepSenseOrigin(BaseModel):

    def __init__(self, preprocess=False, name='DeepSense', n_gpu=1, **kwargs):
        super(DeepSenseOrigin, self).__init__(name=name)
        self.ics_model = None
        self.mcs_model = None
        self.gru_model = None

        # MODEL hyperparameters

        ########################
        # Model Compile
        ########################
        self.LEARNING_RATE = kwargs['lr'] if 'lr' in kwargs.keys() else 0.001
        self.DECAY = kwargs['decay'] if 'decay' in kwargs.keys() else 0.9
        self.OPTIMIZER = kwargs['optimizer'] if 'optimizer' in kwargs.keys() else \
            keras.optimizers.RMSprop(lr=self.LEARNING_RATE, decay=self.DECAY)
        self.LOSS = kwargs['loss'] if 'loss' in kwargs.keys() else 'categorical_crossentropy'
        self.METRICS = kwargs['metrics'] if 'metrics' in kwargs.keys() else ['acc']

        ########################
        # Data Config
        ########################
        self.WIN_SIZE = kwargs['winsize'] if 'winsize' in kwargs.keys() else 10
        self.WIN_WIDTH = kwargs['winwidth'] if 'winwidth' in kwargs.keys() else 11
        # for window const to put win_width
        self.WIN_SHAPE = kwargs['winshape'] if 'winshape' in kwargs.keys() else [1, 1]
        # window width in seconds
        self.WIN_T = kwargs['winT'] if 'winT' in kwargs.keys() else 1
        self.K_SIZE = kwargs['Ksize'] if 'Ksize' in kwargs.keys() else 1
        self.N_COMPS = kwargs['ncomps'] if 'ncomps' in kwargs.keys() else 2*self.WIN_WIDTH

        ########################
        # CNN params
        ########################
        self.CONV1_KERNEL = kwargs['c1kernel'] if 'c1kernel' in kwargs.keys() else [3, 2]
        self.CONV1_STRIDE = kwargs['c1stride'] if 'c1stride' in kwargs.keys() else [1, 1]

        self.CONV2_KERNEL = kwargs['c2kernel'] if 'c2kernel' in kwargs.keys() else [1, 3]
        self.CONV2_STRIDE = kwargs['c2stride'] if 'c2stride' in kwargs.keys() else [1, 1]

        self.CONV3_KERNEL = kwargs['c3kernel'] if 'c3kernel' in kwargs.keys() else [1, 2]
        self.CONV3_STRIDE = kwargs['c3stride'] if 'c3stride' in kwargs.keys() else [1, 1]

        self.CONV4_KERNEL = kwargs['c4kernel'] if 'c4kernel' in kwargs.keys() else [self.K_SIZE, 2]
        self.CONV4_STRIDE = kwargs['c4stride'] if 'c4stride' in kwargs.keys() else [self.K_SIZE, 1]

        self.CONV5_KERNEL = kwargs['c5kernel'] if 'c5kernel' in kwargs.keys() else [1, 3]
        self.CONV5_STRIDE = kwargs['c5stride'] if 'c5stride' in kwargs.keys() else [1, 1]

        self.CONV6_KERNEL = kwargs['c6kernel'] if 'c6kernel' in kwargs.keys() else [1, 2]
        self.CONV6_STRIDE = kwargs['c6stride'] if 'c6stride' in kwargs.keys() else [1, 1]

        self.GRU_DROPOUT = kwargs['grudropout'] if 'grudropout' in kwargs.keys() else 0.5
        self.GRU_UNITS_1 = kwargs['gruunits1'] if 'gruunits1' in kwargs.keys() else 128
        self.GRU_UNITS_2 = kwargs['gruunits2'] if 'gruunits2' in kwargs.keys() else 3

        self.INPUT_SHAPE = (3, self.N_COMPS, self.WIN_SIZE)  # (d, n_features, T)
        self.CONV_FILTER = kwargs['cfilter'] if 'cfilter' in kwargs.keys() else 64

        ########################
        # Model Input Shapes
        ########################
        self.ICS_INPUT_SHAPE = (3, self.N_COMPS, 1)
        self.MCS_INPUT_SHAPE = (1, (self.N_COMPS - (self.CONV1_KERNEL[1] - 1) - (self.CONV2_KERNEL[1] - 1) -
                                    (self.CONV3_KERNEL[1] - 1)) * self.CONV_FILTER, 1)
        self.GRU_INPUT_SHAPE = (self.WIN_SIZE, (self.MCS_INPUT_SHAPE[1] - (self.CONV4_KERNEL[1] - 1) -
                                                (self.CONV5_KERNEL[1] - 1) - (self.CONV6_KERNEL[1] - 1)) *
                                self.CONV_FILTER + 1)

        if preprocess:
            self.pre_processing_ds()
            print('Model, features, labels are done.')

        # build model based on config
        if n_gpu == 1:
            self.cpu_model = None
            self.model = self.build_model()
        else:
            with tf.device('/cpu:0'):
                self.cpu_model = self.build_model()
            self.model = multi_gpu_model(self.cpu_model, gpus=n_gpu)
        return

    def build_model(self, summary=False, detail=False):
        """
        DeepSense framework implementation using keras API
        :param summary: print summary after building the model
        :param detail: print extra info
        :return: DeepSense model object
        """
        if detail:
            print('Initializing DeepSense Model, it may take a while...')

        # create sub module models
        """
        Individual Convolutional Subnet
        """
        self.ics_model = [keras.models.Sequential([
            ##############
            # first conv layer, 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV1_KERNEL,
                                strides=self.CONV1_STRIDE, padding='valid',
                                input_shape=self.ICS_INPUT_SHAPE),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            ##############
            # second conv layer, with 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV2_KERNEL,
                                strides=self.CONV2_STRIDE, padding='valid'),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            ##############
            # third conv layer, with 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV3_KERNEL,
                                strides=self.CONV3_STRIDE, padding='valid'),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            # flatten results
            keras.layers.Reshape(target_shape=(1, int(self.MCS_INPUT_SHAPE[1])))
        ], name='Individual_Convolutional_Subnet_%d' % i) for i in range(self.K_SIZE)]

        """
        Merge Convolutional Subnet
        """
        self.mcs_model = keras.models.Sequential([
            ##############
            # first conv layer, 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV4_KERNEL,
                                strides=self.CONV4_STRIDE, padding='valid',
                                input_shape=self.MCS_INPUT_SHAPE),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            ##############
            # second conv layer, with 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV5_KERNEL,
                                strides=self.CONV5_STRIDE, padding='valid'),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            ##############
            # third conv layer, with 2D convolution
            # with batch normalization and relu activation
            ##############
            keras.layers.Conv2D(filters=self.CONV_FILTER, kernel_size=self.CONV6_KERNEL,
                                strides=self.CONV6_STRIDE, padding='valid'),
            keras.layers.BatchNormalization(scale=False),
            keras.layers.ReLU(),

            # flatten results
            keras.layers.Reshape(target_shape=(1, int(self.GRU_INPUT_SHAPE[1] - 1))),
        ], name='Merge_Convolutional_Subnet')

        """
        GRU model
        """
        self.gru_model = keras.models.Sequential([
            ##############
            # Recurrent Layers
            # using GRU
            ##############
            keras.layers.GRU(units=self.GRU_UNITS_1, return_sequences=True,
                             unroll=True, activation='tanh', input_shape=self.GRU_INPUT_SHAPE),

            # keras.layers.BatchNormalization()(gru1)
            keras.layers.Dropout(rate=self.GRU_DROPOUT),

            keras.layers.GRU(units=self.GRU_UNITS_2, return_sequences=True,
                             unroll=True, activation='tanh')
        ], name='Stacked_GRU_Model')

        individual_subnet_output = []
        merge_subnet_output = []

        # create Input Layers: 1 for window t, k for all sensors
        t = keras.Input(shape=self.WIN_SHAPE)
        k_inputs = [keras.Input(shape=self.INPUT_SHAPE) for _ in range(self.K_SIZE)]
        k_inputs.append(t)

        config = {'trainable': False}
        expand_dim_layer = TFExpandDim(axis=-1, **config)
        unstack_layer = TFUnstack(output_dim=self.GRU_UNITS_2, **config)
        inputs_i_layer = TFGather(axis=3, **config)

        ##############
        # For each time-window, for each sensor data, apply 3 layers of convolution
        # Concat the output of each sensor, apply the 3 layer convolution again
        # Flatten the output of the last layer and concat with window t.
        # Then feed the (win_size, features) tensor to stacked GRU
        ##############
        for win in range(self.WIN_SIZE):
            for sensor_k in range(self.K_SIZE):
                # get specific model for this sensor
                model = self.ics_model[sensor_k]
                # use current sensor's input and window slice to get current window features
                # get one window of the input
                inputs_i_layer.set_window(win)
                inputs_i = inputs_i_layer(k_inputs[sensor_k])

                # get output of individual convolutional subnet for this sensor
                ics_out = model(inputs_i)

                individual_subnet_output.append(ics_out)

            # concat individual_subnet_output to form a new matrix
            if self.K_SIZE != 1:
                merged = keras.layers.concatenate(individual_subnet_output, axis=1)
            else:
                merged = individual_subnet_output[0]

            # prepare merged tensor for mcs model
            merged = expand_dim_layer(merged)
            individual_subnet_output = []

            # get output of merged convolutional subnet for concatenated sensors output
            msc_out = self.mcs_model(merged)

            # concatenate mcs output with window size t
            msc_out = keras.layers.concatenate(inputs=[msc_out, t], axis=2)
            merge_subnet_output.append(msc_out)

        # concatenate all windows outputs
        conv_output = keras.layers.concatenate(merge_subnet_output, axis=1)

        # get output from GRU model
        gru_out = self.gru_model(conv_output)

        ##############
        # Output Layers
        # Using averaging features over time -> final feature
        # Then feed it to softmax layer
        ##############
        unstacked = unstack_layer(gru_out)
        avg_f = keras.layers.average(inputs=unstacked)
        output = keras.layers.Softmax()(avg_f)

        # build the final model
        model = keras.Model(inputs=k_inputs, outputs=output)

        if summary:
            print(model.summary())
        if detail:
            print('DeepSense model created.')
        return model

    def pre_processing_ds(self, root_dir='../data/sensor/', confirm=True,
                          return_raw=False, balance=True, test_split=0.2):
        """
        preprocessing raw data to feed into DeepSense network
        :param root_dir: root directory to store dataset files
        :param confirm: sift directory before reading datasets
        :param segment: window size while creating features
        :param step: number of offset points the window moves
        :param return_raw: return raw data with features & labels
        :param balance: balance every category to have the same quantity of samples
        :param test_split: 0-1 float, fraction of the size for test set
        :return: (raw data), features dataset, labels dataset
        """
        # load files recursively from root folder
        raw = utils.load_folder_recursive(root_dir, confirm=confirm, clean=True)
        # create features, labels
        self.features, labels, self.val_features, val_labels = \
            preprocessing.extract_features_deepsense(raw, segment=self.WIN_WIDTH, step=(self.WIN_WIDTH-1),
                                                     balance=balance, T=self.WIN_SIZE,
                                                     test_split=test_split)
        self.labels_origin = labels
        self.labels = keras.utils.to_categorical(labels)
        self.val_labels = keras.utils.to_categorical(val_labels)

        if return_raw:
            return raw, self.features, self.labels
        return

    def train(self, checkpoint_callback=True, earlystop_callback=False,
              csv_callback=False, patience=40,
              validation_split=0.0, validation_data=False,
              ckpt_period=1, shuffle=False, log='', epoch=50, batch_size=20,
              plateau_callback=False, plateau_patience=50,
              plateau_fractor=0.8, ckpt_path=''):
        """
        training the given model by using given feature set and label set
        :param feature: feature set of shape [(samples,) + INPUT_SHAPE]
        :param label: label set of shape [(samples,) + 3]
        :param model: keras model
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
        if type(self.model) == type(None) or type(self.features) == type(None) or type(self.labels) == type(None):
            print('You have to create features and labels first!')
            return

        # if no batch-size, use full-batch
        batch_size = batch_size if batch_size !=0 else self.features.shape[0]

        """
        Validation
        """
        if validation_data:
            if type(self.val_features) == type(None) or type(self.val_labels) == type(None):
                print('You dont have validation data set prepared yet!')
                return
            validation_data = ([self.val_features, np.full((self.val_features.shape[0],1,1),self.WIN_T)],
                               self.val_labels)
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
        self.hist = self.model.fit([self.features, np.full((self.features.shape[0], 1, 1), self.WIN_T)], self.labels,
                                   batch_size=batch_size, epochs=epoch, callbacks=callbacks,
                                   validation_split=validation_split, validation_data=validation_data, shuffle=shuffle)

        """
        Save history
        """
        self.hist_dump.append(self.hist.history)
        return

    def load_model(self, path):
        """
        load model from local
        :param path: path to the model
        :return: -
        """
        self.model = keras.models.load_model(path, custom_objects={'TFGather':TFGather,
                                                                   'TFExpandDim':TFExpandDim,
                                                                   'TFUnstack':TFUnstack})
        return

    def evaluate(self, validation_data=None, win_t=None, batch_size=None):
        """
        perform evaluate action for the model
        :param validation_data: (features, labels) tuple or use its own validation data
        :param win_t: time window used in input data, if None, use its own default value
        :param batch_size: batch size to use while evaluating
        :return: -
        """
        T = win_t if win_t is not None else self.WIN_T
        val_features, val_labels = validation_data if validation_data is not None \
            else (self.val_features, self.val_labels)
        scores = self.model.evaluate([val_features, np.full((val_features.shape[0],1,1),T)],val_labels,
                                     batch_size=batch_size)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        # evaluate for each category
        cat1, cat2, cat3 = [], [], []
        cat = [cat1, cat2, cat3]
        for l in range(val_labels.shape[0]):
            cat[np.argmax(val_labels[l])].append(l)
        for i, c in enumerate(cat):
            cfeatures, clabels = val_features[c], val_labels[c]
            scores = self.model.evaluate([cfeatures, np.full((clabels.shape[0], 1, 1), T)], clabels,
                                         batch_size=batch_size)
            print('Category %d, Test loss: %.4f' % (i, scores[0]))
            print('Category %d, Test accuracy: %.4f' % (i, scores[1]))
        return

    def reset_model(self):
        self.ics_model = None
        self.mcs_model = None
        self.gru_model = None
        self.clear_session()
        self.cpu_model = None
        self.model = self.build_model()
        self.compile(**{'optimizer': keras.optimizers.RMSprop(lr=self.LEARNING_RATE, decay=self.DECAY),
                        'loss': 'categorical_crossentropy',
                        'metrics': ['acc']})
        return
