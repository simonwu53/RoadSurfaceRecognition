'''
Paper Title: Road Surface Recognition Based on DeepSense Neural Network using Accelerometer Data
Created by ITS Lab, Institute of Computer Science, University of Tartu
'''

from model.Base import BaseModel
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from tensorflow import keras
from functools import reduce


class ConvolutionalNeuralNetwork(BaseModel):
    def __init__(self, name='CNN', preprocess=False, n_gpu=1, **kwargs):
        """
        create CNN instance

        :param name: name of the model
        :param preprocess: start preprocessing data
        :param n_gpu: num of gpu to use
        :param kwargs: model keyword args
        """
        super(ConvolutionalNeuralNetwork, self).__init__(name=name)

        # MODEL hyperparameters
        self.N_COMPS = kwargs['ncomps'] if 'ncomps' in kwargs.keys() else 32
        self.RESHAPE = kwargs['reshape'] if 'reshape' in kwargs.keys() else (2, 16, 1)
        self.WIN_WIDTH = kwargs['winwidth'] if 'winwidth' in kwargs.keys() else 51
        self.WIN_SIZE = kwargs['winsize'] if 'winsize' in kwargs.keys() else 2

        self.CONV1_FILTER = kwargs['c1filter'] if 'c1filter' in kwargs.keys() else 64
        self.CONV1_KERNEL = kwargs['c1kernel'] if 'c1kernel' in kwargs.keys() else [2, 3]
        self.CONV1_STRIDE = kwargs['c1stride'] if 'c1stride' in kwargs.keys() else [1, 1]

        self.CONV2_FILTER = kwargs['c2filter'] if 'c2filter' in kwargs.keys() else 64
        self.CONV2_KERNEL = kwargs['c2kernel'] if 'c2kernel' in kwargs.keys() else [1, 3]
        self.CONV2_STRIDE = kwargs['c2stride'] if 'c2stride' in kwargs.keys() else [1, 1]

        self.CONV3_FILTER = kwargs['c3filter'] if 'c3filter' in kwargs.keys() else 64
        self.CONV3_KERNEL = kwargs['c3kernel'] if 'c3kernel' in kwargs.keys() else [1, 3]
        self.CONV3_STRIDE = kwargs['c3stride'] if 'c3stride' in kwargs.keys() else [1, 1]

        self.DENSE1 = kwargs['dense1'] if 'dense1' in kwargs.keys() else 128
        self.DENSE2 = kwargs['dense2'] if 'dense2' in kwargs.keys() else 32
        self.DENSE3 = kwargs['dense3'] if 'dense3' in kwargs.keys() else 3

        self.LEARNING_RATE = kwargs['lr'] if 'lr' in kwargs.keys() else 1.0
        self.DECAY = kwargs['decay'] if 'decay' in kwargs.keys() else 0.0
        self.OPTIMIZER = kwargs['optimizer'] if 'optimizer' in kwargs.keys() else \
            keras.optimizers.Adadelta(lr=self.LEARNING_RATE, decay=self.DECAY)
        self.LOSS = kwargs['loss'] if 'loss' in kwargs.keys() else 'categorical_crossentropy'
        self.METRICS = kwargs['metrics'] if 'metrics' in kwargs.keys() else ['acc']

        if preprocess:
            self.pre_processing()
            print('Model, features, labels are done.')

        # build model based on kwargs
        self.conv_model = None
        self.dense_model = None
        if n_gpu == 1:
            self.model = self.build_model()
        else:
            with tf.device('/cpu:0'):
                self.cpu_model = self.build_model()
            self.model = multi_gpu_model(self.cpu_model, gpus=n_gpu)
        return

    def build_model(self):
        # First part of the model, contains convolutional layers
        self.conv_model = keras.models.Sequential([
            keras.layers.Reshape(target_shape=self.RESHAPE, input_shape=(self.N_COMPS,)),
            keras.layers.Conv2D(filters=self.CONV1_FILTER, kernel_size=self.CONV1_KERNEL,
                                strides=self.CONV1_STRIDE, padding='valid', activation='relu'),
            keras.layers.Conv2D(filters=self.CONV2_FILTER, kernel_size=self.CONV2_KERNEL,
                                strides=self.CONV2_STRIDE, padding='valid', activation='relu'),
            keras.layers.Conv2D(filters=self.CONV3_FILTER, kernel_size=self.CONV3_KERNEL,
                                strides=self.CONV3_STRIDE, padding='valid', activation='relu',
                                name='conv_out')
        ], name='Convolutional_Subnet')
        # get the output shape of the conv model
        conv_out_shape = self.conv_model.get_layer('conv_out').output_shape
        # Second part of the model, use Dense neuron layers
        self.dense_model = keras.models.Sequential([
            keras.layers.Reshape(target_shape=(reduce(lambda a,b:a*b,conv_out_shape[1:]),),
                                 input_shape=conv_out_shape),
            keras.layers.Dense(units=self.DENSE1, activation='relu'),
            keras.layers.Dense(units=self.DENSE2, activation='relu'),
            keras.layers.Dense(units=self.DENSE3, activation=None),
            keras.layers.Softmax()
        ], name='Full_Connect_Subnet')

        # set up the final combined CNN model
        inputs = keras.Input(shape=(self.N_COMPS,))
        conv = self.conv_model(inputs)
        dense = self.dense_model(conv)

        model = keras.models.Model(inputs=inputs, outputs=dense)
        return model


class NeuralNetwork(BaseModel):
    def __init__(self, name='NN', preprocess=False, n_gpu=1, **kwargs):
        """
        Create NN instance

        :param name: name of the model
        :param preprocess: start preprocessing data
        :param n_gpu: num of gpu to use
        :param kwargs: model keyword args
        """
        super(NeuralNetwork, self).__init__(name=name)

        # MODEL hyperparameters
        self.N_COMPS = kwargs['ncomps'] if 'ncomps' in kwargs.keys() else 32
        self.WIN_WIDTH = kwargs['winwidth'] if 'winwidth' in kwargs.keys() else 51
        self.WIN_SIZE = kwargs['winsize'] if 'winsize' in kwargs.keys() else 2
        self.INPUTSHAPE = (self.N_COMPS,)

        self.DENSE1 = kwargs['dense1'] if 'dense1' in kwargs.keys() else 64
        self.DENSE2 = kwargs['dense2'] if 'dense2' in kwargs.keys() else 128
        self.DENSE3 = kwargs['dense3'] if 'dense3' in kwargs.keys() else 256
        self.DENSE4 = kwargs['dense4'] if 'dense4' in kwargs.keys() else 128
        self.DENSE5 = kwargs['dense5'] if 'dense5' in kwargs.keys() else 32
        self.DENSE6 = kwargs['dense6'] if 'dense6' in kwargs.keys() else 3

        self.LEARNING_RATE = kwargs['lr'] if 'lr' in kwargs.keys() else 1.0
        self.DECAY = kwargs['decay'] if 'decay' in kwargs.keys() else 0.0
        self.OPTIMIZER = kwargs['optimizer'] if 'optimizer' in kwargs.keys() else \
            keras.optimizers.Adadelta(lr=self.LEARNING_RATE, decay=self.DECAY)
        self.LOSS = kwargs['loss'] if 'loss' in kwargs.keys() else 'categorical_crossentropy'
        self.METRICS = kwargs['metrics'] if 'metrics' in kwargs.keys() else ['acc']

        if preprocess:
            self.pre_processing()
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

    def build_model(self):
        model = keras.models.Sequential([
            keras.layers.Dense(units=self.DENSE1, activation='relu',
                               input_shape=self.INPUTSHAPE),
            keras.layers.Dense(units=self.DENSE2, activation='relu'),
            keras.layers.Dense(units=self.DENSE3, activation='relu'),
            keras.layers.Dense(units=self.DENSE4, activation='relu'),
            keras.layers.Dense(units=self.DENSE5, activation='relu'),
            keras.layers.Dense(units=self.DENSE6),
            keras.layers.Softmax()
        ], name='Dense_Model')
        return model
