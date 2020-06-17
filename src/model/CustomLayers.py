'''
Paper Title: Road Surface Recognition Based on DeepSense Neural Network using Accelerometer Data
Created by ITS Lab, Institute of Computer Science, University of Tartu
'''

import tensorflow as tf
from tensorflow import keras


class TFGather(keras.layers.Layer):
    def __init__(self, output_dim=1, axis=3, **kwargs):
        """
        Customized layer to get one of window from input tensor
        """
        self.output_dim = output_dim
        self.axis = axis
        self.w = 1
        super(TFGather, self).__init__(**kwargs)
        return

    def build(self, input_shape):
        super(TFGather, self).build(input_shape)
        return

    def set_window(self, w):
        self.w = w
        return

    def call(self, inputs):
        return tf.gather(inputs, indices=[self.w], axis=self.axis)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(TFGather, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TFExpandDim(keras.layers.Layer):
    def __init__(self, output_dim=1, axis=-1, **kwargs):
        """
        Customized layer to expand input tensor's dimension
        """
        self.output_dim = output_dim
        self.axis = axis
        super(TFExpandDim, self).__init__(**kwargs)
        return

    def build(self, input_shape):
        super(TFExpandDim, self).build(input_shape)
        return

    def call(self, inputs):
        return keras.backend.expand_dims(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape.append(1)
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(TFExpandDim, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TFUnstack(keras.layers.Layer):
    def __init__(self, output_dim, axis=1, **kwargs):
        self.output_dim = output_dim
        self.axis = axis
        super(TFUnstack, self).__init__(**kwargs)
        return

    def build(self, input_shape):
        super(TFUnstack, self).build(input_shape)
        return

    def call(self, inputs):
        return tf.unstack(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [(None, self.output_dim) for _ in range(shape[1])]
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(TFUnstack, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
