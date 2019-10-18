import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Lambda

import numpy as np


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Lambda

import numpy as np

def NameLayer(name):
    return Lambda(lambda i: i, name=name)

def sparse_to_tensor(x, dtype=tf.float32):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, tf.convert_to_tensor(coo.data, dtype=dtype), coo.shape)

class PCA_(Layer):
    def __init__(self, components, mean, **kwargs):
        super(PCA_, self).__init__(**kwargs)
        self.components = tf.Variable(components, trainable = False)
        self.mean = tf.Variable(mean, trainable = False)
        self.output_dim = (K.int_shape(self.mean)[0] / 3, 3)

    def build(self, input_shape):
        super(PCA_, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.reshape(tf.matmul(x, self.components) + self.mean, (-1, K.int_shape(self.mean)[0] / 3, 3))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.output_dim[1])

class Scatter_(Layer):
    def __init__(self, vs, NVERTS, **kwargs):
        super(Scatter_, self).__init__(**kwargs)
        from scipy.sparse import coo_matrix
        self.vs = vs
        q = coo_matrix( (np.ones((len(vs),)), (vs, range(len(vs)))), shape=(NVERTS, len(vs)), dtype = np.float32 )
        self.vs = sparse_to_tensor(q)

        self.NVERTS = NVERTS
        self.output_dim = (NVERTS, 3)

    def build(self, input_shape):
        super(Scatter_, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        def fn(a):
            return tf.sparse_tensor_dense_matmul(self.vs, a)
        return tf.map_fn(fn, x)#self.scatter_offsets(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.output_dim[1])