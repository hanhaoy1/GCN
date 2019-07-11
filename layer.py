from init import *
import tensorflow as tf


_LAYER_UIDS = {}


def get_layer_uids(layer):
    if layer not in _LAYER_UIDS:
        _LAYER_UIDS[layer] = 1
        return 1
    else:
        _LAYER_UIDS[layer] += 1
        return _LAYER_UIDS[layer]


class Layer(object):
    """
    Base Layer class
    # Properties
        name: String
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            i.e. takes input  return output
        __call__(inputs): wrapper for _call()
        _log_vars(): Log all variables
    """
    def __init__(self, **kwargs):
        allowed_kwargs = ['name', 'logging']
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uids(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout=0,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weight'] = glorot([input_dim, output_dim], name='weight')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weight'])
        x = tf.matmul(self.support, x)
        if self.bias:
            x += self.bias
        return self.act(x)






