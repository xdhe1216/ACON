from keras.layers import activations, initializers, regularizers, constraints, InputSpec, Layer, conv_utils
import tensorflow as tf


class ACON_C(Layer):
    """
    data_format: A string,
            input feature must be channel last
    """
    def __init__(self,
                 data_format=None,
                 p1_initializer='glorot_uniform',
                 p1_regularizer=None,
                 p1_constraint=None,
                 p2_initializer='glorot_uniform',
                 p2_regularizer=None,
                 p2_constraint=None,
                 **kwargs):
        super(ACON_C, self).__init__(**kwargs)
        self.supports_masking = True
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.p1_initializer = initializers.get(p1_initializer)
        self.p1_regularizer = regularizers.get(p1_regularizer)
        self.p1_constraint = constraints.get(p1_constraint)
        self.p2_initializer = initializers.get(p2_initializer)
        self.p2_regularizer = regularizers.get(p2_regularizer)
        self.p2_constraint = constraints.get(p2_constraint)

    def build(self, input_shape):

        channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        kernel_shape = (1, 1) + (1, input_dim)
        self.p1 = self.add_weight(shape=kernel_shape,
                                  initializer=self.p1_initializer,
                                  name='kernel_p1',
                                  regularizer=self.p1_regularizer,
                                  constraint=self.p1_constraint)

        self.p2 = self.add_weight(shape=kernel_shape,
                                  initializer=self.p2_initializer,
                                  name='kernel_p2',
                                  regularizer=self.p2_regularizer,
                                  constraint=self.p2_constraint)

    def call(self, inputs):
        x, beta = inputs
        x1 = self.p1 * x
        x2 = self.p2 * x
        temp = x1 - x2
        x = temp * tf.sigmoid(beta*temp) + x2
        return x

    def get_config(self):
        config = {

            'data_format': self.data_format,
            'p1_initializer': initializers.serialize(self.p1_initializer),
            'p1_regularizer': regularizers.serialize(self.p1_regularizer),
            'p1_constraint': constraints.serialize(self.p1_constraint),
            'p2_initializer': initializers.serialize(self.p2_initializer),
            'p2_regularizer': regularizers.serialize(self.p2_regularizer),
            'p2_constraint': constraints.serialize(self.p2_constraint),
        }
        base_config = super(ACON_C, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

