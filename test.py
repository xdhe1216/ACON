from keras.layers import Input, Conv2D, MaxPool2D, Activation, GlobalAvgPool2D, BatchNormalization, \
    Flatten, Dense, Reshape
from keras.models import Model
from keras.datasets import mnist
import cv2
from keras.optimizers import Adam
import keras
import numpy as np
from ACON_C import ACON_C


def meta_acon(inputs, r=16):
    in_dims = int(inputs.shape[-1])
    temp_dims = max(r, in_dims//r)
    x = GlobalAvgPool2D()(inputs)
    x = Reshape((1, 1, in_dims))(x)
    x = Conv2D(temp_dims, 1)(x)
    x = BatchNormalization()(x)
    x = Conv2D(in_dims, 1)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='sigmoid')(x)
    x = ACON_C()([inputs, x])
    return x


class CNN(object):
    def __init__(self, input_shape, cls_num):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.cls_num = cls_num
        self.basic_filter = 16

    def __call__(self, *args, **kwargs):
        inputs = Input(shape=self.input_shape)
        x = inputs
        x = Conv2D(64, 3, padding='same')(x)

        name = 'conv_bn_1'
        x = self._conv_bn(x, self.basic_filter, kernel_size=3, activation='relu', name=name)
        name = 'conv_bn_2'
        x = self._conv_bn(x, self.basic_filter, kernel_size=3, activation='relu', name=name)
        x = MaxPool2D(name='maxpool_1')(x)

        name = 'conv_bn_3'
        x = self._conv_bn(x, self.basic_filter, kernel_size=3, activation='relu', name=name)
        name = 'conv_bn_4'
        x = self._conv_bn(x, self.basic_filter, kernel_size=3, activation='relu', name=name)
        x = MaxPool2D(name='maxpool_2')(x)

        name = 'conv_bn_5'
        x = self._conv_bn(x, self.basic_filter, kernel_size=3, activation='relu', name=name)
        name = 'conv_bn_6'
        x = self._conv_bn(x, self.basic_filter, kernel_size=3, activation='relu', name=name)
        x = MaxPool2D(name='maxpool_3')(x)

        x = Flatten()(x)
        x = Dense(self.cls_num, activation='softmax')(x)
        return Model(inputs, x)

    def _conv_bn(self, inputs, filters, kernel_size, activation, name):
        x = Conv2D(filters, kernel_size, padding='same', name=name + '_Conv2D')(inputs)
        x = BatchNormalization()(x)
        x = meta_acon(x, 16)

        return x


if __name__ == '__main__':
    input_shape = (32, 32, 3)
    (X_train, y_train), (X_test, y_test) = mnist.load_data("../test_data_home")
    X_train, y_train = X_train[:1000], y_train[:1000]  # 训练集1000条
    X_test, y_test = X_test[:100], y_test[:100]  # 测试集100条
    X_train = [cv2.cvtColor(cv2.resize(i, (input_shape[1], input_shape[0])), cv2.COLOR_GRAY2RGB)
               for i in X_train]  # 变成彩色的
    X_test = [cv2.cvtColor(cv2.resize(i, (input_shape[1], input_shape[0])), cv2.COLOR_GRAY2RGB)
              for i in X_test]  # 变成彩色的

    model = CNN(input_shape, 10)()
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-4), metrics=['accuracy'])
    X_train = np.array(X_train).astype(np.float32) / 255
    X_test = np.array(X_test).astype(np.float32) / 255
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=128)
