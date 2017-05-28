from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import numpy.random


class SiameseNet(object):
    def __init__(self, input_size=64):
        self.input_size = input_size
        self.model = self._get_model()

    def _get_model(self):
        input_shape = (self.input_size, self.input_size, 1)
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        convnet = Sequential()
        convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                           kernel_initializer=self._W_init, kernel_regularizer=l2(2e-4)))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(128, (7, 7), activation='relu',
                           kernel_regularizer=l2(2e-4), kernel_initializer=self._W_init, bias_initializer=self._b_init))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=self._W_init, kernel_regularizer=l2(2e-4),
                           bias_initializer=self._b_init))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=self._W_init, kernel_regularizer=l2(2e-4),
                           bias_initializer=self._b_init))
        convnet.add(Flatten())
        convnet.add(
            Dense(4096, activation="sigmoid", kernel_regularizer=l2(1e-3), kernel_initializer=self._W_init,
                  bias_initializer=self._b_init))
        # encode each of the two inputs into a vector with the convnet
        encoded_l = convnet(left_input)
        encoded_r = convnet(right_input)
        # merge two encoded inputs with the l1 distance between them
        L1_distance = lambda x: K.abs(x[0] - x[1])
        both = merge([encoded_l, encoded_r], mode=L1_distance, output_shape=lambda x: x[0])
        prediction = Dense(1, activation='sigmoid', kernel_initializer=self._W_init, bias_initializer=self._b_init)(both)
        siamese_net = Model(input=[left_input, right_input], output=prediction)

        optimizer = Adam(0.00006)
        siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)
        return siamese_net

    def _W_init(self, shape, name=None):
        """Initialize weights as in paper"""
        values = numpy.random.normal(loc=0, scale=1e-2, size=shape)
        return K.variable(values, name=name)

    def _b_init(self, shape, name=None):
        """Initialize bias as in paper"""
        values = numpy.random.normal(loc=0.5, scale=1e-2, size=shape)
        return K.variable(values, name=name)
