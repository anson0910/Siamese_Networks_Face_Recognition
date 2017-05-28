from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import os, numpy.random


class SiameseNet(object):
    def __init__(self, data_loader, weights_path, input_size=64):
        self.data_loader = data_loader
        self.weights_path = weights_path
        self.input_size = input_size
        self.model = self._get_model()

        if os.path.exists(weights_path):
            print('Found weights file: {}, loading weights'.format(weights_path))
            self.model.load_weights(weights_path)
        else:
            print('Could not find weights file, initialized parameters randomly')

    def train(self, num_batches=900000, starting_batch=0, batch_size=32,
              loss_every=250, evaluate_every=1000, num_way=40, num_val_trials=10):
        best = 10.0
        for i in range(starting_batch, num_batches):
            (inputs, targets) = self.data_loader.get_training_batch(batch_size)
            loss = self.model.train_on_batch(inputs, targets)
            if i % evaluate_every == 0:
                val_acc = self.data_loader.test_oneshot(self.model, num_way, num_val_trials, verbose=True)
                if val_acc >= best:
                    print("Saving weights to {}".format(self.weights_path))
                    self.model.save(self.weights_path)
                    best = val_acc

            if i % loss_every == 0:
                print("iteration {}, training loss: {:.2f},".format(i, loss))

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
        l1_distance = lambda x: K.abs(x[0] - x[1])
        both = merge([encoded_l, encoded_r], mode=l1_distance, output_shape=lambda x: x[0])
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
