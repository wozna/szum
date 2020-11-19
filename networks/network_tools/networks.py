import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import InceptionV3, MobileNetV2
from efficientnet.model import EfficientNetB0


class Network:
    def __init__(self, setting, data):
        self.__input_x = setting.input_x
        self.__input_y = setting.input_y
        self.__input_z = setting.input_z
        self._n_classes = setting.n_classes
        self._data = data
        self._input_shape = (self.__input_x, self.__input_y, self.__input_z)
        self._steps_per_epoch = self._data.train_n() // self._data.batch_size()
        self._validation_steps_per_epoch = self._data.validate_n() // self._data.batch_size()
        self._model = models.Sequential()
        self._metrics = ['acc']

    def fit(self, epochs, callbacks):
        history = self._model.fit_generator(
            self._data.train_data(),
            steps_per_epoch=self._steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=self._data.validate_data(),
            validation_steps=self._validation_steps_per_epoch, workers=16)
        return history

    def evaluate(self):
        return self._model.evaluate_generator(self._data.test_data(), self._data.test_n() // self._data.batch_size())


class NetworkInceptionV3(Network):
    def __init__(self, setting, data):
        Network.__init__(self, setting, data)
        self.__weights = 'imagenet'
        self.__include_top = False
        self.__conv_base = InceptionV3(weights=self.__weights, include_top=self.__include_top,
                                       input_shape=self._input_shape)
        self.__loss = 'categorical_crossentropy'
        self.__optimizer = optimizers.SGD(lr=.01, momentum=.9)

    def create_model(self):
        self.__conv_base.trainable = False
        self._model.add(self.__conv_base)
        self._model.add(layers.GlobalAveragePooling2D())
        self._model.add(layers.Dense(self._n_classes, activation='softmax'))
        self._model.compile(
            loss=self.__loss, optimizer=self.__optimizer, metrics=self._metrics)
        return self._model

class NetworkMobileNetV2(Network):
    def __init__(self, setting, data):
        Network.__init__(self, setting, data)
        self.__weights = 'imagenet'
        self.__include_top = False
        self.__conv_base = MobileNetV2(weights=self.__weights, include_top=self.__include_top,
                                       input_shape=self._input_shape)
        self.__loss = 'categorical_crossentropy'
        self.__optimizer = optimizers.SGD(lr=.01, momentum=.9)

    def create_model(self):
        self.__conv_base.trainable = False
        self._model.add(self.__conv_base)
        self._model.add(layers.GlobalAveragePooling2D())
        self._model.add(layers.Dense(self._n_classes, activation='softmax'))
        self._model.compile(
            loss=self.__loss, optimizer=self.__optimizer, metrics=self._metrics)
        return self._model


class NetworkEfficientNetB0(Network):
    def __init__(self, setting, data):
        Network.__init__(self, setting, data)
        self.__weights = 'imagenet'
        self.__include_top = False
        self.__conv_base = EfficientNetB0(weights=self.__weights, include_top=self.__include_top,
                                          input_shape=self._input_shape,  backend=keras.backend,
                                          layers=keras.layers, models=keras.models, utils=keras.utils)
        self.__loss = 'categorical_crossentropy'
        self.__optimizer = optimizers.SGD(lr=.01, momentum=.9)

    def create_model(self):
        self.__conv_base.trainable = False
        self._model.add(self.__conv_base)
        self._model.add(layers.GlobalAveragePooling2D())
        self._model.add(layers.Dense(self._n_classes, activation='softmax'))
        self._model.compile(
            loss=self.__loss, optimizer=self.__optimizer, metrics=self._metrics)
        return self._model
