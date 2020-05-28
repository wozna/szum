from keras import layers
from keras import models
from keras import optimizers
# Scientific Math
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers


class Network:
    def __init__(self, label_value):
        self.__input_shape = (8000, 1)
        self._metrics = ['accuracy']
        self.__loss = keras.losses.categorical_crossentropy
        self.__optimizer = keras.optimizers.Adam(lr=0.001)
        self.__drop_out_rate = 0.5
        self._model = self.create_model(label_value)

    def fit(self, train_wav, train_label, test_wav, test_label, epochs, batch_size, callbacks):
        self._model.compile(loss=self.__loss,
                      optimizer=self.__optimizer,
                      metrics=self._metrics)

        history = self._model.fit(
            train_wav, 
            train_label,
            validation_data=[test_wav, test_label],
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks)
        return history
    
    def create_model(self, label_value):
        input_tensor = Input(shape=(self.__input_shape))
        x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides=1)(input_tensor)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(self.__drop_out_rate)(x)
        # x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
        # x = layers.MaxPooling1D(2)(x)
        # x = layers.Dropout(self.__drop_out_rate)(x)
        # x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1)(x)
        # x = layers.MaxPooling1D(2)(x)
        # x = layers.Dropout(self.__drop_out_rate)(x)
        # x = layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x)
        # x = layers.MaxPooling1D(2)(x)
        # x = layers.Dropout(self.__drop_out_rate)(x)
        x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.__drop_out_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.__drop_out_rate)(x)
        output_tensor = layers.Dense(len(label_value), activation='softmax')(x)
        model = keras.Model(input_tensor, output_tensor)
        return model