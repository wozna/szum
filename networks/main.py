from network_tools.utils import Util, TimeHistory
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from network_tools.data import DataGenerator, DataGeneratorSimple
from network_tools.networks import NetworkInceptionV3
from network_tools.architecture import Architecture
from network_tools.settings import Setting
from contextlib import redirect_stdout

# Scientific Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

#Deep learning
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

# Visualization
import matplotlib.pyplot as plt
import tensorflow as tf




class Teacher:
    def __init__(self):
        self.__base_dir = 'data'
        self.__network_name = 'NetworkInceptionV3_19_October'
        self.__model_path = 'models/' + self.__network_name + '/'
        self.__path_log = self.__model_path + "log.csv"
        self.__path_name = "."
        self.__epochs = 2

    def __schedule(self, epoch):
        if epoch < 5:
            return 0.01
        if epoch < 10:
            return 0.02
        else:
            return 0.004

    def __schedule_fine_tune(self, epoch):
        if epoch < 5:
            return 0.0008
        if epoch < 10:
            return 0.00016
        else:
            return 0.000032

    def __get_callbacks(self):
        csv_logger = CSVLogger(self.__path_log, append=True, separator=';')
        check_pointer = ModelCheckpoint(filepath=self.__model_path + self.__network_name + '.hdf5', verbose=1,
                                        save_best_only=True)
        lr_scheduler = LearningRateScheduler(self.__schedule)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        time_callback = TimeHistory()
        tensorboard = TensorBoard(
                    log_dir=self.__path_log,
                    write_graph=self.__model_path,
                )
        return [lr_scheduler, csv_logger, check_pointer, early_stopping, time_callback, tensorboard]

    def start(self):
        # create util
        Util.create_dir(self.__model_path)

        # create settings
        setting = Setting()

        # data generate
        data = DataGeneratorSimple()
        data.preprocessing()

        train_wav, test_wav, train_label, test_label = train_test_split(data.get_wav_vals(), data.get_label_vals(),
                                                                        test_size=0.2,
                                                                        random_state=1993,
                                                                        shuffle=True)

        # Parameters
        lr = 0.001
        generations = 20000
        num_gens_to_wait = 250
        batch_size = 512
        drop_out_rate = 0.5
        input_shape = (8000, 1)

        # For Conv1D add Channel
        train_wav = train_wav.reshape(-1, 8000, 1)
        test_wav = test_wav.reshape(-1, 8000, 1)

        label_value = data.get_target_list()
        label_value.append('unknown')
        label_value.append('silence')

        new_label_value = dict()
        for i, l in enumerate(label_value):
            new_label_value[l] = i
        label_value = new_label_value

        # Make Label data 'string' -> 'class num'
        temp = []
        for v in train_label:
            temp.append(label_value[v[0]])
        train_label = np.array(temp)

        temp = []
        for v in test_label:
            temp.append(label_value[v[0]])
        test_label = np.array(temp)

        # Make Label data 'class num' -> 'One hot vector'
        train_label = keras.utils.to_categorical(train_label, len(label_value))
        test_label = keras.utils.to_categorical(test_label, len(label_value))

        print('Train_Wav Demension : ' + str(np.shape(train_wav)))

        # Conv1D Model
        input_tensor = Input(shape=(input_shape))

        x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides=1)(input_tensor)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(drop_out_rate)(x)
        output_tensor = layers.Dense(len(label_value), activation='softmax')(x)

        model = tf.keras.Model(input_tensor, output_tensor)

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        history = model.fit(train_wav, train_label, validation_data=[test_wav, test_label],
                            batch_size=batch_size,
                            epochs=100,
                            verbose=1)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # # create network and get model
        # network = NetworkInceptionV3(setting, data)
        # model = network.create_model()
        #
        # # create util for model, logging
        # architecture = Architecture(model, self.__model_path)
        # architecture.log()
        #
        # # save settings and layers
        # Util.save_setting(self.__model_path, setting, self.__epochs)
        # with open(self.__model_path + 'model_summary.txt', 'w') as f:
        #     with redirect_stdout(f):
        #         model.summary()  # save network layers
        #
        # callbacks = self.__get_callbacks()
        #
        # # train and validation
        # history = network.fit(epochs=self.__epochs, callbacks=callbacks)
        #
        # # test
        # test_loss, test_acc = network.evaluate()
        # print('test acc:', test_acc)
        #
        # # save model, please check file name
        # Util.save_model(model, self.__path_name, test_acc)
        # Util.save_history(self.__model_path, history)
        # Util.save_time(self.__model_path, callbacks[4])


if __name__ == "__main__":
    teacher = Teacher()
    teacher.start()
