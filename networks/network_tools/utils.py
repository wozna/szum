import matplotlib.pyplot as plt
import os
import pickle
import keras
import time



class Util:

    @staticmethod
    def save_model(model, path_name, test_acc):
        model_json = model.to_json()
        file_name = path_name + "_" + str(test_acc);
        with open(file_name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(file_name + ".h5")
        print("Saved model to disk")
        pass

    @staticmethod
    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print("directory already exists")

    @staticmethod
    def save_setting(path, setting, epochs):
        file = open(path + 'settings.txt', 'w')
        file.write('input=[ ' + str(setting.input_x) + ', ' + str(setting.input_y) + ', ' + str(setting.input_z) + ' ]')
        file.write(' batch_size= ' + str(setting.batch_size))
        file.write(' epochs= ' + str(epochs))
        file.close()

    @staticmethod
    def save_layers(path, model):
        file = open(path + 'layers.txt', 'w')
        file.write(str(model.summary()))
        file.close()

    @staticmethod
    def save_history(path, history):
        with open(path + '/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    @staticmethod
    def save_time(path, time_to_save):
        file = open(path + 'time.txt', 'w')
        file.write(str(time_to_save.times))
        sum_to_save = sum(time_to_save.times)
        file.write(str(sum_to_save))
        file.close()

    @staticmethod
    def save_plot(history, series1, series2, title, ylabel, xlabel, legend, file_name):
        plt.plot(history.history[series1])
        plt.plot(history.history[series2])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend(legend, loc='upper left')
        plt.savefig(file_name)


class TimeHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.times = []

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
