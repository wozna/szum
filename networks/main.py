from network_tools.utils import Util, TimeHistory
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from network_tools.data import DataGeneratorSimple
from network_tools.networks import Network

# Scientific Math
import numpy as np
from sklearn.model_selection import train_test_split

#Deep learning
import tensorflow.keras as keras


class Teacher:
    def __init__(self):
        self.__network_name = "NetworkSimple"
        self.__model_path = 'models/' + self.__network_name + '/'
        self.__log_name = "log.csv"
        self.__base_dir =  'data'
        self.__batch_size = 512
        self.__input_shape = (8000, 1)
        self.__epochs = 100

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
        csv_logger = CSVLogger(self.__log_name, append=True, separator=';')
        check_pointer = ModelCheckpoint(filepath=self.__model_path + self.__network_name + '.hdf5', verbose=1,
                                        save_best_only=True)
        lr_scheduler = LearningRateScheduler(self.__schedule)
        #early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
        time_callback = TimeHistory()
        return [lr_scheduler, csv_logger, check_pointer, time_callback]

    def start(self):
        # create util
        Util.create_dir(self.__model_path)

        # data generate
        data = DataGeneratorSimple()
        data.preprocessing()

        train_wav, test_wav, train_label, test_label = train_test_split(data.get_wav_vals(), data.get_label_vals(),
                                                                        test_size=0.2,
                                                                        random_state=1993,
                                                                        shuffle=True)

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

        # network creating and training 

        network = Network(label_value)
        callbacks = self.__get_callbacks()
        history = network.fit(train_wav, train_label, test_wav, test_label, self.__epochs, self.__batch_size, callbacks)

        Util.save_plot(history, 'acc', 'val_acc', 'model accuracy', 'accuracy',
                       'epoch', ['train', 'test'],'acc.png')
   
        Util.save_plot(history, 'loss', 'val_loss', 'model loss', 'loss',
                       'epoch', ['train', 'test'],'loss.png')



if __name__ == "__main__":
    teacher = Teacher()
    teacher.start()
