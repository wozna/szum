from network_tools.utils import Util, TimeHistory
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from network_tools.data import DataGenerator
from network_tools.networks import NetworkInceptionV3
from network_tools.architecture import Architecture
from network_tools.settings import Setting
from contextlib import redirect_stdout


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
        data = DataGenerator(self.__base_dir, setting)

        # create network and get model
        network = NetworkInceptionV3(setting, data)
        model = network.create_model()

        # create util for model, logging
        architecture = Architecture(model, self.__model_path)
        architecture.log()

        # save settings and layers
        Util.save_setting(self.__model_path, setting, self.__epochs)
        with open(self.__model_path + 'model_summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()  # save network layers

        callbacks = self.__get_callbacks()

        # train and validation
        history = network.fit(epochs=self.__epochs, callbacks=callbacks)

        # test
        test_loss, test_acc = network.evaluate()
        print('test acc:', test_acc)

        # save model, please check file name
        Util.save_model(model, self.__path_name, test_acc)
        Util.save_history(self.__model_path, history)
        Util.save_time(self.__model_path, callbacks[4])


if __name__ == "__main__":
    teacher = Teacher()
    teacher.start()
