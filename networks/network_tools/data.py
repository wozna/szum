import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input  # change model


class DataGenerator:
    def __init__(self, base_dir, setting):
        self.__base_dir = base_dir
        self.__input_x = setting.input_x
        self.__input_y = setting.input_y
        self.__train_dir = os.path.join(base_dir, 'training')
        self.__validation_dir = os.path.join(base_dir, 'validation')
        self.__test_dir = os.path.join(base_dir, 'evaluation')
        self.__batch_size = setting.batch_size
        self.__class_mode = 'categorical'
        self.__train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                                  rotation_range=30,
                                                  width_shift_range=0.2,
                                                  height_shift_range=0.2,
                                                  shear_range=0.2,
                                                  zoom_range=0.2,
                                                  horizontal_flip=True)

        self.__test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.__train_generator = self.__train_datagen.flow_from_directory(
            self.__train_dir,
            target_size=(self.__input_x, self.__input_y),
            batch_size=self.__batch_size,
            class_mode=self.__class_mode)

        self.__validation_generator = self.__test_datagen.flow_from_directory(
            self.__validation_dir,
            target_size=(self.__input_x, self.__input_y),
            batch_size=self.__batch_size,
            class_mode=self.__class_mode)

        self.__test_generator = self.__test_datagen.flow_from_directory(
            self.__test_dir,
            target_size=(self.__input_x, self.__input_y),
            batch_size=self.__batch_size,
            class_mode=self.__class_mode)

    def train_data(self):
        return self.__train_generator

    def train_n(self):
        return self.__train_generator.n

    def validate_data(self):
        return self.__validation_generator

    def validate_n(self):
        return self.__validation_generator.n

    def test_data(self):
        return self.__test_generator

    def test_n(self):
        return self.__test_generator.n

    def batch_size(self):
        return self.__batch_size
