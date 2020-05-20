# path
import os
from os.path import isdir, join
from pathlib import Path

# Scientific Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

# Deep learning
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

import random
import copy
import librosa


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


class DataGeneratorSimple:
    def __init__(self):

        self.__train_audio_path = 'D:\\a__a\\projects\\data\\train\\audio'
        self.__dirs = [f for f in os.listdir(self.__train_audio_path) if isdir(join(self.__train_audio_path, f))]
        self.__dirs.sort()
        self.__max_ratio = 0.1
        print('Number of labels: ' + str(len(self.__dirs[1:])))
        print(self.__dirs)
        self.__all_wav = []
        self.__unknown_wav = []
        self.__noised_wav = []
        self.__wav_vals = []
        self.__label_all = []
        self.__label_value = {}
        self.__labels = []
        self.__unknown = []
        self.__silence_wav = []
        self.__silence_label = []
        self.__target_list = ['yes', 'no']   #, 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        self.__unknown_list = [d for d in self.__dirs if d not in self.__target_list and d != '_background_noise_']
        print('target_list : ', end='')
        print(self.__target_list)
        print('unknowns_list : ', end='')
        print(self.__unknown_list)
        print('silence : _background_noise_')
        i = 0;
        self.__background = [f for f in os.listdir(join(self.__train_audio_path, '_background_noise_')) if
                             f.endswith('.wav')]
        self.__background_noise = []
        for wav in self.__background:
            samples, sample_rate = librosa.load(join(join(self.__train_audio_path, '_background_noise_'), wav))
            samples = librosa.resample(samples, sample_rate, 8000)
            self.__background_noise.append(samples)

        for direct in self.__dirs[1:]:
            waves = [f for f in os.listdir(join(self.__train_audio_path, direct)) if f.endswith('.wav')]
            self.__label_value[direct] = i
            i = i + 1
            print(str(i) + ":" + str(direct) + " ", end="")
            for wav in waves:
                samples, sample_rate = librosa.load(join(join(self.__train_audio_path, direct), wav), sr=16000)
                samples = librosa.resample(samples, sample_rate, 8000)
                if len(samples) != 8000:
                    continue

                if direct in self.__unknown_list:
                    self.__unknown_wav.append(samples)
                else:
                    self.__label_all.append(direct)
                    self.__all_wav.append([samples, direct])

        self.__wav_all = np.reshape(np.delete(self.__all_wav, 1, 1), (len(self.__all_wav)))
        self.__label_all = [i for i in np.delete(self.__all_wav, 0, 1).tolist()]

        # Random pick start point

    def get_one_noise(self, noise_num=0):
        selected_noise = self.__background_noise[noise_num]
        start_idx = random.randint(0, len(selected_noise) - 1 - 8000)
        return selected_noise[start_idx:(start_idx + 8000)]

    def preprocessing(self):
        augment = 1
        delete_index = []
        for i in range(augment):
            new_wav = []
            noise = self.get_one_noise(i)
            for i, s in enumerate(self.__wav_all):
                if len(s) != 8000:
                    delete_index.append(i)
                    continue
                s = s + (self.__max_ratio * noise)
                self.__noised_wav.append(s)
        np.delete(self.__wav_all, delete_index)
        np.delete(self.__label_all, delete_index)
        self.__wav_vals = np.array([x for x in self.__wav_all])
        self.__label_vals = [x for x in self.__label_all]
        self.__labels = copy.deepcopy(self.__label_vals)
        for _ in range(augment):
            self.__label_vals = np.concatenate((self.__label_vals, self.__labels), axis=0)
        self.__label_vals = self.__label_vals.reshape(-1, 1)
        # knowns audio random sampling
        self.__unknown = self.__unknown_wav
        np.random.shuffle(self.__unknown_wav)
        self.__unknown = np.array(self.__unknown)
        self.__unknown = self.__unknown[:2000 * (augment + 1)]
        self.__unknown_label = np.array(['unknown' for _ in range(2000 * (augment + 1))])
        self.__unknown_label = self.__unknown_label.reshape(2000 * (augment + 1), 1)
        delete_index = []
        for i, w in enumerate(self.__unknown):
            if len(w) != 8000:
                delete_index.append(i)
        self.__unknown = np.delete(self.__unknown, delete_index, axis=0)
        # silence audio
        num_wav = (2000 * (augment + 1)) // len(self.__background_noise)
        for i, _ in enumerate(self.__background_noise):
            for _ in range((2000 * (augment + 1)) // len(self.__background_noise)):
                self.__silence_wav.append(self.get_one_noise(i))
        self.__silence_wav = np.array(self.__silence_wav)
        self.__silence_label = np.array(['silence' for _ in range(num_wav * len(self.__background_noise))])
        self.__silence_label = self.__silence_label.reshape(-1, 1)
        self.__wav_vals = np.reshape(self.__wav_vals, (-1, 8000))
        self.__noised_wav = np.reshape(self.__noised_wav, (-1, 8000))
        self.__unknown = np.reshape(self.__unknown, (-1, 8000))
        self.__silence_wav = np.reshape(self.__silence_wav, (-1, 8000))
        print(self.__wav_vals.shape)
        print(self.__noised_wav.shape)
        print(self.__unknown.shape)
        print(self.__silence_wav.shape)
        print(self.__label_vals.shape)
        print(self.__unknown_label.shape)
        print(self.__silence_label.shape)
        self.__wav_vals = np.concatenate((self.__wav_vals, self.__noised_wav), axis=0)
        self.__wav_vals = np.concatenate((self.__wav_vals, self.__unknown), axis=0)
        self.__wav_vals = np.concatenate((self.__wav_vals, self.__silence_wav), axis=0)
        self.__label_vals = np.concatenate((self.__label_vals, self.__unknown_label), axis=0)
        self.__label_vals = np.concatenate((self.__label_vals, self.__silence_label), axis=0)
        print(len(self.__wav_vals))
        print(len(self.__label_vals))

    def get_wav_vals(self):
        return self.__wav_vals

    def get_label_vals(self):
        return self.__label_vals

    def get_target_list(self):
        return self.__target_list
