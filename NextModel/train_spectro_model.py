import tensorflow as tf
from pathlib import Path
from sklearn.utils import class_weight

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import numpy as np
from tensorflow import keras
from utils import word2index, get_model, prepare_data, get_data, batch_generator, \
    get_noises, get_data_and_classes, batch_simple_generator, get_specgram, get_melspecgram, get_spectrogram_shape, \
    get_log_specgram, get_melspecgram


def train(batch_size=8, epochs=50, model_dir_name='default_model_name', with_noise=True,
          spectrogram_function=get_specgram):
    index2word = [word for word in word2index]
    print(index2word)
    unique_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_classes = len(word2index)

    print("loading dataset...")
    train_data = prepare_data(get_data("../data/orginal/training"))
    valid_data = prepare_data(get_data("../data/orginal/validation"))
    print("Size of training dataset: " + str(len(train_data)))
    print("Size of validation dataset: " + str(len(valid_data)))

    train_noises = None
    valid_noises = None
    if with_noise:
        train_noises = np.array(get_noises("../data/noise_new/training"))
        valid_noises = np.array(get_noises("../data/noise_new/validation"))

    train_data, train_classes = get_data_and_classes(train_data)
    valid_data, valid_classes = get_data_and_classes(valid_data)

    class_weights = class_weight.compute_class_weight('balanced', unique_classes, train_classes)
    class_weights = {i: class_weights[i] for i in range(len(unique_classes))}
    print(class_weights)

    # shape = (128, 32)  # check shape that spectrogram returns
    shape = get_spectrogram_shape(spectrogram_function, train_data[0])  # or use this function to check

    train_gen = batch_simple_generator(train_data, train_classes, train_noises, batch_size=batch_size,
                                       spectrogram_function=spectrogram_function)
    valid_gen = batch_simple_generator(valid_data, valid_classes, valid_noises, batch_size=batch_size,
                                       spectrogram_function=spectrogram_function)

    keras.backend.clear_session()

    # change function get_model to different if you want to change network model
    model = get_model(num_classes, shape=shape)

    model_path = "models/" + model_dir_name  # path where save model
    Path(model_path).mkdir(parents=True, exist_ok=True)

    check_pointer = tf.keras.callbacks.ModelCheckpoint(filepath=model_path + 'model.{epoch:02d}.hdf5',
                                                       verbose=1,
                                                       save_best_only=False)
    csv_logger = tf.keras.callbacks.CSVLogger(model_path + 'training.log')

    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    history = model.fit_generator(
        generator=train_gen,
        epochs=epochs,
        steps_per_epoch=train_data.shape[0] // batch_size,
        validation_data=valid_gen,
        validation_steps=valid_data.shape[0] // batch_size,
        callbacks=[check_pointer, csv_logger],
        class_weight=class_weights)


if __name__ == "__main__":
    batch_size = 8
    epochs = 25
    model_dir_name = 'model_with_get_specgram_with_orginal_and_class_weights/'
    with_noise = False
    # choose spectrogram function [get_specgram, get_log_specgram, get_melspecgram, get_stft]
    spectrogram_function = get_specgram
    train(batch_size=batch_size, epochs=epochs, model_dir_name=model_dir_name, with_noise=with_noise,
          spectrogram_function=spectrogram_function)
