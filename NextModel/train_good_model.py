import tensorflow as tf
from pathlib import Path

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import numpy as np
from tensorflow import keras
from utils import word2index, get_model, prepare_data, get_data, batch_generator, \
    get_noises, get_data_and_classes


def train(batch_size=8, epochs=50, model_dir_name='default_model_name', with_noise=True):
    index2word = [word for word in word2index]
    print(index2word)
    num_classes = len(word2index)

    print("loading dataset...")
    train_data = prepare_data(get_data("../data/data_new/training"))
    valid_data = prepare_data(get_data("../data/data_new/validation"))
    print("Size of training dataset: " + str(len(train_data)))
    print("Size of validation dataset: " + str(len(valid_data)))

    train_noises = None
    valid_noises = None
    if with_noise:
        train_noises = np.array(get_noises("../data/noise_new/training"))
        valid_noises = np.array(get_noises("../data/noise_new/validation"))

    train_data, train_classes = get_data_and_classes(train_data)
    valid_data, valid_classes = get_data_and_classes(valid_data)

    keras.backend.clear_session()
    model = get_model(num_classes)

    model_name = "models/model_with_high_noise/"  # path where save model
    Path(model_name).mkdir(parents=True, exist_ok=True)

    check_pointer = tf.keras.callbacks.ModelCheckpoint(filepath=model_name + 'model.{epoch:02d}.hdf5',
                                                       verbose=1,
                                                       save_best_only=False)
    csv_logger = tf.keras.callbacks.CSVLogger(model_name + 'training.log')

    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    train_gen = batch_generator(train_data, train_classes, train_noises, batch_size=batch_size)
    valid_gen = batch_generator(valid_data, valid_classes, valid_noises, batch_size=batch_size)

    history = model.fit_generator(
        generator=train_gen,
        epochs=epochs,
        steps_per_epoch=train_data.shape[0] // batch_size,
        validation_data=valid_gen,
        validation_steps=valid_data.shape[0] // batch_size,
        callbacks=[check_pointer, csv_logger])


if __name__ == "__main__":
    batch_size = 8
    epochs = 10
    model_dir_name = 'model_good_with_noise/'
    with_noise = False
    train(batch_size=batch_size, epochs=epochs, model_dir_name=model_dir_name, with_noise=with_noise)
