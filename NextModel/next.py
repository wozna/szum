import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                         [tf.config.experimental.VirtualDeviceConfiguration(
#                                                             memory_limit=1024)])

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils_next import word2index, get_model, prepare_data, get_data, batch_generator, \
    get_noises
from tqdm import tqdm

index2word = [word for word in word2index]
print(index2word)
num_classes = len(word2index)
batch_size = 8
epochs = 50
speech_commands_dataset_basepath = "data_new/training"

print("loading dataset...")
samples = []
classes = []

train = prepare_data(get_data(speech_commands_dataset_basepath))
noises = get_noises("noise")
noises = np.array(noises)
# noises = None
# validation = prepare_data(get_data("/macierz/home/s165554/pg/szum/data/data_new/validation/"))
print("Size of training dataset: " + str(len(train)))

for class_name in train.word:
    classes.append(word2index[class_name])
classes = np.array(classes, dtype=np.int)
samples = train.path
features = np.array(samples.values)
print(word2index)

train_data, validation_data, train_classes, validation_classes = train_test_split(features, classes,
                                                                                  test_size=0.30, random_state=42,
                                                                                  shuffle=True)

keras.backend.clear_session()  # clear previous model (if cell is executed more than once)

model = get_model(num_classes)

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

check_pointer = tf.keras.callbacks.ModelCheckpoint(filepath="models/NextModel/" + 'model_January_11_.{epoch:02d}.hdf5',
                                                   verbose=1,
                                                   save_best_only=False)

csv_logger = tf.keras.callbacks.CSVLogger('training.log')

train_gen = batch_generator(train_data, train_classes, noises, batch_size=batch_size)
valid_gen = batch_generator(validation_data, validation_classes, noises, batch_size=batch_size)

history = model.fit_generator(
    generator=train_gen,
    epochs=epochs,
    steps_per_epoch=train_data.shape[0] // batch_size,
    validation_data=valid_gen,
    validation_steps=validation_data.shape[0] // batch_size,
    callbacks=[check_pointer, csv_logger])
