import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from pathlib import Path
from tqdm import tqdm
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils import wav2feature, prepare_data, get_data, word2index, get_audio, add_noise, \
    get_random_noise_audio, get_cut_noise_audio, get_specgram, get_simple_model, get_spectrogram_shape, get_model, \
    get_data_and_classes


def signaltonoise(command, noise):
    c = np.mean(command)
    n = np.mean(noise)

    return np.abs(c / n)


def get_nr_noises(path, samples_nr):
    ''' Returns paths to noise samples.'''
    datadir = Path(path)
    files = [str(f) for f in datadir.glob('**/*.wav') if f]
    return files[0:samples_nr]


index2word = [word for word in word2index]
print(index2word)

print("loading dataset...")
test = prepare_data(get_data('../data/data_new/evaluation'))
test, classes = get_data_and_classes(test)
# noises = get_nr_noises("../data/noise", len(test))
noises = None
with_noise = False
shape = get_spectrogram_shape(get_specgram, test[0])
print(shape)

#model = get_model(11, shape)
path = "models/good_model_specgram/model.31.hdf5"
#model.load_weights(path)
model = keras.models.load_model(path)
model.summary()



snr_ratio = 0.25
snr = []
samples = []
# 5, 4, 3, 2, 1, 0.5, 0.33, 0.25, 0.20
for id, path in tqdm(enumerate(test)):
    command = get_audio(path)
    if with_noise:
        noise = get_cut_noise_audio([id])
        c = np.mean(command)
        n = np.mean(noise)
        noise_i = np.abs(c * 1.0 / (n * snr_ratio))
        snr.append(signaltonoise(command, noise * noise_i))
        samples.append(add_noise(command, noise, command_i=1.0, noise_i=noise_i))
    else:
        samples.append(command)
test = np.array(samples)
classes = np.array(classes, dtype=np.int)

correct = 0
predicted = []

for id, rec in enumerate(tqdm(test)):
    recorded_feature = get_specgram(rec)
    recorded_feature = np.expand_dims(recorded_feature, 0)
    prediction = model.predict(recorded_feature).reshape((11,))
    prediction /= prediction.sum()
    max_class_id = prediction.argmax()
    predicted.append(max_class_id)
    if index2word[max_class_id] == index2word[classes[id]]:
        correct += 1

acc = correct / len(test)
print("TOP1 acc = " + str(acc))

cm = confusion_matrix(classes, predicted, normalize="all")
plt.figure(figsize=(8, 8))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(np.arange(0, 11, 1), index2word, rotation=90)
plt.yticks(np.arange(0, 11, 1), index2word)
plt.tick_params(labelsize=12)
plt.title('Confusion matrix ')
plt.colorbar()
plt.savefig(path + "_matrix.png")
