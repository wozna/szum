import matplotlib
import numpy as np
from tensorflow import keras
from pathlib import Path
from tqdm import tqdm
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils import wav2feature, prepare_data, get_data, word2index, get_audio, add_noise, \
    get_random_noise_audio, get_cut_noise_audio, get_specgram


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
# train = prepare_data(get_data('/macierz/home/s165554/pg/szum/data/evaluation/'))
test = prepare_data(get_data('../data/orginal/evaluation'))
# noises = get_nr_noises("../data/noise", len(test))
noises = None
with_noise = False
model = keras.models.load_model("models/model_with_get_specgram/model.17.hdf5")

samples = []
classes = []
snr = []

for class_name in test.word:
    classes.append(word2index[class_name])
classes = np.array(classes, dtype=np.int)

test = np.array(test.path)
snr_ratio = 0.25
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

# test_data = []
# for id, rec in enumerate(test):
#     test_data.append(np.expand_dims(get_specgram(rec), 0))
#
# y = np.argmax(model.predict(test_data), axis=1)
# cm = confusion_matrix(classes, y, normalize="all")
# # %matplotlib inline
# plt.close()
# plt.figure(figsize=(8, 8))
# plt.imshow(cm, cmap=plt.cm.Blues)
# plt.xlabel("Predicted labels")
# plt.ylabel("True labels")
# plt.xticks(np.arange(0, 20, 1), index2word, rotation=90)
# plt.yticks(np.arange(0, 20, 1), index2word)
# plt.tick_params(labelsize=12)
# plt.title('Confusion matrix ')
# plt.colorbar()
# plt.show()
predicted = []

for id, rec in enumerate(test):
    recorded_feature = get_specgram(rec)

    recorded_feature = np.expand_dims(recorded_feature, 0)  # add "fake" batch dimension 1
    prediction = model.predict(recorded_feature).reshape((11,))

    # normalize prediction output to get "probabilities"
    prediction /= prediction.sum()

    # print the 3 candidates with highest probability
    prediction_sorted_indices = prediction.argsort()
    predicted.append(int(prediction_sorted_indices[0]))
    #print("candidates:\n------------  " + index2word[classes[id]])
    #print("SNR = " + str(snr[id]))
    for k in range(3):
        i = int(prediction_sorted_indices[-1 - k])
        #print("%d.)\t%s\t:\t%2.1f%%" % (k + 1, index2word[i], prediction[i] * 100))
        if k == 0 and index2word[i] == index2word[classes[id]]:
            correct += 1

    #print("-----------------------------")

acc = correct / len(samples)
print("TOP1 acc = " + str(acc))
cm = confusion_matrix(classes, predicted, normalize="all")
%matplotlib inline
plt.close()
plt.figure(figsize = (8,8))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(np.arange(0, 20, 1), index2word, rotation=90)
plt.yticks(np.arange(0, 20, 1), index2word)
plt.tick_params(labelsize=12)
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()