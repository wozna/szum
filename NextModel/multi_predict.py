import numpy as np
from tensorflow import keras
from pathlib import Path
from tqdm import tqdm

from utils_next import wav2feature, prepare_data, get_data, word2index, get_audio, add_noise, \
    get_random_noise_audio, get_cut_noise_audio


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
num_classes = len(word2index)
num_samples_per_class = 2000

print("loading dataset...")
# train = prepare_data(get_data('/macierz/home/s165554/pg/szum/data/evaluation/'))
train = prepare_data(get_data('../data/data_new/evaluation'))
train_size = len(train)
noises = get_nr_noises("../data/noise", train_size)
noise_size = len(noises)
model_1 = keras.models.load_model("models/no_noise/model.09.hdf5")
model_2 = keras.models.load_model("models/model_January_14/model.11.hdf5")
model_3 = keras.models.load_model("models/model_January_16/model.11.hdf5")
# samples = []
classes = []
snr = []

for class_name in train.word:
    classes.append(word2index[class_name])
classes = np.array(classes, dtype=np.int)

train = np.array(train.path)
ratios = {5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.33, 0.25, 0.20, 0.16, 0.14}
# snr_ratio =  5.0

for snr_ratio in ratios:
    samples = []
    for id, path in enumerate(tqdm(train)):
        noise = get_cut_noise_audio(noises[id % noise_size])
        command = get_audio(path)
        c = np.mean(command)
        n = np.mean(noise)
        noise_i = np.abs(c * 1.0 / (n * snr_ratio))

        snr.append(signaltonoise(command, noise * noise_i))
        samples.append(add_noise(command, noise, command_i=1.0, noise_i=noise_i))
        # samples.append(command)
    samples = np.array(samples)
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    for id, rec in enumerate(samples):
        recorded_feature = wav2feature(rec)

        recorded_feature = np.expand_dims(recorded_feature, 0)  # add "fake" batch dimension 1
        prediction_1 = model_1.predict(recorded_feature).reshape((11,))
        prediction_2 = model_2.predict(recorded_feature).reshape((11,))
        prediction_3 = model_3.predict(recorded_feature).reshape((11,))
        # normalize prediction output to get "probabilities"
        prediction_1 /= prediction_1.sum()
        prediction_2 /= prediction_2.sum()
        prediction_3 /= prediction_3.sum()
        # print the 3 candidates with highest probability
        prediction_sorted_indices = prediction_1.argsort()
        prediction_sorted_indices_2 = prediction_2.argsort()
        prediction_sorted_indices_3 = prediction_3.argsort()

        # print("candidates:\n------------  " + index2word[classes[id]])
        # print("SNR = " + str(snr[id]))
        for k in range(3):
            i = int(prediction_sorted_indices[-1 - k])
            i_2 = int(prediction_sorted_indices_2[-1 - k])
            i_3 = int(prediction_sorted_indices_3[-1 - k])

            # print("%d.)\t%s\t:\t%2.1f%%" % (k + 1, index2word[i], prediction[i] * 100))
            if k == 0 and index2word[i] == index2word[classes[id]]:
                correct_1 += 1
            if k == 0 and index2word[i_2] == index2word[classes[id]]:
                correct_2 += 1
            if k == 0 and index2word[i_3] == index2word[classes[id]]:
                correct_3 += 1

                # print("-----------------------------")
    print(str(snr_ratio))
    acc_1 = correct_1 / len(samples)
    print("TOP1 acc = " + str(acc_1))
    acc_2 = correct_2 / len(samples)
    print("TOP1 acc = " + str(acc_2))
    acc_3 = correct_3 / len(samples)
    print("TOP1 acc = " + str(acc_3))
