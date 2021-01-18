import librosa
import numpy as np
from tensorflow import keras
from pathlib import Path
from tqdm import tqdm

from utils_next import wav2feature, prepare_data, get_data, word2index, get_audio, add_noise, \
    get_random_noise_audio


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

model = keras.models.load_model("models/model_January_16/model.11.hdf5")

samples = []
classes = []
snr = []

class_name = "right"
command_path = "../data/data_new/evaluation/right/a24cf51c_nohash_1.wav"
noise_path = "../data/noise/a001_80_90.wav"
classes.append(word2index[class_name])
classes = np.array(classes, dtype=np.int)

snr_ratio = 0.009
# 5, 4, 3, 2, 1, 0.5, 0.33, 0.25, 0.20
noise = get_random_noise_audio([noise_path])
command = get_audio(command_path)
c = np.mean(command)
n = np.mean(noise)
noise_i = np.abs(c * 1.0 / (n * snr_ratio))

snr.append(signaltonoise(command, noise * noise_i))
audio_with_noise = add_noise(command, noise, command_i=1.0, noise_i=noise_i)
samples.append(audio_with_noise)
librosa.output.write_wav(path="output2.wav", y=audio_with_noise, sr=16000)

from playsound import playsound

playsound("output2.wav")

samples = np.array(samples)
classes = np.array(classes, dtype=np.int)

correct = 0

for id, rec in enumerate(samples):
    recorded_feature = wav2feature(rec)

    recorded_feature = np.expand_dims(recorded_feature, 0)  # add "fake" batch dimension 1
    prediction = model.predict(recorded_feature).reshape((11,))
    # normalize prediction output to get "probabilities"
    prediction /= prediction.sum()

    # print the 3 candidates with highest probability
    prediction_sorted_indices = prediction.argsort()
    print("candidates:\n------------  " + index2word[classes[id]])
    print("SNR = " + str(snr[id]))
    for k in range(3):
        i = int(prediction_sorted_indices[-1 - k])
        print("%d.)\t%s\t:\t%2.1f%%" % (k + 1, index2word[i], prediction[i] * 100))
        if k == 0 and index2word[i] == index2word[classes[id]]:
            correct += 1

    print("-----------------------------")

acc = correct / len(samples)
print("TOP1 acc = " + str(acc))
