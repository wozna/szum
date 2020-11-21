import librosa
from tensorflow import keras
from pathlib import Path
import python_speech_features
import numpy as np
import pandas as pd

word2index = {
    # core words
    "yes": 0,
    "no": 1,
    "up": 2,
    "down": 3,
    "left": 4,
    "right": 5,
    "on": 6,
    "off": 7,
    "stop": 8,
    "go": 9,
    "unknown": 10
}


def get_audio(path, sample_rate=16000):
    wav, wav_sr = librosa.load(path, sr=sample_rate)
    if wav.size < 16000:
        return np.pad(wav, (16000 - wav.size, 0), mode='constant')
    else:
        return wav[0:16000]


def get_noise_audio(path, sample_rate=16000):
    wav, wav_sr = librosa.load(path, sr=sample_rate)
    return wav


# compute MFCC features from audio signal
def audio2feature(audio):
    audio = audio.astype(np.float)
    # normalize data
    audio -= audio.mean()
    audio /= np.max((audio.max(), -audio.min()))
    # compute MFCC coefficients
    features = python_speech_features.mfcc(audio, samplerate=16000, winlen=0.025, winstep=0.01, numcep=20, nfilt=40,
                                           nfft=512, lowfreq=100, highfreq=None, preemph=0.97, ceplifter=22,
                                           appendEnergy=True, winfunc=np.hamming)
    return features


# load .wav-file, add some noise and compute MFCC features
def wav2feature(wav):
    data = wav.astype(np.float)
    # normalize data
    data -= data.mean()
    data /= np.max((data.max(), -data.min()))
    # add gaussian noise
    data += np.random.normal(loc=0.0, scale=0.025, size=data.shape)
    # compute MFCC coefficients
    features = python_speech_features.mfcc(data, samplerate=16000, winlen=0.025, winstep=0.01, numcep=20, nfilt=40,
                                           nfft=512, lowfreq=100, highfreq=None, preemph=0.97, ceplifter=22,
                                           appendEnergy=True, winfunc=np.hamming)
    return features


def get_model(num_classes):
    model = keras.models.Sequential()

    model.add(keras.layers.Input(shape=(99, 20)))

    model.add(keras.layers.Conv1D(64, kernel_size=8, activation="relu", input_shape=(99, 20)))
    model.add(keras.layers.MaxPooling1D(pool_size=3))

    model.add(keras.layers.Conv1D(128, kernel_size=8, activation="relu"))
    model.add(keras.layers.MaxPooling1D(pool_size=3))

    model.add(keras.layers.Conv1D(256, kernel_size=5, activation="relu"))
    model.add(keras.layers.GlobalMaxPooling1D())

    model.add(keras.layers.Dense(128, activation="relu"))

    model.add(keras.layers.Dense(64, activation="relu"))

    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    return model


def get_data(path):
    ''' Returns dataframe with columns: 'path', 'word'.'''
    datadir = Path(path)
    files = [(str(f), f.parts[-2]) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'word'])

    return df


def prepare_data(df):
    '''Transform data into something more useful.'''
    train_words = ['yes', 'no', 'up', 'down',
                   'left', 'right', 'on', 'off', 'stop', 'go']

    df.loc[~df.word.isin(train_words), 'word'] = 'unknown'
    return df


def stretch(command):
    speed_rate = np.random.uniform(0.7, 1.3)
    data = librosa.effects.time_stretch(command, speed_rate)
    if len(data) > 16000:
        return data[:16000]
    else:
        return np.pad(data, (0, max(0, 16000 - len(data))), "constant")


def change_pitch(data):
    y_pitch = data.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    return librosa.effects.pitch_shift(y_pitch.astype('float64'),
                                       16000, n_steps=pitch_change,
                                       bins_per_octave=bins_per_octave)


def get_noises(path):
    ''' Returns paths to noise samples.'''
    datadir = Path(path)
    files = [str(f) for f in datadir.glob('**/*.wav') if f]
    return files


def add_augmentation(command):
    # stretch command from (0.7, 1.3)
    data = stretch(command)
    data = change_pitch(data)
    return data


def add_noise(command, noises):
    noise_id = np.random.randint(0, len(noises))
    noise = get_noise_audio(noises[noise_id])

    start_ = np.random.randint(noise.shape[0] - 16000)
    bg_slice = noise[start_: start_ + 16000]
    wav_with_bg = command * np.random.uniform(0.8, 1.2) + \
                  bg_slice * np.random.uniform(0, 0.5)

    return wav_with_bg


def get_augmented_data(paths, noises=None):
    '''
    Given list of paths, return specgrams.
    '''
    data = []

    for path in paths:
        wav = get_audio(path)
        wav = add_augmentation(wav)

        if noises is not None:
            data.append(add_noise(wav, noises))
        else:
            data.append(wav)

    features = []
    for k, sample in enumerate(data):
        features.append(wav2feature(sample))
    features = np.array(features)

    return features


def batch_generator(X, y, noises=None, batch_size=16):
    '''
    Return a random image from X, y
    '''

    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, X.shape[0], batch_size)
        im = X[idx]
        label = y[idx]

        data = get_augmented_data(im, noises=noises)

        yield np.concatenate([data]), label
