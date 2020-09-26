import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
from PIL import Image
from scipy.fftpack import fft


def log_spectrogram(audio, sample_rate, window_size=20,
                    step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)


def wav2img(wav_path, targetdir='', figsize=(4, 4)):
    """
    takes in wave file path
    and the fig size. Default 4,4 will make images 288 x 288
    """
    plt.figure(figsize=figsize)
    # use soundfile library to read in the wave files
    samplerate, test_sound = wavfile.read(wav_path)
    _, spectrogram = log_spectrogram(test_sound, samplerate)

    # create output path
    output_file = wav_path.split('/')[-1].split('.wav')[0]
    output_file = targetdir + '/' + output_file
    #plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.imsave('%s.png' % output_file, spectrogram)
    plt.close()


def generate_spec(audio_path, save_path):
    all_files = [y for y in os.listdir(audio_path) if '.wav' in y]
    for file in all_files:
        wav2img(audio_path + '/' + file, save_path)


def spectrogram_from_directory(directory_path, save_path):
    subdirectories = [f for f in os.scandir(directory_path) if f.is_dir()]
    for sub in subdirectories:
        print(sub.name)
        save_directory = save_path + '/' + sub.name
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        generate_spec(sub.path, save_directory)
    print("Success! Generated spectograms from all directories")


audio_path = '/macierz/home/s165554/pg/szum/data/commands'
save_path = '/macierz/home/s165554/pg/szum/data/train'

spectrogram_from_directory(audio_path, save_path)
