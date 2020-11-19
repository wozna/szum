import librosa.display
import librosa
from scipy.fftpack import fft
from PIL import Image
import numpy as np
import os
from scipy.io import wavfile
from scipy import signal
import matplotlib
from matplotlib.backend_bases import RendererBase
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def log_spectrogram(audio, sample_rate, window_size=16,
                    step_size=8, eps=1e-10):
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


def wav_2_mel(wav_file_path, mel_directory, name):
    samples, sample_rate = librosa.load(wav_file_path)  # your file
    #samples = librosa.resample(samples, sample_rate, 8000)
    plt.axis('off')  # no axis
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    S = librosa.feature.melspectrogram(
        y=samples, sr=sample_rate, n_mels=128, fmax=8000)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), fmax=8000)
    #name = os.path.basename(wav_file_path).split('.')[0]
    plt.savefig(mel_directory + "/" + name + ".png",
                bbox_inches='tight', pad_inches=0)
    plt.close()

def get_spectrogram(wav):
    D = librosa.stft(wav, n_fft=480, hop_length=160,
                     win_length=480, window='hamming')
    spect, phase = librosa.magphase(D)
    return spect


def save_spectogram(src_path, save_path, name):
    wav, sr = librosa.load(src_path, sr=None)
    log_spect = np.log(get_spectrogram(wav))
    plt.imshow(log_spect, aspect='auto', origin='lower',)
    plt.imsave('%s.png' % (save_path + name), log_spect)


def generate_spec(audio_path, save_path):
    all_files = [y for y in os.listdir(audio_path) if '.wav' in y]
    i = 0
    for file in all_files:
        i = i + 1
        wav_2_mel(audio_path + '/' + file, save_path, str(i))


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
save_path = '/macierz/home/s165554/pg/szum/data/train-4'

spectrogram_from_directory(audio_path, save_path)