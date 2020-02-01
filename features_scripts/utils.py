import os
import logging
import librosa
import numpy as np
import matplotlib.pyplot as plt


def extract_mfcc_features(filename_path: str, offset: float, duration: float, sample_rate: int):
    x, sample_rate = librosa.load(filename_path,
                                  res_type='kaiser_fast',
                                  duration=duration,
                                  sr=sample_rate,
                                  offset=offset)
    sample_rate = np.array(sample_rate)

    # Extract audio features by taking an arithmetic mean of 13 identified MFCCs.
    # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    return np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13), axis=0)

def extract_mel_spec_as_image(src_path: str, dst_path: str):
    x, sample_rate = librosa.load(src_path,
                                  res_type='kaiser_fast',
                                  duration=1.5,
                                  sr=22050 * 2,
                                  offset=0.5)
    n_fft = 1024
    hop_length = 256
    n_mels = 40
    f_min = 20
    f_max = 8000
    mel_spec = librosa.feature.melspectrogram(x, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, sr=sample_rate, power=1.0, 
    fmin=f_min, fmax=f_max)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    plt.imsave(dst_path, mel_spec_db)
