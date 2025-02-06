#  Copyright (c) 2024 Robert Lieck

import IPython.display
import numpy as np
from scipy.io.wavfile import write as write_wav
import librosa
import matplotlib.pyplot as plt


sampling_rate = 44000


def normalise_wave(wave, max_amp=0.95):
    wave /= abs(wave).max()
    wave *= max_amp


def fade_wave(wave, time=0.01, start=True, end=True):
    # ramp of correct length
    fade_vals = np.linspace(0, 1, int(np.ceil(time * sampling_rate)))
    if start:
        wave[:len(fade_vals)] *= fade_vals
    if end:
        wave[-len(fade_vals):] *= np.flip(fade_vals)


def render(wave, normalise=True, fade=True):
    wave = wave.copy()
    if normalise is True:
        normalise = dict()
    if normalise is not False:
        normalise_wave(wave, **normalise)
    if fade is True:
        fade = dict()
    if fade is not False:
        fade_wave(wave, **fade)
    return wave


def save(wave, file_name, normalise=True, fade=True):
    wave = render(wave=wave, normalise=normalise, fade=fade)
    # convert to 16bit integer
    wave = np.int16(wave * (np.iinfo(np.int16).max - 1))
    write_wav(file_name, sampling_rate, wave)


def load(file):
    x, _ = librosa.load(file, sr=sampling_rate)
    return x


def audio(wave, fade=True):
    wave = render(wave=wave, normalise=False, fade=fade)
    IPython.display.display(IPython.display.Audio(data=wave, rate=sampling_rate))


def audio_add(wave, start_time, audio=None, min_total_time=0):
    offset = int(start_time * sampling_rate)
    # extend audio if required
    required_length = max(offset + len(wave), int(min_total_time * sampling_rate))
    if audio is None or len(audio) < required_length:
        new_audio = np.zeros(required_length)
        if audio is not None:
            new_audio[:len(audio)] = audio
        audio = new_audio
    # add audio
    audio[offset:offset + len(wave)] += wave
    return audio


def sound(func, phases=0., duration=1.):
    # time vector
    time = np.arange(0, duration, 1 / sampling_rate)
    # array of unit-angle steps (corresponding to frequency of 1Hz)
    angle_steps = np.full_like(time, duration * 2 * np.pi / len(time))
    # get frequencies and amplitudes over time
    if callable(func):
        freq_amps = func(time)
    else:
        freq_amps = func
    if not isinstance(freq_amps, tuple):
        freqs = freq_amps
        amps = 1.
    else:
        freqs, amps = freq_amps
    freqs = np.atleast_2d(freqs)
    amps = np.atleast_2d(amps)
    # effective change of angle for different frequencies (time-dependent)
    angle_steps = angle_steps[:, None] * freqs
    # actual angle at given time corresponds to the accumulated angle steps
    angles = np.cumsum(angle_steps, axis=0) + phases
    # generate oscillations, multiply by amplitudes and sum up
    return (amps * np.sin(angles)).sum(axis=1)


def spectrogram(wave, ylim=None, figsize=(14, 5), **kwargs):
    X = librosa.stft(wave)
    Xdb = librosa.amplitude_to_db(abs(X))
    Xdb += Xdb.min()
    plt.figure(figsize=figsize)
    kwargs = {**dict(sr=sampling_rate, x_axis='time', y_axis='hz'),
              **dict(kwargs)}
    librosa.display.specshow(Xdb, **kwargs)
    if ylim is not None:
        plt.ylim(None, ylim)
    plt.show()


def harmonic_tone(f0, decay=1, n=20, **kwargs):
    return sound(lambda time: (
        [f0 * i for i in range(1, n + 1)],
        [np.exp(-i*decay) for i in range(0, n)],
    ), **kwargs)
