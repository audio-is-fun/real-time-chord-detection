'''
Automatic chords detection algorithm (real time).

Implementation of the algorithm described in Described in the Bachelor Thesis: 
Design and Evaluation of a Simple Chord Detection Algorithm by Christoph Hausner.

Implemented by Haldo Sponton (haldosponton@gmail.com).

Note: this functionality requires pyaudio. 
I had to install it using `conda install pyaudio`. Pip installation failed.
'''

# imports
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import sys

import pyaudio
import struct

import queue
from matplotlib.animation import FuncAnimation
import sounddevice as sd

import config
import utils

# initial screen
print(config.SPLASH)

# show audio devices 
print('  ## Audio devices')
print(sd.query_devices())
print('\n')

# constants
chunk = 2 * 1024            # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
channels = 1                 # single channel for microphone
fs = 44100                   # samples per second

# pyaudio class instance
p = pyaudio.PyAudio()

# stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=channels,
    rate=fs,
    input=True,
    output=True,
    frames_per_buffer=chunk
)

samples = []

while True:
    
    try:
        # binary data
        data = stream.read(chunk)  
        # convert data to integers, make np array, then offset it by 127
        data_int = struct.unpack(str(2 * chunk) + 'B', data)
        # create np array and normalize
        data_np = np.array(data_int, dtype='b')[::2] / 128

        if len(samples) < config.NFFT:
            samples = np.concatenate([samples, data_np])
        else:
            # compute spectrogram
            f, t, sp = utils.compute_spectrogram(samples, fs)

            # harmonic content extraction
            sp_filtered = utils.harmonic_content_extraction(sp, f)

            # compute chromagram
            chromagram = utils.compute_chromagram(sp, f)

            # compute chordgram
            chordgram, detected_chords, detected_weights = utils.compute_chordgram(chromagram)
            print('  ## Detected chords: ' + ' '.join(detected_chords))
            print('  ## Detected weights: ' + ' '.join(['%.2f' % w for w in detected_weights]) + '\n')

            # save chordgram image
            utils.save_chordgram_image(chordgram, t, detected_chords, 'test.png')

            samples = []

    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))