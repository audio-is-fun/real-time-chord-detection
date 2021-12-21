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
import librosa
import librosa.display
import scipy
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import sys

import sounddevice as sd

import config
import utils

import cv2

# arguments
device = 1

# clear output
os.system('clear')
# initial screen
print(config.SPLASH)
# show audio devices 
print('  ## Audio devices')
print(sd.query_devices())
print('\n')

try:
    samplerate = sd.query_devices(device, 'input')['default_samplerate']
    fftsize = config.NFFT
    # block_duration = config.NFFT/samplerate * 1000 # miliseconds

    def callback(indata, frames, time, status):
        if status:
            text = ' ' + str(status) + ' '
        if any(indata):
            samples = indata[:, 0]
            #global chordgram
            #global detected_chords
            f, t, sp = utils.compute_spectrogram(samples, samplerate)
            sp_filtered = utils.harmonic_content_extraction(sp, f)
            chromagram = utils.compute_chromagram(sp_filtered, f)
            chordgram, detected_chords, detected_weights = utils.compute_chordgram(chromagram)
            #print('  ## Detected chords: ' + ' '.join(detected_chords))
            #print('  ## Detected weights: ' + ' '.join(['%.2f' % w for w in detected_weights]) + '\n')
            if detected_chords[0] != 'NC' and sum(chromagram)[0] > config.chromagram_accum_threshold:
                # clear output
                os.system('clear')
                # initial screen
                print(config.SPLASH)
                # show audio devices 
                print('  ## Audio devices')
                print(sd.query_devices())
                print('\n')
                # print detection
                utils.print_chromagram(chromagram)
                print(config.ascii_text[detected_chords[0]] + '\n', flush=True)
        else:
            print('no input')

    with sd.InputStream(device=device, channels=1, callback=callback,
                        blocksize=config.NFFT,
                        samplerate=samplerate):
        while True:
            response = input()
            if response in ('', 'q', 'Q'):
                break

except KeyboardInterrupt:
    print('Interrupted by user')
except Exception as e:
    print(type(e).__name__ + ': ' + str(e))