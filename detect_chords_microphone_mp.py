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

import sounddevice as sd
import matplotlib.animation as animation
#from multiprocessing import Process
import multiprocessing as mp
from ctypes import c_float, c_double

import config
import utils

# initial screen
print(config.SPLASH)

# show audio devices 
print('  ## Audio devices')
print(sd.query_devices())
print('\n')

# arguments
device = 1
gain = 10 # initial gain factor

chordgram = np.zeros((len(config.chords), 1))
detected_chords = ['NC']
def runGraph(queue):

    fig = plt.figure(figsize=(3, 7))
    ax = plt.subplot(111)
    image = ax.pcolormesh(chordgram)
    plt.yticks(np.arange(len(config.chords)) + 0.5, config.chords)
    plt.ylabel('Chords (American notation) ->')

    # annotate with detected chords
    bbox_props = dict(boxstyle="round4", fc="white", ec="black")
    tt = np.arange(1) + 0.5
    text = ax.text(tt[0], config.chords.index(detected_chords[0]) + 0.5, detected_chords[0], 
                size=16, bbox=bbox_props, ha="center", va="center")

    # This function is called periodically from FuncAnimation
    def animate(i):
        # print(i)
        # Update line with new Y values
        ax.cla()
        # chordgram = np.random.randn(len(config.chords), 1)
        if queue.empty():
            chordgram = np.zeros((len(config.chords), 1))
            detected_chords = ['NC']
        else:
            chordgram, detected_chords = queue.get()
        image = ax.pcolormesh(chordgram)
        text = ax.text(tt[0], config.chords.index(detected_chords[0]) + 0.5, detected_chords[0], 
                size=16, bbox=bbox_props, ha="center", va="center")
        plt.yticks(np.arange(len(config.chords)) + 0.5, config.chords)
        plt.ylabel('Chords (American notation) ->')
        return image,text,
    
    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig,
        animate,
        interval=50
        )
    plt.draw()
    plt.show(block=True)

def mainProgram(queue):
    try:
        samplerate = sd.query_devices(device, 'input')['default_samplerate']

        def callback(indata, frames, time, status):
            if status:
                text = ' ' + str(status) + ' '
            if any(indata):
                samples = indata[:, 0]
                #global chordgram
                #global detected_chords
                f, t, sp = utils.compute_spectrogram(samples, samplerate)
                sp_filtered = utils.harmonic_content_extraction(sp, f)
                chromagram = utils.compute_chromagram(sp, f)
                chordgram, detected_chords, detected_weights = utils.compute_chordgram(chromagram)
                if detected_chords[0] != 'NC' and sum(chromagram)[0] > config.chromagram_accum_threshold:
                    print('  ## Detected chords: ' + ' '.join(detected_chords))
                    print('  ## Detected weights: ' + ' '.join(['%.2f' % w for w in detected_weights]) + '\n')
                    queue.put((chordgram, detected_chords))
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

if __name__ == '__main__':
    if plt.get_backend() == "MacOSX":
        mp.set_start_method("forkserver", force=True)

    queue = mp.Queue()

    p = mp.Process(target=runGraph, args=(queue, ))
    p.start()
    mainProgram(queue)
    p.join()
    # runGraph()