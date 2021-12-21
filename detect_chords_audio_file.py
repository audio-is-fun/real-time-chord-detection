'''
Automatic chords detection algorithm.

Implementation of the algorithm described in Described in the Bachelor Thesis: 
Design and Evaluation of a Simple Chord Detection Algorithm by Christoph Hausner.

Implemented by Haldo Sponton (haldosponton@gmail.com).
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

import config
import utils

# parse arguments
parser = argparse.ArgumentParser(description='Detect chords from audio file. Implementation of the algorithm described in Described in the Bachelor Thesis: Design and Evaluation of a Simple Chord Detection Algorithm by Christoph Hausner. Implemented by Haldo Sponton (haldosponton@gmail.com).')
parser.add_argument('--input_file', '-i', type=str, required=True,
                help='Path to input audio file.')
parser.add_argument('--output_file', '-o', type=str, default=None,
                help='Output chordgram image (default will be the same filename of the audio file but .png).')
args = parser.parse_args()

if not args.output_file:
    base = os.path.basename(args.input_file)
    args.output_file = 'output/' + os.path.splitext(base)[0] + '.png'

# initial screen
print(config.SPLASH)

# load audio file
samples, fs = utils.load_audio(args.input_file)
print('  ## Audio loaded. Duration %.2f seconds - Samples %d - Sampling rate (fs) %d Hz\n' %
      (len(samples)/fs, len(samples), fs))

# compute spectrogram
f, t, sp = utils.compute_spectrogram(samples, fs)
print('  ## Spectrogram computed. Frequency bins %d, time segments %d.\n' % sp.shape)

# harmonic content extraction
sp_filtered = utils.harmonic_content_extraction(sp, f)
print('  ## Harmonic content extraction finished.\n')

# compute chromagram
chromagram = utils.compute_chromagram(sp, f)
print('  ## Chromagram computed.\n')

# compute chordgram
chordgram, detected_chords, detected_weights = utils.compute_chordgram(chromagram)
print('  ## Chordgram computed.\n')
print('  ## Detected chords: ' + ' '.join(detected_chords))
print('  ## Detected weights: ' + ' '.join(['%.2f' % w for w in detected_weights]) + '\n')

# save chordgram image
utils.save_chordgram_image(chordgram, t, detected_chords, args.output_file)
print('  ## Chordgram image saved in %s.\n' % args.output_file)