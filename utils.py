import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
import warnings
warnings.filterwarnings('ignore')

import config

def load_audio(input_file):
    '''
    Load audio from file
    '''
    return librosa.load(input_file, sr = None, offset = 0.0, duration = None)

def compute_spectrogram(samples, fs):
    '''
    Compute spectrogram using configurated parameters
    '''
    f, t, sp = scipy.signal.spectrogram(x=samples, fs=fs, nfft=config.NFFT, 
                                    noverlap=config.NOVERLAP, nperseg=config.NFFT, 
                                    mode='magnitude', scaling='spectrum')
    return f, t, sp

def get_harmonics_decay_model():
    '''
    Calculate harmonics decay model
    '''
    A = [1.0]
    for i in range(config.n_harmonics):
        A.append(A[0]*(config.ex**(i+1)))
    return A

def harmonic_content_extraction(sp, f, n_harmonics = config.n_harmonics, 
                                fmin = config.fmin, fmax = config.fmax, 
                                ex = config.ex):
    '''
    Perform harmonic content extraction
    '''
    sp_filtered = np.full_like(sp, 0)
    flim = f[-1]
    
    for i in range(len(f)):
        if f[i]<fmin:
            continue
        if f[i]>fmax:
            break
        sp_in_hamronics = []
        for n in range(1, n_harmonics):
            if n*f[i] < flim:
                idx = (np.abs(f - n*f[i])).argmin()
                sp_in_hamronics.append(sp[idx,:]*((1/ex)**(n-1)))
        sp_in_hamronics = np.array(sp_in_hamronics)
        sp_filtered[i,:] = np.min(sp_in_hamronics, axis=0)
    return sp_filtered

def compute_chromagram(sp, f, first_note = config.first_note, 
                       last_note = config.last_note, f_A4 = config.f_A4):
    '''
    Compute chromagram for notes between <first_note> and <last_note>
    '''
    def freq_midi(N, f_A4=f_A4):
        return 2**((N-69)/12) * f_A4

    chromagram = np.zeros((12, sp.shape[1]))
    pitches = config.pitches
    midi_codes_zero = config.midi_codes_zero
    for i in range(12):
        pitch = pitches[i]
        midi_code = midi_codes_zero[i]
        sp_in_note = []
        for j in range(10):
            if midi_code + j*12 < first_note:
                continue
            if midi_code + j*12 > last_note:
                break
            idx = (np.abs(f - freq_midi(midi_code + j*12))).argmin()
            sp_in_note.append(sp[idx,:])
        sp_in_note = np.array(sp_in_note)
        chromagram[i, :] = np.mean(sp_in_note, axis=0)
    return chromagram

def compute_chordgram(chromagram, w_major = config.w_major, 
                      w_minor = config.w_minor, w_nc = config.w_nc):
    '''
    Compute chordgram for chords in config.chords.
    '''
    chordgram = np.zeros((25,chromagram.shape[1]))
    detected_chords = []
    detected_weights = []
    chord_templates = np.array(config.chord_templates)
    for i in range(chromagram.shape[1]):
        chroma = chromagram[:,i]
        for j in range(25):
            a = chroma
            b = chord_templates[j]
            chordgram[j,i] = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        detected_chords.append(config.chords[chordgram[:,i].argmax()])
        detected_weights.append(chordgram[chordgram[:,i].argmax(),i])
    return chordgram, detected_chords, detected_weights

def save_chordgram_image(chordgram, t, detected_chords, output_file):
    '''
    Get chordgram visualization as image
    '''
    # generate figure
    fig, ax = plt.subplots(figsize=(20, 16))
    ax.pcolormesh(chordgram)
    plt.yticks(np.arange(len(config.chords)) + 0.5, config.chords)
    plt.xticks(np.arange(len(t)) + 0.5, ['%.2f' % x for x in t])
    plt.ylabel('Chords (American notation) ->')
    plt.xlabel('Time (seconds) ->')

    # annotate with detected chords
    bbox_props = dict(boxstyle="round4", fc="white", ec="black")
    tt = np.arange(len(t)) + 0.5
    for i in range(len(t)):
        ax.text(tt[i], config.chords.index(detected_chords[i]) + 0.5, detected_chords[i], 
                size=36, bbox=bbox_props, ha="center", va="center")

    # save figure
    plt.savefig(output_file, bbox_inches='tight')

def print_chromagram(chromagram):
    for note, value in zip(config.pitches, chromagram):
        print(note, '\t' + '%.3f' % (100*value[0]), '\t|', (int(5000*value[0]))*'#')
    print('\n')