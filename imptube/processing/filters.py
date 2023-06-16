from scipy.signal import stft, istft, windows
import numpy as np

from .signal_proc import stereo_to_spectra, transfer_function

def spectral_filtering(arr, low_pass):
    spectrum = stft(arr, nperseg=64)
    for s in spectrum[2].T:
        s[low_pass:] = 0
    return istft(spectrum[2])[1][:len(arr)]

def extend_with_zeros(p_time):
    if len(p_time.shape) == 2:
        extended = np.concatenate((p_time, np.zeros_like(p_time)), axis=1)
    elif len(p_time.shape) == 1:
        extended = np.concatenate((p_time, np.zeros_like(p_time)))
    return extended

def tf_filtering(tf, lowcut=10, highcut=1000, fs=48000):
    '''
    This function windows a transfer function to a limited bandwidth.
    For some reason, when used for calibration TF filtering, 
    lowcut should be put high enough to prevent problems with low S/N ratio.
    '''
    freqs = np.fft.fftfreq(len(tf), 1/fs)
    window = np.zeros_like(freqs)
    f_step = freqs[1]-freqs[0]
    
    idx_low = int(lowcut/f_step)
    idx_high = int(highcut/f_step)
    
    l_win = windows.hann(int(idx_low*2**(1/3)))
    l_win = l_win[:int(len(l_win)/2)]
    r_win = windows.hann(int(idx_high/2**(1/3)))
    r_win = r_win[int(len(r_win)/2):]
    
    window[idx_low:idx_low+len(l_win)] = l_win
    window[idx_low:idx_high] = 1
    window[idx_high-len(r_win):idx_high] = r_win

    window[-(idx_low+len(l_win)):-idx_low] = l_win[::-1]
    window[-idx_high:-idx_low] = 1
    window[-idx_high:-(idx_high-len(r_win))] = r_win[::-1]
    return tf*window, window

def ir_filtering(impulse_response):
    '''
    This function cuts out second half of the IR, which contains harmonic distortion products.
    '''    
    cut = int(len(impulse_response)/2)
    impulse_response[cut:] = 0 
    return impulse_response

def count_octaves(f0, f1):
    '''
    This function counts octaves between two provided frequencies.
    '''
    return np.log(f1/f0)/np.log(2)

def harmonic_distortion_filter(p_time, p_ref, f_low=10, f_high=1000, fs=48000, roll_fwd=True):
    '''
    Combination of IR and TF filtering for removing harmonic distortion from measured signal.
    p_time  - time domain recording
    p_ref   - reference (source time sequence)
    f_low   - lower frequency limit of the reference (used for TF filtering and rolling)
        For some reason, to filter calibration audio, 
        this should be higher then the original frequency limit of the sweep.
    f_high  - higher frequency limit of the reference
    fs      - sampling frequency
    roll_fwd - sets the direction for reference signal rolling. This is done to keep the end of the measured sweep.
    '''
    octave_span = len(p_ref)/count_octaves(f_low, f_high)

    p_time = extend_with_zeros(p_time)
    p_ref = extend_with_zeros(p_ref)
    if roll_fwd == True:
        p_ref = np.roll(p_ref, -int(octave_span/2))
    else:
        p_ref = np.roll(p_ref, int(octave_span/2))
    
    p_1, p_2 = stereo_to_spectra(p_time)
    p_ref = np.fft.fft(p_ref)
    tf_1 = transfer_function(p_ref, p_1)
    tf_2 = transfer_function(p_ref, p_2)
    tf_1 = tf_filtering(tf_1, lowcut=f_low*2, highcut=f_high)[0]
    tf_2 = tf_filtering(tf_2, lowcut=f_low*2, highcut=f_high)[0]
    ir_1 = np.fft.ifft(tf_1)
    ir_2 = np.fft.ifft(tf_2)

    ir_1 = ir_filtering(ir_1)
    ir_2 = ir_filtering(ir_2)

    tff_1 = np.fft.fft(ir_1)
    tff_2 = np.fft.fft(ir_2)

    pf_1 = tff_1 * p_ref
    pf_2 = tff_2 * p_ref

    pf_1_time = np.fft.ifft(pf_1)[:int(len(p_ref)/2)]
    pf_2_time = np.fft.ifft(pf_2)[:int(len(p_ref)/2)]


    return np.asarray([pf_1_time, pf_2_time]).astype(np.float32)