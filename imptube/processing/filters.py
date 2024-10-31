"""
This module contains functions for filtering signals obtained from the impedance tube measurements.
"""
from scipy.signal import stft, istft, windows
import numpy as np
from typing import Tuple

from .signal_proc import stereo_to_spectra, transfer_function

def spectral_filtering(arr: np.ndarray, low_pass: int, nperseg: int=64) -> np.ndarray:
    """
    Apply spectral filtering to an array.

    Example usage: On a measured transfer function, to remove high-frequency noise.

    The functioning might be slightly counterintuitive, since we use FFT to get
    H(f) ->FFT-> H(t*), so it is more equivalent to windowing in the time domain.

    Parameters
    ----------
    arr : np.ndarray
        The input array.
    low_pass : int
        The cutoff frequency for the low-pass filter.

    Returns
    -------
    np.ndarray
        The filtered array.
    """
    spectrum = stft(arr, nperseg=nperseg)
    for s in spectrum[2].T:
        s[low_pass:] = 0
    return istft(spectrum[2])[1][:len(arr)]

def noise_filtering(arr: np.ndarray) -> np.ndarray:
    """Suitable for H(f) noise filtering.
    It takes the input array, calculates the iFFT, windows the peak, and then FFTs it back.
    """
    arr = np.fft.ifft(arr)
    # 10ms window
    win = windows.tukey(480)
    window = np.zeros_like(arr)
    window[:len(win)] = win
    window = np.roll(window, -int(len(window)/2))

    arr = arr * window
    return np.fft.fft(arr)

def extend_with_zeros(p_time : np.ndarray) -> np.ndarray:
    """
    Extend the input array with zeros.

    Parameters
    ----------
    p_time : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        The extended array with zeros.
    """
    if len(p_time.shape) == 2:
        extended = np.concatenate((p_time, np.zeros_like(p_time)), axis=1)
    elif len(p_time.shape) == 1:
        extended = np.concatenate((p_time, np.zeros_like(p_time)))
    return extended

def tf_filtering(
        tf : np.ndarray, 
        lowcut : int=10, 
        highcut : int=1000, 
        fs : int=48000
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Window a transfer function to a limited bandwidth.

    Parameters
    ----------
    tf : np.ndarray
        The transfer function to be filtered.
    lowcut : int, optional
        The lower cutoff frequency in Hz. Default is 10.
    highcut : int, optional
        The upper cutoff frequency in Hz. Default is 1000.
    fs : int, optional
        The sampling frequency in Hz. Default is 48000.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the filtered transfer function and the filter window.

    Notes
    -----
    For some reason, when used for calibration TF filtering, 
    lowcut should be put high enough to prevent problems with low S/N ratio.
    """
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
    window[idx_low+len(l_win):idx_high-len(r_win)] = 1
    window[idx_high-len(r_win):idx_high] = r_win

    window[-(idx_low+len(l_win)):-idx_low] = l_win[::-1]
    window[-(idx_high-len(r_win)):-(idx_low+len(l_win))] = 1
    window[-idx_high:-(idx_high-len(r_win))] = r_win[::-1]
    return tf*window, window

def ir_filtering(impulse_response: np.ndarray) -> np.ndarray:
    """
    This function cuts out the second half of the impulse response, which contains harmonic distortion products.

    Parameters
    ----------
    impulse_response : np.ndarray
        The input impulse response.

    Returns
    -------
    np.ndarray
        The filtered impulse response with the second half cut out.
    """
    cut = int(len(impulse_response) / 2)
    impulse_response[cut:] = 0 
    return impulse_response

def count_octaves(f0: float, f1: float) -> float:
    """
    This function counts octaves between two provided frequencies.

    Parameters
    ----------
    f0 : float
        The lower frequency.
    f1 : float
        The higher frequency.

    Returns
    -------
    float
        The number of octaves between f0 and f1.
    """
    return np.log(f1/f0) / np.log(2)

def harmonic_distortion_filter(
        p_time : np.ndarray, 
        p_ref : np.ndarray, 
        f_low : float=10, 
        f_high : float=1000, 
        fs : int=48000, 
        roll_fwd : bool=True
        ) -> np.ndarray:
    '''
    Combination of IR and TF filtering for removing harmonic distortion from measured signal.
    Note: Use with reasonably slow sweep rates to avoid artifacts 
    due to harmonics appearing in the beginning of the IR.

    Parameters
    ----------
    p_time : np.ndarray
        Time domain recording.
    p_ref : np.ndarray
        Reference (source time sequence).
    f_low : float
        Lower frequency limit of the reference (used for TF filtering and rolling).
        For some reason, to filter calibration audio, this should be higher than the original frequency limit of the sweep.
    f_high : float
        Higher frequency limit of the reference.
    fs : int
        Sampling frequency.
    roll_fwd : bool | None
        Sets the direction for reference signal rolling. This is done to avoid data loss in the IR. Use None for no rolling (shorter sweeps)

    Returns
    -------
    np.ndarray
        Array containing the filtered time domain signals for both channels.
    '''
    octave_span = len(p_ref)/count_octaves(f_low, f_high)

    p_time = extend_with_zeros(p_time)
    p_ref = extend_with_zeros(p_ref)
    if roll_fwd == True:
        p_ref = np.roll(p_ref, -int(octave_span/2))
    elif roll_fwd == False:
        p_ref = np.roll(p_ref, int(octave_span/2))
    else:
        p_ref = p_ref
    
    p_1, p_2 = stereo_to_spectra(p_time)
    p_ref = np.fft.fft(p_ref)
    tf_1 = transfer_function(p_ref, p_1)
    tf_2 = transfer_function(p_ref, p_2)
    tf_1 = tf_filtering(tf_1, lowcut=f_low, highcut=f_high)[0]
    tf_2 = tf_filtering(tf_2, lowcut=f_low, highcut=f_high)[0]
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

    return np.asarray([pf_1_time, pf_2_time]).real