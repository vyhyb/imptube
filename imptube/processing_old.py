import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
import os
from scipy.fft import fft, fftfreq
from scipy.signal import stft, istft, windows

from .utils import filter, read_folder

NORMAL_DENSITY = 1.186
NORMAL_PRESSURE = 101325
NORMAL_TEMPERATURE = 293

def read_file(path):
    return wav.read(path)

def read_audio(file):
    return file[1].T

def separate_mono(stereo_data):
    p1_time = stereo_data[0]
    p2_time = stereo_data[1]
    return p1_time, p2_time

def time_to_freq(p_time):
    return fft(p_time)

def frequencies(p, fs):
    return fftfreq(p.shape[0], 1/fs)

def stereo_to_spectra(stereo_data):
    p1_time, p2_time = separate_mono(stereo_data)
    p1 = fft(p1_time)
    p2 = fft(p2_time)
    return p1, p2

def auto_spectrum(p1):
    return p1 * np.conj(p1)

def cross_spectrum(p1, p2):
    return p2 * np.conj(p1)

def transfer_function(p1, p2):
    return cross_spectrum(p1, p2) / auto_spectrum(p1)

def load_bc(parent_folder):
    file_paths = read_folder(parent_folder)
    bc_path = filter(
            file_paths,
            ["bound_cond.csv"]
    )
    bc_df = pd.read_csv(bc_path[0])
    return bc_df

def load_freqs(parent_folder):
    file_paths = read_folder(parent_folder)
    freqs_path = filter(
            file_paths,
            ["freqs.npy"]
    )
    freqs = np.load(freqs_path[0])
    return freqs

def load_alpha(parent_folder):
    file_paths = read_folder(parent_folder)
    alpha_path = filter(
            file_paths,
            ["_alpha"]
    )
    alpha = []
    unique_d = []
    for p in alpha_path:
        alpha.append(np.load(p))
        idx1 = p.rfind("d")
        idx2 = p.rfind(".")
        unique_d.append(p[idx1:idx2])
    return alpha, unique_d

def transfer_function_from_file(path=None, folder=None, filter_str="_"):
    if folder is None:
        if path is None:
            print("You need to define path to file (path), or to folder (folder)")
        else:
            f = read_file(path)
            a = read_audio(f)
            p1, p2 = stereo_to_spectra(a)
            tf = transfer_function(p1, p2)
    else:
        files = read_folder(folder)
        f_paths = filter(files, [filter_str])
        f = []
        a = []
        p1 = []
        p2 = []
        for fp in f_paths:
            f.append(read_file(fp))
            a.append(read_audio(f[-1]))
            p = stereo_to_spectra(a[-1])
            p1.append(p[0])
            p2.append(p[1])
        p1_np = np.array(p1)
        p2_np = np.array(p2)
        p1_np = p1_np.mean(axis=0)
        p2_np = p2_np.mean(axis=0)
        tf = transfer_function(p1_np, p2_np)
    return tf

def calibration_factor(p11, p12, p21, p22):
    return transfer_function(p11, p12) / transfer_function(p22, p21)

def calibration_from_files(path1=None, path2=None, parent_folder=None, export=True):
    if parent_folder is None:
        f1 = read_file(path1)
        f2 = read_file(path2)
        a1 = read_audio(f1)
        a2 = read_audio(f2)
        p11, p12 = stereo_to_spectra(a1)
        p21, p22 = stereo_to_spectra(a2)
        cf = calibration_factor(p11, p12, p21, p22)
    else:
        folder = os.path.join(parent_folder, "calibration", "audio")
        export_folder = os.path.join(parent_folder, "calibration", "cal_factor")
        files = read_folder(folder)
        f1_paths = filter(files, ["conf1"])
        f2_paths = filter(files, ["conf2"])
        f1 = []
        f2 = []
        a1 = []
        a2 = []
        p11 = []
        p12 = []
        p21 = []
        p22 = []
        for f_1, f_2 in zip(f1_paths, f2_paths):
            f1.append(read_file(f_1))
            f2.append(read_file(f_2))
            a1.append(read_audio(f1[-1]))
            a2.append(read_audio(f2[-1]))
            p1 = stereo_to_spectra(a1[-1])
            p2 = stereo_to_spectra(a2[-1])
            p11.append(p1[0])
            p12.append(p1[1])
            p21.append(p2[0])
            p22.append(p2[1])
        p11_np = np.array(p11)
        p12_np = np.array(p12)
        p21_np = np.array(p21)
        p22_np = np.array(p22)
        
        p11_np = p11_np.mean(axis=0)
        p21_np = p21_np.mean(axis=0)
        p22_np = p22_np.mean(axis=0)
        p12_np = p12_np.mean(axis=0)
        
        cf = calibration_factor(p11_np, p12_np, p21_np, p22_np)
        freqs = fftfreq(f1[0][1].shape[0], 1/f1[0][0])
        if export == True:
            base_name = os.path.split(parent_folder)[1]
            np.save(os.path.join(parent_folder, base_name+"_freqs.npy"), freqs) #freq export
            np.save(os.path.join(export_folder, base_name+"_cal_f_12.npy"), cf) #cf export
    return cf, freqs

def calc_speed_sound(tempC):
    return 343.2 * np.sqrt((tempC+273) / NORMAL_TEMPERATURE)

def calc_char_impedance_air(temperature, atm_pressure):
    char_impedance = NORMAL_DENSITY*(
        (atm_pressure*NORMAL_TEMPERATURE)/(NORMAL_PRESSURE*(temperature+273))
    )
    return char_impedance

def k_0(speed_sound, freqs):
    return 2 * np.pi * freqs / speed_sound

def tf_i_r(temperature, freqs, s):
    tf_I = np.exp(- 1j * k_0(calc_speed_sound(temperature), freqs) * s)
    tf_R = np.exp(1j * k_0(calc_speed_sound(temperature), freqs) * s)
    return tf_I, tf_R

def reflection_factor(tf_I, tf_R, tf_12, temperature, freqs, x1):
    return (tf_12 - tf_I)/(tf_R - tf_12) * np.exp(2j * k_0(calc_speed_sound(temperature), freqs) * x1)

def absorption_coefficient(reflection_f):
    return 1 - np.abs(reflection_f)**2

def surface_impedance(reflection_f, temperature, atm_pressure):
    char_impedance_air = calc_char_impedance_air(temperature, atm_pressure)
    surf_impedance = (1+reflection_f)/(1-reflection_f)*char_impedance_air
    return surf_impedance

def transfer_function_from_path(parent_folder):
    bc_df = load_bc(parent_folder)
    limit = int(bc_df["lim"].iloc[0])
    audio_folder = os.path.join(parent_folder, "measurement", "audio")
    tf_folder = os.path.join(parent_folder, "measurement", "transfer_func")
    audio_files = read_folder(audio_folder)
    fs, a = read_file(audio_files[0])
    limit_idx = int(fftfreq(a.shape(0), 1/fs)[1]*limit)
    audio_files_d_filtered = []
    for f in audio_files:
        idx1 = f.rfind("d")
        idx2 = f.rfind("_")
        audio_files_d_filtered.append(f[idx1:idx2])
    unique_d = np.unique(np.array(audio_files_d_filtered))
    
    tfs = []
    for d in unique_d:
        tf = transfer_function_from_file(folder=audio_folder, filter_str=d)
        tfs.append(tf[:limit_idx])
        np.save(arr=tfs[-1], file=os.path.join(tf_folder, os.path.split(parent_folder)[1]+"_tf_"+d+".npy"))

    return unique_d, tfs

def alpha_from_path(parent_folder, fs=48000, cal=None, tfs=None, bc=None, return_f=False):
    # boundary conditions
    if bc is None:
        bc_df = load_bc(parent_folder)
    else:
        bc_df = bc
    temp_c = bc_df["temp"].iloc[0]
    x1_dist = bc_df["x1"].iloc[0]
    x2_dist = bc_df["x2"].iloc[0]
    limit = int(bc_df["lim"].iloc[0])
    s_dist = x1_dist - x2_dist

    # read calibration transfer function
    if cal is None:
        tf_cal = np.load(
            read_folder(
                os.path.join(
                    parent_folder, "calibration", "cal_factor"
                )
            )[0]
        )
    else:
        tf_cal = cal
    freqs = load_freqs(parent_folder)
    limit_idx = int(freqs[1]*limit)
    tf_cal = tf_cal[:limit_idx]
    freqs = freqs[:limit_idx]

    # read all transfer functions from folder
    tf_paths = read_folder(
        os.path.join(parent_folder, "measurement", "transfer_func")
    )
    # print(tf_paths)
    if tfs is None:
        tfs = []
        for t in tf_paths:
            tfs.append(np.load(t)[:limit_idx])

    # calculate transfer function for incident and reflected wave
    tf_incident, tf_reflected = tf_i_r(temp_c, freqs, s_dist)
    tf_incident = tf_incident
    tf_reflected = tf_reflected

    # calculate reflection factor and absorption coefficient
    rfs = []
    for t in tfs:
        tf_corrected = t/tf_cal
        rfs.append(reflection_factor(tf_incident, tf_reflected, tf_corrected, temp_c, freqs, x1_dist))
    alpha_n = []
    for r in rfs:
        alpha_n.append(absorption_coefficient(r))

    # save all alpha courses
    for a, p in zip(alpha_n, tf_paths):
        print(p)
        idx = str(p).rfind("d")
        np.save(arr=a,
            file=os.path.join(
                parent_folder, "measurement", "alpha", os.path.split(parent_folder)[1]+"_alpha_"+str(p)[idx:-4]+".npy"
            )
        )
    np.save(arr=freqs, file=os.path.join(
                parent_folder, "measurement", "alpha", os.path.split(parent_folder)[1]+"_freqs.npy"
        )
    )
    if return_f == True:
        ret = (alpha_n, freqs)
    else:
        ret = alpha_n
    return ret

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