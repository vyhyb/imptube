import numpy as np
from scipy.fft import fft, fftfreq

NORMAL_DENSITY = 1.186
NORMAL_PRESSURE = 101325
NORMAL_TEMPERATURE = 293

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

def calibration_factor(p11, p12, p21, p22):
    return transfer_function(p11, p12) / transfer_function(p22, p21)

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