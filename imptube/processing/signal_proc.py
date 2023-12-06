"""
This module contains functions for processing audio signals from the impedance tube measurements.
"""
import numpy as np
from scipy.fft import fft, fftfreq

NORMAL_DENSITY = 1.186
NORMAL_PRESSURE = 101325
NORMAL_TEMPERATURE = 293

def read_audio(file : tuple[np.ndarray, int]) -> np.ndarray:
    """
    Read audio from a file.

    Parameters
    ----------
    file : tuple[np.ndarray, int]
        A tuple containing the audio data as a numpy array and the sample rate.

    Returns
    -------
    np.ndarray
        The audio data as a numpy array.
    """
    return file[1].T

def separate_mono(stereo_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separates the stereo audio data into two mono channels.

    Parameters
    ----------
    stereo_data : np.ndarray
        The stereo audio data to be separated.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two mono channels, where the first element is the
        audio data for the first channel and the second element is the audio
        data for the second channel.
    """
    p1_time = stereo_data[0]
    p2_time = stereo_data[1]
    return p1_time, p2_time

def time_to_freq(p_time : np.ndarray) -> np.ndarray:
    """
    Convert time domain signal to frequency domain using Fast Fourier Transform (FFT).

    Parameters
    ----------
    p_time : np.ndarray
        The time domain signal to be converted.

    Returns
    -------
    np.ndarray
        The frequency domain representation of the input signal.
    """
    return fft(p_time)

def frequencies(p : np.ndarray, fs : int) -> np.ndarray:
    """
    Calculate the frequencies of a signal.

    Parameters
    ----------
    p : np.ndarray
        The signal data.
    fs : int
        The sampling frequency.

    Returns
    -------
    np.ndarray
        An array of frequencies corresponding to the signal.
    """
    return fftfreq(p.shape[0], 1/fs)

def stereo_to_spectra(stereo_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert stereo audio data to spectra.

    Parameters
    ----------
    stereo_data : np.ndarray
        Stereo audio data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the spectra of the first and second channels.
    """
    p1_time, p2_time = separate_mono(stereo_data)
    p1 = fft(p1_time)
    p2 = fft(p2_time)
    return p1, p2

def auto_spectrum(p1 : np.ndarray) -> np.ndarray:
    """
    Calculate the auto spectrum of a signal.

    Parameters
    ----------
    p1 : np.ndarray
        The input signal.

    Returns
    -------
    np.ndarray
        The auto spectrum of the input signal.
    """
    return p1 * np.conj(p1)

def cross_spectrum(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Calculate the cross spectrum between two signals.

    Parameters
    ----------
    p1 : np.ndarray
        The first signal.
    p2 : np.ndarray
        The second signal.

    Returns
    -------
    np.ndarray
        The cross spectrum of the two signals.
    """
    return p2 * np.conj(p1)

def transfer_function(p1 : np.ndarray, p2 : np.ndarray) -> np.ndarray:
    return cross_spectrum(p1, p2) / auto_spectrum(p1)

def calibration_factor(
        p11 : np.ndarray, 
        p12 : np.ndarray, 
        p21 : np.ndarray, 
        p22 : np.ndarray
        ) -> np.ndarray:
    """Calculate the calibration factor.

    This function calculates the calibration factor based on the given input arrays.
    The calibration factor is computed as the ratio of the transfer functions
    between p11 and p12, and p22 and p21.

    Parameters
    ----------
    p11 : np.ndarray
        The input array representing p11.
    p12 : np.ndarray
        The input array representing p12.
    p21 : np.ndarray
        The input array representing p21.
    p22 : np.ndarray
        The input array representing p22.

    Returns
    -------
    np.ndarray
        The calibration factor as an array.

    """
    return transfer_function(p11, p12) / transfer_function(p22, p21)

def calc_speed_sound(tempC : float) -> float:
    """Calculate the speed of sound in air at a given temperature.

    Parameters
    ----------
    tempC : float
        The temperature in degrees Celsius.

    Returns
    -------
    float
        The speed of sound in meters per second.
    """
    return 343.2 * np.sqrt((tempC+273) / NORMAL_TEMPERATURE)

def calc_char_impedance_air(temperature : float, atm_pressure : float) -> float:
    """Calculate the characteristic impedance of air.

    This function calculates the characteristic impedance of air based on 
    the given temperature and atmospheric pressure.

    Parameters
    ----------
    temperature : float
        The temperature of the air in degrees Celsius.
    atm_pressure : float
        The atmospheric pressure in pascals.

    Returns
    -------
    float
        The characteristic impedance of air in ohms.
    """
    char_impedance = NORMAL_DENSITY*(
        (atm_pressure*NORMAL_TEMPERATURE)/(NORMAL_PRESSURE*(temperature+273))
    )
    return char_impedance

def k_0(speed_sound: float, freqs: np.ndarray) -> np.ndarray:
    """
    Calculate the wavenumber for a given speed of sound and frequencies.

    Parameters
    ----------
    speed_sound : float
        The speed of sound.
    freqs : np.ndarray
        An array of frequencies.

    Returns
    -------
    np.ndarray
        An array of wavenumbers.
    """
    return 2 * np.pi * freqs / speed_sound

def tf_i_r(
        temperature : float, 
        freqs : np.ndarray, 
        s : float
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the transfer functions for the incident and reflected sound.

    Parameters
    ----------
    temperature : float
        The temperature in degrees Celsius.
    freqs : np.ndarray
        The array of frequencies.
    s : float
        The distance between the microphones.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the transfer functions for the incident and reflected sound.

    Notes
    -----
    The transfer functions are calculated using the speed of sound at the given temperature and the array of frequencies.
    The transfer function for the incident sound is calculated as exp(-1j * k_0 * s), where k_0 is the wavenumber.
    The transfer function for the reflected sound is calculated as exp(1j * k_0 * s), where k_0 is the wavenumber.
    """
    tf_I = np.exp(- 1j * k_0(calc_speed_sound(temperature), freqs) * s)
    tf_R = np.exp(1j * k_0(calc_speed_sound(temperature), freqs) * s)
    return tf_I, tf_R

def reflection_factor(
        tf_I : np.ndarray, 
        tf_R : np.ndarray, 
        tf_12 : np.ndarray, 
        temperature : float, 
        freqs : np.ndarray, 
        x1 : float
        ) -> np.ndarray:
    """Calculate the reflection factor for a given set of parameters.

    Parameters
    ----------
    tf_I : np.ndarray
        The transfer function of the incident wave.
    tf_R : np.ndarray
        The transfer function of the reflected wave.
    tf_12 : np.ndarray
        The transfer function of the medium between the incident and reflected waves.
    temperature : float
        The temperature of the medium.
    freqs : np.ndarray
        The frequencies at which the reflection factor is calculated.
    x1 : float
        The distance between the incident and reflected waves.

    Returns
    -------
    np.ndarray
        The reflection factor at each frequency.
    """
    return (tf_12 - tf_I)/(tf_R - tf_12) * np.exp(2j * k_0(calc_speed_sound(temperature), freqs) * x1)

def absorption_coefficient(reflection_f: np.ndarray) -> np.ndarray:
    """
    Calculate the absorption coefficient.

    Parameters
    ----------
    reflection_f : np.ndarray
        The reflection coefficient in the frequency domain.

    Returns
    -------
    np.ndarray
        The absorption coefficient in the frequency domain.
    """
    return 1 - np.abs(reflection_f)**2

def surface_impedance(
        reflection_f : np.ndarray, 
        temperature : float, 
        atm_pressure : float
        ) -> float:
    """Calculate the surface impedance.

    This function calculates the surface impedance based on the reflection factor, temperature, and atmospheric pressure.

    Parameters
    ----------
    reflection_f : np.ndarray
        The reflection factor.
    temperature : float
        The temperature in degrees Celsius.
    atm_pressure : float
        The atmospheric pressure in Pascal.

    Returns
    -------
    float
        The surface impedance.

    """
    char_impedance_air = calc_char_impedance_air(temperature, atm_pressure)
    surf_impedance = (1+reflection_f)/(1-reflection_f)*char_impedance_air
    return surf_impedance