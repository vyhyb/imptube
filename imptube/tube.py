'''In this module, three main classes are defined - Measurement, Tube and Sample.
'''

import sys
import sounddevice as sd
import numpy as np
import pandas as pd
import soundfile as sf
import os
from scipy.io import wavfile
from scipy.signal import chirp
from scipy.signal.windows import hann
from time import sleep, strftime
from imptube.utils import make_foldertree
from imptube.processing import (
    calibration_from_files,
    transfer_function_from_path,
    alpha_from_path,
    harmonic_distortion_filter,
    calc_rms_pressure_level
)
from typing import Protocol
import logging


class Measurement:
    """Contains information about measurement from the perspective of
    signal and boundary conditions.

    Attributes
    ----------
    fs : int
        measurement sampling frequency
    channels_in : list[int]
        list of input channel numbers
    channels_out : list[int]
        list of output channel numbers (usually one member list)
    device : str
        string specifying part of sound card name
        List of available devices can be obtained with
        `python3 -m sounddevice` command.
    samples : int
        number of samples in the generated log sweep
        typically 2**n
    window_len : int
        length of the Hann half-window applied to the ends of the sweep
    sub_measurements : int
        number of measurements taken for each specimen
        Normally, no differences between sweep measurements should occur,
        this attribute mainly compensates for potential playback artifacts.
    f_low : int
        lower frequency limit for the generated sweep
    f_high : int
        higher frequency limit for the generated sweep
    fs_to_spl : float
        conversion level from dBFS to dB SPL for microphone 1
    sweep_lvl : float
        level of the sweep in dBFS
    """

    def __init__(
            self, 
            fs : int=48000, 
            channels_in : list[int]=[1,2], 
            channels_out : list[int]=[1], 
            device : str='Scarlett',
            samples : int=131072, 
            window_len : int=8192,
            sub_measurements : int=2,
            f_low : int=10,
            f_high : int=1000,
            fs_to_spl : float=130,
            sweep_lvl : float=-6  
        ):
        self.fs = fs
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.device = device
        self.samples = samples
        self.window_len = window_len
        self.sub_measurements = sub_measurements
        self.f_limits = [f_low, f_high]
        self.fs_to_spl = fs_to_spl
        self.sweep_lvl = sweep_lvl

        self.boundary_df = pd.DataFrame({"fs_to_spl": [fs_to_spl]})
        self.boundary_df.to_csv(
            strftime("%y-%m-%d_%H-%M") + "_lvl_calib.csv"
        )

        self.make_sweep()
        sd.default.samplerate = fs
        sd.default.channels = len(channels_in), len(channels_out)
        sd.default.device = device


    def make_sweep(self, windows=True) -> np.ndarray:
        """Generates numpy array with log sweep.

        Parameters
        ----------
        fs : int 
            measurement sampling frequency
        samples : int
            number of samples in the generated log sweep
            typically 2**n
        window_len : int
            length of the Hann half-window applied to the ends
            of the sweep
        f_low : int
            lower frequency limit for the generated sweep
        f_high : int
            higher frequency limit for the generated sweep

        Returns
        -------
        log_sweep : np.ndarray
            numpy array containing mono log sweep
        """
        t = np.linspace(0,self.samples/self.fs,self.samples, dtype=np.float32)
        
        half_win = int(self.window_len)
        log_sweep = chirp(t, self.f_limits[0], t[-1], self.f_limits[1], method="log", phi=90)
        
        if windows:
            window = hann(int(self.window_len*2))
            log_sweep[:half_win] = log_sweep[:half_win]*window[:half_win]
            log_sweep[-half_win:] = log_sweep[-half_win:]*window[half_win:]
        
        lvl_to_factor = 10**(self.sweep_lvl/20)
        log_sweep = log_sweep*lvl_to_factor

        self.sweep = log_sweep
        return log_sweep
    
    def regen_sweep(self):
        """Regenerates the sweep."""
        self.make_sweep()

    def update_sweep_lvl(self):
        """Updates the sweep level."""
        self.sweep = self.sweep/np.max(np.abs(self.sweep))
        self.sweep = self.sweep * 10**(self.sweep_lvl/20)

    def filter_sweep(
        self,
        rfft_incident_pressure: np.ndarray,
        f_limits=(10, 400),
        ) -> np.ndarray:
        """Filters the sweep with respect to incident pressure measured beforehand.

        Parameters
        ----------
        rfft_incident_pressure : np.ndarray
            rfft of incident pressure
        f_lim : tuple[int, int]
            frequency limits for the filtering
        
        Returns
        -------
        filtered_sweep : np.ndarray
            filtered sweep
        """
        # generate sweep without windows
        sweep_wo_win = self.make_sweep(windows=False)

        # apply blackman window to the ends of the sweep
        win = np.blackman(self.window_len//4)
        sweep_wo_win[:len(win)//2] = sweep_wo_win[:len(win)//2] * win[:len(win)//2]
        sweep_wo_win[-len(win)//2:] = sweep_wo_win[-len(win)//2:] * win[-len(win)//2:]

        # calculate rfft of the sweep
        rfft_sweep = np.fft.rfft(sweep_wo_win)
        rfft_freqs = np.fft.rfftfreq(len(sweep_wo_win), d=1/self.fs)
        
        # find indices of frequency limits
        f_low_idx = np.argmin(np.abs(rfft_freqs-f_limits[0]))
        f_high_idx = np.argmin(np.abs(rfft_freqs-f_limits[1]))

        # calculate amplitude of the filtered sweep spectrum
        amplitude = np.abs(rfft_sweep) / np.abs(rfft_incident_pressure)
        filtered_sweep_spectrum = amplitude * np.exp(1j*np.angle(rfft_sweep))

        # construct and apply filter based on blackman windows
        filt = np.blackman(50)
        filter = np.ones_like(filtered_sweep_spectrum)
        filter[:f_low_idx] = 0
        filter[f_low_idx:f_low_idx+len(filt)//2] = filt[:len(filt)//2]
        filter[-f_high_idx-len(filt)//2:-f_high_idx] = filt[len(filt)//2:]
        filter[-f_high_idx:] = 0
        filtered_sweep_spectrum *= filter

        # calculate ifft of the filtered sweep spectrum
        filtered_sweep = np.fft.irfft(filtered_sweep_spectrum)

        # apply hanning window to the ends of the filtered sweep
        window = np.hanning(self.window_len)
        filtered_sweep[:len(window)//2] = filtered_sweep[:len(window)//2] * window[:len(window)//2]
        filtered_sweep[-len(window)//2:] = filtered_sweep[-len(window)//2:] * window[len(window)//2:]

        # normalize the filtered sweep
        filtered_sweep = filtered_sweep/np.max(np.abs(filtered_sweep))

        # apply level to the filtered sweep based on measurement level
        lvl_to_factor = 10**(self.sweep_lvl/20)
        filtered_sweep = filtered_sweep*lvl_to_factor

        self.sweep = filtered_sweep
        return filtered_sweep

    def measure(self,
            out_path : str='',
            thd_filter : bool=True,
            export : bool=True
            ) -> tuple[np.ndarray, int]:
        """Performs measurement and saves the recording. 
        
        Parameters
        ----------
        out_path : str
            path where the recording should be saved, including filename
        thd_filter : bool
            enables harmonic distortion filtering
            This affects the files saved.
        export : bool
            enables export to specified path

        Returns
        -------
        data : np.ndarray
            measured audio data
        fs : int
            sampling rate
        """
        data = sd.playrec(
            self.sweep, 
            input_mapping=self.channels_in, 
            output_mapping=self.channels_out,
            dtype=np.float32)
        sd.wait()
        data = np.asarray(data)

        #filtration
        if thd_filter:
            data = data.T
            data = harmonic_distortion_filter(
                data, 
                self.sweep, 
                f_low=self.f_limits[0], 
                f_high=self.f_limits[1]
                )
            data = data.T
        
        if export:    
            sf.write(
                file=out_path,
                data=data,
                samplerate=self.fs,
                format='WAV',
                subtype='FLOAT'
                )
        
        return data, self.fs
    
    def calc_incident_pressure_filter(
        self,
        spectrum: np.ndarray,
        r: np.ndarray,
        f: np.ndarray,
        distance: float,
        speed_of_sound: float = 343,
        f_limits=(10, 400)
        ) -> np.ndarray:
        def calculate_incident_pressure(
            pressure, 
            reflection_factor, 
            distance,
            wavenumber
        ):
            return pressure / (
                np.exp(-1j * wavenumber * distance) 
                + reflection_factor * np.exp(1j * wavenumber * distance)
            )
        
        sweep_spectrum = np.fft.rfft(self.sweep.copy())
        # calculate incident pressure
        f_low_idx = np.argmin(np.abs(f-f_limits[0]))
        f_high_idx = np.argmin(np.abs(f-f_limits[1]))
        incident_pressure = calculate_incident_pressure(
            pressure=spectrum,
            reflection_factor=r[f_low_idx:f_high_idx],
            distance=distance,
            wavenumber=2*np.pi*f[f_low_idx:f_high_idx]/speed_of_sound
        )

        #extend incident pressure to have the same length as sweep_spectrum
        incident_pressure = np.concatenate([np.ones(f_low_idx)*incident_pressure[0], incident_pressure])
        incident_pressure = np.concatenate([incident_pressure, np.ones(len(sweep_spectrum)-len(incident_pressure))*incident_pressure[-1]])

        # smoothen incident pressure by applying a moving average convolution filter with a window of 10 samples
        incident_pressure = np.convolve(np.abs(incident_pressure), np.hanning(20), mode="same")*np.exp(1j*np.angle(incident_pressure))
        # normalize incident by the actual amplitude of the sweep used in the measurement
        incident_pressure = incident_pressure / np.abs(sweep_spectrum)
        return incident_pressure

class Tube:
    """Class representing tube geometry.

    Attributes
    ----------
    further_mic_dist : float
        further microphone distance from sample
    closer_mic_dist : float
        closer mic distance from sample
    freq_limit : int
        higher frequency limit for exports
    """
    def __init__(self,
            further_mic_dist : float=0.400115, #x_1
            closer_mic_dist : float=0.101755, #x_2
            freq_limit : int=2000,
            ):
        self.further_mic_dist = further_mic_dist
        self.closer_mic_dist = closer_mic_dist
        self.mic_spacing = further_mic_dist - closer_mic_dist
        self.freq_limit = freq_limit

class Sample:
    """A class representing sample and its boundary conditions.
    
    Attributes
    ----------
    name : str
        name of the sample
    temperature : float
        ambient temperature in degC
    rel_humidity : float
        ambient relative humidity in %
    tube : Tube
        impedance tube definition object
    timestamp : str
        strftime timestamp in a format '%y-%m-%d_%H-%M'
    folder : str
        path to project data folder, defaults to "data"
    """
    def __init__(self,
            name : str,
            temperature : float,
            rel_humidity : float,
            atm_pressure : float = 101325,
            tube : Tube=Tube(),
            timestamp : str = strftime("%y-%m-%d_%H-%M"),
            folder = "data",
            ):
        self.name = name
        self.timestamp = timestamp
        self.temperature = temperature
        self.atm_pressure = atm_pressure
        self.rel_humidity = rel_humidity
        self.tube = tube
        self.folder = folder
        self.trees = make_foldertree(
            self.name, 
            self.folder, 
            self.timestamp
            )
        bound_dict = {
            'temp': [self.temperature],
            'RH': [self.rel_humidity],
            'atm_pressure': [self.atm_pressure],
            'x1': [self.tube.further_mic_dist],
            'x2': [self.tube.closer_mic_dist],
            'lim': [self.tube.freq_limit]
            }
        self.boundary_df = pd.DataFrame(bound_dict)
        self.boundary_df.to_csv(
            os.path.join(
                self.trees[2],self.trees[1]+"_bound_cond.csv"
            )
        )
            
    def migrate_cal(self, cal_name, cal_stamp, cal_parent="data"):
        """Migrates calibration files from different measurement.
        
        Parameters
        ----------
        cal_name : str
            calibration sample name
        cal_stamp : str
            calibration sample timestamp i a '%y-%m-%d_%H-%M' format
        cal_parent : str
            parent data folder, defaults to 'data'
        """
        cal_trees = make_foldertree(
            variant=cal_name,
            time_stamp=cal_stamp,
            parent=cal_parent
            )
        cal_parent_folder = cal_trees[2]
        import_folder = cal_trees[3][1]
        freqs = np.load(
            os.path.join(cal_parent_folder, cal_trees[1]+"_freqs.npy")
            ) #freq import
        cf = np.load(
            os.path.join(import_folder, cal_trees[1]+"_cal_f_12.npy")
            ) #cf import

        parent_folder = self.trees[2]
        export_folder = self.trees[3][1]
        np.save(
            os.path.join(parent_folder, self.trees[1]+"_freqs.npy"),
            freqs
            ) #freq export
        np.save(
            os.path.join(export_folder, self.trees[1]+"_cal_f_12.npy"),
            cf
            ) #cf export

def calibration(
        sample : Sample,
        measurement : Measurement,
        thd_filter : bool=True,
        export : bool=True,
        noise_filter : bool=False,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Performs CLI calibration measurement.
    
    Parameters
    ----------

    sample : imptube.tube.Sample
        
    measurement : Measurement

    thd_filter : bool
        Enables harmonic distortion filtering
    """
    caltree = sample.trees[3][0]
    if not os.path.exists(caltree):
        os.makedirs(caltree)

    m = measurement
    running = True
    while running:
        for c in range(1, 3):
            ready = input(f"Calibrate in configuration {c}? [Y/n]")
            if ready.lower() == "n":
                break
            else:
                for s in range(m.sub_measurements):
                    f = os.path.join(caltree, sample.trees[1]+f"_cal_wav_conf{c}_{s}.wav")
                    print(f)
                    m.measure(f, thd_filter=thd_filter)
                    sleep(0.5)
        if input("Repeat calibration process? [y/N]").lower() == "y":
            continue
        else:
            running = False
        input("Move the microphones to original position before measurement!")
    
    cal = calibration_from_files(parent_folder=sample.trees[2], export=export, noise_filter=noise_filter)

    return cal

def single_measurement(
        sample : Sample,
        measurement : Measurement,
        depth : float,
        thd_filter : bool= True,
        calc_spl : bool = True
        ) -> tuple[list[np.ndarray], int]:
    """Performs measurement.
    
    Parameters
    ----------

    sample : imptube.tube.Sample
        
    measurement : Measurement

    depth : float
        current depth of the sample
    thd_filter : bool
        Enables harmonic distortion filtering

    Returns
    -------
    sub_measurement_data : list[np.ndarray]
        list of audio recordings taken
    fs : float
        sampling rate of the recording
    """
    m = measurement
    sub_measurement_data = []
    for s in range(m.sub_measurements):
        f = os.path.join(sample.trees[4][0], sample.trees[1]+f"_wav_d{depth}_{s}.wav")
        data, fs = m.measure(f, thd_filter=thd_filter)
        sub_measurement_data.append(data)
        sleep(0.5)

    if calc_spl:
        rms_spl = calc_rms_pressure_level(data.T[0], m.fs_to_spl)
        logging.info(f"RMS SPL: {rms_spl} dB")
        m.rms_spl = rms_spl
    return sub_measurement_data, fs

def calculate_alpha(
        sample : Sample,
        return_r : bool = False,
        return_z : bool = False,
        noise_filter : bool = False
        ) -> tuple[np.ndarray, np.ndarray]:
    """Performs transfer function and alpha calculations from audio data
    found in a valid folder structure.

    Parameters
    ----------
    sample : Sample

    Returns
    -------
    alpha : np.ndarray
        sound absorption coefficient for frequencies lower than 
        limit specified in sample.tube.freq_limit
    freqs : np.ndarray
        frequency values for the alpha array
    """
    sample.unique_d, sample.tfs = transfer_function_from_path(sample.trees[2], noise_filter=noise_filter)
    results = alpha_from_path(
        sample.trees[2],
        return_f=True,
        return_r=return_r,
        return_z=return_z
        )
    return results

class Sensor(Protocol):
    """A protocol for Sensor class implementation."""
    def read_temperature(self) -> float:
        ...
    
    def read_humidity(self) -> float:
        ...

    def read_pressure(self) -> float:
        ...
    
def read_env_bc(sensor : Sensor) -> tuple[float, float, float]:
    for i in range(5):
        try:
            temperature = sensor.read_temperature()
            rel_humidity = sensor.read_humidity()
            atm_pressure = sensor.read_pressure()
            break
        except:
            print(f"Reading {i+1} not succesful.")
        if i == 4:
            print("Unable to read data from sensor, try manually enter temperature and RH on initialization.")
            sys.exit()
    return temperature, rel_humidity, atm_pressure

def calculate_spectrum(
    sample: Sample,
    substring: str,
    f_limits=(10, 400)
):
    audio_files = os.listdir(sample.trees[4][0])
    filtered_files = [f for f in audio_files if substring in f]
    audio_data = []
    for f in filtered_files:
        fs, data = wavfile.read(f"{sample.trees[4][0]}/{f}")
        audio_data.append(data.T[0])
    audio_data = np.array(audio_data)
    audio_data = np.mean(audio_data, axis=0)
    
    audio_spectrum = np.fft.rfft(audio_data)
    audio_freqs = np.fft.rfftfreq(len(audio_data), d=1/fs)
    flow_idx = np.argmin(np.abs(audio_freqs-f_limits[0]))
    fhigh_idx = np.argmin(np.abs(audio_freqs-f_limits[1]))
    audio_spectrum = audio_spectrum[flow_idx:fhigh_idx]
    audio_freqs = audio_freqs[flow_idx:fhigh_idx]
    return audio_spectrum, audio_freqs
    #  TODO save bc as config file...
    #  bound_dict = {
    #     'temp': [self.temperature],
    #     'RH': [self.RH],
    #     'x1': [self.x_1],
    #     'x2': [self.x_2],
    #     'lim': [self.limit],
    #     }
    # self.boundary_df = pd.DataFrame(bound_dict)
    # self.trees = make_foldertree(self.name, self.folder)
    # self.boundary_df.to_csv(
    #     os.path.join(
    #         self.trees[2],self.trees[1]+"_bound_cond.csv"
    #     )
    # )
    