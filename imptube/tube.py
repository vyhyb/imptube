'''In this module, three main classes are defined - Measurement, Tube and Sample.
'''

import sys
import sounddevice as sd
import numpy as np
import pandas as pd
import soundfile as sf
import os
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
            fs_to_spl : float=130  
        ):
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.device = device
        self.fs = fs
        self.sub_measurements = sub_measurements
        self.samples = samples
        self.f_limits = [f_low, f_high]
        self.fs_to_spl = fs_to_spl

        self.sweep = self.make_sweep(
            samples=samples,
            window_len=window_len,
            f_low=f_low,
            f_high=f_high
        )
        sd.default.samplerate = fs
        sd.default.channels = len(channels_in), len(channels_out)
        sd.default.device = device


    def make_sweep(self,
            fs : int=48000,
            samples : int=65536,
            window_len : int=8192,
            f_low : int=10,
            f_high : int=1000
            ) -> np.ndarray:
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
        t = np.linspace(0,samples/fs,samples, dtype=np.float32)
        
        half_win = int(window_len)
        log_sweep = chirp(t, f_low, t[-1], f_high, method="log", phi=90)
        window = hann(int(window_len*2))

        log_sweep[:half_win] = log_sweep[:half_win]*window[:half_win]
        log_sweep[-half_win:] = log_sweep[-half_win:]*window[half_win:]
        log_sweep = log_sweep/4

        return log_sweep

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
            output_mapping=self.channels_out)
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
            tube : Tube=Tube(),
            timestamp : str = strftime("%y-%m-%d_%H-%M"),
            folder = "data",
            ):
        self.name = name
        self.timestamp = timestamp
        self.temperature = temperature
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
            'x1': [self.tube.further_mic_dist],
            'x2': [self.tube.closer_mic_dist],
            'lim': [self.tube.freq_limit],
            'dbfs_to_spl': [130],
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
        thd_filter = True
        ):
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

    return calibration_from_files(parent_folder=sample.trees[2])

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
        logging.INFO(f"RMS SPL: {rms_spl} dB")
        m.rms_spl = rms_spl
    return sub_measurement_data, fs

def calculate_alpha(
        sample : Sample
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
    sample.unique_d, sample.tfs = transfer_function_from_path(sample.trees[2])
    sample.alpha, sample.freqs = alpha_from_path(sample.trees[2], return_f=True)
    return sample.alpha, sample.freqs

class Sensor(Protocol):
    """A protocol for Sensor class implementation."""
    def read_temperature(self) -> float:
        ...
    
    def read_humidity(self) -> float:
        ...
    
def read_env_bc(sensor : Sensor) -> tuple[float, float]:
    for i in range(5):
        try:
            temperature = sensor.read_temperature()
            rel_humidity = sensor.read_humidity()
            break
        except:
            print(f"Reading {i+1} not succesful.")
        if i == 4:
            print("Unable to read data from sensor, try manually enter temperature and RH on initialization.")
            sys.exit()
    return temperature, rel_humidity

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
    