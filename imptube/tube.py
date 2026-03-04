'''In this module, three main classes are defined - Measurement, Tube and Sample.
'''

import sys
import sounddevice as sd
import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import hann
from time import sleep, strftime
from imptube.processing import (
    harmonic_distortion_filter,
    calc_rms_pressure_level,
    stereo_to_spectra,
    calibration_factor,
    noise_filtering,
    transfer_function,
    frequencies,
    tf_i_r,
    reflection_factor,
    absorption_coefficient,
    surface_impedance
)
from typing import Protocol
import logging


class Measurement:
    """Contains information about a measurement, sweep settings and
    boundary conditions.

    Attributes
    ----------
    fs : int
        measurement sampling frequency.
    channels_in : list[int]
        list of input channel numbers.
    channels_out : list[int]
        list of output channel numbers (usually a single‑element list).
    device : str
        substring used to select the sound‑card device for playback/record.
        A full list of devices can be obtained with
        ``python3 -m sounddevice``.
    samples : int
        number of samples in the generated log sweep (typically a power of
        two).
    window_len : int
        length of the Hann half‑window applied to the start and end of the
        sweep.
    sub_measurements : int
        number of recordings taken for each specimen; individual sweeps are
        averaged to reduce playback/recording artefacts.
    f_limits : list[int]
        two‑element list containing the lower and upper frequency limits for
        sweep generation.
    fs_to_spl : float
        conversion level from dBFS to dB SPL for microphone 1.
    sweep_lvl : float
        level of the sweep in dBFS.
    sweep : np.ndarray
        the most recently generated excitation sweep waveform (set by
        ``make_sweep``/``regen_sweep``/``filter_sweep``).
    data : np.ndarray | None
        last measured audio data; populated by ``measure`` and modified by
        ``filter_harmonic_distortion``.
    rms_spl : float | None
        root‑mean‑square sound‑pressure level calculated during the last
        measurement (set in ``single_measurement``).
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

        self.make_sweep()
        sd.default.samplerate = fs
        sd.default.channels = len(channels_in), len(channels_out)
        sd.default.device = device


    def make_sweep(self, windows=True) -> np.ndarray:
        """Generate and store a log‑frequency sweep using current settings.

        The sweep is constructed from the attributes of the
        :class:`Measurement` instance: ``fs``, ``samples``,
        ``f_limits`` (``f_low``/``f_high``), ``window_len`` and
        ``sweep_lvl``.  A logarithmic chirp running from the low to the
        high limit with duration ``samples/fs`` is created.  By default the
        first and last ``window_len`` samples are tapered with a Hann
        half‑window; this behaviour can be disabled
        with the ``windows`` argument.  The generated signal is then
        scaled to the level specified by ``sweep_lvl`` (dBFS) and stored in
        ``self.sweep``.

        Parameters
        ----------
        windows : bool, optional
            Apply Hann half‑windows to the start and end of the sweep.
            Defaults to ``True``.  If ``False`` the sweep is returned
            without any windowing.

        Returns
        -------
        np.ndarray
            The generated mono log sweep.  The same array is assigned to
            ``self.sweep``.
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
        """Regenerates the sweep with the current settings."""
        self.make_sweep()

    def update_sweep_lvl(self):
        """Updates the sweep level according to ``self.sweep_lvl``."""
        self.sweep = self.sweep/np.max(np.abs(self.sweep))
        self.sweep = self.sweep * 10**(self.sweep_lvl/20)

    def filter_sweep(
        self,
        rfft_incident_pressure: np.ndarray,
        f_limits=(10, 400),
        ) -> np.ndarray:
        """Filters the sweep with respect to incident pressure measured beforehand.

        This method applies frequency-domain filtering to the sweep signal by:
        1. Generating a windowed sweep
        2. Computing its FFT
        3. Calculating the amplitude response relative to incident pressure
        4. Applying blackman window-based filters at frequency limits
        5. Converting back to time domain with hann windowing
        6. Normalizing and scaling to the sweep level

        Parameters
        ----------
        rfft_incident_pressure : np.ndarray
            rfft of incident pressure spectrum
        f_limits : tuple[int, int], optional
            frequency limits for the filtering. Defaults to (10, 400).
        
        Returns
        -------
        filtered_sweep : np.ndarray
            filtered sweep in time domain, scaled to sweep_lvl
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
            ) -> tuple[np.ndarray, int]:
        """Performs measurement using playrec with current sweep settings.
        
        Returns
        -------
        data : np.ndarray
            measured audio data from all input channels
        fs : int
            sampling rate
        """
        data = sd.playrec(
            self.sweep, 
            input_mapping=self.channels_in, 
            output_mapping=self.channels_out,
            dtype=np.float32)
        sd.wait()
        self.data = np.asarray(data)
        
        return self.data, self.fs
    
    def filter_harmonic_distortion(
        self,
        ) -> np.ndarray:
        """Filters the harmonic distortion products from the measured data.
        A wrapper for the `harmonic_distortion_filter` function 
        from `imptube.processing` module. 
        
        It filters the last measured data stored in `self.data`,
        so it should be called right after `measure` method.

        Returns
        -------
        filtered_data : np.ndarray
            filtered data
        fs : int
            sampling rate
        """

        self.data = harmonic_distortion_filter(
            p_time=self.data,
            p_ref=self.sweep,
            f_low=self.f_limits[0],
            f_high=self.f_limits[1],
            fs=self.fs
            )
        return self.data, self.fs

    def calc_incident_pressure_filter(
        self,
        spectrum: np.ndarray,
        r: np.ndarray,
        f: np.ndarray,
        distance: float,
        f_limits: tuple[int, int],
        speed_of_sound: float = 343,
        ) -> np.ndarray:
        """Calculates incident pressure filter based on the measured 
        spectrum and reflection factor.
        
        Such filter can be used to filter the input sweep
        to compensate for the loudspeaker frequency response.

        Parameters
        ----------
        spectrum : np.ndarray
            measured pressure spectrum
        r : np.ndarray
            reflection factor
        f : np.ndarray
            frequency values
        distance : float
            distance between the sample and the microphone
        f_limits : tuple[int, int]
            frequency limits for the filtering
        speed_of_sound : float, optional
            speed of sound in air. Defaults to 343 m/s.

        Returns
        -------
        incident_pressure : np.ndarray
            incident pressure filter (complex spectrum)
        """
        def calculate_incident_pressure(
            pressure: np.ndarray, 
            reflection_factor: np.ndarray, 
            distance: float,
            wavenumber: np.ndarray
        ):
            """Calculate incident pressure from measured spectrum and reflection factor.

            Parameters
            ----------
            pressure : np.ndarray
                measured pressure spectrum
            reflection_factor : np.ndarray
                reflection factor
            distance : float
                distance between the sample and the microphone
            wavenumber : np.ndarray
                wavenumber values

            Returns
            -------
            np.ndarray
                incident pressure spectrum
            """
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
        incident_pressure = np.concatenate([
            np.ones(f_low_idx)*incident_pressure[0], 
            incident_pressure
            ])
        incident_pressure = np.concatenate([
            incident_pressure, 
            np.ones(len(sweep_spectrum)-len(incident_pressure))*incident_pressure[-1]
            ])

        # smoothen incident pressure by applying a moving average convolution filter with a window of 20 samples
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
        closer microphone distance from sample
    mic_spacing : float
        distance between the two microphones
    freq_limit : int
        higher frequency limit for exports
    """
    def __init__(self,
            further_mic_dist : float, #x_1
            closer_mic_dist : float, #x_2
            freq_limit : int,
            ):
        self.further_mic_dist = further_mic_dist
        self.closer_mic_dist = closer_mic_dist
        self.mic_spacing = further_mic_dist - closer_mic_dist
        self.freq_limit = freq_limit

class Sample:
    """A class representing a sample and its boundary conditions as well as
    the data from the measurement and calibration.
    
    Attributes
    ----------
    name : str
        name of the sample
    tube : Tube
        impedance tube definition object
    measurement : Measurement
        measurement settings and sweep configuration
    temperature : float
        ambient temperature in °C
    rel_humidity : float
        ambient relative humidity in %
    atm_pressure : float
        atmospheric pressure in Pa. Defaults to 101325.
    timestamp : str
        strftime timestamp in format '%y-%m-%d_%H-%M'
    freqs : np.ndarray | None
        frequency values from measurement
    cf : np.ndarray | None
        calibration factor spectrum
    tf : np.ndarray | None
        transfer function spectrum
    tf_corrected : np.ndarray | None
        calibration-corrected transfer function spectrum
    """
    def __init__(self,
            name : str,
            tube : Tube,
            measurement : Measurement,
            temperature : float,
            rel_humidity : float,
            atm_pressure : float = 101325,
            timestamp : str = strftime("%y-%m-%d_%H-%M"),
            ):
        self.name = name
        self.timestamp = timestamp
        self.temperature = temperature
        self.atm_pressure = atm_pressure
        self.rel_humidity = rel_humidity
        self.tube = tube
        self.measurement = measurement
        self.freqs = None
        self.cf = None
        self.tf = None


    def calibration(
            self,
            thd_filter : bool=True,
            noise_filter : bool=False,
            ) -> tuple[np.ndarray, np.ndarray]:
        """Performs CLI calibration measurement with two microphone configurations.
        
        Parameters
        ----------
        thd_filter : bool, optional
            Enables harmonic distortion filtering. Defaults to True.
        noise_filter : bool, optional
            Enables noise filtering on calibration factor. Defaults to False.
        
        Returns
        -------
        cf : np.ndarray
            Calibration factor spectrum stored in self.cf
        """
        cal_data = [[], []]
        m = self.measurement
        running = True
        while running:
            for c in range(1, 3):
                ready = input(f"Calibrate in configuration {c}? [Y/n]")
                if ready.lower() == "n":
                    break
                else:
                    cal_data[c-1] = []
                    for _ in range(m.sub_measurements):
                        data, _ = m.measure()
                        cal_data[c-1].append(data)
                        sleep(0.5)
            if input("Repeat calibration process? [y/N]").lower() == "y":
                continue
            else:
                running = False
            input("Move the microphones to original position before measurement!")
        
        cal_data = [np.mean(cal_data[0], axis=0), np.mean(cal_data[1], axis=0)]

        if thd_filter:
            cal_data[0] = harmonic_distortion_filter(
                p_time=cal_data[0],
                p_ref=m.sweep,
                f_low=m.f_limits[0],
                f_high=m.f_limits[1],
                fs=m.fs
            )
            cal_data[1] = harmonic_distortion_filter(
                p_time=cal_data[1],
                p_ref=m.sweep,
                f_low=m.f_limits[0],
                f_high=m.f_limits[1],
                fs=m.fs
            )

        p11, p12 = stereo_to_spectra(cal_data[0])
        p21, p22 = stereo_to_spectra(cal_data[1])
        self.cf = calibration_factor(p11, p12, p21, p22)
        
        if noise_filter:
            self.cf = noise_filtering(self.cf)
        
        return self.cf

    def single_measurement(
            self,
            thd_filter : bool= True,
            noise_filter : bool = False,
            calc_spl : bool = True
            ) -> tuple[list[np.ndarray], int]:
        """Performs measurement with optional filtering and SPL calculation.
        
        Parameters
        ----------
        thd_filter : bool, optional
            Enables harmonic distortion filtering. Defaults to True.
        noise_filter : bool, optional
            Enables noise filtering on transfer function. Defaults to False.
        calc_spl : bool, optional
            Enables SPL calculation and logging. Defaults to True.

        Returns
        -------
        tf_corrected : np.ndarray
            Calibration-corrected transfer function spectrum
        fs : int
            Sampling rate of the recording
        """
        m = self.measurement
        def _measure():
            sub_measurement_data = []
            for _ in range(m.sub_measurements):
                data, fs = m.measure()
                sub_measurement_data.append(data)
                sleep(0.5)
            
            avg_data = np.mean(sub_measurement_data, axis=0)
            if thd_filter:
                sub_measurement_data = harmonic_distortion_filter(
                    p_time=avg_data,
                    p_ref=m.sweep,
                    f_low=m.f_limits[0],
                    f_high=m.f_limits[1],
                    fs=m.fs
                )


            self.freqs = frequencies(avg_data, m.fs)
            p1, p2 = stereo_to_spectra(avg_data)

            return p1, p2
        
        input("Ready to perform measurement? [Enter]")

        p11, p12 = _measure()
        self.tf = transfer_function(p11, p12)

        if self.cf is not None:
            self.tf_corrected = self.tf / self.cf
        else:
            input("Calibration needed. Switch the microphones and press Enter.")
            p21, p22 = _measure()
            self.cf = calibration_factor(p11, p12, p21, p22)
            self.tf_corrected = self.tf / self.cf
            input("Calibration complete. \nSwitch the microphones back to original configuration and press Enter to proceed.")

        if noise_filter:
            self.tf_corrected = noise_filtering(self.tf_corrected)

        if calc_spl:
                rms_spl = calc_rms_pressure_level(m.data.T[0], m.fs_to_spl)
                logging.info(f"RMS SPL: {rms_spl} dB")
        m.rms_spl = rms_spl
        return self.tf_corrected, m.fs

    def calculate_alpha(
            self,
            return_r : bool = False,
            return_z : bool = False,
            ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates sound absorption coefficient and surface impedance from measured transfer function.

        Parameters
        ----------
        return_r : bool, optional
            If True, also returns the reflection factor. Defaults to False.
        return_z : bool, optional
            If True, also returns the surface impedance. Defaults to False.

        Returns
        -------
        alpha : np.ndarray
            Sound absorption coefficient
        freqs : np.ndarray
            Frequency values corresponding to alpha
        r : np.ndarray, optional
            Reflection factor (returned only if return_r=True)
        z : np.ndarray, optional
            Surface impedance (returned only if return_z=True)
        """

        if self.tf is None:
            raise ValueError("No transfer function found for this sample." \
                " Perform measurement first.")

        tf_incident, tf_reflected = tf_i_r(self.temperature, self.freqs, self.tube.mic_spacing)
        tf_incident = tf_incident
        tf_reflected = tf_reflected

        rf = reflection_factor(tf_incident, tf_reflected, self.tf_corrected, self.temperature, self.freqs, self.tube.further_mic_dist)
        an = absorption_coefficient(rf)
        zs = surface_impedance(rf, self.temperature, self.atm_pressure)

        ret = [an, self.freqs]
        if return_r:
            ret.append(rf)
        if return_z:
            ret.append(zs)

        return 

class Sensor(Protocol):
    """A protocol for Sensor class implementation."""
    def read_temperature(self) -> float:
        ...
    
    def read_humidity(self) -> float:
        ...

    def read_pressure(self) -> float:
        ...
    
def read_env_bc(sensor : Sensor) -> tuple[float, float, float]:
    """Read environmental data from sensor with retry logic.
    
    Attempts to read temperature, humidity, and atmospheric pressure from the 
    sensor up to 5 times. Exits the program if all attempts fail.

    Parameters
    ----------
    sensor : Sensor
        An instance of a class that implements the Sensor protocol, providing methods to read temperature, humidity, and pressure.

    Returns
    -------
    temperature : float
        The temperature reading from the sensor.
    rel_humidity : float
        The relative humidity reading from the sensor.
    atm_pressure : float
        The atmospheric pressure reading from the sensor.
    
    Raises
    ------
    SystemExit
        If unable to read sensor data after 5 attempts.


    """
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

# def calculate_spectrum(
#     sample: Sample,
#     substring: str,
#     f_limits=(10, 400)
# ):
#     audio_files = os.listdir(sample.trees[4][0])
#     filtered_files = [f for f in audio_files if substring in f]
#     audio_data = []
#     for f in filtered_files:
#         fs, data = wavfile.read(f"{sample.trees[4][0]}/{f}")
#         audio_data.append(data.T[0])
#     audio_data = np.array(audio_data)
#     audio_data = np.mean(audio_data, axis=0)
    
#     audio_spectrum = np.fft.rfft(audio_data)
#     audio_freqs = np.fft.rfftfreq(len(audio_data), d=1/fs)
#     flow_idx = np.argmin(np.abs(audio_freqs-f_limits[0]))
#     fhigh_idx = np.argmin(np.abs(audio_freqs-f_limits[1]))
#     audio_spectrum = audio_spectrum[flow_idx:fhigh_idx]
#     audio_freqs = audio_freqs[flow_idx:fhigh_idx]
#     return audio_spectrum, audio_freqs
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
    