"""
This module contains functions for reading and processing audio files.
"""
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
import os
from scipy.fft import fftfreq

from .signal_proc import (
    read_audio,
    stereo_to_spectra,
    transfer_function, 
    calibration_factor, 
    reflection_factor, 
    absorption_coefficient,
    surface_impedance,
    tf_i_r
)
from ..utils import filter, read_folder


def read_file(path : str) -> tuple[int, np.ndarray]:
    """
    Read a WAV file from the given path.

    Parameters
    ----------
    path : str
        The path to the WAV file.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - The audio data as a 1-D numpy array.
        - The sample rate of the audio.

    """
    return wav.read(path)

import numpy as np
import pandas as pd

def load_bc(parent_folder : str) -> pd.DataFrame:
    """
    Load the bound condition data from a given parent folder.

    Parameters:
    -----------
    parent_folder : str
        The path to the parent folder containing the bound condition file.

    Returns:
    --------
    pd.DataFrame
        The loaded bound condition data as a pandas DataFrame.
    """
    file_paths = read_folder(parent_folder)
    bc_path = filter(
            file_paths,
            ["bound_cond.csv"]
    )
    bc_df = pd.read_csv(bc_path[0])
    return bc_df

def load_freqs(parent_folder : str) -> np.ndarray:
    """
    Load frequency data from the specified parent folder.

    Parameters
    ----------
    parent_folder : str
        The path to the parent folder containing the frequency data.

    Returns
    -------
    numpy.ndarray
        The loaded frequency data as a NumPy array.
    """
    file_paths = read_folder(parent_folder)
    freqs_path = filter(
            file_paths,
            ["freqs.npy"]
    )
    freqs = np.load(freqs_path[0])
    return freqs

import numpy as np

def load_alpha(parent_folder : str) -> tuple[list[np.ndarray], list[str]]:
    """Load sound absorption coefficient data from the specified parent folder.

    Parameters
    ----------
    parent_folder : str
        The path to the parent folder containing the sound absorption coefficient data files.

    Returns
    -------
    tuple
        A tuple containing two lists:
        - alpha: A list of numpy arrays, each representing an sound absorption coefficient data file.
        - unique_d: A list of unique 'd' values extracted from the file paths.
    """
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

def transfer_function_from_file(
        path : str=None, 
        folder : str=None, 
        filter_str : str="_"
        ) -> np.ndarray:
    """Calculate the transfer function from a file or a folder of files.

    Parameters
    ----------
    path : str, optional
        Path to the file, by default None.
    folder : str, optional
        Path to the folder containing files, by default None.
    filter_str : str, optional
        Filter string to select specific files in the folder, by default "_".

    Returns
    -------
    ndarray
        The transfer function calculated from the input file(s).

    Raises
    ------
    ValueError
        If both `path` and `folder` are None.

    Notes
    -----
    - If `folder` is None, the function will calculate the transfer function from a single file specified by `path`.
    - If `folder` is provided, the function will calculate the average transfer function from all files in the folder that match the `filter_str`.

    """
    if folder is None:
        if path is None:
            raise ValueError("You need to define path to file (path), or to folder (folder)")
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

def calibration_from_files(
        path1: str = None,
        path2: str = None, 
        parent_folder: str = None, 
        export: bool = True
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform calibration from audio files.

    Parameters
    ----------
    path1 : str, optional
        Path to the first audio file, by default None.
    path2 : str, optional
        Path to the second audio file, by default None.
    parent_folder : str, optional
        Path to the parent folder containing calibration and audio folders, by default None.
    export : bool, optional
        Whether to export calibration data, by default True.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the calibration factor and frequency data.
    """
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

def transfer_function_from_path(
        parent_folder: str
        ) -> tuple[np.ndarray, np.ndarray]:
    """Calculate transfer functions from audio files in the given parent folder.

    Parameters
    ----------
    parent_folder : str
        The path to the parent folder containing the audio files.

    Returns
    -------
    tuple
        A tuple containing the unique values of 'd' extracted from the audio file names,
        and a list of transfer functions corresponding to each unique 'd' value.
    """
    bc_df = load_bc(parent_folder)
    limit = int(bc_df["lim"].iloc[0])
    audio_folder = os.path.join(parent_folder, "measurement", "audio")
    tf_folder = os.path.join(parent_folder, "measurement", "transfer_func")
    audio_files = read_folder(audio_folder)
    fs, a = read_file(audio_files[0])
    limit_idx = int(limit/fftfreq(a.shape[0], 1/fs)[1])
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

def alpha_from_path(
        parent_folder : str, 
        fs : int=48000, 
        cal : np.ndarray=None, 
        tfs : np.ndarray=None, 
        bc : pd.DataFrame=None, 
        return_f : bool=False,
        return_r : bool=False,
        return_z : bool=False
        ) -> tuple[np.ndarray]:
    """Calculate the absorption coefficient from the given parent folder path.

    Parameters
    ----------
    parent_folder : str
        The path of the parent folder.
    fs : int, optional
        The sampling frequency, by default 48000.
    cal : np.ndarray, optional
        The calibration transfer function, by default None.
    tfs : np.ndarray, optional
        The transfer functions, by default None.
    bc : pd.DataFrame, optional
        The boundary conditions, by default None.
    return_f : bool, optional
        Whether to return the absorption coefficient and frequencies, by default False.
    return_r : bool, optional
        Whether to return the reflection factor, by default False.
    return_z : bool, optional
        Whether to return the impedance, by default False.

    Returns
    -------
    tuple[np.ndarray] or np.ndarray
        If return_f is True, returns a tuple containing the absorption coefficient and frequencies.
        Otherwise, returns only the absorption coefficient.
    """
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
    try:
        atm_pressure = bc_df["atm_pressure"].iloc[0]
    except:
        Exception("atm_pressure not provided with boundary condition file.")
    if return_z and atm_pressure is None:
        raise Exception("You need to provide valid atmospheric pressure to calculate surface impedance.")
    print(f"atm pressure: {atm_pressure}")
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
    limit_idx = int(limit/freqs[1])
    tf_cal = tf_cal[:limit_idx]
    freqs = freqs[:limit_idx]

    # read all transfer functions from folder
    tf_paths = read_folder(
        os.path.join(parent_folder, "measurement", "transfer_func")
    )
    tf_paths = sorted(tf_paths)
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
    impedances = []
    for t in tfs:
        tf_corrected = t/tf_cal
        rfs.append(reflection_factor(tf_incident, tf_reflected, tf_corrected, temp_c, freqs, x1_dist))
    alpha_n = []
    for r in rfs:
        alpha_n.append(absorption_coefficient(r))
    
    if return_z:
        for r in rfs:
            impedances.append(surface_impedance(r, temp_c, atm_pressure))
    # save all alpha courses
    for a, p in zip(alpha_n, tf_paths):
        # print(p)
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

    # save all reflection factors
    for r, p in zip(rfs, tf_paths):
        # print(p)
        idx = str(p).rfind("d")
        np.save(arr=r,
            file=os.path.join(
                parent_folder, "measurement", "refl_factor", os.path.split(parent_folder)[1]+"_refl_factor_"+str(p)[idx:-4]+".npy"
            )
        )
    np.save(arr=freqs, file=os.path.join(
                parent_folder, "measurement", "refl_factor", os.path.split(parent_folder)[1]+"_freqs.npy"
        )
    )

    # save all impedances
    for i, p in zip(impedances, tf_paths):
        # print(p)
        idx = str(p).rfind("d")
        np.save(arr=i,
            file=os.path.join(
                parent_folder, "measurement", "impedance", os.path.split(parent_folder)[1]+"_impedance_"+str(p)[idx:-4]+".npy"
            )
        )
    np.save(arr=freqs, file=os.path.join(
                parent_folder, "measurement", "impedance", os.path.split(parent_folder)[1]+"_freqs.npy"
        )
    )

    
    ret = [alpha_n]
    if return_f:
        ret = [alpha_n, freqs]
    if return_r:
        ret.append(rfs)
    if return_z:
        ret.append(impedances)
        
    return tuple(ret)