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
    tf_i_r
)
from ..utils import filter, read_folder


def read_file(path):
    return wav.read(path)

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