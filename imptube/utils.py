import os
import re
from os.path import isfile, join
from time import strftime
import pandas as pd


def filter(string, substr): 
    return [st for st in string if
        any(sub in st for sub in substr)]

def check_dir(path):
    if not os.path.exists(path):
            os.makedirs(path)

def read_folder(path_to_dir):
    '''
    return path of files in a given folder
    '''
    file_list = [f for f in os.listdir(path_to_dir) if isfile(join(path_to_dir, f))]
    file_list.sort()
    path_list = []
    for f in file_list:
        path_list.append(os.path.join(path_to_dir, f))
    return path_list

def make_foldertree(variant, parent="data", time_stamp=None):
    '''
    makes the whole tree of folders necessary for measurement
    returns time_stamp, name (time+variant name)
    '''
    if time_stamp is None:
        time_stamp = strftime("%y-%m-%d_%H-%M")

    name = (
        time_stamp + '_' + variant
    )
    #parent folder
    parenttree = os.path.join(parent, name)

    # kalibrace
    calibtree = os.path.join(parenttree, "calibration")
    calibtrees = [
        os.path.join(calibtree, "audio"),
        os.path.join(calibtree, "cal_factor")
    ]

    # mereni
    measuretree = os.path.join(parenttree, "measurement")
    measuretrees = [
        os.path.join(measuretree, "audio"),
        os.path.join(measuretree, "transfer_func"),
        # os.path.join(measuretree, "impedance"),
        # os.path.join(measuretree, "wave_number"),
        os.path.join(measuretree, "alpha"),
        os.path.join(measuretree, "fig"),
        os.path.join(measuretree, "fig", "impedance"),
        os.path.join(measuretree, "fig", "wave_number"),
        os.path.join(measuretree, "fig", "alpha")
    ]
    trees = calibtrees + measuretrees
    for t in trees:
        check_dir(t)

    return time_stamp, name, parenttree, calibtrees, measuretrees


def foldertree_from_parent(parent):
    time_stamp = parent[0:14]
    variant = parent[15:]
    return make_foldertree(variant, parent='data', time_stamp=time_stamp)

def check_dir(path):
    if not os.path.exists(path):
            os.makedirs(path)

def deal_with_outliers(x):
    for c in list(x):
        x[c] = x[c][x[c].between(x[c].min(),x[c].min()*2)]
    return x

def merge_t60_csv(path_list):
    t60 = []
    idx = []
    pi = -len(path_list[0])+path_list[0].find(re.findall('s.m.', path_list[0])[0])
    if any(path_list[0][:pi-1]+'_t60_merged.csv' in p for p in path_list):
        path_list.remove(path_list[0][:pi-1]+'_t60_merged.csv')

    for p in path_list:
        t60.append(pd.read_csv(p))
        idx.append(p[pi:pi+4])
    df_t60 = pd.concat(t60).reset_index(drop=True).drop('Unnamed: 0', axis=1)

    df_t60.insert(0, 'index', pd.Series(idx))
    
    df_t60.to_csv(path_list[0][:pi-1]+'_t60_merged.csv')
    return df_t60

def read_folder(path_to_dir):
    '''
    return path of files in a given folder
    '''
    file_list = [f for f in os.listdir(path_to_dir) if isfile(join(path_to_dir, f))]
    file_list.sort()
    path_list = []
    for f in file_list:
        path_list.append(os.path.join(path_to_dir, f))
    return path_list