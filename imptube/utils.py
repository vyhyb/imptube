import os
import re
from os.path import isfile, join
from time import strftime
import pandas as pd


def filter(string: str, substr: str) -> list:
    """Filter a list of strings based on the presence of a substring.

    Parameters
    ----------
    string : str
        The list of strings to be filtered.
    substr : str
        The substring to search for in each string.

    Returns
    -------
    list
        A list of strings that contain the specified substring.
    """
    return [st for st in string if any(sub in st for sub in substr)]

def check_dir(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Parameters
    ----------
    path : str
        The path of the directory to be checked/created.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_folder(path_to_dir: str) -> list:
    """
    Return a list of file paths in the given folder.

    Parameters
    ----------
    path_to_dir : str
        The path to the directory.

    Returns
    -------
    list
        A list of file paths in the given folder.
    """
    file_list = [f for f in os.listdir(path_to_dir) if isfile(join(path_to_dir, f))]
    file_list.sort()
    path_list = []
    for f in file_list:
        path_list.append(os.path.join(path_to_dir, f))
    return path_list

def make_foldertree(
        variant : str, 
        parent : str="data", 
        time_stamp : str=None
        ) -> tuple:
    """
    makes the whole tree of folders necessary for measurement

    Parameters
    ----------
    variant : str
        The name of the variant.
    parent : str, optional
        The parent folder. Defaults to "data".
    time_stamp : str, optional
        The time stamp. Defaults to None.

    Returns
    -------
    tuple
        A tuple containing the time stamp, name, parent tree, calibration trees, and measurement trees.
    """
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


def foldertree_from_parent(parent: str) -> tuple:
    """Create a folder tree from a parent directory.

    This function takes a parent directory path and creates a folder tree structure
    based on the given parent directory. The folder tree structure is created by
    extracting the time stamp and variant from the parent directory path and passing
    them to the `make_foldertree` function.

    Parameters
    ----------
    parent : str
        The parent directory path.

    Returns
    -------
    tuple
        A tuple containing the folder tree structure.

    """
    time_stamp = parent[0:14]
    variant = parent[15:]
    return make_foldertree(variant, parent='data', time_stamp=time_stamp)

def check_dir(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Parameters
    ----------
    path : str
        The path of the directory to be checked/created.
    """
    if not os.path.exists(path):
        os.makedirs(path)

import pandas as pd

def deal_with_outliers(x: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame.

    Parameters
    ----------
    x : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with outliers removed.
    """
    for c in list(x):
        x[c] = x[c][x[c].between(x[c].min(), x[c].min() * 2)]
    return x

def merge_t60_csv(path_list : list[str]) -> pd.DataFrame:
    """
    Merge multiple T60 CSV files into a single DataFrame.

    Parameters
    ----------
    path_list : list[str]
        List of file paths to the T60 CSV files.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing data from all T60 CSV files.

    Raises
    ------
    FileNotFoundError
        If any of the specified file paths do not exist.

    Notes
    -----
    - The T60 CSV files should have the same structure.
    - The merged DataFrame will have an additional 'index' column containing the file index.
    - The merged DataFrame will be saved as a new CSV file with the name '<first_file_name>_t60_merged.csv'.
    """
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

def read_folder(path_to_dir : str) -> list[str]:
    '''
    Return a list of file paths in the given folder.

    Parameters:
    path_to_dir (str): The path to the directory.

    Returns:
    list: A list of file paths in the directory.
    '''
    file_list = [f for f in os.listdir(path_to_dir) if isfile(join(path_to_dir, f))]
    file_list.sort()
    path_list = []
    for f in file_list:
        path_list.append(os.path.join(path_to_dir, f))
    return path_list