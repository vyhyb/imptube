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
    