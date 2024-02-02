# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:54:30 2023

@author: vadan
"""
# %%
# imports
import numpy as np

import math as m
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# %%
# INPUT DATA FILE NAME HERE
name = "Extrapolated Values, No Fixed Cell (csv)"
fmt = ".csv"
full_name = name + fmt

# %%


def read_data(data_input_name, delimiter_input, comment_marker_input):
    """
    Reads 3 column data file (csv or txt), with a given delimiter and
    comment marker. Converts to numpy array, with 3 columns.
    Then sorts the data according to the first column. This will remove lines
    of data that are not numbers, indicated those that have been eliminated.
    Parameters
    ----------
    data_input_name : string
    delimiter_input : string
    comment_marker_input : string
    Returns
    -------
    numpy_array[floats]
    """
    print("\nReading File: ", data_input_name)
    print("==================================================")
    try:
        data_intake = np.genfromtxt(
            data_input_name, delimiter=delimiter_input,
            comments=comment_marker_input)
    except ValueError:
        return 1
    except NameError:
        return 1
    except TypeError:
        return 1
    except IOError:
        return 1
    index = 0
    eliminated_lines = 0
    initial_length = len(data_intake[:, 0])
    for line in data_intake:
        if np.isnan(line[0]) or np.isnan(line[1]) or np.isnan(line[2]):
            print('Deleted line {0}: '.format(index + 1 + eliminated_lines))
            print(line)
            data_intake = np.delete(data_intake, index, 0)
            index -= 1
            eliminated_lines += 1
        index += 1
    if eliminated_lines == 0:
        print("[no lines eliminated]")
    print("==================================================")
    print('Initial Array Length: {0}'.format(initial_length))
    print('Final Array Length: {0}'.format(len(data_intake[:, 0])))
    print('Data Read with {0} lines removed'.format(eliminated_lines))
    print("==================================================")
    return data_intake


# %% Main
data_raw = read_data(full_name, ',', '%')
# print(data_raw)
freqs = data_raw[:, 0]
peaks_raw = data_raw[:, 1]
fwhms_raw = data_raw[:, 2]

# %%
angfreqs = np.pi * 2 * freqs
print(angfreqs)
peaks = peaks_raw * 10**-9
print(peaks)
fwhms = fwhms_raw * 10**-9
print(fwhms)

Ls = 1/((angfreqs)**2 * (peaks))
print(Ls)
