#%%
#imports
import numpy as np

import math as m
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#%%
#INPUT DATA FILE NAME HERE
name = "70khzcsv"
fmt = ".csv"

#%%
def read_data(data_input_name, delimiter_input, comment_marker_input):
    """
    Reads 2 column data file (csv or txt), with a given delimiter and
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
        if np.isnan(line[0]) or np.isnan(line[1]):
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

def sort_data(data_output):
    """
    Combines 2 numpy arrays, adding the bottom_data to the bottom of the first.
    Then sorts the data according to the first column. This will terminate
    if the arrays do not have the same width.
    Parameters
    ----------
    top_data : numpy_array
    bottom_data : numpy_array
    Returns
    -------
    sorted_array : numpy_array
    or if unsucessful:
        1 : float
    """
    sorted_array = data_output[data_output[:, 0].argsort()]
    return sorted_array


def fun(x, I, y, x_0):
    '''
    ASSUMED FUNTIONAL FORM THAT THE CODE WILL ATTEMPT TO ADJUST PARAMETERS FOR
    '''
    num = I*(y**2)
    den = (x-x_0)**2 + y**2
    
    return  num/den

#%% Main READ DATA
raw_data = read_data(name + fmt, ',', '%')
#print(data)
data = sort_data(raw_data)
# print(data )
xvals_raw = data[:,0]
xvals = xvals_raw * 10**9
# xvals = xvals[10:38]
yvals = data[:,1]
# yvals = yvals[10:38]

#%% Main CURVE FIT
guess_mast = [6, 0.040, 1]

opt, acc = curve_fit(fun, xvals, yvals, p0=guess_mast, maxfev = 1000000)


print('Parameter Values [I, y, x_0] = ', opt)
print("==================================================")

#%% Main ANALYSE
plt.scatter(xvals, yvals, label='RMS Voltage', s=7)

last_val = xvals[np.argmax(xvals)]
#print(last_val)
xrange = np.linspace(0,last_val, 777)


maxvalinx = np.argmax(yvals)
resonant_peak = yvals[maxvalinx]
print("Resonant Peak =", resonant_peak)
half_resonant_peak = resonant_peak / 2
resontant_peak_pos = xvals[maxvalinx]

plt.scatter(xvals[maxvalinx], yvals[maxvalinx], label='Resonant Peak', marker='x', s=50)

fittedvals = fun(xrange, opt[0], opt[1], opt[2])
resonant_peak_fitted_inx = np.argmax(fittedvals)
# print(resonant_peak_fitted_inx)
resonant_peak_fitted = fittedvals[resonant_peak_fitted_inx]
print("Resonant Peak Fitted =", resonant_peak_fitted)
fittedvalsU = fittedvals[resonant_peak_fitted_inx:]
fittedvalsL = fittedvals[:resonant_peak_fitted_inx]

# print(fittedvals)
print("size of fitted array:", np.shape(fittedvals))
# print(fittedvalsL)
print("size of fitted array lower half:",np.shape(fittedvalsL))
# print(fittedvalsU)
print("size of fitted array upper half:",np.shape(fittedvalsU))

dC_U_index = np.argmin(abs(fittedvalsU-half_resonant_peak))
dC_U = fittedvalsU[dC_U_index]
print("Upper Half Maximum: ", dC_U)
dC_L_index = np.argmin(abs(fittedvalsL-half_resonant_peak))
dC_L = fittedvalsL[dC_L_index]
print("Lower Half Maximum: ", dC_U)
plt.scatter(xvals[maxvalinx], yvals[maxvalinx], label='Resonant Peak', marker='x', s=50)
plt.scatter(xvals[maxvalinx], yvals[maxvalinx], label='Resonant Peak', marker='x', s=50)

print("==================================================")

#%% Main PLOT
try:
    # plt.xlim(0.75, 1.25)
    plt.plot(xrange, fittedvals,  
             label='Fitted function')
except Exception:
    print("couldn't plot curve")
    pass

plt.legend(loc='best')
plt.savefig('curvefit.png', dpi=777)
plt.show()