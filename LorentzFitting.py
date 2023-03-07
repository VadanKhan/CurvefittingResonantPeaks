#%%
#imports
import numpy as np

import math as m
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#%%
#INPUT DATA FILE NAME HERE
name = "60khzparaffincsv"
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

def width(fraction_input, uncertainty_input):
    '''
    REQUIRED TO BE AFTER "Main fit funciton and split array"

    Parameters
    ----------
    fraction_input : TYPE
        DESCRIPTION.
    uncertainty_input : TYPE
        DESCRIPTION.

    Returns
    -------
    x_difference : TYPE
        DESCRIPTION.
    propagated_halfpoint_unc : TYPE
        DESCRIPTION.

    '''
    print("FINDING WIDTH FOR FRACTION:", fraction_input)
    x_resonant_peak = fraction_input * resonant_peak
    dC_a_index = np.argmin(abs(fittedvalsL-x_resonant_peak))
    dC_a = fittedvalsL[dC_L_index]
    # print("Sub Lower Half Maximum index: ", dC_L_index)
    # print("Lower Half Maximum: ", dC_L)
    dC_b_index = np.argmin(abs(fittedvalsU-x_resonant_peak))
    dC_b = fittedvalsU[dC_U_index]
    # print("Sub Upper Half Maximum index: ", dC_U_index)
    # print("Upper Half Maximum: ", dC_U)
    
    dC_a_indext = dC_a_index
    dC_b_indext = dC_b_index + resonant_peak_fitted_inx
    print("C_a index: ", dC_a_indext)
    print("C_b index: ", dC_b_indext)
    dC_at = fittedvals[dC_a_indext]
    dC_bt = fittedvals[dC_b_indext]
    print("V(C_a): ", dC_at)
    print("V(C_b): ", dC_bt)
    Capacitance_a = xrange[dC_a_indext]
    Capacitance_b = xrange[dC_b_indext]
    print("Capacitance_a: ", Capacitance_a)
    print("Capacitance_b: ", Capacitance_b)

    x_difference = Capacitance_b - Capacitance_a


    #%%PLOT Ca Cb
    plt.scatter(Capacitance_a, dC_at, marker='x',
                s=50)
    plt.scatter(Capacitance_b, dC_bt, marker='x',
                s=50)
    # plt.axvline(Capacitance_a + uncertainty_input, c='grey', alpha = 0.5,
    #             linestyle='--')
    # plt.axvline(Capacitance_a - uncertainty_input, c='grey', alpha = 0.5,
    #             linestyle='--')
    # plt.axvline(Capacitance_b + uncertainty_input, c='grey', alpha = 0.5,
    #             linestyle='--')
    # plt.axvline(Capacitance_b - uncertainty_input, c='grey', alpha = 0.5,
    #             linestyle='--')
    plt.axhline(x_resonant_peak, c='pink', alpha = 0.5,
                linestyle='--')
    propagated_halfpoint_unc = np.sqrt(2*(uncertainty_input)**2)
    print("C_b-C_a: ", x_difference)
    print("C_b-C_a uncertainty: ", propagated_halfpoint_unc)
    print("==================================================")
    return x_difference, propagated_halfpoint_unc

#%% Main READ DATA
raw_data = read_data(name + fmt, ',', '%')
#print(data)
data = sort_data(raw_data)
# print(data )
xvals_raw = data[:,0]
xvals = xvals_raw
# xvals = xvals[10:38]
yvals = data[:,1]
# yvals = yvals[10:38]

#%% Main CURVE FIT
guess_mast = [6, 0.040E-9, 1E-9]

opt, acc = curve_fit(fun, xvals, yvals, p0=guess_mast, maxfev = 1000000)


print('Parameter Values [I, y, x_0] = ', opt)
print("==================================================")

#%% Main ANALYSE PEAKS
plt.scatter(xvals, yvals, label='RMS Voltage', c='grey', s=2)

last_val = xvals[np.argmax(xvals)]
#print(last_val)
xrange = np.linspace(0,last_val, 10000)


maxvalinx = np.argmax(yvals)
# maxvalinx = maxvalinx +1
resonant_peak = yvals[maxvalinx]
print("RESONANT PEAK =", resonant_peak)
resonant_peak_pos = xvals[maxvalinx]
print("RESONANT PEAK CAPCITANCE =", resonant_peak_pos)
print("==================================================")



plt.scatter(xvals[maxvalinx], yvals[maxvalinx], label='Measured Resonant Peak',
            marker='x', s=50)

#%% Main FIT FUNCTION AND SPLIT ARRAY
fittedvals = fun(xrange, opt[0], opt[1], opt[2])
resonant_peak_fitted_inx = np.argmax(fittedvals)
print("Resonant Peak Fitted index =",resonant_peak_fitted_inx)
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

#%% Main FIND HALF POINTS
half_resonant_peak = resonant_peak / 2
dC_L_index = np.argmin(abs(fittedvalsL-half_resonant_peak))
dC_L = fittedvalsL[dC_L_index]
# print("Sub Lower Half Maximum index: ", dC_L_index)
# print("Lower Half Maximum: ", dC_L)
dC_U_index = np.argmin(abs(fittedvalsU-half_resonant_peak))
dC_U = fittedvalsU[dC_U_index]
# print("Sub Upper Half Maximum index: ", dC_U_index)
# print("Upper Half Maximum: ", dC_U)

dC_L_indext = dC_L_index
dC_U_indext = dC_U_index + resonant_peak_fitted_inx
print("Lower Half Maximum index: ", dC_L_indext)
print("Upper Half Maximum index: ", dC_U_indext)
dC_Lt = fittedvals[dC_L_indext]
dC_Ut = fittedvals[dC_U_indext]
print("Lower Half Maximum: ", dC_Lt)
print("Upper Half Maximum: ", dC_Ut)
Lower_Half_Capacitance = xrange[dC_L_indext]
Upper_Half_Capacitance = xrange[dC_U_indext]
print("Lower Half C: ", Lower_Half_Capacitance)
print("Upper Half C: ", Upper_Half_Capacitance)

print("==================================================")
fwhm_capacitance = Upper_Half_Capacitance - Lower_Half_Capacitance
print("RESONANT PEAK CAPCITANCE =", resonant_peak_pos)
print("FULL WIDTH HALF MAXIMUM: ", fwhm_capacitance)
print("==================================================")

#%% Main Plot FWHM
plt.scatter(Lower_Half_Capacitance, dC_Lt, label='Lower Half Peak', marker='x',
            s=50)
plt.scatter(Upper_Half_Capacitance, dC_Ut, label='Upper Half Peak', marker='x',
            s=50)
plt.axvline(Lower_Half_Capacitance + 0.003E-9, c='grey', alpha = 0.5,
            linestyle='--')
plt.axvline(Lower_Half_Capacitance - 0.003E-9, c='grey', alpha = 0.5,
            linestyle='--')
plt.axvline(Upper_Half_Capacitance + 0.003E-9, c='grey', alpha = 0.5,
            linestyle='--')
plt.axvline(Upper_Half_Capacitance - 0.003E-9, c='grey', alpha = 0.5,
            linestyle='--')
fhwm_propagated_halfpoint_unc = np.sqrt(2*(0.003E-9)**2)

#%% Main FWHM  Uncertainties
resonantnextspace = xvals[maxvalinx+1]-xvals[maxvalinx]
resonantbackspace = xvals[maxvalinx]-xvals[maxvalinx-1]
averagespacing = (resonantnextspace + resonantbackspace)/2
print("Resonant Peak Uncertainty: ", averagespacing)
print("FWHM Uncertainty: ", fhwm_propagated_halfpoint_unc)
plt.axvline(resonant_peak_pos + averagespacing, c='grey', alpha = 0.5,
            linestyle='--')
plt.axvline(resonant_peak_pos - averagespacing, c='grey', alpha = 0.5,
            linestyle='--')
print("==================================================")

#%% Main Find Ca and Cb general
diff19, diff19unc = width(19/20, 0.001E-9)
diff18, diff18unc = width(18/20, 0.001E-9)
diff17, diff17unc = width(17/20, 0.002E-9)
diff16, diff16unc = width(16/20, 0.002E-9)
diff15, diff15unc = width(15/20, 0.003E-9)
diff14, diff14unc = width(14/20, 0.003E-9)
diff13, diff13unc = width(13/20, 0.003E-9)
diff12, diff12unc = width(12/20, 0.003E-9)
diff11, diff11unc = width(11/20, 0.003E-9)

#%% Main PLOT
try:
    plt.xlim(1.16E-9, 1.27E-9)
    plt.plot(xrange, fittedvals,  
             label='Fitted function', c='grey')
except Exception:
    print("couldn't plot curve")
    pass

plt.legend(loc = 'lower right', borderaxespad=0, fontsize='5')
plt.savefig('curvefit.png', dpi=1000)
plt.show()

#%% Main Aggregate C differences and fractions
fractions = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
v1v2 = 1/fractions
xvals = np.sqrt(v1v2**2 - 1)
differences = np.array([fwhm_capacitance, diff11, diff12, diff13, diff14, 
                        diff15, diff16, diff17, diff18, diff19])
differences_unc = np.array([fhwm_propagated_halfpoint_unc, diff11unc, diff12unc,
                            diff13unc, diff14unc, diff15unc, diff16unc, 
                            diff17unc, diff18unc, diff19unc])
powerlossarray = np.column_stack((xvals, differences, differences_unc))
print(powerlossarray)
np.savetxt('powerlossarray' + name + '.csv', powerlossarray, delimiter = ',')
