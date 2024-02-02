# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:30:55 2023

@author: m23510vk
"""

#%% IMPORT STATEMENTS
import numpy as np

#%% GLOBAL CONSTANTS
L = 4.40E-2
Cs = 0.18E-9
Ct = np.array([1.426672032e-09, 9.866994236e-10, 7.233963885e-10])
Cfwhm = np.array([8.105927835051544e-11, 5.5583505154639274e-11,  4.4003608247422664e-11])
Cfwhm_unc = np.array([4.2426406871192855e-12,  4.2426406871192855e-12, 4.2426406871192855e-12])
frac_unc = Cfwhm_unc / Cfwhm
print("Fractional Uncertainties: ", frac_unc)


Rvals = np.empty(0)
for i in range(0, 3):
    R = (Cfwhm[i]/2) * np.sqrt(L/(Ct[i]+Cs)**3) 
    Rvals = np.append(Rvals, R)
print("R Values: ", Rvals)
Rvals_unc = frac_unc * Rvals
print("R uncertainties: ", Rvals_unc)
weighting = 1/Rvals_unc**2
uncertainty_avg = 1/np.sqrt(np.sum(weighting))

avg = np.average(Rvals, weights=Cfwhm_unc)
print(r'Average Resistance is {0:1.3g} ohms +- {1:1.3g}'.format(avg, uncertainty_avg))