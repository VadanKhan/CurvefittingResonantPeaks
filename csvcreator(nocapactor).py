# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:01:56 2023

@author: m23510vk
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

'Python file format converter'
#%%
import numpy as np
#%%
#IMPORT FILE NAME HERE
# name = "16-12_39_54, VK second testsavg"
#IMPORT FILE INPUT FORMAT HERE
# fmt = ".txt"
#IMPUT FILE OUTPUT FORMAT HERE
outfmt = ".csv"

#%%
frequencies = np.array([60000, 70000, 80000])
xvals = 1/(2*np.pi*frequencies)**2
# print(xvals)
yvals = np.array([1.426672032e-09, 9.866994236e-10, 7.233963885e-10])
unc = np.array([ 1.9087095000000173e-12, 1.182765050000072e-12, 1.0702468999999867e-12])
raw = np.column_stack((xvals, yvals, unc))
print(raw)
np.savetxt('InductanceCSV_NoCell' + outfmt, raw, delimiter = ',')