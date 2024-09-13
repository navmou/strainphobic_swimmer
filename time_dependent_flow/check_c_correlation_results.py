#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:37:25 2022

@author: n
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

n_fields = 2
n_time_steps = 500
corr_l = 500
d = h5py.File('st_v3_u_A_No1.h5' , 'r')
avg_corr=np.zeros(corr_l)
for i in range(n_time_steps):
    avg_corr+= np.array(d.get(f'corr{0}'))
avg_corr = avg_corr/n_time_steps
eta = 0.1
X = np.linspace(0, corr_l*(0.0025), corr_l)
plt.figure(figsize=(15,10))
plt.semilogx(X,(np.exp(-X**2/(2*eta**2))/(eta**2))*((eta**2 - X**2)+X**2) , '--', label = r'The. $u_xu_x$')
plt.semilogx(X,(np.exp(-X**2/(2*eta**2))/(eta**2))*((eta**2 - X**2)) , '--', label = r'The. $u_yu_y$')

plt.semilogx(X, avg_corr, label = 'sim.')

plt.legend()



plt.figure()
A = np.array(d.get('As'))

for i in range(n_fields):
    plt.plot(A[i])

plt.figure()

u = np.array(d.get('U'))
v = np.array(d.get('V'))
plt.streamplot(np.linspace(0,25,10000), np.linspace(0,1,400), u, v, density=6)
plt.xlim(0,1)
plt.ylim(0,1)


d.close()



d = h5py.File('st_v3_Lx25_xi0.1_cG3_u01/st_v3_u_A_No1.hdf5', 'r')
u = np.array(d.get('flowData/uxs'))
