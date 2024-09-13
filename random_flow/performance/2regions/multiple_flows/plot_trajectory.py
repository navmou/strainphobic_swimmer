#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:27:49 2021

@author: navid
"""



import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.titlesize'] = 25 
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 22 
mpl.rcParams['xtick.labelsize'] = 22 
mpl.rcParams['ytick.labelsize'] = 22 
mpl.rcParams['legend.fontsize'] = 22

#Loading data
strategy = 'grady'
beta = 0
L0 = 0.01

d = h5py.File(f'{strategy}/{beta}/beta{beta}.h5' , 'r')

plt.figure(figsize=(12,12))
for realization in range(0,10000,500):
    traj = np.array(d.get(f'Episodes/traj{realization}'))%(L0)
    plt.scatter(traj[:,0] , traj[:,1] , color = 'red' , alpha = 0.1 , s = 0.001)

plt.xlim(0,L0)
plt.ylim(0,L0)
plt.xlabel('X')
plt.ylabel('Y')


plt.figure(figsize=(12,12))
traj = np.array(d.get('Episodes/traj0'))%(L0)
plt.scatter(traj[:,0] , traj[:,1] , color = 'red' , alpha = 0.2 , s = 0.1)

plt.xlim(0,L0)
plt.ylim(0,L0)
plt.xlabel('X')
plt.ylabel('Y')



plt.figure(figsize=(12,12))
for realization in range(0,10000,500):
    traj = np.array(d.get(f'Episodes/traj{realization}'))
    plt.plot(traj[:,0] , traj[:,1])

plt.xlabel('X')
plt.ylabel('Y')



d.close()