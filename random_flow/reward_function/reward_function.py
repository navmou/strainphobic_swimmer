#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 11:24:26 2022

@author: navmou
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.titlesize'] = 25 
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 22 
mpl.rcParams['xtick.labelsize'] = 22 
mpl.rcParams['ytick.labelsize'] = 22 
mpl.rcParams['legend.fontsize'] = 22





a , b = np.meshgrid(np.linspace(0,1,50) , np.linspace(-1, 1,51))

betas = np.linspace(0,1,11)

for beta in betas:
    plt.figure(figsize=(15,10))
    R = -beta*(a+b)+b
    plt.contourf(R)
    plt.xlabel('Strain penalty')
    plt.ylabel('Vertical reward')
    plt.title(fr'$\beta=${beta:.1f}' , fontsize=22)
    plt.xticks([0,25,49],['0','-0.5','-1'])
    plt.yticks([0,25,50],['-1','0','1'])
    plt.colorbar()
    plt.savefig(f'beta_{beta:.1f}.png')

