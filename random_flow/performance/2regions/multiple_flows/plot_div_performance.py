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

signals = ['gradx' , 'grady' , 'gradxy' , 'onlygrad']

strategies = ['passive' , 'naive' , 'random' , 'upward' , r'$\beta = 0.0$' , r'$\beta = 0.1$' , r'$\beta = 0.2$' ,  r'$\beta=0.3$' , 
              r'$\beta = 0.4$' , r'$\beta=0.5$' , r'$\beta=0.6$' , r'$\beta=0.7$' , r'$\beta=0.8$' , r'$\beta=0.9$' , r'$\beta=1.0$']

n_realization = 10000

ys = np.zeros((4,15))
trs = np.zeros((4,15))
tot_times = np.zeros((4,15))

signal_counter = 0
for signal in signals:
    
    paths = [ 'passive/passive' , 'naive/naive' , f'random/{signal}/random_{signal}' , 'upward/upward' ,  f'{signal}/0/beta0' ,  f'{signal}/1/beta1',  f'{signal}/2/beta2',  
         f'{signal}/3/beta3' ,  f'{signal}/4/beta4',  f'{signal}/5/beta5',  f'{signal}/6/beta6',  
         f'{signal}/7/beta7',  f'{signal}/8/beta8' ,  f'{signal}/9/beta9' ,  f'{signal}/10/beta10']
    
    t1 = []
    t2 = []
    y = []
    tr = []
    
    for path in paths:
        d = h5py.File(f'{path}.h5' , 'r')
        times = np.array(d.get('times'))
        for i in range(n_realization):
            times[i] /= np.sum(times[i])
        t1.append(np.mean(times[:,0]))
        t2.append(np.mean(times[:,1]))
        y.append(np.mean(np.array(d.get('vertical_migration'))))
        tr.append(np.mean(np.array(d.get('trSS_sum'))))
        d.close()
        
    t1 = np.array(t1)
    t2 = np.array(t2)
    tr = np.array(tr)
    y = np.array(y)
    tot_times[signal_counter] = t1
    ########################################################
    #plotting times compariosn
    plt.figure(figsize=(15,13))
    plt.plot(np.arange(1,16) , np.array(t1) , '^--' , label = r'$S_0$' , color = 'green' , MarkerSize=15)
    plt.plot(np.arange(1,16) , np.array(t2) , 'o--' , label = r'$S_1$', MarkerSize = 15, color = 'red')
    plt.plot(np.arange(1,16) , np.array(y) , '*--' , label = r'$\langle \Delta y \rangle/v_s T$' , MarkerSize = 15, color = 'blue')
    plt.plot(np.arange(1,16) , np.array(tr) ,'h--' , label = r'$\overline{tr S^2 }$' , MarkerSize = 15, color = 'purple')
    plt.xticks(np.arange(1,16) , strategies, rotation = 90)
    plt.ylabel('')
    plt.legend()
    plt.grid()
    plt.savefig(f'times_{signal}.png')
    
    ########################################################
    #Plotting normalized times (normalized by advected swimmer time)
    n_t1 = t1/t1[0]
    n_t2 = t2/t2[0]
    n_tr = tr/tr[0]
    n_y = y/y[3]
    ys[signal_counter] = n_y 
    trs[signal_counter] = n_tr
    plt.figure(figsize=(15,13))
    plt.plot(np.arange(1,16) , np.array(n_t1) , '^--' , label = r'$S_0$' , MarkerSize = 15 , color = 'green')
    plt.plot(np.arange(1,16) , np.array(n_t2) , 'o--' , label = r'$S_1$', MarkerSize = 15, color = 'red')
    #plt.plot(np.arange(1,16) , np.array(n_tr) , 'h--' , label = r'$\overline{tr S^2 }$' , MarkerSize = 15, color = 'purple')
    #plt.plot(np.arange(1,16) , np.array(n_y) , '*--' , label =  r'$\langle \Delta y \rangle/v_s T$'  , MarkerSize = 15, color = 'blue')
    plt.xticks(np.arange(1,16) , strategies , rotation = 90)
    plt.ylabel(r'$T/T_{passive}$')
    plt.legend()
    plt.grid()
    plt.savefig(f'normalized_times_{signal}.png')
    
    signal_counter+=1
    
    
    
plt.figure(figsize=(15,13))
plt.plot(np.arange(1,16) , ys[0] , '^--' , label = r'$(\nabla \cdot {S})_x , Tr(S^2)$' , MarkerSize = 15 , color = 'green')
plt.plot(np.arange(1,16) , ys[1] , 'o--' , label = r'$(\nabla \cdot {S})_y , Tr(S^2)$' , MarkerSize = 15, color = 'red')
plt.plot(np.arange(1,16) , ys[2] , 'h--' , label = r'$(\nabla \cdot {S}) , Tr(S^2)$'  , MarkerSize = 15, color = 'purple')
plt.plot(np.arange(1,16) , ys[3] , '*--' , label = r'$(\nabla \cdot S)$'   , MarkerSize = 15, color = 'blue')
plt.xticks(np.arange(1,16) , strategies , rotation = 90)
plt.ylabel(r'$\langle \Delta y \rangle/\langle \Delta y_{up} \rangle$'  )
plt.legend()
plt.grid()
plt.savefig('normalized_migration.png')
   

plt.figure(figsize=(15,13))
plt.plot(np.arange(1,16) , trs[0] , '^--' , label = r'$(\nabla \cdot {S})_x , Tr(S^2)$' , MarkerSize = 15 , color = 'green')
plt.plot(np.arange(1,16) , trs[1] , 'o--' , label = r'$(\nabla \cdot {S})_y , Tr(S^2)$' , MarkerSize = 15, color = 'red')
plt.plot(np.arange(1,16) , trs[2] , 'h--' , label = r'$(\nabla \cdot {S}) , Tr(S^2)$'  , MarkerSize = 15, color = 'purple')
plt.plot(np.arange(1,16) , trs[3] , '*--' , label = r'$(\nabla \cdot S)$'  , MarkerSize = 15, color = 'blue')
plt.xticks(np.arange(1,16) , strategies , rotation = 90)
plt.ylabel(r'$\frac{\overline{Tr(S^2) }}{\overline{Tr(S^2)_{passive}}}$')
plt.legend()
plt.grid()
plt.savefig('normalized_mean_TrSS.png')
   


plt.figure(figsize=(15,13))
plt.plot(np.arange(1,16) , tot_times[0] , '^--' , label = r'$(\nabla \cdot {S})_x , Tr(S^2)$' , MarkerSize = 15 , color = 'green')
plt.plot(np.arange(1,16) , tot_times[1] , 'o--' , label = r'$(\nabla \cdot {S})_y , Tr(S^2)$' , MarkerSize = 15, color = 'red')
plt.plot(np.arange(1,16) , tot_times[2] , 'h--' , label = r'$(\nabla \cdot {S}) , Tr(S^2)$'  , MarkerSize = 15, color = 'purple')
plt.plot(np.arange(1,16) , tot_times[3] , '*--' , label = r'$(\nabla \cdot S)$'  , MarkerSize = 15, color = 'blue')
plt.xticks(np.arange(1,16) , strategies , rotation = 90)
plt.ylabel(r'Percentage of time in favorable region')
plt.legend()
plt.grid()
plt.savefig('percentage_time.png')


# =============================================================================
# 
# ########################################################
# #Plotting the trajcetories (density plots)
# L0 = 0.01
# 
# flow = 11
# d = h5py.File(f"../../../../../../../../FLOW/new-flow/st_data_v3_No{flow}.hdf5", 'r')
# x , y = np.meshgrid(np.linspace(0,0.01 , 1024) , np.linspace(0,0.01 , 1024))
# A11s = np.array(d.get('flowData/A11s'))
# A12s = np.array(d.get('flowData/A12s'))
# A21s = np.array(d.get('flowData/A21s'))
# A22s = np.array(d.get('flowData/A22s'))
# A11 = A11s.reshape(1024,1024)
# A12 = A12s.reshape(1024,1024)
# A21 = A21s.reshape(1024,1024)
# A22 = A22s.reshape(1024,1024)
# t = []
# for i in range(1024):
#     for j in range(1024):
#         t.append(A11[i,j]*A11[i,j] + 
#              0.5*(A12[i,j]+A21[i,j])*(A12[i,j]+A21[i,j]) + 
#              A22[i,j]*A22[i,j]) 
# t = np.array(t)
# print(np.mean(t))
# print(np.max(t))
# print(np.min(t))
# t2d = np.reshape(t , (1024,1024))
# 
# for path in range(len(paths)):
#     d = h5py.File(f'{paths[path]}.h5' , 'r')
#     plt.figure(figsize=(15,12))
#     plt.contourf(x,y , t2d,levels=[0,1.7,20] , colors=['green' , 'yellow'])
#     plt.colorbar()
#     for realization in range(0,500,50):
#         traj = np.array(d.get(f'Episodes/traj{realization}'))%L0
#         plt.scatter(traj[:,0] , traj[:,1] , color = 'red' , alpha = 0.2 , s = 0.1)
#     plt.title(f'{strategies[path]}',fontsize=25)    
#     plt.xlim(0,0.01)
#     plt.ylim(0,0.01)
# 
#     plt.savefig(f'density_plot_{names[path]}.png')
# 
# =============================================================================
