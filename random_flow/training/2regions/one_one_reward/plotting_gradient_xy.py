#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:33:51 2021

@author: naivd
"""

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.titlesize'] = 25 
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 22 
mpl.rcParams['xtick.labelsize'] = 22 
mpl.rcParams['ytick.labelsize'] = 22 
mpl.rcParams['legend.fontsize'] = 18


#####################################################################
#Functions#

def smooth(v, window):
    N = v.shape[0]
    smoothed = []
    for i in range(N):
        if i < window:
            smoothed.append(np.mean(v[0:i]))
        else:
            smoothed.append(np.mean(v[i-window:i]))
    return np.array(smoothed)
    



####################################################################

# Parameters
n_states = 25
n_actions = 5
beta = 0
window = 100
n_train = 50
T = 5500

eps_0_episode = 2500


###################################################################
# Plotting




top_trains = []

for b in range(11):
    d = h5py.File(f'beta{b}/1/Data.h5' , 'r')
    reward_sum = np.array(d.get('reward_sum'))
    maximum = np.mean(reward_sum[eps_0_episode:])        
    top_train = 0
    for train in range(n_train):
        d = h5py.File(f'beta{b}/{train+1}/Data.h5' , 'r')
        reward_sum = np.array(d.get('reward_sum'))
        mean_r = np.mean(reward_sum[eps_0_episode:])
        #print(f'training {i} - normalized : {np.mean(mean_r)}')
        d.close()
        if mean_r > maximum:
            maximum = mean_r
            top_train = train
        
    top_trains.append(top_train)



plt.subplots(3,1 , figsize=(15,20), sharex=True)
plt.suptitle(rf'$\beta $= {beta/10}', fontsize=22)
avg_dy = np.zeros(3000)
avg_trSS = np.zeros(3000)
avg_reward_sum = np.zeros(3000)

for train in range(n_train):
    d = h5py.File(f'beta{beta}/{train+1}/Data.h5' , 'r')
    reward_sum = np.array(d.get('reward_sum'))
    dy = np.array(d.get('vertical_migration'))
    trSS = np.array(d.get('trSS_sum'))
    d.close()
    plt.subplot(3,1,1)
    v = smooth(reward_sum, window)
    avg_reward_sum += v
    if train == top_trains[beta]:
        plt.plot(v, color='green')
    else:
        plt.plot(v, color='blue' , linewidth=0.1)
    plt.ylabel(r'$\frac{1}{T}\sum_t^T R(t)$')
    plt.subplot(3,1,2)
    v = smooth(dy , window)
    avg_dy += v
    if train == top_trains[beta]:
        plt.plot(v, color='green')
    else:
        plt.plot(v, color='blue' , linewidth=0.1)
    plt.ylabel(r'$\frac{1}{v_s T}\sum_t^T \Delta y(t)$')
    plt.subplot(3,1,3)
    v = smooth(trSS , window)
    avg_trSS += v
    if train == top_trains[beta]:
        plt.plot(v, color='green')
    else:
        plt.plot(v, color='blue' , linewidth=0.1)    
    plt.ylabel(r'$\frac{1}{Tr S^2_{max}T}\sum_t^T Tr S^2(t)$')


plt.subplot(3,1,1)
plt.plot(avg_reward_sum/n_train , color = 'red')
plt.plot()
plt.subplot(3,1,2)
plt.plot(avg_dy/n_train , color = 'red')
plt.subplot(3,1,3)
plt.plot(avg_trSS/n_train , color = 'red')
plt.xlabel('Episode')

plt.savefig(f'training_plot_beta_{beta/10}.png')





policies = np.zeros((11,n_states))       
    
total_R = []
dy = []
trSS_R = []
episode = 3000-1#input('Enter the episode: ')
f = open('policies.txt' ,'w+')
for beta in range(11):
    d = h5py.File(f'beta{beta}/{top_trains[beta]+1}/Data.h5' , 'r')
    
    reward_sum = np.array(d.get('reward_sum'))
    mean_r = np.mean(reward_sum[eps_0_episode:])
    total_R.append(mean_r)
    
    Q = np.array(d.get(f'Qs/{episode}'))
    policy = np.zeros((n_states))
    for j in range(n_states):
        policy[j] = np.argmax(Q[j])
    print(f'tot_R : beta {beta} , trian {top_train} : Policy {policy} , {mean_r}')
    f.write(f'tot_R : beta {beta} , trian {top_train} : Policy {policy} , {mean_r}\n')
    policies[beta] = policy
    
    
    reward_sum = np.array(d.get('vertical_migration'))
    mean_r = np.mean(reward_sum[eps_0_episode:])
    dy.append(mean_r)
    
    
    reward_sum = np.array(d.get('trSS_sum'))
    mean_r = np.mean(reward_sum[eps_0_episode:])
    trSS_R.append(mean_r)
    
    d.close()
    
    print('####################')
    f.write('####################\n')
    
    

plt.subplots(3,1,figsize=(15,10) , sharex = True)
plt.subplot(3,1,1) 
plt.scatter(np.linspace(0,1,11) , total_R)
plt.ylabel(r'$\frac{1}{T}\sum_t^T R(t)$')
plt.subplot(3,1,2) 
plt.scatter(np.linspace(0,1,11) , dy)   
plt.ylabel(r'$\frac{1}{v_s T}\sum_t^T \Delta y(t)$')
plt.subplot(3,1,3) 
plt.scatter(np.linspace(0,1,11) , trSS_R)   
plt.ylabel(r'$\frac{1}{Tr S^2_{max}T}\sum_t^T Tr S^2(t)$')
plt.xlabel(r'$\beta$')
plt.savefig('beta_dependency.png')





for i in range(11):
    f.write(f'beta {i/10} : ')
    f.write('{')
    for j in range(n_states):
        f.write(f'{int(policies[i,j])}')
        if j < n_states-1:
            f.write(',')
    f.write('};\n')






f.write('########################### To use in Bash ###########################\n')
f.write("(")
for i in range(11):
    f.write('"{')
    for j in range(n_states):
        f.write(f'{int(policies[i,j])}')
        if j < n_states-1:
            f.write(',')
    f.write('}" ')
f.write(")")


f.close()


















