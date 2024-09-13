import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


mpl.rcParams['figure.titlesize'] = 25 
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 22 
mpl.rcParams['xtick.labelsize'] = 22 
mpl.rcParams['ytick.labelsize'] = 22 
mpl.rcParams['legend.fontsize'] = 18



n_states = 10
n_actions = 5
episode = 2999
n_trains = 50
top_policies = np.zeros((11,n_states))


for beta in range(11):
    
    d = h5py.File(f'beta{beta}/1/Data.h5' , 'r')
    reward_sum = np.array(d.get('reward_sum'))
    maximum = np.mean(reward_sum[2500:])
    train = 0
    policies = np.zeros((n_states,n_actions))
    for i in range(n_trains):
        d = h5py.File(f'beta{beta}/{i+1}/Data.h5' , 'r')
        reward_sum = np.array(d.get('reward_sum'))
        mean_r = np.mean(reward_sum[2500:])
        #print(f'training {i} - normalized : {np.mean(mean_r)}')
        if mean_r > maximum:
            maximum = mean_r
            train = i
        
        Q = np.array(d.get(f'Qs/{episode}'))
        policy = np.zeros((n_states))
        for j in range(n_states):
            policy[j] = np.argmax(Q[j])
        #print(f'trian {i} : Policy {policy} , {np.mean(mean_r)}')
        for j in range(n_states):
            policies[j,int(policy[j])] += 1
        
        d.close()
        
    d = h5py.File(f'beta{beta}/{train+1}/Data.h5' , 'r')
    Q = np.array(d.get(f'Qs/{episode}'))
    policy = np.zeros((n_states))
    for j in range(n_states):
        policy[j] = np.argmax(Q[j])
    top_policies[beta] = policy 
    d.close()


    
    #plt.figure(figsize=(15,10))
    #plt.imshow(policies)
    #plt.colorbar()
    #plt.xticks(np.arange(5))
    #plt.xlabel('Actions')
    #plt.ylabel('States')
    #plt.savefig('heat_map.png')
    



    
    states = ['0,-2' , '0,-1' , '0,0' , '0,+1' , '0,+2' , '1,-2' , '1,-1' , '1,0' , '1,+1' , '1,+2' ]
    actions = [r'$\rightarrow$' , r'$\uparrow$' , r'$\leftarrow$' , r'$\downarrow$' , r'$stop$']
    size_y = n_states
    size_x = n_actions
    data = policies
    
    # Limits for the extent
    x_start = -0.5
    x_end = n_actions - 0.5
    y_start = -0.5
    y_end = n_states - 0.5
    
    extent = [x_start, x_end, y_start, y_end]
    
    # The normal figure
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, origin='lower', interpolation='None', cmap='Blues')
    plt.yticks(np.arange(n_states),states)
    plt.ylabel(r'$\rm{Tr}(\mathbb{S}^2),\partial_y \rm{Tr}(\mathbb{S}^2)$')
    plt.xticks(np.arange(n_actions), actions)
    plt.xlabel(r'action')
    
    # Add the text
    jump_x = (x_end - x_start) / (2.0 * size_x)
    jump_y = (y_end - y_start) / (2.0 * size_y)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size_x, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size_y, endpoint=False)
    
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = data[y_index, x_index]
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(text_x, text_y, label, color='black', ha='center', va='center')
    
    
    for i in range(n_states):
        ax.add_patch(Rectangle((-0.5+top_policies[beta,i], -0.5+i), 1, 1, edgecolor='red', fill=False, lw=3))
        
    fig.colorbar(im)
    plt.title(fr'Training with $\beta={beta/10}$', fontsize=20)
    plt.savefig(f'heat_map_beta_{beta}.png')
    plt.show()
