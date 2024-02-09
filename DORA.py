# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:49:31 2024

@author: Dr. Manish Yadav


Educational implementation of a linear single-mass damped oscillator
(mass-spring-damper system) with optional external harmonic forcing.

d^2x/d^2t + c*dx/dt + k*x = f*sin(omega*t+phi)

conversion into 1st-order ordinary differential equation (state-space representation)

New set of variables:
    q1 = x
    q2 = dq1/dt = dx/dt

Transformation of ODE:
    dq1/dt = q2
    dq2/dt = -c*q2 -k*q1 +f*sin(omega*t+phi)

Notation in vector form:
    q = [q1, q2]^T
    F = [0, f*sin(omega*t+phi)]^T

    [dq/dt] = [[A]]*[q] + [F]

    state matrix A
    [[A]] = [[0, 1], [-k, -c]]

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import time
from tqdm.notebook import tqdm, trange
from scipy.integrate import odeint

def one_dof_oscillator(t, q: np.ndarray,
                       c: float = 0.1,
                       k: float = 1.0,
                       f: float = 0,
                       omega: float = 1.0,
                       phi: float = 0.0,
                       beta: float=0
                       ) -> np.ndarray:
    """ ODE one-dimensional oscillator. """

    A = np.array([[0, 1], [-k, -c]])
    B = np.array([[0, 0], [-beta, 0]])
    F = np.array([0, f*np.cos(omega*t+phi)])
    
    return np.dot(A, q) + np.dot(B, q*q*q) + F

def FreqGenerator(f, omega, t, phi):
    return np.array([0, f*np.sin(omega*t+phi)])

def SolveDuffing(q0, t_eval, c, k, f, omega, phi, beta):
    # numerical time integration
    sol_forced = solve_ivp(one_dof_oscillator, t_span=[t_eval[0], t_eval[-1]], y0=q0,\
                           t_eval=t_eval, args=(c, k, f, omega, phi, beta))

    # display of trajectories
    q = sol_forced.y.T
    t = sol_forced.t
    forcing = f*np.cos(omega*t+phi)  
    # Plot(q[:, 0], q[:, 1], forcing) 
    return q, forcing

def PhaseSpace_Plot(Q,f):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.plot(Q[500:,0], Q[500:,1], lw=1.0)
    plt.title('$f={:.2f}$'.format(f),fontsize=22)
    # plt.scatter(q[0,0], q[0,1], marker='o',c='r')
    ax.set_xlabel(r'$q_1(t)$',fontsize=22); ax.set_ylabel(r'$q_2(t)$',fontsize=22)
#     ax.set_xlim(-1.6,1.6); ax.set_ylim(-1.2,1.2)
    ax.tick_params(labelsize=18)
    plt.show()
    

#### Response Analysis
def ForcingRespose(q0, t_eval, c, k, f_steps, omega, phi, beta):
    # numerical time integration
    
    sol = solve_ivp(one_dof_oscillator, t_span=[t_eval[0], t_eval[-1]], y0=q0,\
                           t_eval=t_eval, args=(c, k, f_steps[0], omega, phi, beta))
    f_amp = np.repeat(f_steps[0], sol.t.shape[0])
    sols = sol.y; f_amps = f_amp
    ####Setting Initial condition for next forcing
    y0 = sol.y[:,-1]
    for i in trange(1,len(f_steps)):
        sol = solve_ivp(one_dof_oscillator, t_span=[t_eval[0], t_eval[-1]], y0=q0,\
                           t_eval=t_eval, args=(c, k, f_steps[i], omega, phi, beta))
        f_amp = np.repeat(f_steps[i], sol.t.shape[0])
        
        sols=np.append(sols, sol.y, axis=1)
        f_amps=np.append(f_amps, f_amp)
        
        ####Setting Initial condition for next forcing
        y0 = sol.y[:,-1]
    return sols, f_amps
    
def Plot(X, ts, f_amps):
    # fig_size = plt.rcParams["figure.figsize"]  
    # fig_size[0] = 12; fig_size[1] = 8
    # plt.rcParams["figure.figsize"] = fig_size 
    # plt.figure()
    # plt.subplot(3, 1, 1)
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    ax.plot(ts[::2], X[0,::2], lw=0.5, color='b')
    ax.set_ylabel(r'$q_1(t)$', fontsize=30)
    ax.set_xlabel(r'$t$', fontsize=30)
    ax.set_xlim(ts[0]-2,ts[-1]+2)
    ax.tick_params(labelsize=20)
    plt.title('Forced Duffing Oscillator', fontsize=22)
    plt.show()
    # plt.subplot(3, 1, 2)
    
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    ax.plot(ts[::2], X[1,::2], lw=0.5, color='b')
    ax.set_xlim(ts[0]-2,ts[-1]+2)
    ax.set_ylabel(r'$q_2(t)$', fontsize=30)
    ax.set_xlabel(r'$t$', fontsize=30)
    ax.tick_params(labelsize=20)
    plt.show()
    
    # plt.subplot(3, 1, 3)
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    ax.plot(ts[::2], f_amps[::2], lw=3, color='green', label='Test sets')
    ax.axhline(y = 0.46, c='grey', ls='--')
    ax.axhline(y = 0.49, c='grey', ls='--', label='Training sets')
    ax.legend(fontsize=18)
    ax.set_xlim(ts[0]-2,ts[-1]+2)
    ax.set_xlabel(r'$t$', fontsize=30)
    ax.set_ylabel(r'$f$ (amp)', fontsize=30)
    ax.tick_params(labelsize=20)
    # plt.tight_layout()
    plt.show()

    fig_size = plt.rcParams["figure.figsize"]  
    fig_size[0] = 1.5; fig_size[1] = 1.5
    plt.rcParams["figure.figsize"] = fig_size 
    plt.plot(X[0], X[1], lw=0.5)
    plt.scatter(X[0], X[1], marker='o',c='r')
    plt.xlabel(r'$q_1(t)$', fontsize=25); plt.ylabel(r'$q_2(t)$', fontsize=25)
    plt.show()

    
###Forced case: initial transient phase, then following forcing by some phase shift
###induced by the damping

######### Duffing Oscillator parameters####################################
c = 0.3    # damping
k = -1.0     # linear stiffness
fs = [0.46, 0.49]    # forced case
omega = 1.5
phi = 0
beta=1

# initial conditions
q0 = np.array([0.05, 0.05])

T = 250
t_eval = np.arange(start=0, stop=T, step=0.1)

# numerical time integration
# time integration interval
%time solA = solve_ivp(one_dof_oscillator, t_span=[t_eval[0], t_eval[-1]], y0=q0,\
                      t_eval=t_eval, args=(c, k, fs[0], omega, phi, beta))

%time solB = solve_ivp(one_dof_oscillator, t_span=[t_eval[0], t_eval[-1]], y0=q0,\
                      t_eval=t_eval, args=(c, k, fs[1], omega, phi, beta))

# display of trajectories
qa = solA.y.T; t = solA.t
forcingA = fs[0]*np.sin(omega*t+phi)

qb = solB.y.T;
forcingB = fs[1]*np.sin(omega*t+phi)

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, qa[:, 0], color='black')
plt.ylabel(r'$q_1(t)$')
plt.title('Forced oscillator')

plt.subplot(3, 1, 2)
plt.plot(t, qa[:, 1], color='black')
plt.ylabel(r'$q_2(t)$')

plt.subplot(3, 1, 3)
plt.plot(t, forcingA, color='red')
plt.xlabel(r'$t$')
plt.ylabel(r'$f(t)$')
plt.tight_layout()
plt.show()
   
PhaseSpace_Plot(qa,fs[0])
PhaseSpace_Plot(qb,fs[1])


#### Response Analysis
T = 250; h=0.1
t_eval = np.arange(start=0, stop=T, step=0.1)

f_steps = [0.2, 0.35, 0.48, 0.58, 0.75]

sols, f_amps =ForcingRespose(q0, t_eval, c, k, f_steps, omega, phi, beta)
T_evals = np.arange(start=0, stop=T*len(f_steps), step=h)
Plot(sols, T_evals, f_amps)

print(sols.shape, T_evals.shape, f_amps.shape)
    
    
### Save and Load data
######1. Prepare Data
## Train
Fs_a = np.tile(fs[0], qa.shape[0])
Tr_a = np.concatenate((qa.T, np.expand_dims(Fs_a, axis=1).T), axis=0)
Fs_b = np.tile(fs[1], qb.shape[0])
Tr_b = np.concatenate((qb.T, np.expand_dims(Fs_b, axis=1).T), axis=0)

Tr_Dat = np.concatenate((Tr_a, Tr_b), axis=1)

## Test
Fs = np.repeat(f_steps, qa.shape[0])
Tst_Dat = np.concatenate((sols, np.expand_dims(Fs, axis=1).T), axis=0)

print(Tr_Dat.shape, Tst_Dat.shape)



####2. Save data
# DORA_Tr = np.save('DORA_Train.npy', Tr_Dat) 
# DORA_Tst = np.save('DORA_Test.npy', Tst_Dat) 


####3. Load data
DORA_Tr1 = np.load('DORA_Train.npy') 
DORA_Tst1 = np.load('DORA_Test.npy') 

print(DORA_Tr1.shape, DORA_Tst1.shape)
    

#### Plot loaded data
T=2500

fig_size = plt.rcParams["figure.figsize"]  
fig_size[0] = 5; fig_size[1] = 5
    
plt.plot(DORA_Tr1[0,:T], DORA_Tr1[1,:T])
plt.title('$f=${:}'.format(DORA_Tr1[2,0]), fontsize=16)
plt.xlabel(r'$q_1(t)$',fontsize=22); plt.ylabel(r'$q_2(t)$',fontsize=22)
plt.show()
plt.plot(DORA_Tr1[0,T:], DORA_Tr1[1,T:])
plt.title('$f=${:}'.format(DORA_Tr1[2,T]), fontsize=16)
plt.xlabel(r'$q_1(t)$',fontsize=22); plt.ylabel(r'$q_2(t)$',fontsize=22)
plt.show()


fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
ax.plot(DORA_Tst1[0,::1], lw=0.5, color='b', label='Test $f$ response $q1$')
ax.plot(DORA_Tst1[2], lw=2, color='green', label='Test set $f$')
ax.axhline(y = DORA_Tr1[2,0], c='grey', ls='--')
ax.axhline(y = DORA_Tr1[2,T], c='grey', ls='--', label='Train set $f$')
ax.legend(fontsize=14)
ax.set_xlabel(r'$t$', fontsize=30)
ax.set_ylabel(r'$f$ (amp)', fontsize=30)
ax.tick_params(labelsize=20)
# plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
ax.plot(DORA_Tst1[1,::1], lw=0.5, color='b', label='Test $f$ response $q2$')
ax.plot(DORA_Tst1[2], lw=2, color='green', label='Test set $f$')
ax.axhline(y = DORA_Tr1[2,0], c='grey', ls='--')
ax.axhline(y = DORA_Tr1[2,T], c='grey', ls='--', label='Train set $f$')
ax.legend(fontsize=14)
ax.set_xlabel(r'$t$', fontsize=30)
ax.set_ylabel(r'$f$ (amp)', fontsize=30)
ax.tick_params(labelsize=20)
# plt.tight_layout()
plt.show()