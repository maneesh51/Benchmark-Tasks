# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:49:31 2024

@author: Dr. Manish Yadav


Duffing Oscillator:
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
import pandas as pd
from tqdm.notebook import tqdm, trange
from scipy.integrate import odeint
import argparse

#### Load data
DORA_Tr1 = pd.read_csv('DORA_Train.csv').to_numpy().T
DORA_Tst1 = pd.read_csv('DORA_Test.csv').to_numpy().T

print(DORA_Tr1.shape, DORA_Tst1.shape)

### Plot loaded data
T=2500

fig_size = plt.rcParams["figure.figsize"]  
fig_size[0] = 5; fig_size[1] = 5
    
plt.plot(DORA_Tr1[1,500:T], DORA_Tr1[2,500:T])
plt.title('$f=${:}'.format(DORA_Tr1[3,0]), fontsize=16)
plt.xlabel(r'$q_1(t)$',fontsize=22); plt.ylabel(r'$q_2(t)$',fontsize=22)
plt.show()

fig, ax = plt.subplots(figsize=(5, 5))
plt.plot(DORA_Tr1[1,T+500:], DORA_Tr1[2,T+500:])
plt.title('$f=${:}'.format(DORA_Tr1[3,T]), fontsize=16)
plt.xlabel(r'$q_1(t)$',fontsize=22); plt.ylabel(r'$q_2(t)$',fontsize=22)
plt.show()


fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
ax.plot(DORA_Tst1[3], lw=0.5, color='g', label='Test $f$')
ax.legend(fontsize=14)
ax.set_xlabel(r'$t$', fontsize=30)
ax.set_ylabel(r'$f$ (amp)', fontsize=30)
ax.tick_params(labelsize=20)
# plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
ax.plot(DORA_Tst1[1,::1], lw=0.5, color='b', label='Test $f$ response $q1$')
ax.plot(DORA_Tst1[4], lw=2, color='green', label='Test set $f$')
ax.axhline(y = DORA_Tr1[3,0], c='grey', ls='--')
ax.axhline(y = DORA_Tr1[3,T], c='grey', ls='--', label='Train set $f$')
ax.legend(fontsize=14)
ax.set_xlabel(r'$t$', fontsize=30)
ax.set_ylabel(r'$f$ (amp)', fontsize=30)
ax.tick_params(labelsize=20)
# plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
ax.plot(DORA_Tst1[2,::1], lw=0.5, color='b', label='Test $f$ response $q2$')
ax.plot(DORA_Tst1[4], lw=2, color='green', label='Test set $f$')
ax.axhline(y = DORA_Tr1[3,0], c='grey', ls='--')
ax.axhline(y = DORA_Tr1[3,T], c='grey', ls='--', label='Train set $f$')
ax.legend(fontsize=14)
ax.set_xlabel(r'$t$', fontsize=30)
ax.set_ylabel(r'$f$ (amp)', fontsize=30)
ax.tick_params(labelsize=20)
# plt.tight_layout()
plt.show()