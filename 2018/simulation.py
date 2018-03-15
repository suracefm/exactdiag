import numpy as np
from scipy import sparse
import scipy.sparse.linalg as la
import time
from random import randint
from operators import *
from spectrum_analysis import *
from io import *
from hamiltonians import *
import sys


def simulation(dim_loc, L, n_dis, data, Hfunc, Kfunc, Zfunc, time_set, EVOLVEZ=True):
    dim=dim_loc**L

    # Setting cycle
    steps=len(time_set)
    Z=np.zeros((n_dis, L, steps), dtype=complex)
    Znew=np.zeros((n_dis, L, steps), dtype=complex)
    spectral_matrix= np.zeros((n_dis, 7))

    # Disorder cycle
    for counter in range(n_dis):
        start = time.time()

        H = Hfunc(**data)
        kick = Kfunc(**data)
        U_F = np.dot(la.expm(-1j*H),kick)
        eigval, eigvec = np.linalg.eig(U_F)

        # Spectral properties
        spectrum = np.angle(eigval)
        gaps = gap(spectrum)
        shifted_gaps = shifted_gap(spectrum, dim_loc)
        shifted_gaps_2 = shifted_gap(spectrum, 2)
        log10_gaps = np.log10(gap(spectrum))
        log10_shifted_gaps = np.log10(shifted_gap(spectrum, dim_loc))
        log10_shifted_gaps_2 = np.log10(shifted_gap(spectrum, 2))
        r = ratio(spectrum)
        spectral_matrix[counter]=np.array([np.mean(gaps), np.mean(shifted_gaps),\
         np.mean(shifted_gaps_2), np.mean(log10_gaps), np.mean(log10_shifted_gaps),\
         np.mean(log10_shifted_gaps_2), r ])

        if EVOLVEZ:
            #Initial state
            initial_state = np.zeros(dim)
            initial_state[randint(0, dim-1)] = 1

            final_state = evolve(time_set, initial_state, eigvec, eigval)

            for i in range(L):
             Z[counter, i], Znew[counter, i] = Zfunc(initial_state, final_state, i, time_set, **data)

        elapsed = time.time()-start
        print('size', L, '\tdisorder realization', counter,'\ttime elapsed', elapsed)

    Z_mean=np.mean(Z, axis=(0,1))
    Z_var=np.var(Z, axis=(0,1))
    Znew_mean=np.mean(Znew, axis=(0,1))
    Znew_var=np.var(Znew, axis=(0,1))
    spectral_data=np.mean(spectral_matrix, axis=0)
    spectral_data_var=np.var(spectral_matrix, axis=0) #not really the variance!!!!
    return Z_mean, Z_var, Znew_mean, Znew_var, spectral_data, spectral_data_var
