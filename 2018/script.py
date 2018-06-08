import numpy as np
import matplotlib.pyplot as plt
from myio import *
from simulation import *

spec_names=('gap', 'shifted_gap','shifted_gap_2', 'log10_gap', 'log10_shifted_gap', 'log10_shifted_gap_2', 'ratio')
time_set= np.power(2, np.arange(40))
lenarr=len(time_set)

for L in [2]:
    for lambd in [0.01,0.99]:
        dim_loc=4
        n_dis=100
        simdict={'dim_loc': dim_loc, 'L': L, 'n_dis': n_dis}
        phi=np.pi/3
        eps=0.1
        time_set= np.power(2, np.arange(40))
        idata={'JZZ': 1.0, 'hZ': 1.0, 'hX': 1.0, 'alphas': np.array([(1-lambd)*np.exp(1j*phi)/2, 1, (1-lambd)*np.exp(-1j*phi)/2]),\
               'betas': np.array([eps,lambd, eps]), 'lambdas': np.array([eps,1,eps]),\
               'phi': phi, 'lambd': lambd}
        filename='clock4/clock4_%d_%.2f.txt' %(L,lambd)

        clockH, clockK, clockZ = clock(dim_loc, L)
        Z_mean, Z_var, Y_mean, Y_var, spectral_data, spectral_data_var = simulation(dim_loc, L, n_dis, idata, clockH, clockK, clockZ, time_set)

        with open(filename, 'wb') as f:
                for key, value in simdict.items():
                    f.write(('\n# '+key+' '+str(value)).encode('utf-8'))
                for key, value in idata.items():
                    f.write(('\n# '+key+' '+str(value)).encode('utf-8'))
                for i in range(7):
                    f.write(('\n# '+spec_names[i]+' '+str(spectral_data[i])+' '+str(spectral_data_var[i])).encode('utf-8'))
                f.write('\n# time\tRe(Z)\Im(Z)\tVar(Re(Z))\tVar(Im(Z))\n'.encode('utf-8'))
                np.savetxt(f, np.stack((time_set, np.real(Z_mean), np.imag(Z_mean),\
                            np.real(Z_var), np.imag(Z_var), np.real(Y_mean), np.imag(Y_mean),\
                                        np.real(Y_var), np.imag(Y_var)), axis=-1))
