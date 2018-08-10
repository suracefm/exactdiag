import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
from operators import *

h=0.99
m=1.0
J=1.0
hz=0.1
phiz=0.0
hx=0.0
phix=0.0
n_dis=100
#index=1
f=1.0
sizes = [2,3,4,5,6,7]
#time_set = np.power(2, np.arange(40))
#time_set=np.arange(60000, step=19)
time_set = (np.power(1.2, np.arange(100))).astype(int)

omega=np.exp(1j*2*np.pi/3)
sigma = np.array([[1,0,0],[0,omega,0],[0,0,omega**2]])
tau = np.array([[0,0,1],[1,0,0],[0,1,0]])
kick = la.expm(h*2*np.pi/(3*np.sqrt(3))*(tau-np.transpose(tau)))

def build_H(J,m,hz,hx,L):
    dim = 3**L
    H = np.zeros((dim, dim), dtype=float)
    g = np.random.choice([1,omega, omega**2], size=L)
    for site in range(L):
        H=H+2*np.real((J-1j*g[site]*m)*two_site_op(np.conj(sigma), sigma, site, site+1, L))
        H=H+hx*2*np.real(np.exp(1j*phix)*one_site_op(tau, site, L))
        H=H+hz*2*np.real(np.exp(1j*phiz)*one_site_op(sigma, site, L))
    return H


ReZ=np.zeros((len(sizes), len(time_set)), dtype=float)
ImZ=np.zeros((len(sizes), len(time_set)), dtype=float)
varReZ=np.zeros((len(sizes), len(time_set)), dtype=float)
varImZ=np.zeros((len(sizes), len(time_set)), dtype=float)

for countL, L in enumerate(sizes):
    filename='Z3/Z3_L{}_f{}_h{}_m{}_J{}_hz{}_phiz{}_hx{}_phix{}_ndis{}.txt'.format(L,f,h,m,J,hz,phiz,hx,phix,n_dis)
    initial_state=np.zeros(3**L)
    flips = L-int(L*f)
    index=np.sum(np.power(3, np.random.randint(L, size=flips)))
    initial_state[index]=1
    print("Initial state (local magnetization) ",[expect_val(one_site_op(sigma, i, L), initial_state) for i in range(L)])

    K = kick
    for _ in range(L-1):
        K = np.kron(K, kick)


    for counter in range(n_dis):
        start = time.time()

        H = build_H(J,m,hz,hx,L)
        U_F = np.dot(la.expm(-1j*H),K)
        eigval, eigvec = np.linalg.eig(U_F)
        spectrum = np.angle(eigval)

        final_state = evolve(time_set, initial_state, eigvec, eigval)
        for i in range(L):
            val=expect_val(one_site_op(sigma, i, L), initial_state)*\
                     expect_val(one_site_op(np.conj(sigma), i, L),final_state)\
                     *(omega)**(time_set%3)
            ReZ[countL]+=np.real(val)
            varReZ[countL]+=np.real(val)**2
            ImZ[countL]+=np.imag(val)
            varImZ[countL]+=np.imag(val)**2
        elapsed = time.time()-start
        #print('size', L, '\tdisorder realization', counter,'\ttime elapsed', elapsed)

    ReZ[countL]=ReZ[countL]/(L*n_dis)
    varReZ[countL]=(varReZ[countL]/(L*n_dis)-ReZ[countL]**2)/(L*n_dis-1)
    ImZ[countL]=ImZ[countL]/(L*n_dis)
    varImZ[countL]=(varImZ[countL]/(L*n_dis)-ImZ[countL]**2)/(L*n_dis-1)
    with open(filename, 'wb') as outfile:
        outfile.write(('# '+filename+'\n').encode('utf-8'))
        np.savetxt(outfile, np.stack((time_set, ReZ[countL], ImZ[countL], varReZ[countL], varImZ[countL]), axis=-1),\
         header='time\tRe(Z)\Im(Z)\tVar(Re(Z))\tVar(Im(Z))')
