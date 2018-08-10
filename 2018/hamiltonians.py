import numpy as np
import itertools
from operators import *

def clock(dim_loc, L):
    dim = dim_loc**L

    # Build operators of the local Hilbert space: Index m from 1 to dim_loc-1
    sigma_m = np.empty((dim_loc-1, dim_loc, dim_loc), dtype=complex) # m row column
    if dim_loc == 2:
        sigma_m[0]=(np.array([[1,0],[0,-1]]))
    else:
        for m in range(dim_loc-1):
            sigma_m[m]=np.diag([np.exp(1j*2*(m+1)*n*np.pi/dim_loc) \
            for n in range(dim_loc)])

    tau_m = np.empty((dim_loc-1, dim_loc, dim_loc))
    for m in range(dim_loc-1): tau_m[m] = np.roll(np.identity(dim_loc), -m-1, axis=1)
    tau = tau_m[0]

    def buildH(JZZ, m, hZ, hX, alphas, betas, lambdas, **kwargs):
        if "JZZ_array" not in kwargs:
            if dim_loc == 2:
                JZZ_array = JZZ*np.ones(L)+1j*m*np.random.choice([-1,1], size=L)
            else:
                choices = [np.exp(1j*2*(m+1)*n*np.pi/dim_loc) \
                for n in range(dim_loc)]
                JZZ_array = JZZ*np.ones(L)+1j*m*np.random.choice(choices, size=L)
        else: JZZ_array=kwargs['JZZ_array']
        if "hZ_array" not in kwargs:
            hZ_array = hZ*np.ones(L)
        else: hZ_array=kwargs['hZ_array']
        if "hX_array" not in kwargs:
            hX_array = hX*np.ones(L)
        else: hX_array=kwargs['hX_array']
        Hamiltonian = np.zeros((dim, dim), dtype=complex)
        for m, site in itertools.product(range(dim_loc-1), range(L)):
                Hamiltonian=Hamiltonian+JZZ_array[site]*alphas[m]*\
                two_site_op(sigma_m[-m-1], sigma_m[m], site, site+1, L).toarray()\
                +np.conj(JZZ_array[site])*alphas[m]*\
                two_site_op(sigma_m[m], sigma_m[-m-1], site, site+1, L).toarray()
                Hamiltonian=Hamiltonian+hX_array[site]*betas[m]*\
                one_site_op(tau_m[m], site, L).toarray()
                Hamiltonian=Hamiltonian+hZ_array[site]*lambdas[m]*\
                one_site_op(sigma_m[m], site, L).toarray()
        return Hamiltonian

    def buildK(**kwargs):
        if "h" not in kwargs:
            single_kick=tau
        else:
            h=kwargs['h']
            single_kick = la.expm(h*tau-h*tau_m[1])
        kick=single_kick
        for _ in range(L-1):
            kick = np.kron(kick,single_kick)
        return kick

    def Z(initial_state, final_state, i, time_set, **kwargs):
            Zval=expect_val(one_site_op(sigma_m[-1], i, L).toarray(), initial_state)*\
                 expect_val(one_site_op(sigma_m[0], i, L).toarray(),final_state)\
                 *np.exp(-1j*2*np.pi*(time_set%dim_loc)/dim_loc)
            #Znew=expect_val(one_site_op(sigma_m[1], i, L).toarray(), initial_state)*\
            #     expect_val(one_site_op(sigma_m[1], i, L).toarray(),final_state)\
            #     *np.exp(-1j*4*np.pi*(time_set%dim_loc)/dim_loc)
            return Zval#, Znew
    return buildH, buildK, Z
