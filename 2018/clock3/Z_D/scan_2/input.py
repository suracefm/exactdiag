import numpy as np

# Parameters
sizes = [2,3,4,5,6]
dim_loc = 3
n_dis = 100

lr = False
spectral_properties = True
evolution = True

Jzz = 1.0
hz = 0.9

phi_vec=(np.pi/6)*np.array([0.0,1.0,3.0,6.0])

hx_vec = np.linspace(0.10, 0.30, 5)

betas = [1]*(dim_loc-1)
lambdas = [1]*(dim_loc-1)

time_set = np.power(2, np.arange(40))
