import numpy as np

def rearrange(eignew, eigold):
    IPR=0
    dim=len(eigold)
    i=0
    while i<dim:
        pvec=np.abs(np.dot(np.conj(eigold[:,i]),eignew[:,i:]))
        j=np.argmax(pvec)
        IPR+=pvec[j]**4/dim
        if j!=0:
            eignew[:,[i, i+j]] = eignew[:,[i+j, i]]
        i+=1
    return eignew, IPR

def IPR_func(eignew, eigold):
    return np.mean(np.abs(np.sum(np.conj(eigold)*eignew, axis=0))**4)

def sum_IPR(eignew, eigold):
    dim=len(eigold)
    sIPR=0
    for i in range(dim):
        sIPR+=IPR_func(np.roll(eigold, i, axis=1), eignew)
    return sIPR
