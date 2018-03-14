import numpy as np

def gap(array):
    array = np.sort(array)
    gap = np.diff(array)
    gap = np.append(gap, array[0]-array[-1]+2*np.pi)
    return gap

def shifted_gap(array, k):
    array = np.sort(array)
    shift = len(array)//k
    pi_k_gaps = (array-np.roll(array, shift))%(2*np.pi)-2*np.pi/k
    return np.abs(pi_k_gaps)

def ratio(array):
    gaps = gap(array)
    ratio = gaps/np.roll(gaps, -1)
    ratio = np.minimum(ratio, 1/ratio)
    return np.mean(ratio)
