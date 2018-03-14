import numpy as np
from scipy import sparse
import scipy.sparse.linalg as la
from functools import reduce


FORMAT='csc'

def mult_kron(matrices, sparse_format=None):
    return reduce((lambda x, y: sparse.kron(x,y, format=FORMAT)), matrices)

def one_site_op(matrix, i, L):
    dim_loc=np.shape(matrix)[0]
    i_left = sparse.eye(dim_loc**(i))
    i_right = sparse.eye(dim_loc**(L-i-1))
    return mult_kron([i_left, matrix, i_right])

def two_site_op(matrix_i, matrix_j, i, j, L):
    i=i%L
    j=j%L
    if i>j:
        i,j=j,i
        matrix_i, matrix_j = matrix_j, matrix_i
    dim_loc=np.shape(matrix_i)[0]
    i_left = sparse.eye(dim_loc**(i))
    i_center = sparse.eye(dim_loc**(j-i-1))
    i_right = sparse.eye(dim_loc**(L-j-1))
    return mult_kron([i_left, matrix_i, i_center, matrix_j, i_right])

def normalize(vector):
    return vector/np.linalg.norm(vector, axis=0)

def np_prod(matrices):#np.linalg.multi_dot
    return reduce((lambda x, y: np.dot(x,y)), matrices)

#np array, index=time_set
def expect_val(operator, state):
    return np.sum(state.conj().T*np.dot(operator, state.T), axis=0)

#np array containing  final_state[time_set]
def evolve(time_set, initial_state, eigvec, eigval):
    state_0_diag = np.dot(eigvec.conj().T, initial_state)
    state_evol = np.array(normalize(np.dot(eigvec,\
    state_0_diag[:, np.newaxis]*(eigval[:, np.newaxis]**time_set))))
    return state_evol.T
