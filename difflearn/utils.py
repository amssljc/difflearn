# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:55:08 2021

@author: jcleng
"""


import numpy as np
from numpy import diag



def triu_vec(M):
    """
    extract upper triangle elements to vectors.

    Parameters
    ----------
    M : TYPE
        DESCRIPTION.

    Returns
    -------
    vec : TYPE
        DESCRIPTION.

    """
    # not batch case, (p, p)
    if len(M.shape) == 2:
        if isinstance(M, np.ndarray):
            vec = M[np.triu(np.ones_like(M), k=1) == 1]
        
        vec = vec[np.newaxis, :]
    # batch case, (B, p, p)
    elif len(M.shape) == 3:
        if isinstance(M, np.ndarray):
            vec = [m[np.triu(np.ones_like(m), k=1) == 1] for m in M]
            vec = np.stack(vec)
        
    else:
        print("input M dim should <= 3.")
        return
    return vec



def vec2mat(V):

    p = int(np.sqrt(V.shape[-1]))
    M = V.reshape(-1, p, p)
    return M


def remove_diag(theta):
    if isinstance(theta, np.ndarray):
        theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return theta_


def theta2partial(theta):
    sqrt_diag = np.sqrt(diag(1.0 / theta.diagonal()))
    partial = -np.dot(np.dot(sqrt_diag, theta), sqrt_diag)
    np.fill_diagonal(partial, 1)
    return partial


def validate_posdef(X):
    # =============================================================================
    #     validate if datasets are positive definate matrices
    # =============================================================================
    eig_min = np.min(np.linalg.eigvals(X))
    if eig_min > 0:
        print("X is positive definate.")
        return True
    elif eig_min == 0:
        print("X is semi-positive definate.")
        return False
    else:
        print("X is not positive definate!!!" )
        return False


def keep_largest_k(X, k):

    l = len(X.flatten())
    X_ = X.copy()
    if k == 0:
        X_[::] = 0
        return X_
    indices = np.argpartition(abs(X_), l - k - 1, axis=None)
    X_[tuple(np.array(np.unravel_index(indices, X_.shape, "C"))[:, :-k])] = 0
    return X_
