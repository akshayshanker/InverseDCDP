
""" 
Simple implementation of the roof-top cut algorithm without iterative search
and without intersection points. 

"""

import numpy as np
from numba import njit, prange
import numba
import copy
import time
import numba as nb
from pykdtree.kdtree import KDTree as KDTree



def _rfc_vectorized(M, gradM, Qval, sigma, M_bar, radius, closest_indices, dd, k):   
    d = M.shape[1]  #dimension of grid
    n = M.shape[0]
    dp = sigma.shape[1] #number of policies
 
    # closest indices has shape (n, k-1)
    # Evaluate values on tangent plane and check which points are below
    gradrs = gradM.reshape(n,d,1)
    Md = M[closest_indices].reshape(n*(k-1),d) - np.repeat(M, k-1, axis=0) # obtain difference between gid points and neigbours
    Mdn = Md.reshape(n,k-1,d)
    tngv = Mdn@gradrs
    tngv = tngv.reshape(n,k-1) #tangent plane values for neighoburing points
    vdiff = Qval[closest_indices].reshape(n,k-1)-Qval
    I1 = vdiff < tngv #Indicator matrix (n x k-1) for each  neighobur if point below tangent plane

    #Check for jump in policy
    Pd = sigma[closest_indices].reshape(n*(k-1),dp) - np.repeat(sigma, k-1, axis=0) # obtain differences between poicy vals and neighbours
    Pdnorm = np.linalg.norm(Pd, axis = 1)
    delsig = np.abs((Pdnorm.reshape(n,k-1))/(np.linalg.norm(Md,axis=1).reshape(n,k-1)))
    I2 = delsig>M_bar  #Indicator matrix (n x k-1) for each grid point if jump in policy

    #Only include neighoburs within radius 
    I3 = dd[:,1:]<radius

    #Select sub optimal points
    sub_optimal_points = np.unique(I1*I2*I3*(closest_indices+1))
    sub_optimal_points = sub_optimal_points[1:]-1

    return sub_optimal_points, tngv, closest_indices


def Rfc(M, gradM, Qval, sigma, M_bar, radius):
    """Implements vectorized version of roof-top cut and eliminates points below the roof top.

    Parameters
    ----------
    M : 2D array
    Irregular grid.
    gradM : 2D array
    Gradient of value function at each point on the grid (each column represents a dimension).
    Qval : 2D array
    Value function at each point on the grid.
    sigma : 2D array 
    Policy function at each point on the grid (each policy represented as a column).
    M_bar : float 
    Jump detection threshold.
    radius : float 
    Neigbour distance threshold.

    Returns
    -------
    sub_optimal_points : list
    List of indices of sub-optimal points.
    tngv: 2D array
    Tangent plane values for neighbouring points
    closest_indices: 2D array
    Indices for neighbouring points
    """

    k = 5    #number of closest neighbours 
    tree = KDTree(M)
    #obtain k nearest neighbours
    dd, closest_indices = tree.query(M, k)
    closest_indices = closest_indices[:,1:] #Keep only strict neighbours
    sub_optimal_points, tngv, closest_indices = _rfc_vectorized(M, gradM, Qval, sigma, M_bar, radius, closest_indices, dd, k)

    return sub_optimal_points, tngv, closest_indices