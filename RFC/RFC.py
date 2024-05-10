"""
Module to implement vectorized roof-top cut by Dobrescu and Shanker (2024)

Code author: Akshay Shanker, University of New South Wales, Sydney
Email: akshay.shanker@me.com.

Credits
-------

Anant Mathur for providing the guidance and practical coding
help on vectorization of the RFC algorithm.

"""

import numpy as np
from numba import njit
import time

# Scipy stuff 
from scipy.interpolate import interp1d

# Fast KDTree
from pykdtree.kdtree import KDTree as KDTree1

# RFC helpers
from RFC.math_funcs import interpolateEGM
from RFC.RFCIntersect import IntersectionPoints


def _rfcVectorized(M, gradM, Qval, sigma, test_inds, closest_indices, dd,
                    J_bar=1 + 1e-10, radius=5e-1, radius_I=5e-1, k=10, intersect = False):
    """
    Implements vectorized version of roof-top cut. 
    
    Implements Equation (20) in Dobrescrue and Shanker (2024)

    Parameters
    ----------
    M : 2D array
        Irregular flat cartesian grid.
    gradM : 2D array
        Gradient of value function at each point on the grid.
        delta_{p}V(m(i)) = gradM[i, p]
    Qval : 2D array
        Value function at each point on the grid.
    sigma : 2D array
       Grid distance functions at each point on the grid. 
    test_inds: 1D array
        Indices of points from which to apply RFC.
    closest_indices : 2D array
        Indices of the k closest neighbors to each point in test_inds.
    dd : 2D array
        Pairwise distances between points in the grid and its neighbours.
    J_bar : float
        Jump detection threshold.
    radius : float
        Search radius for RFC. (rho_r in DS(2024)

    Notes
    -----
    Recall grid distance function can typically be the policy function 
    used to test for jumps due to discrete choice changes in the future. 
    """
    k_minus_1 = k - 1
    M_test = M[test_inds]  # Test points
    d = M_test.shape[1]  # Dimension of grid
    n = M_test.shape[0]  # Number of grid points to test
    dp = sigma.shape[1]        # Number of policies (dim grid dist function range)

    # Note: closest indices has shape (n, k-1) since KDTree returns the point 
    # itself as the first closest point

    # 1. Evaluate values neighbor values on tangent plane and check which points are below

    # Reshape gradients to 3D array
    gradrs = gradM[test_inds].reshape(n, d, 1)
    
    # Obtain difference between grid points and neighbors
    Md = M[closest_indices].reshape(n * (k_minus_1), d) - np.repeat(M_test, k_minus_1, axis=0) 
    Mdn = Md.reshape(n, k-1, d)

    # Get tangent plane values for neighboring points
    tngv = (Mdn @ gradrs).reshape(n, k_minus_1)
    #tngv = tngv.reshape(n, k-1)

    # Difference between value function at each point and its neighbors
    vdiff = Qval[closest_indices].reshape(n, k_minus_1) - Qval[test_inds]
   
    # Indicator matrix (n x k-1) of neighboring if point below tangent plane
    I1 = vdiff < tngv 
    
    # 2. Check for jump in policy
    
    # Obtain differences between grid distance func vals and neighbors
    Pd = sigma[closest_indices].reshape(n * (k_minus_1), dp) - np.repeat(sigma[test_inds], k_minus_1, axis=0) 

    # Calculating pointwise delsig and checking if jump in any of the grid dist has jumps
    Pd_abs = np.abs(Pd)
    Md_norm = dd[:, 1:]
    delsig = Pd_abs / (Md_norm.reshape(-1, 1))
    jump_any = np.sum(delsig > J_bar, axis=1) > 0
    
    # Indicator matrix (n x k-1) for each grid point and neighbor if jump in any of the policies
    I2 = jump_any.reshape(n, k_minus_1)

    # Only evaluate neighbors within radius
    I3 = dd[:, 1:] < radius

    # Select suboptimal points
    mask = I1 & I2 & I3
    sub_optimal_points = np.unique(closest_indices[mask] + 1)
    sub_optimal_points = sub_optimal_points[1:] - 1

    # Indicator array for neighbors that lie above the tangent plane from test_inds
    if intersect:
        I_above = I2 * (dd[:, 1:] < radius_I) * (vdiff > tngv)
    else:
        I_above = None

    return sub_optimal_points, I_above, closest_indices


def RFC(M, gradM, Qval,sigma,\
                          grid,
                          J_bar,
                          radius,
                          radius_I,
                          GdistInds,
                          k=5,
                          k1 =10,
                          max_iter = 50,
                          intersection=False,
                          interp_intersect = False,
                          s= 0.08, 
                          s_decay = 0.9,
                          k_decay1 = 5, 
                          k_decay2 = 15,
                          decay_n = 4,
                          max_iter_intersect = 7,
                          n_closest = 4):
    
    """
    Recursively implements the vectorized version of roof-top cut over subset
    and nearest neighbours of test points in the grid. 

    Function also implements a search for optimal points that jump from 
    neighboring points and evaluates the intersection points between them. 
    
    Parameters
    ----------
    M : 2D array
        Irregular grid.
    gradM : 2D array
        Gradient of value function at each point on the grid 
            (each column represents a of teh grid dimension).
    Qval : 2D array
        Value function at each point on the grid.
    sigma : 2D array 
        Policy function at each point on the grid (each policy represented as a column).
    J_bar : float 
        Jump detection threshold.
    radius : float 
        Max search radius. 
    radius_I : float
        Neighbor distance threshold for intersection points.
    polInds : list
        List of policy indices to test for jumps (i.e. grid distance function index).
    k : int
        Number of nearest neighbours to consider for RFC.
    k1 : int
        Number of nearest neighbours to consider to calculate intersection points.
    max_iter : int
        Maximum number of iterations in Random search for suboptimal points.
    intersection : bool
        If True, the function returns clean grid plus intersection points.
    interp_intersect : bool
        If True, the function interpolates the intersection points. Otherwise returns NaNs. 
    s : float
        Fraction of grid points to consider in each iteration as centers for search. 
    s_decay : float
        Reduction factor for s after each iteration.
    k_decay1 : int
        Min number of neighbours to consider for suboptimal points.
    k_decay2 : int
        Reduction constant for k after each iteration.
    n_closest : int

    Returns
    -------
    M_clean: 2D array
        Cleaned grid points.
    Qval_clean: 2D array
        Cleaned value function.
    sigma_clean: 2D array
        gradM_clean
    M_intersect_2: 2D array
        Intersection points.
    Qval_intersect_2: 2D array
    sigma_intersect_2: 2D array

    Todo
    ----
    1. Make separate function for intersection point approximation. 

    """

    n = 0
    

    # Initialize grids to be cleanded and returned as clean grids 
    M_clean = M
    Qval_clean = Qval
    sigma_clean = sigma
    gradM_clean = gradM
    
    # Initial array of suboptimal points
    sub_optimal_points = np.array([-1]) 
    
    # Pull out random test points from full grid 
    s1 = int(s*len(M_clean))
    test_inds = np.random.choice(np.arange(len(M_clean)),\
                                  size = int(s1*1), replace=True)
    test_points = M_clean[test_inds]
    
    while  n<max_iter and len(M_clean)>0:
        
        # A. Build KDTree with current clean points and perform RFC 

        #start = time.time()
        M_clean32 = np.float32(M_clean)
        tree= KDTree1(M_clean32, leafsize=70)
        dd, closest_indices = tree.query(np.float32(test_points), k, eps = 0.001)
        
        closest_indices[closest_indices>len(M_clean)] = int(len(M_clean)-1)

        # Keep only k-1 strict neighbours
        closest_indices = closest_indices[:, 1:]  
        #closest_indices[:,0] = test_inds
       
        # Get sub-optimal points using RFC -- using latest M_clean 
        # indices for indexing the points 
        # test_inds are indices of test points from which RFC is applied
        #  in latest M_clean

        new_sub_optimal_points, I_above, pairs\
                                        = _rfcVectorized(M_clean32,
                                                        np.float32(gradM_clean),
                                                        np.float32(Qval_clean),
                                                        np.float32(sigma_clean[:,GdistInds]),
                                                        test_inds, 
                                                        closest_indices,
                                                        dd,
                                                        J_bar = J_bar,
                                                        radius = radius,
                                                        radius_I = radius_I,k = k)

        # indicies of M_clean before sub-optimal points are removed
        #M_clean_inds = np.arange(len(M_clean))
        
        if len(new_sub_optimal_points) == 0:
            # If new suboptimal points are not founds, break 
            break
        else:
            
            # Update suboptimal points and make clean array
            sub_optimal_points = np.array(new_sub_optimal_points).astype(int)

            # B. Get indices for next test points
            # we pick points in neighbourhood clusterrs are not not sub-optima
            # but have sub-optimal points within the neighbourhood cluster 

            # remove rows in closest indices with no sub optimal points 
            closest_indices = np.array(closest_indices).astype(int)
            mask1 = np.isin(closest_indices, sub_optimal_points)
            row_mask = np.any(mask1, axis=1)
            closest_indices1 = closest_indices[row_mask]
            # get cloest indices that were optimal but in row with a suboptimal point
            optimal_points_inds = np.setdiff1d(closest_indices1.ravel(),\
                                                     new_sub_optimal_points)
            # create new test poitns 

            if len(optimal_points_inds) > s1:
                #s = len(optimal_points_inds)
                optimal_points_inds = np.unique(np.random.choice(optimal_points_inds,\
                                                    size = s1, replace = False))
                

            
            
            optimal_points = M_clean[optimal_points_inds]
            test_inds_old = optimal_points_inds # testinds indexed by previous M_clean
            test_points = np.float32(optimal_points)  # new test points 

            #C. Update clean grids
            mask = np.ones(M_clean.shape[0], dtype=bool)
            mask[sub_optimal_points] = False
            M_clean_inds = np.arange(len(M_clean))[mask]
            
            M_clean = M_clean[M_clean_inds]
            Qval_clean = Qval_clean[M_clean_inds]
            sigma_clean = sigma_clean[M_clean_inds]
            gradM_clean = gradM_clean[M_clean_inds]


            # update the new test points to be indexed by the new M_clean indices
            test_inds = np.where(np.in1d(M_clean_inds, test_inds_old))[0]

            #test_inds1 = np.random.choice(np.arange(len(M_clean)),\
            #                      size = int(s1*.05), replace=True)
            
            # make 1d array out of testind1 nd testind
            #test_inds = np.unique(np.hstack((test_inds, test_inds1)))

            test_inds_fresh = np.random.choice(np.arange(len(M_clean)),\
                                  size = int(s1*0.1), replace=True)
                
            optimal_points_inds = np.unique(np.append(optimal_points_inds, test_inds_fresh))

            
            
            # update iteration, reduce number of search points and neihbors
            n += 1

            if n>decay_n:
                s1 = int(s1*s_decay)
            
            k = int(max(k_decay1,k-k_decay2))

    if intersection:
        # Append intersection points to the clean arrays
        n = 0 
        s1 = int(len(M_clean)*s)
        tree = KDTree1(M_clean, leafsize=60)
        
        test_inds = np.unique(np.random.choice(np.arange(len(M_clean)),\
                                                 size = int(s1), replace=True))

        # Make empty arrays to cumulate results
        test_inds_cumul = np.empty(shape=(0,  ))  # indices of test points for intersection 
        I_above = np.empty(shape=(0, int(k1-1)  )) # indicator matrix for jumps in policy
        closest_indices_cumul = np.empty(shape=(0, int(k1-1)  )) # indices of closest neighbours
        new_sub_optimal_points = np.empty(shape=[0]*len(sub_optimal_points.shape)) # suboptimal points

        # 1. Collect intersection points
        while  n<max_iter_intersect:
            
            dd, closest_indices = tree.query(np.float32(M_clean[test_inds]), k1, eps = 0.01)
            closest_indices = closest_indices[:, 1:]
            #print(closest_indices.shape)
            #start = time.time()

            # I_above_prime is a mask that gives indices in closest_indices
            # that have jumped from the point associated with each row 
            # in closest_indices (i.e. the test points)
            sub_optimal_points, I_above_prime, pairs\
                                            = _rfcVectorized(M_clean,
                                                            gradM_clean,
                                                            Qval_clean,\
                                                            sigma_clean[:,GdistInds],
                                                            test_inds,
                                                            closest_indices,
                                                            dd,
                                                            J_bar = J_bar,
                                                            radius= radius,
                                                            radius_I = radius_I,k=k1,
                                                            intersect = True)
            
            
            closest_indices_cumul = np.vstack((closest_indices_cumul, closest_indices))
            new_sub_optimal_points = np.hstack((new_sub_optimal_points, sub_optimal_points))
            test_inds_cumul = np.hstack((test_inds_cumul, test_inds))
            
            # break if no new intersection points are found
            if np.sum(I_above_prime) == 0:
                break 
            
            else:

                # Update jumped points 
                I_above_u = np.unique(I_above, axis=0)
                I_above_prime_u = np.unique(I_above_prime, axis=0)
                I_above = np.vstack((I_above, I_above_prime))
                
                if len(I_above_prime_u) == len(I_above_u):
                    break

                else:
                    # new indices that are above the tangent plane
                    jumped_indices = closest_indices[I_above_prime].ravel()

                    if n>1:
                        s1 = int(s1*s_decay)
                        #k1 = int(max(5,k1+1))
                    
                    # use points that were in the neighbourhood of  points
                    # that were jumped from as the new test points
                    mask1 = np.isin(closest_indices, jumped_indices)
                    row_mask = np.any(mask1, axis=1)
                    closest_indices1 = closest_indices[row_mask]\
                    
                    try:
                        test_inds = np.unique(np.random.choice(closest_indices1,\
                                                                replace = True,\
                                                                size = s1))
                    except:
                        test_inds = np.unique(np.random.choice(np.arange(len(M_clean)),\
                                                                replace = True, size = s1))
            n += 1
        
        # 2. Interpolate intersection points
        if len(I_above)>0:
            
            # A. Get pairs of indicies and points that jumped from
            # index and are optimal 
            I_above_final = np.zeros((len(M_clean), k1-1))
            closest_indices_final = np.zeros((len(M_clean), k1-1))

            # get unique indices of center points that were tested for jumps 
            # and mask of jump points for the unique indices
            test_inds_cumul, unique_indices = np.unique(test_inds_cumul.astype(int), return_index=True)
            I_above = I_above[unique_indices]

            # get unique indicies of center points and nearest neighbours
            # that were tested for jumps 
            closest_indices_cumul = closest_indices_cumul[unique_indices]

            # Whenever a point in M_clean was tested for a jump, replace
            # that row in I_above_final with the indicator array representing
            # neighbours that jumped from that point
            I_above_final[test_inds_cumul,:] = I_above

            closest_indices_final[test_inds_cumul,:] = closest_indices_cumul
            closest_indices_final = closest_indices_final.astype(int)
            I_above_final = I_above_final.astype(bool)
            
            start = time.time()
            # B. Get intersection points
            M_intersect, Qval_intersect, sigma_intersect, intersect_pairs\
                                = IntersectionPoints(M_clean,M_clean, gradM_clean, Qval_clean,\
                                                        closest_indices_final, I_above_final,\
                                                        new_sub_optimal_points,\
                                                        k1, sigma_clean,\
                                                        J_bar,\
                                                        interpolate = interp_intersect,
                                                        n_closest=n_closest)
            
            # Uncommment if you want to remove suboptimal points from the clean grid
            # that are found when searching for intersection jumps
            # get list of non-nan suboptimal points
            #new_sub_optimal_points = new_sub_optimal_points[~np.isnan(new_sub_optimal_points)]
            #if len(new_sub_optimal_points)>0:
            #    new_sub_optimal_points = new_sub_optimal_points.astype(int)
            #    mask = np.ones(M_clean.shape[0], dtype=bool)
            #    mask[new_sub_optimal_points] = False
            
            # Attach intersection points to the clean grids
            M_clean = np.vstack((M_clean, M_intersect))
            Qval_clean = np.vstack((Qval_clean, Qval_intersect))
            sigma_clean = np.vstack((sigma_clean, sigma_intersect))

    if intersection:
        return M_clean, Qval_clean, sigma_clean,M_intersect,Qval_intersect,sigma_intersect
    else:

        return M_clean, Qval_clean, sigma_clean, None, None,None
