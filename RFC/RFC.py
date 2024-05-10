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


@njit
def _closestPairs(M, intersect_pairs, n_closest=4):
    """
    Given a list of  pairs (2-tuples), this function finds the pairs that agree with the
    the n_closest in terms of distance in M to each point in the second column

    Parameters
    ----------
    M : 2D array
        The grid points.
    intersect_pairs : 2D array
        The pairs of indices of the grid points.
    n_closest : int
        The number of closest points to consider.
    
    Returns
    -------
    new_intersect_pairs : 2D array
        The closest pairs of indices of the grid points.

    Notes
    -----
    This function is used to find the closest points to the intersection points
    where the second grid point in a pair is the point with the value function that
    is above the tangent plane from the first grid point in the pair.

    Restricting intersections to closest pairs only implemented
    for 2 dimensional grids. 

    """
    
    # Initialize a new list to store the modified intersect pairs
    new_intersect_pairs = []

    # For each unique j in intersect_pairs
    for j in np.unique(intersect_pairs[:, 1]):
        # Get the indices of all points s.t. j is above the tangent plane from the point
        # or `connected points`
        connected_points = intersect_pairs[intersect_pairs[:, 1] == j, 0]
        
        # Calculate the distances from j to the connected points
        connected_distances = np.sqrt(np.sum((M[connected_points] - M[j])**2, axis=-1))
        
        # Find the indices of the two closest points
        closest_points = connected_points[np.argsort(connected_distances)[:n_closest]]
        
        # Add the pairs (l, j) and (k, j) to the new list
        new_intersect_pairs.extend([(l, j) for l in closest_points])

    # Convert the list to a numpy array
    new_intersect_pairs = np.array(new_intersect_pairs)

    return new_intersect_pairs


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
    Md = M[closest_indices].reshape(n * (k-1), d) - np.repeat(M_test, k-1, axis=0) 
    Mdn = Md.reshape(n, k-1, d)

    # Get tangent plane values for neighboring points
    tngv = Mdn @ gradrs
    tngv = tngv.reshape(n, k-1)

    # Difference between value function at each point and its neighbors
    vdiff = Qval[closest_indices].reshape(n, k-1) - Qval[test_inds]
   
    # Indicator matrix (n x k-1) of neighboring if point below tangent plane
    I1 = vdiff < tngv 
    
    # 2. Check for jump in policy
    
    # Obtain differences between grid distance func vals and neighbors
    Pd = sigma[closest_indices].reshape(n * (k-1), dp) - np.repeat(sigma[test_inds], k-1, axis=0) 

    # Calculating pointwise delsig and checking if jump in any of the grid dist has jumps
    Pd_abs = np.abs(Pd)
    Md_norm = dd[:, 1:]
    delsig = Pd_abs / (Md_norm.reshape(-1, 1))
    jump_any = np.sum(delsig > J_bar, axis=1) > 0
    
    # Indicator matrix (n x k-1) for each grid point and neighbor if jump in any of the policies
    I2 = jump_any.reshape(n, k-1)

    # Only evaluate neighbors within radius
    I3 = dd[:, 1:] < radius

    # Select suboptimal points
    sub_optimal_points = np.unique(I1 * I2 * I3 * (closest_indices + 1))
    sub_optimal_points = sub_optimal_points[1:] - 1

    # Indicator array for neighbors that lie above the tangent plane from test_inds
    if intersect:
        I_above = I2 * (dd[:, 1:] < radius_I) * (vdiff > tngv)
    else:
        I_above = None

    return sub_optimal_points, I_above, closest_indices


def IntersectionPoints(M,M_old,gradM, Qval, closest_indices, I_above,\
                        sub_optimal_points, k,sigma, J_bar,\
                        factor_n = 10, interpolate= True, n_closest=1):
    
    d = M.shape[1]  #dimension of grid
    n = M.shape[0]
    dp = sigma.shape[1]

    #sub select ki nearest neighbours (ki <= k-1)
    ki = k-1
    
    original_shape = closest_indices.shape
    flattened_indices = np.array(closest_indices.flatten()).astype(int)
    sub_optimal_points = np.array(sub_optimal_points).astype(int)
    start = time.time()
    #flattened_result = in1d_vec_nb(flattened_indices, np.array(sub_optimal_points))
    flattened_result = np.isin(flattened_indices, sub_optimal_points)
    #print(f"Time taken for  setyup: {time.time() - start:.2f} seconds")
    result = flattened_result.reshape(original_shape)
    intersect_neighbours = closest_indices * (1 - result) * I_above
    
    
    # Check if any value in each row is non-zero
    rows_have_non_zero = np.any(intersect_neighbours != 0, axis=1)
    # Get indices of rows that have at least one non-zero value
    non_zero_row_indices = np.where(rows_have_non_zero)[0]
    
    #non_zero_row_indices = np.setdiff1d(non_zero_row_indices, sub_optimal_points)
    
    intersect_sub = intersect_neighbours[non_zero_row_indices,:]
    
    # Convert to boolean: True for non-zero, False for zero
    bool_array = intersect_sub != 0
    # Find the index of the first non-zero value in each row
    first_non_zero_index = np.argmax(bool_array, axis=1)
    
    # Extract the first non-zero value from each row
    first_non_zero = intersect_sub[np.arange(intersect_sub.shape[0]), first_non_zero_index]


    intersect_pairs = np.c_[non_zero_row_indices,first_non_zero]
    intersect_pairs = np.sort(intersect_pairs, axis=1)
    intersect_pairs = np.unique(intersect_pairs, axis = 0)
    
    start = time.time()
    #print(intersect_pairs)
    intersect_pairs = _closestPairs(M, intersect_pairs, n_closest = n_closest)
    #print(f"Time taken for  _closestPairs: {time.time() - start:.2f} seconds")
    npairs = intersect_pairs.shape[0]
    # Computing the equal tangent equation
    x1s = M[intersect_pairs[:,0]]
    x2s = M[intersect_pairs[:,1]]

    # keep only intersect pais such that for each M in the first col, the second col is the closest neighbour
    # to the first col
    #M_inter_d = np.linalg.norm(x1s-x2s,axis=1)
    
    
    M_inter_d = x1s-x2s
    M_inter_d = M_inter_d.reshape(npairs,1,d)

    grad1s = gradM[intersect_pairs[:,0],:]
    grad1s_rs = grad1s.reshape(npairs,d,1)
    grad2s = gradM[intersect_pairs[:,1],:]
    grad_diff = grad2s-grad1s
    grad_diff_rs = grad_diff.reshape(npairs,d,1)

    denom = M_inter_d@grad_diff_rs
    denom = denom.reshape(npairs)

    y1s = Qval[intersect_pairs[:,0]]
    y2s = Qval[intersect_pairs[:,1]]
    y_diffs = y1s-y2s
    y_diffs = y_diffs.reshape(npairs)

    lterm = M_inter_d@grad1s_rs
    lterm = lterm.reshape(npairs)

    # Solve for t in the equal tangent equation
    t = (y_diffs-lterm)/denom
    t = t.reshape(npairs,1)
    

    #Removing t that lie outside interval [0,1]
    # Boolean array for values in the interval [0, 1] in the specified column
    in_interval = (t[:, 0] >= 0) & (t[:, 0] <= 1)
    # Find row indices where the condition is True
    t_row_indices = np.where(in_interval)[0]

    #Point on the line segment x1,x2 that satisfy the equal tangent equation
    M_intersect = x1s*t+(1-t)*x2s
    M_intersect1 = M_intersect[~np.isnan(M_intersect).any(axis=1)]

    # find closest point in grid to intersect 
    #tree2 = KDTree1(grid)
    #ddM, closest_indicesM = tree2.query(M_intersect1, 1)
    #print(closest_indicesM.shape)
    #closest_indicesM = np.array(closest_indicesM).astype(int)
    #M_intersect[~np.isnan(M_intersect).any(axis=1)] = grid[closest_indicesM]

    if interpolate:
        M_inter_neighbour,M_intersect, Qval_intersect,sigma_inter_neighbour =\
              _get_interp_neighbours(M_old, M_intersect,grad1s,t,sigma,\
                                      intersect_pairs,x1s,y1s,x2s,t_row_indices,\
                                       npairs,factor_n,d,dp, J_bar)
        start = time.time()

        ####Interpolation#########
        sigma_intersect = _interp_intesections(M_intersect, M_inter_neighbour,sigma_inter_neighbour,d,dp)
    else: 
        sigma_intersect = np.empty((M_intersect.shape[0],dp))
        sigma_intersect[:] = np.nan
        Qval_intersect = np.empty((M_intersect.shape[0],1))
        Qval_intersect[:] = np.nan

    # remove any nan in M_intersect if either of the poitns are nans along rows
    mask = np.isnan(M_intersect).any(axis=1)
    M_intersect = M_intersect[~mask]
    Qval_intersect = Qval_intersect[~mask]
    sigma_intersect = sigma_intersect[~mask]

    return M_intersect, Qval_intersect, sigma_intersect, intersect_pairs

def _get_interp_neighbours(M_old,M_intersect,grad1s, t,sigma, intersect_pairs,x1s,y1s,x2s,t_row_indices,npairs,factor_n,d,dp,J_bar):
    
    X_diff = M_intersect-x1s
    #Obtain Qval by evaluating values on tangent
    Qval_intersect = (X_diff.reshape(npairs,1,d)@grad1s.reshape(npairs,d,1)).reshape(npairs,1)+y1s

    M_intersect =  M_intersect[t_row_indices,:]
    Qval_intersect = Qval_intersect[t_row_indices,:]
    intersect_pairs = intersect_pairs[t_row_indices,:]

    t_closest = (t[t_row_indices]*2-1)>0
    x_closest = x1s[t_row_indices]*t_closest+(1-t_closest)*x2s[t_row_indices]
    
    min_number_of_neighbours = (d+2)
    #number of neighbours to search around intersection point
    kp = int(min_number_of_neighbours*factor_n)
    
    tree = KDTree1(M_old, leafsize=70)
    dd_pi, closest_indices_pi = tree.query(M_intersect, kp)
    n_pi = t_closest.shape[0]

    # Array of column indices to select, one for each row
    # For example, to select the 1st column from the 1st row, the 3rd from the 2nd, and the 2nd from the 3rd:
    column_indices = (1-t_closest).ravel()
    # Create an array of row indices
    row_indices = np.arange(intersect_pairs.shape[0])
    # Index the 2D array to get specific elements
    selected_elements = intersect_pairs[row_indices, column_indices]


    #Calculating pointwise delsig and checking if jump in any of the policies has jumps
    Pd_pi = sigma[closest_indices_pi].reshape(n_pi*(kp),dp) - np.repeat(sigma[selected_elements], kp, axis=0) # obtain differences between poicy vals and neighbours
    Pd_pi_abs = np.abs(Pd_pi)
    Md_pi = M_old[closest_indices_pi].reshape(n_pi*(kp),d)- np.repeat(M_old[selected_elements], kp, axis=0) # obtain difference between

    Md_pi_norm = np.linalg.norm(Md_pi,axis=1)

    delsig_pi = Pd_pi_abs/(Md_pi_norm.reshape(-1, 1))
    #Check if each policy has jumped
    jump_all_pi = delsig_pi>=J_bar 
    #Check if jump occurs at least once across any of the policies
    jump_any_pi = np.sum(delsig_pi>=J_bar,axis = 1)>0
    #Indicator matrix (n x k-1) for each grid point and neighbour if jump in any of the policies

    #Indicator matrix (n x k-1) for each grid point if no jump in policy
    I2_pi =  1-jump_any_pi.reshape(n_pi,kp)


    ##Extracting indices of neighbours that are on the same discrete choice
    array = I2_pi*closest_indices_pi
    array = array.astype(float) 
    non_zero_mask = array != 0
    
    #change right hand side to include more points in the extrapolation/interpolation
    k_mask = np.cumsum(non_zero_mask, axis=1) <= min_number_of_neighbours
   
    final_mask = non_zero_mask & k_mask
    result = np.full((array.shape[0], min_number_of_neighbours), np.nan)
    for i in range(array.shape[0]):
        result[i, :np.sum(final_mask[i])] = array[i, final_mask[i]]
    result = result.astype(int)

    # mask if any row of results is nan
    mask = np.isnan(result).any(axis=1)
    # add mask if any result is leses than 0
    mask1 = (result<0).any(axis=1)
    mask = mask | mask1
    result = result[~mask]
    M_intersect = M_intersect[~mask]
    Qval_intersect = Qval_intersect[~mask]
    #Index grid points and policies for neighbours on same discrete choice\
    M_inter_neighbour= M_old[result]
    sigma_inter_neighbour = sigma[result]
    #Qval_intersect = 
    #Qval_intersect = Qval_intersect

    return M_inter_neighbour,M_intersect,Qval_intersect,sigma_inter_neighbour

def _interp_intesections(M_intersect, M_inter_neighbour,sigma_inter_neighbour,d,dp):

    """ 
    Interpolates policy at CS=VF intersection points using the optimal policy
    values at the neighbouring points on the grid.

    Parameters
    ----------
    M_intersect : 2D array
        Intersection points on the grid.
    M_inter_neighbour : 3D array
        Neighbouring points on the grid for each intersection point.
    sigma_inter_neighbour : 3D array
        Policy values at each neighbouring point for each intersection point.
    
    Returns
    -------
    sigma_intersect : 2D array
        Interpolated policy values at each intersection point.
    
        
    
    """

     
    nrows = M_inter_neighbour.shape[0]
    nneighbours = M_inter_neighbour.shape[1]
    p_interp = np.zeros((nrows,dp))
    
    for j in range(M_inter_neighbour.shape[0]):
        Y = sigma_inter_neighbour[j]
        if d == 1:
            X = M_inter_neighbour[j].reshape(nneighbours)
            interp = interp1d(X, Y.T)
            p_interp[j,:] = interp(M_intersect[j,0]).T
        else:
            X = M_inter_neighbour[j]

            try:

                p_interp[j,:] = interpolateEGM(X[:,0],X[:,1],Y, M_intersect[j,0], M_intersect[j,1])
                p_interp[j,:] = interp(M_intersect[j,:])

            except:
                p_interp[j,:] = Y[0].reshape(1,dp)

    sigma_intersect = p_interp


    return sigma_intersect
    
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
