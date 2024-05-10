from numba import njit
import numpy as np
import time
from scipy.interpolate import interp1d
from scipy.spatial import KDTree



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
    