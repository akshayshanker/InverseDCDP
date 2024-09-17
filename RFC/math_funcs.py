import math
from numba import njit
import numpy as np
from scipy.interpolate import LinearNDInterpolator as linearND
from scipy.interpolate import NearestNDInterpolator as nearestND
from scipy.spatial import Delaunay
from pykdtree.kdtree import KDTree as KDTree1


# Define interpolation wrapper 
def interpolateEGM(x, z, grid,t, nearest_nans = True, switch_premergeT= 4):

    z = z.reshape((z.shape[0], -1))

    # Create a Delaunay triangulation of the data points
    if t <switch_premergeT:
        try:
            tri = Delaunay(x,qhull_options='Q6')
        except:
            #print("Delaunay failed")
            tri = Delaunay(x, incremental= False)
    else:
        try:
            tri = Delaunay(x,qhull_options='Q0 Qz')
        except:
            #print("Delaunay failed")
            tri = Delaunay(x, incremental= False)

    interp_function_ND = linearND(tri, z)
    policies = interp_function_ND(grid)
    
    #valid_data = ~np.isnan(z).any(axis=1)
    #invalid_data = np.isnan(z).any(axis=1)

    # Remove NaNs from the input data
    if nearest_nans and np.any(np.isnan(policies)):
        x = x[valid_data]
        z = z[valid_data]

        try:
            nan_inds = invalid_data
            nans_grid = x[nan_inds]
            clean_grid = x[~nan_inds]
            tree = KDTree1(clean_grid)
            dd, closest_indices = tree.query(nans_grid, 1)
            policies[nan_inds] = z[closest_indices]
        except:
            nearest_function_ND = nearestND(x,z)
            pols_nearest = nearest_function_ND(grid)
            policies[np.isnan(policies)] = pols_nearest[np.isnan(policies)]

    return policies

@njit
def rootsearch(f,a,b,dx, h_prime,z, Ud_prime_a, Ud_prime_h,t):
    x1 = a; f1 = f(a, h_prime,z, Ud_prime_a, Ud_prime_h,t)
    x2 = a + dx; f2 = f(x2, h_prime,z, Ud_prime_a, Ud_prime_h,t)
    
    while f1*f2 > 0.0:
        if x1 >= b:
            return np.nan,np.nan
        x1 = x2; f1 = f2
        x2 = x1 + dx; f2 = f(x2,h_prime,z, Ud_prime_a, Ud_prime_h,t)
        #print(x2)
    return x1,x2

def bisect(f,x1,x2,switch=0,epsilon=1.0e-9):
    f1 = f(x1)
    if f1 == 0.0:
        return x1
    f2 = f(x2)
    if f2 == 0.0:
        return x2
    if f1*f2 > 0.0:
        print('Root is not bracketed')
        return None
    n = int(math.ceil(math.log(abs(x2 - x1)/epsilon)/math.log(2.0)))
    for i in range(n):
        x3 = 0.5*(x1 + x2); f3 = f(x3)
        if (switch == 1) and (abs(f3) >abs(f1)) and (abs(f3) > abs(f2)):
            return None
        if f3 == 0.0:
            return x3
        if f2*f3 < 0.0:
            x1 = x3
            f1 = f3
        else:
            x2 =x3
            f2 = f3
    return (x1 + x2)/2.0


@njit 
def f(x):
    return x * np.cos(x-4)


@njit
def interp_as(xp, yp, x, extrap=True):
    """Function  interpolates 1D
    with linear extraplolation

    Parameters
    ----------
    xp : 1D array
            points of x values
    yp : 1D array
            points of y values
    x  : 1D array
            points to interpolate

    Returns
    -------
    evals: 1D array
            y values at x

    """

    evals = np.zeros(len(x))
    if extrap and len(xp) > 1:
        for i in range(len(x)):
            if x[i] < xp[0]:
                if (xp[1] - xp[0]) != 0:
                    evals[i] = yp[0] + (x[i] - xp[0]) * (yp[1] - yp[0])\
                        / (xp[1] - xp[0])
                else:
                    evals[i] = yp[0]

            elif x[i] > xp[-1]:
                if (xp[-1] - xp[-2]) != 0:
                    evals[i] = yp[-1] + (x[i] - xp[-1]) * (yp[-1] - yp[-2])\
                        / (xp[-1] - xp[-2])
                else:
                    evals[i] = yp[-1]
            else:
                evals[i] = np.interp(x[i], xp, yp)
    else:
        evals = np.interp(x, xp, yp)
    return evals


def upper_envelope(segments,  calc_crossings=False):
    """

    Cloned HARK line segment upper_envelope function  

    Finds the upper envelope of a list of non-decreasing segments
    Parameters
    ----------
    segments : list of segments. Segments are tuples of arrays, with item[0]
        containing the x coordninates and item[1] the y coordinates of the
        points that confrom the segment item.
    calc_crossings : Bool, optional
        Indicates whether the crossing points at which the "upper" segment
        changes should be computed. The default is True.
    Returns
    -------
    x : np.array of floats
        x coordinates of the points that conform the upper envelope.
    y : np.array of floats
        y coordinates of the points that conform the upper envelope.
    env_inds : np array of ints
        Array of the same length as x and y. It indicates which of the
        provided segments is the "upper" one at every returned (x,y) point.
    """
    n_seg = len(segments)

    # Collect the x points of all segments in an ordered array, removing duplicates
    x = np.unique(np.concatenate([x[0] for x in segments]))

    # Interpolate all segments on every x point, without extrapolating.
    y_cond = np.zeros((n_seg, len(x)))
    for i in range(n_seg):

        if len(segments[i][0]) == 1:
            # If the segment is a single point, we can only know its value
            # at the observed point.
            row = np.repeat(np.nan, len(x))
            ind = np.searchsorted(x, segments[i][0][0])
            row[ind] = segments[i][1][0]
        else:
            # If the segment has more than one point, we can interpolate
            row = np.interp(x,segments[i][0], segments[i][1])
            extrap = np.logical_or(x < segments[i][0][0], x > segments[i][0][-1])
            row[extrap] = np.nan

        y_cond[i, :] = row

    # Take the maximum to get the upper envelope.
    env_inds = np.nanargmax(y_cond, 0)
    y = y_cond[env_inds, range(len(x))]

    # Get crossing points if needed
    if calc_crossings:

        xing_points, xing_lines = calc_cross_points(x, y_cond, env_inds)

        if len(xing_points) > 0:

            # Extract x and y coordinates
            xing_x = np.array([p[0] for p in xing_points])
            xing_y = np.array([p[1] for p in xing_points])

            # To capture the discontinuity, we'll add the successors of xing_x to
            # the grid
            succ = np.nextafter(xing_x, xing_x + 1)

            # Collect points to add to grids
            xtra_x = np.concatenate([xing_x, succ])
            # if there is a crossing, y will be the same on both segments
            xtra_y = np.concatenate([xing_y, xing_y])
            xtra_lines = np.concatenate([xing_lines[:, 0], xing_lines[:, 1]])

            # Insert them
            idx = np.searchsorted(x, xtra_x)
            x = np.insert(x, idx, xtra_x)
            y = np.insert(y, idx, xtra_y)
            env_inds = np.insert(env_inds, idx, xtra_lines)

    return x, y, env_inds

@njit
def get_closest(sorted_indices, idx, p, n):
    start = max(idx - p // 2, 0)
    end = min(idx + p // 2 + 1, n)
    closest_range = sorted_indices[start:end]
    if len(closest_range) < p:
        if start == 0:
            closest_range = np.concatenate((closest_range, sorted_indices[:p - len(closest_range)]))
        else:
            closest_range = np.concatenate((sorted_indices[-(p - len(closest_range)):], closest_range))
    return closest_range[:p]

@njit
def find_closest_indices(data, p):
    n = data.shape[0]
    # Initialize the result arrays
    closest_indices_x = np.empty((n, p), dtype=np.int64)
    closest_indices_y = np.empty((n, p), dtype=np.int64)
    
    # Extract x and y coordinates
    x_coords = data[:, 0]
    y_coords = data[:, 1]
    
    # Sort indices based on x and y coordinates
    sorted_indices_x = np.argsort(x_coords)
    sorted_indices_y = np.argsort(y_coords)
    
    # Get original indices sorted by x and y
    sorted_x = x_coords[sorted_indices_x]
    sorted_y = y_coords[sorted_indices_y]
    
    # Find the p closest indices along each axis
    for i in range(n):
        point_x = x_coords[i]
        point_y = y_coords[i]

        # Locate the index of the point in the sorted arrays
        idx_x = np.searchsorted(sorted_x, point_x, side='left')
        idx_y = np.searchsorted(sorted_y, point_y, side='left')
        
        # Calculate ranges to slice out p closest points
        start_x = max(idx_x - p // 2, 0)
        end_x = min(start_x + p, n)
        start_x = max(end_x - p, 0)  # Adjust start if end overflow
        
        start_y = max(idx_y - p // 2, 0)
        end_y = min(start_y + p, n)
        start_y = max(end_y - p, 0)  # Adjust start if end overflow
        
        # Retrieve indices of the closest points
        closest_indices_x[i] = sorted_indices_x[start_x:end_x]
        closest_indices_y[i] = sorted_indices_y[start_y:end_y]

    return closest_indices_x, closest_indices_y
