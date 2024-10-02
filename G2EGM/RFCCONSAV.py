
"""
Druedhal and Jorgensen (2017) pension savings model modified to include RFC. 


1. Optimal constraint choice from the exogenous grid. 
Recall the inversion functions are: 

- invert_ucon: Invert unconstrained region
- invert_dcon: Invert d constrained region
- invert_acon: Invert a asset constrained region
- invert_con: Invert fully constrained region

Each inversion function is modified to eliminate points that do not satisfy 
the complementary slackness conditions as per Theorem 2 in Dobrescu and Shanker (2024).

To account for the case when p_L < \infty, the following region is added:
- conBar: Invert fully constrained region with y_p (d) = p_L and a = 0.

The region where y_p = p_L and a> 0 is accounted for by points in ucon.

2. Implementation of RFC. 

The RFC is implemented in the function rfc_upper_envelope_inplace over 
a combined set of points from the inversion functions that satisfy the
complementary slackness conditions and Euler equation inverse. 


"""


import numpy as np
from numba import njit
import os

from RFC.RFC import RFC 
from RFC.math_funcs import interpolateEGM

# consav
from consav import linear_interp # for linear interpolation
from consav.grids import nonlinspace_jit # grids

# local modules
import G2EGM.pens as pens
import G2EGM.utility as utility
import G2EGM.upperenvelope as upperenvelope
from G2EGM.upperenvelope import fill_holes

import time
from pykdtree.kdtree import KDTree as KDTree1
import dill as pickle
import gc

import numpy as np
from numba import njit


@njit
def calculate_gradient_1d(data, x):
    gradients = np.empty_like(data, dtype=np.float64)
    for i in range(1, len(data)):
        gradients[i] = (data[i] - data[i - 1]) / (x[i] - x[i - 1])
    gradients[0] = gradients[1]  # assuming continuous gradient at the start
    return gradients

@njit
def correct_jumps_gradient_2d(data, x, y, gradient_jump_threshold):
    """
    Removes jumps in a 2D array based on gradient jump threshold, ensuring that jumps must occur
    in both directions before correction, applied row-wise and column-wise.
    
    Args:
        data (numpy.ndarray): The input 2D data array.
        x (numpy.ndarray): The 1D array of x values corresponding to rows in `data`.
        y (numpy.ndarray): The 1D array of y values corresponding to columns in `data`.
        gradient_jump_threshold (float): Threshold for detecting jumps in gradient.
    
    Returns:
        numpy.ndarray: Corrected 2D data array.

    Notes:
        This is only an option to implement after interpolation 
        and is not a part of the RFC implementation. 
    """
    corrected_data = np.copy(data)
    
    # Correct row-wise using x values
    for row in range(data.shape[0]):
        gradients = calculate_gradient_1d(data[row], x)
        for col in range(1, data.shape[1]):
            left_jump = np.abs(gradients[col]) > gradient_jump_threshold
            right_jump = np.abs(gradients[col + 1]) > gradient_jump_threshold

            if (corrected_data[row - 1, col] <= 0 or corrected_data[row - 2, col] <= 0 or
                corrected_data[row - 3, col] <= 0) and (
                corrected_data[row + 1, col] <= 0 or corrected_data[row + 2, col] <= 0 or
                corrected_data[row + 3, col] <= 0 or corrected_data[row + 4, col] <= 0) and (
                corrected_data[row, col] > 0):
                corrected_data[row, col] = 0
            elif left_jump and right_jump:
                slope = (corrected_data[row, col - 1] - corrected_data[row, col - 2]) / (
                        x[col - 1] - x[col - 2])
                corrected_data[row, col] = corrected_data[row, col - 1] + slope * (x[col] - x[col - 1])
            elif corrected_data[row, col] == np.nan:
                slope = (corrected_data[row, col - 2] - corrected_data[row, col - 3]) / (
                        x[col - 2] - x[col - 3])
                corrected_data[row, col] = corrected_data[row, col - 2] + slope * (x[col] - x[col - 2])

    # Correct column-wise using y values
    for col in range(data.shape[1]):
        gradients = calculate_gradient_1d(data[:, col], y)
        for row in range(1, data.shape[0]):
            up_jump = np.abs(gradients[row]) > gradient_jump_threshold
            down_jump = np.abs(gradients[row + 1]) > gradient_jump_threshold

            if (corrected_data[row, col - 1] <= 0 or corrected_data[row, col - 2] <= 0 or
                corrected_data[row - 3, col] <= 0) and (
                corrected_data[row, col + 1] <= 0 or corrected_data[row, col + 2] <= 0 or
                corrected_data[row, col + 3] <= 0 or corrected_data[row, col + 4] <= 0) and (
                corrected_data[row, col] > 1e-10):
                corrected_data[row, col] = 0
            elif up_jump or down_jump:
                slope = (corrected_data[row - 1, col] - corrected_data[row - 2, col]) / (
                        y[row - 1] - y[row - 2])
                corrected_data[row, col] = corrected_data[row - 1, col] + slope * (y[row] - y[row - 1])
            elif corrected_data[row, col] == np.nan:
                slope = (corrected_data[row - 2, col] - corrected_data[row - 3, col]) / (
                        y[row - 2] - y[row - 3])
                corrected_data[row, col] = corrected_data[row - 2, col] + slope * (y[row] - y[row - 2])

    return corrected_data

def rfc_upper_envelope_inplace(n,m,v,c,b,d,gr_n,gr_m,par,t):

    """ 
    Wrapper function implements RFC and for the EGM and interpolation. 
    
    """

    nm_raw = np.column_stack((n,m))
    o_raw = np.column_stack((c, b,d))
    v_raw = np.array([v]).T
    gr_raw = np.column_stack((gr_n, gr_m))

    # remove NaNs and infs
    mask = ~np.isfinite(nm_raw).all(axis=1)
    nm_raw = nm_raw[~mask]
    o_raw = o_raw[~mask]
    v_raw = v_raw[~mask]
    gr_raw = gr_raw[~mask]

    # ray endogenous grids if saving 
    e_grids_clean = {}
    e_grids_intersect = {}
        
    start = time.time()

    print("Number of raw EGM points..{}.".format(len(nm_raw)))
    
    # clean grids via roof-top cut for periods before T-2 
    if t<par.T-2:

        
        out_rfc = RFC(nm_raw,gr_raw,v_raw,o_raw,par.grid,
                                        par.J_bar,par.rad, par.rad_I,
                                        par.GdistInds,
                                        par.k,par.k1, max_iter =par.max_iter,
                                        intersection= par.intersection,
                                        interp_intersect = par.interp_intersect,
                                        s= par.s, n_closest = par.n_closest)
        
        # name intersection points if intersection is True
        if par.intersection:
            nm_clean, vf_clean , o_clean,M_intersect_1, Qval_intersect_1,sigma_intersect_1 = out_rfc 
        else:
            nm_clean, vf_clean , o_clean,_,_,_ = out_rfc
        
    else:
        nm_clean = nm_raw
        vf_clean = v_raw
        o_clean = o_raw 
    rfc_time = time.time()-start
    # sort data 
    nm_clean_sort = np.lexsort((nm_clean[:,1], nm_clean[:,0]))
    nm_clean = nm_clean[nm_clean_sort]
    vf_clean = vf_clean[nm_clean_sort]
    o_clean = o_clean[nm_clean_sort]
    
    if par.save_data:
        e_grids_clean['n'] = nm_clean[:,0]
        e_grids_clean['m'] = nm_clean[:,1]
        e_grids_clean['v'] = vf_clean
        e_grids_clean['o'] = o_clean
            
    tree = KDTree1(nm_clean)
    dd, closest_indices = tree.query(par.grid,par.k2)
    closest_indices = np.unique(closest_indices.flatten())
    nm_clean_relerr = nm_clean[closest_indices]
    vf_clean_relerr = vf_clean[closest_indices]
    o_clean_relerr = o_clean[closest_indices]

    if t<par.T-2 and par.intersection == True:
        
        #nm_clean_relerr = np.vstack((nm_clean_relerr, M_intersect_1))
        #vf_clean_relerr = np.vstack((vf_clean_relerr, Qval_intersect_1))
        #o_clean_relerr = np.vstack((o_clean_relerr, sigma_intersect_1))
        
        if par.save_data:
            e_grids_intersect['n'] = M_intersect_1[:,0]
            e_grids_intersect['m'] = M_intersect_1[:,1]
            e_grids_intersect['v'] = Qval_intersect_1
            e_grids_intersect['o'] = sigma_intersect_1

    #start1 = time.time()
    # interpolate policies (note 0 is consumption c and 2 is end deposits d)
    policies_clean = interpolateEGM(nm_clean_relerr,\
                            np.column_stack([vf_clean_relerr,o_clean_relerr[:,[0,2]]]),\
                            par.grid,t,
                            nearest_nans=par.nearest_fill)
    #print('interp time: ',time.time()-start1)


    

    del nm_clean, nm_raw , o_clean, o_raw, v_raw, vf_clean, gr_raw
    gc.collect()
    
    return policies_clean, rfc_time, e_grids_clean, e_grids_intersect

@njit
def inv_mn_and_v(c,d,a,b,w,par):

    v = utility.func(c,par) + w
    m = a+c+d
    n = b-d-pens.func(d,par)

    return m,n,v

@njit
def deviate_d_con(valt,n,c,a,w,par):
    
    for i_b in range(par.Nb_pd):
        for i_a in range(par.Na_pd):
        
            # a. choices
            d_x = par.delta_con*c[i_b,i_a]
            c_x = (1.0-par.delta_con)*c[i_b,i_a]
        
            # b. post-decision states            
            b_x = n[i_b,i_a] + d_x + pens.func(d_x,par)

            if not np.imag(b_x) == 0:
                valt[i_b,i_a] = np.nan
                continue
            
            # c. value
            w_x = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,w,b_x,a[i_b,i_a])
            v_x = utility.func(c_x,par) + w_x

            if not np.imag(v_x) == 0:
                valt[i_b,i_a] = np.nan
            else:
                valt[i_b,i_a] = v_x

@njit
def invert_ucon(w,wa,wb,par):
    """ 

    Constrained region defined by:
    
    mu_f = mu_p = 0 since y_f >0 and p_bar>y_p > m_p

    """

    num = 1

    # i. choices
    c = utility.inv_marg_func(wa,par)
    d = (par.chi*wb)/(wa-wb)-1
    a = par.grid_a_pd_nd
    a_prime  = np.copy(par.grid_a_pd_nd)
    LL = 0
    
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if d[i,j] >= par.p_L:
                d[i,j] = par.p_L

            elif wa[i,j] - wb[i,j]<0 and np.abs(d[i,j])> par.p_L:
                d[i,j] = par.p_L
            
    # ii. states and value
    b = par.grid_b_pd_nd
    a_prime = a 
    wb_prime = np.zeros(w.shape)
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,wb,b.ravel(),a_prime.ravel(), wb_prime.ravel())

    m,n,v = inv_mn_and_v(c,d,a_prime,b,w,par)

    # iii. upperenvelope and interp to common
    gr_n_now = wb_prime
    gr_m_now = utility.marg_func(c,par)


    # Eliminate points that do not satisfy complementary slackness
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            if d[i,j] < 0:
                m[i,j] = np.nan
            elif d[i,j]>=0:
                mu_p =  gr_m_now[i,j] - wb[i,j]*(1+par.chi/(1+d[i,j]))
                if mu_p>0:
                    m[i,j] = np.nan
            

    return n,m,v,c,b,d,gr_n_now,gr_m_now

@njit
def invert_dcon(w,wa,wb,par):
    
    """ 

    Constrained region defined by:

    mu_f = 0, mu_p>=0, y_f>0, y_p= m_p 
    
    """

    num = 2

    # i. decisions                
    c = utility.inv_marg_func(wa,par) # implies mu_f = 0
    d = par.d_dcon
        
    # ii. states and value
    a = par.grid_a_pd_nd
    b = par.grid_b_pd_nd
    m,n,v = inv_mn_and_v(c,d,a,b,w,par)
    gr_n_now = wb
    gr_m_now = utility.marg_func(c,par)

    # Eliminate points that do not satisfy the complementary slackness
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            mu_p =  gr_m_now[i,j] - wb[i,j]*(1+par.chi) 
            if mu_p<0:
                m[i,j] = np.nan
            

    return n,m, v,c,b,d,gr_n_now, gr_m_now

@njit
def invert_dconBar(w,wa,wb,par):
    
    """ 

    Constrained region defined by:

    mu_f = 0, mu_pBar>=0, y_f>0, y_p= m_p 
    
    """

    num = 2

    # i. decisions                
    c = utility.inv_marg_func(wa,par) # implies mu_f = 0
    
    d = par.d_dcon + par.p_L
        
    # ii. states and value
    a = par.grid_a_pd_nd
    b = par.grid_b_pd_nd 
    
    m,n,v = inv_mn_and_v(c,d,a,b,w,par)
   
    gr_m_now = utility.marg_func(c,par)
    
    wb_acon = np.zeros(m.shape)
    gr_n_now = wb

    # Eliminate points that do not satisfy complementary slackness
    for i in range(n.shape[0]):
    
         for j in range(n.shape[1]):
            mu_p =  gr_m_now[i,j] - wb[i,j]*(1+par.chi/(1+d[i,j]))
            if mu_p>=0:
                m[i,j] = np.nan
            #if n[i,j]<1.5 or m[i,j]<1.5:
            #    m[i,j] = np.nan
            if (par.chi*wb[i,j])/(wa[i,j]-wb[i,j])-1>par.p_L:
                m[i,j] = np.nan

    return n,m, v,c,b,d,gr_n_now, gr_m_now

@njit
def invert_acon(w,wa,wb,par):

    """ 

    Constrained region defined by:

    mu_f >= 0, mu_p=0, y_f=0, y_p>=m_p 
    
    """

    num = 3
    # i. allocate
    c = np.zeros((par.Nc_acon,par.Nb_acon))
    d = np.zeros((par.Nc_acon,par.Nb_acon))
    a = np.zeros((par.Nc_acon,par.Nb_acon))
    b = np.zeros((par.Nc_acon,par.Nb_acon))
    w_acon = np.zeros((par.Nc_acon,par.Nb_acon))
    wb_acon = np.zeros((par.Nc_acon,par.Nb_acon))
    wa_acon = np.zeros((par.Nc_acon,par.Nb_acon))

    for i_b in range(par.Nb_acon):
        
        # ii. setup
        wb_acon_float = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,wb,par.b_acon[i_b],0)
        wb_acon[:,i_b] = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,wb,par.b_acon[i_b],0)

        c_min = utility.inv_marg_func((par.chi+1)*wb_acon_float,par)
        c_max = utility.inv_marg_func(wb_acon_float,par)

        c[:,i_b] = nonlinspace_jit(c_min,c_max,par.Nc_acon,par.phi_m)
        
        # iii. choices
        d[:,i_b] = par.chi/(utility.marg_func(c[:,i_b],par)/(wb_acon_float)-1)-1 # ensures mu_p = 0

        for l in range((len(d[:,i_b]))):
            if d[l,i_b] > par.p_L:
                d[l,i_b] = par.p_L
                c[l,i_b] = c[l,i_b]  + d[l,i_b] - par.p_L
                        
        # iv. post-decision states and value function
        b[:,i_b] = par.b_acon[i_b]
        w_acon[:,i_b] = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,w,par.b_acon[i_b],0)
        
        wa_acon[:,i_b] = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,wa,par.b_acon[i_b],0)

                
    # v. states and value
    m,n,v = inv_mn_and_v(c,d,a,b,w_acon,par)

    gr_n_now = wb_acon
    gr_m_now = utility.marg_func(c,par)

    # Eliminate points that do not satisfy complementary slackness
    for i in range(n.shape[0]):       
       for j in range(n.shape[1]):
            mu_f = gr_m_now[i,j] - wa_acon[i,j]
            if mu_f<0:
                m[i,j] = np.nan
            if d[i,j] > par.p_L:
                m[i,j] = np.nan
            
    # vi. upperenvelope and interp to common
    return n,m, v,c,b,d, gr_n_now, gr_m_now

@njit
def invert_con(w,wa,wb,par):

    """ 

    Constrained region defined by:

    mu_f >= 0, mu_p>=0, y_f=0, y_p=m_p 
    
    """
                        
    # i. choices
    c = par.grid_m_nd 
    d = np.zeros(c.shape)
        
    # ii. post-decision states
    a = np.zeros(c.shape)
    b = par.grid_n_nd

    # iii. post decision value
    w_con = np.zeros(c.shape)
    wb_con = np.zeros(c.shape)
    wa_con = np.zeros(c.shape)
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,wa,b.ravel(),a.ravel(),wa_con.ravel())
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,w,b.ravel(),a.ravel(),w_con.ravel())
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,wb,b.ravel(),a.ravel(),wb_con.ravel())

    # iv. value
    v = utility.func(c,par) + w_con     

    gr_n_now = wb_con
    gr_m_now = utility.marg_func(c,par)
    n = np.copy(par.grid_n_nd)
    m = np.copy(par.grid_m_nd)

    # Eliminate points that do not satisfy complementary slackness
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            mu_a = gr_m_now[i,j] - wa_con[i,j]
            if d[i,j] < 0:
                m[i,j] = np.nan
            
            if mu_a<0:
                m[i,j] = np.nan
            
            mu_p = gr_m_now[i,j]  - wb_con[i,j]*(1+par.chi/(1+d[i,j]))
            if mu_p<0:
                m[i,j] = np.nan
                
    return n,m,v,c,b,d,gr_n_now, gr_m_now

@njit
def invert_conBar(w,wa,wb,par):

    """ 

    Constrained region defined by:

    mu_f >= 0, mu_pBar>=0, y_f=0, y_p=p_L
    
    """
                        
    # i. choices
    c = par.grid_m_nd - par.p_L
    d = np.zeros(c.shape) + par.p_L
    #d = np.full(c.shape, par.p_L)
        
    # ii. post-decision states
    a = np.zeros(c.shape)
    b = par.grid_n_nd  + pens.func(par.p_L,par)

    # iii. post decision value
    w_con = np.zeros(c.shape)
    wb_con = np.zeros(c.shape)
    wa_con = np.zeros(c.shape)
    
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,wa,b.ravel(),a.ravel(),wa_con.ravel())
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,w,b.ravel(),a.ravel(),w_con.ravel())
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,wb,b.ravel(),a.ravel(),wb_con.ravel())

    # iv. value
    v = utility.func(c,par) + w_con     

    gr_n_now = wb_con
    gr_m_now = utility.marg_func(c,par)
    n = np.copy(par.grid_n_nd)
    m = np.copy(par.grid_m_nd)

    # Eliminate points that do not satisfy complementary slackness
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            m[i,j] = np.nan
            mu_a = gr_m_now[i,j] - wa_con[i,j]
            if mu_a<0:
                m[i,j] = np.nan
            
            mu_p = gr_m_now[i,j]  - wb_con[i,j]*(1+par.chi/(1+d[i,j]))
            if mu_p>0:
                m[i,j] = np.nan
                
    return n,m,v,c,b,d,gr_n_now, gr_m_now


@njit
def apply_mask(n, m,valid_mask):
    for i in range(n.shape[0]):  # Assuming n.shape[0] is the number of rows
        for j in range(n.shape[1]):  # Assuming n.shape[1] is the number of columns
            if not valid_mask[i, j]:
                n[i, j] = np.nan
                m[i, j] = np.nan

@njit 
def clean_with_mask(n,m,v,c,b,d, gr_n, gr_m ):

    mask = np.where(np.isnan(n))
    n = n[~mask]
    m = m[~mask]
    v = v[~mask]
    c = c[~mask]
    d = d[~mask]
    gr_n = gr_n[~mask]
    gr_m = gr_m[~mask]
    return n,m,v,c,b,d, gr_n, gr_m

@njit
def combine_invert(w,wa,wb,par):

    """ 
    Inverts exogenous grid regions and combines admissible inverse points
    into one set of arrays. 
    
    """

    n,m,v,c,b,d, gr_n, gr_m = invert_ucon(w,wa,wb,par)
    valid_mask = upperenvelope.compute_valid_only(m,n,c,d,v,par)
    apply_mask(n, m, valid_mask)

    n_d,m_d,v_d,c_d,b_d,d_d, gr_n_d, gr_m_d = invert_dcon(w,wa,wb,par)
    valid_mask = upperenvelope.compute_valid_only(m_d,n_d,c_d,d_d,v_d,par)
    apply_mask(n_d, m_d,valid_mask)

    n_a,m_a,v_a,c_a,b_a,d_a, gr_n_a, gr_m_a = invert_acon(w,wa,wb,par)
    valid_mask = upperenvelope.compute_valid_only(m_a,n_a,c_a,d_a,v_a,par)
    apply_mask(n_a, m_a,  valid_mask)

    n_c,m_c,v_c,c_c,b_c, d_c, gr_n_c, gr_m_c = invert_con(w,wa,wb,par)
    valid_mask = upperenvelope.compute_valid_only(m_c,n_c,c_c,d_c,v_c,par)
    apply_mask(n_c, m_c, valid_mask)
    
    if par.p_L<100:
        #n_dBar,m_dBar,v_dBar,c_dBar,b_dBar,d_dBar, gr_n_dBar, gr_m_dBar = invert_dconBar(w,wa,wb,par)
        #valid_mask = upperenvelope.compute_valid_only(m_dBar,n_dBar,c_dBar,d_dBar,v_dBar,par)
        #apply_mask(n_dBar, m_dBar,valid_mask)

        n_cBar,m_cBar,v_cBar,c_cBar,b_cBar,d_cBar, gr_n_cBar, gr_m_cBar = invert_conBar(w,wa,wb,par)
        valid_mask = upperenvelope.compute_valid_only(m_cBar,n_cBar,c_cBar,d_cBar,v_cBar,par)
        apply_mask(n_cBar, m_cBar,valid_mask)

        n1 = np.concatenate((n.ravel(), n_d.ravel(), n_cBar.ravel(),\
                                 n_a.ravel(), n_c.ravel()), axis=0)
        m1 = np.concatenate((m.ravel(), m_d.ravel(),  m_cBar.ravel(),\
                                 m_a.ravel(), m_c.ravel()), axis=0)
        v1 = np.concatenate((v.ravel(), v_d.ravel(), v_cBar.ravel(),\
                                v_a.ravel(), v_c.ravel()), axis=0)
        c1 = np.concatenate((c.ravel(), c_d.ravel(), c_cBar.ravel(),\
                                 c_a.ravel(), c_c.ravel()), axis=0)
        d1 = np.concatenate((d.ravel(), d_d.ravel(),d_cBar.ravel(),\
                                d_a.ravel(), d_c.ravel()), axis=0)
        gr_m1 = np.concatenate((gr_m.ravel(), gr_m_d.ravel(),gr_m_cBar.ravel(),\
                                     gr_m_a.ravel(), gr_m_c.ravel()), axis=0)
        gr_n1 = np.concatenate((gr_n.ravel(), gr_n_d.ravel(), gr_n_cBar.ravel(),\
                                    gr_n_a.ravel(), gr_n_c.ravel()), axis=0)
        b1 = np.concatenate((b.ravel(), b_d.ravel(), b_cBar.ravel(),\
                                 b_a.ravel(), b_c.ravel()), axis=0)
    
    else:
        n1 = np.concatenate((n.ravel(), n_d.ravel(), n_a.ravel(), n_c.ravel()), axis=0)
        m1 = np.concatenate((m.ravel(), m_d.ravel(), m_a.ravel(), m_c.ravel()), axis=0)
        v1 = np.concatenate((v.ravel(), v_d.ravel(), v_a.ravel(), v_c.ravel()), axis=0)
        c1 = np.concatenate((c.ravel(), c_d.ravel(), c_a.ravel(), c_c.ravel()), axis=0)
        d1 = np.concatenate((d.ravel(), d_d.ravel(), d_a.ravel(), d_c.ravel()), axis=0)
        gr_m1 = np.concatenate((gr_m.ravel(), gr_m_d.ravel(), gr_m_a.ravel(), gr_m_c.ravel()), axis=0)
        gr_n1 = np.concatenate((gr_n.ravel(), gr_n_d.ravel(),gr_n_a.ravel(), gr_n_c.ravel()), axis=0)
        b1 = np.concatenate((b.ravel(), b_d.ravel(), b_a.ravel(), b_c.ravel()), axis=0)


    return n1,m1,v1,c1,b1,d1,gr_n1,gr_m1

@njit
def post_prep(t,sol,par, policies):
    w = sol.w[t]
    
    sol.c[t,:,:] = correct_jumps_gradient_2d(policies[:,:,1],\
                                                par.grid_m_nd.reshape((par.Nn,par.Nm))[1,:],\
                                                par.grid_n_nd.reshape((par.Nn,par.Nm))[:,1],par.J_bar)
    sol.d[t,:,:] = correct_jumps_gradient_2d(policies[:,:,2],\
                                                par.grid_m_nd.reshape((par.Nn,par.Nm))[1,:],\
                                            par.grid_n_nd.reshape((par.Nn,par.Nm))[:,1], par.J_bar)
    

    # replace NaNs in value function wth continuation value
    b_end = par.grid_n_nd + sol.d[t,:,:] + pens.func(sol.d[t,:,:],par)
    a_end = par.grid_m_nd - sol.c[t,:,:] - sol.d[t,:,:]
    w_nxt = np.zeros(sol.c[t,:,:].shape)
    
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,\
                                w,b_end.ravel(),a_end.ravel(),w_nxt.ravel())
    
    continuation_value = w_nxt.reshape((par.Nn,par.Nm))\
                                +  utility.func(sol.c[t,:,:],par) 
    
    for i in range(policies.shape[0]):
       for j in range(policies.shape[1]):
            if np.isnan(policies[i, j, 0]):
                policies[i, j, 0] = continuation_value[i, j]

    holes = np.zeros((par.Nn,par.Nm))

    out_c = sol.c[t,:,:]
    out_d = sol.d[t,:,:]
    out_v = policies[:,:,0]
    
    for i in range(holes.shape[0]):
        for j in range(holes.shape[1]):
            if np.isnan(out_c[i, j]):
                holes[i, j] = 1
            if np.isnan(out_d[i, j]):
                holes[i, j] = 1
    Iucon = holes == 1

    fill_holes(out_c,out_d,out_v,Iucon,w,1,par)

    return out_c, out_d, out_v

#@njit
def solve(t,sol,par):

    w = sol.w[t]
    wa = sol.wa[t]
    wb = sol.wb[t]

    # a. invert each segment
    start = time.time()
    n,m,v,c,b,d,gr_n,gr_m = combine_invert(w,wa,wb,par)
    time_invert = time.time()-start

    start = time.time()
    policies, rfc_time, e_grids_clean, e_grids_intersect = rfc_upper_envelope_inplace(n,m,v,c,b,d,gr_n,gr_m,par,t)
    policies = policies.reshape((par.Nn,par.Nm,3))

    if par.save_data:
        e_grid_raw = {}
        e_grid_raw['n'] = n
        e_grid_raw['m'] = m
        e_grid_raw['v'] = v
        e_grid_raw['c'] = c
        e_grid_raw['d'] = d
        e_grid_raw['gr_n'] = gr_n
        e_grid_raw['gr_m'] = gr_m

    
    if par.correct_jumps:
        """ 
        Remove interpolated points between future choice specific policy functions
        by replacing with local extrapolants. 
        
        """

        policies[:,:,2][np.isnan(policies[:,:,2])] = 0
        sol.d[t,:,:][sol.d[t,:,:]<1e-10] = 0
        sol.d[t,:,:][sol.d[t,:,:]>par.p_L] = par.p_L
        out_c, out_d, out_v = post_prep(t,sol,par, policies)

        sol.d[t,:,:][sol.d[t,:,:]<1e-10] = 0
        sol.d[t,:,:][sol.d[t,:,:]>par.p_L] = par.p_L
        sol.c[t,:,:][sol.c[t,:,:]<1e-100] = 1e-100
        
    else:
        sol.c[t,:,:] = policies[:,:,1]
        sol.d[t,:,:] = policies[:,:,2]
    
    time_RFC = time.time()-start
    
    sol.inv_v[t,:,:] = - 1.0/policies[:,:,0]

    # c. derivatives 
    
    # i. m
    vm = utility.marg_func(sol.c[t],par)
    sol.inv_vm[t,:,:] = 1.0/vm

    # ii. n         
    a = par.grid_m_nd - sol.c[t] - sol.d[t]
    b = par.grid_n_nd + sol.d[t] + pens.func(sol.d[t],par)

    wb_now = np.zeros(a.shape)
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,wb,b.ravel(),a.ravel(),wb_now.ravel())
    
    vn = wb_now
    sol.inv_vn[t,:,:] = 1.0/vn

    if t == par.t_save and par.save_data:
        if not os.path.exists(par.scrpath):
            os.makedirs(par.scrpath)
        pickle.dump(e_grid_raw, open('{}/e_grid_raw.pkl'.format(par.scrpath), 'wb'))
        pickle.dump(e_grids_clean, open('{}/e_grids_clean.pkl'.format(par.scrpath), 'wb'))
        pickle.dump(e_grids_intersect, open('{}/e_grids_intersect.pkl'.format(par.scrpath), 'wb'))
        #print("Saved EGM grids to "

    return time_invert, rfc_time