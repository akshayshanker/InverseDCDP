"""
Computes the Euler errors and K regions for the benchmark pension savings
model by Druedhal and Jorgensen (2017) using RFC and compares to G2EGM.

Author: Akshay Shanker, University of New South Wales, Sydney
Email: akshay.shanker@me.com

Acknowledgements:
-----------------

The pensions module (G2EGM) is a modification 
of the original G2EGM mod written by Druedhal and Jorgensen. The code original 
was modified to:

 i) invert the Euler equation according to the necessary conditions 
    in Dobrescu and Shanker (2024) and 
 ii) implement roof-top cut algorithm. 
"""

import numpy as np
import dill as pickle
from G2EGM.G2EGMModel import G2EGMModelClass
from G2EGM.figs import euler_hist, Kregions, decision_functions
from timingPensions import timing

# Suppress all numpy errors (overflows, divisions by zero, etc.)
np.seterr(all='ignore')

if __name__ == '__main__':
    
    # Simulation parameters
    Nm = 800  # Grid points for each axis
    T = 20  # Periods
    k = 75  # Nearest neighbors points for RFC search 
    s = 0.05  # Proportion of the grid to be used in the RFC at each iteration of RFC
    rho_r = 0.33  # Max radius for RFC to eliminate points 
    rho_rI = 0.5  # Radius for RFC to search for intersections
    J_bar = 1 + 1e-5 # Jump detection threshold
    k1 = 30  # Neighbors for intersection point search
    k2 = 1  # Neighbors of uniform grid to construct triangulation for interpolation
    segplot_t = 13  # t for plotting constrained regions
    do_print = False
    
    # File paths for saving plots and data
    scrpath = '/scratch/tp66/dcdp/data/' # drive where raw egmgrids are saved 
    plotpath = 'plots/pensions/' 

    # Initialize and configure the RFC model
    model_RFC = G2EGMModelClass(
        name='RFC',
        par={
            'solmethod': 'RFC',
            'T': T,
            'do_print': do_print,
            'k': k,
            'Nm': Nm,
            'rad': rho_r,
            'rad_I': rho_rI,
            'J_bar': J_bar,
            'k1': k1,
            'k2': k2,
            'intersection': False,
            'interp_intersect': False,
            's': s,
            'correct_jumps': True, 
            't_save': segplot_t, 
            'do_print': do_print
        }
    )
    model_RFC.precompile_numba()
    model_RFC = timing(model_RFC, rep=5)

    # Initialize and configure the G2EGM model
    model_G2EGM = G2EGMModelClass(
        name='G2EGM',
        par={
            'solmethod': 'G2EGM',
            'T': T,
            'do_print': do_print,
            'Nm': Nm
        }
    )
    
    model_G2EGM.precompile_numba()
    model_G2EGM = timing(model_G2EGM, rep=5)

    # Load endogenous grid data
    egrids_intersect = pickle.load(open(f"{scrpath}/e_grids_intersect.pkl", "rb"))
    egrids_clean = pickle.load(open(f"{scrpath}/e_grids_clean.pkl", "rb"))
    egrids_raw = pickle.load(open(f"{scrpath}/e_grid_raw.pkl", "rb"))

    # Create a histogram of the Euler errors
    #smodelDict = {'RFC': model_RFC, 'G2EGM': model_G2EGM}
    #euler_hist(modelDict, Nm, plotpath)

    # Plot K regions
    Kregions(model_RFC, egrids_raw, egrids_clean, egrids_intersect, plotpath, segplot_t)
    decision_functions(model_RFC,3,'RFC_t3')

    models = [model_RFC, model_G2EGM]
    # tab

    postfix = '_G2EGM_vs_RFC'
    
    # b. euler erros
    lines = []
    txt = 'All (average)'
    for i,model in enumerate(models):
        txt += f' & {np.nanmean(model.sim.euler):.3f}'
    txt += '\\\\ \n'
    lines.append(txt)

    txt = '\\,\\,5th percentile'
    for i,model in enumerate(models):
        txt += f' & {np.nanpercentile(model.sim.euler,5):.3f}'
    txt += '\\\\ \n'    
    lines.append(txt)

    txt = '\\,\\,95th percentile'
    for i,model in enumerate(models):
        txt += f' & {np.nanpercentile(model.sim.euler,95):.3f}'
    txt += '\\\\ \n'   
    lines.append(txt)

    txt = '\\,\\,Median'
    for i,model in enumerate(models):
        txt += f' & {np.nanpercentile(model.sim.euler,50):.3f}'
    txt += '\\\\ \n'   
    lines.append(txt)

    with open(f'tabs_euler_errors{postfix}.tex', 'w') as txtfile:
        txtfile.writelines(lines)
        
    # c. timings
    lines = []
    txt = 'Total'
    for model in models:
        txt += f' & {np.sum(model.par.time_work)/60:.2f}'
    txt += '\\\\ \n'
    lines.append(txt)


    with open(f'tabs_timings{postfix}.tex', 'w') as txtfile:
        txtfile.writelines(lines)

