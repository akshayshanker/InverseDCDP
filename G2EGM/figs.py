import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
plt.rcParams.update({"axes.grid" : True, "grid.color": "black", "grid.alpha":"0.25", "grid.linestyle": "--"})
plt.rcParams.update({'font.size': 14})

color1 = np.array([3.0/255.0,103.0/255.0,166.0/255.0])
color2 = np.array([242.0/255.0,62.0/255.0,46.0/255.0])
color3 = np.array([3.0/255.0,166.0/255.0,166.0/255.0])
color4 = np.array([242.0/255.0,131.0/255.0,68.0/255.0])
color5 = np.array([242.0/255.0,242.0/255.0,68.0/255.0])
color6 = np.array([242.0/255.0,68.0/255.0,68.0/255.0])

sns.set(style="white", rc={
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})

def retirement(model):
    
    # a. unpack
    par = model.par
    sol = model.sol

    # b. settings
    fig_max_m = 5

    # c. figure
    for varname in ['c_ret','inv_v_ret']:
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        for t in range(par.T):
            I = sol.m_ret[t] < fig_max_m
            y = getattr(sol,varname)[t,I]
            ax.plot(sol.m_ret[t,I],y)

        # details
        ax.set_title(varname)
        ax.grid(True)
        ax.set_xlabel('$m_t$')
        ax.set_xlim([0,fig_max_m])

    plt.show()

def decision_functions(model,t, title):

    # a. settings
    fig_max_m = 5
    fig_max_n = 5
        
    # b. unpack
    par = model.par
    sol = model.sol
    
    # c. varnames
    for varname in ['c','d', 'inv_v']:
        
        if 'w' in varname:

            if t == par.T-1:
                continue
            else:
                x = par.grid_a_pd_nd.ravel()
                y = par.grid_b_pd_nd.ravel()
                I = (x < fig_max_m) & (y < fig_max_n)
            
        else:
            
            x = par.grid_m_nd.ravel()
            y = par.grid_n_nd.ravel()
            I = (x < fig_max_m) & (y < fig_max_n)
                
        # i. value
        value = getattr(sol,varname)[t].ravel()
        
        # ii. figure
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.scatter(x[I],y[I],value[I],s=1,c=value[I],cmap=cm.viridis)
        
        # iii. details
        ax.set_title(f't = {t}, {varname}')
        ax.grid(True)
        ax.set_xlabel('$m_t$')
        ax.set_xlim([0,fig_max_m])
        ax.set_ylabel('$n_t$')
        ax.set_xlim([0,fig_max_m])
    
        plt.savefig('plots/pensions/decision_functions_{}_{}.png'.format(title,varname))

def decision_functions_diff(model,model2, t, title):
    
        # a. settings
        fig_max_m = 5
        fig_max_n = 5
            
        # b. unpack
        par = model.par
        sol = model.sol
        sol2 = model2.sol
        
        # c. varnames
        for varname in ['c','d', 'inv_v']:
            
            if 'w' in varname:
    
                if t == par.T-1:
                    continue
                else:
                    x = par.grid_a_pd_nd.ravel()
                    y = par.grid_b_pd_nd.ravel()
                    I = (x < fig_max_m) & (y < fig_max_n)
                
            else:
                
                x = par.grid_m_nd.ravel()
                y = par.grid_n_nd.ravel()
                I = (x < fig_max_m) & (y < fig_max_n)
                    
            # i. value
            value = getattr(sol,varname)[t].ravel()
            value2 = getattr(sol2,varname)[t].ravel()
            
            # ii. figure
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1,projection='3d')
            ax.scatter(x[I],y[I],value[I] - value2[I],s=1,c=value[I] - value2[I],cmap=cm.viridis)
            
            # iii. details
            ax.set_title(f't = {t}, {varname}')
            ax.grid(True)
            ax.set_xlabel('$m_t$')
            ax.set_xlim([0,fig_max_m])
            ax.set_ylabel('$n_t$')
            ax.set_xlim([0,fig_max_m])
        
            plt.savefig('plots/pensions/decision_functions_diff_{}_{}.png'.format(title,varname))
    

def segments(model,t, title):
    
    # a. unpack
    par = model.par
    sol = model.sol

    # b. settings
    fig_max_m = 5
    fig_max_n = 5

    # c. variables
    a = par.grid_m_nd - sol.c[t] - sol.d[t]
    d = np.fmax(sol.d[t],0)
    m = par.grid_m_nd
    n = par.grid_n_nd

    # d. indicators
    I = a < 1e-7
    a[I] = 0

    I = (m < fig_max_m) & (n < fig_max_n)
    Icon = (a == 0) & (d == 0) & (I == 1)
    Iucon = (a > 0) & (d > 0) & (I == 1)
    Iacon = (a <= 1e-7) & (d > 0) & (I == 1)
    Idcon = (a > 0) & (d == 0) & (I == 1)

    # e. figure
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)

    if Icon.sum() > 0: ax.scatter(m[Icon],n[Icon],s=4,color=color3,label='con')
    if Iacon.sum() > 0: ax.scatter(m[Iacon],n[Iacon],s=4,color=color1,label='acon')
    if Idcon.sum() > 0: ax.scatter(m[Idcon],n[Idcon],s=4,color=color2,label='dcon')
    if Iucon.sum() > 0: ax.scatter(m[Iucon],n[Iucon],s=4,color='black',label='ucon')

    # f. details
    ax.grid(True)
    legend = ax.legend(frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')    
    frame.set_alpha(1)    

    ax.set_xlabel('$m_t$')
    ax.set_xlim([0,fig_max_m])
    ax.set_ylabel('$n_t$')
    ax.set_xlim([0,fig_max_m])
        
    plt.savefig('plots/pensions/segments_{}.png'.format(title))

def euler_hist(modelDict,Nm, plot_path): 

    # plot histogram of RFC vs G2EGM 
    palette = sns.color_palette("cubehelix", len(modelDict.keys()))

    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    
    i = 0
    for key in modelDict.keys():

        model = modelDict[key]
        errors = np.ravel(model.sim.euler)
        ax.hist(errors,bins=50, label = key,edgecolor='none', alpha = 0.75, density=True, color=palette[i])
        i = i + 1
    
    ax.set_xlim(-12.5, 0.5)
    ax.set_xlabel('Lg of relative Euler error')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)
    
    plt.savefig('{}/euler_hist_{}.png'.format(plot_path,Nm))


def Kregions(model_RFC, egrids_raw, egrids_clean, e_grids_intersect, plot_path, t):

    # 1. color palette
    palette = sns.color_palette("cubehelix",6)
    color1 = palette[0]
    color2 = palette[1]
    color3 = palette[2]
    color4 = palette[3]
    color5 = palette[4]
    color6 = palette[5]

    # 2. model
    par = model_RFC.par
    sol = model_RFC.sol

    # 3. Figure settings
    fig_max_m = 5
    fig_max_n = 5

    # 4. Interpolated policy 
    a = par.grid_m_nd - sol.c[t] - sol.d[t]
    d = np.fmax(sol.d[t],0)
    m = par.grid_m_nd
    n = par.grid_n_nd
    
    # 4.a. indicators
    IS = a < 1e-7
    a[IS] = 0
    IS = (m < fig_max_m) & (n < fig_max_n)
    
    # Interpolated policy functions
    IconS = (a  <= 1e-10) & (d <= 1e-10) & (IS == 1)
    IconBarS = (a  <= 1e-10) & (d  >= par.p_L) & (IS == 1)
    IuconS = (a > 0) & (d > 0) & (IS == 1)
    IaconS = (a  <= 1e-7) & (d <  par.p_L) & (IS == 1) & (d >  0)
    IdconS = (a > 0) & (d  <= 1e-10) & (IS == 1)
    IdconBarS = (a > 0)  & (IS == 1)& (d  >= par.p_L)

    # 4.b. figure
    
    fig = plt.figure(figsize=(10,5))
    # Create the first subplot in the first column
    ax1 = fig.add_subplot(121)  # 121 means 1 row, 2 columns, first plot
    ax1.scatter(m[IconS], n[IconS], d[IconS], color=color3)
    ax1.set_xlabel(r'Start of time $t$ financial assets')
    ax1.set_ylabel(r'Start of time $t$ pension assets')

    marker2 = '.'

    
    # plot LHS interpolated policy 
    if IconS.sum() > 0: ax1.scatter(m[IconS],n[IconS],s=4,color=color1,label='dcon')
    if IaconS.sum() > 0: ax1.scatter(m[IaconS],n[IaconS],s=4,color=color2,label='acon')
    if IconBarS.sum() > 0: ax1.scatter(m[IconBarS],n[IconBarS],s=4,color=color6,label='uconl')
    if IdconS.sum() > 0: ax1.scatter(m[IdconS],n[IdconS],s=4,color=color3,label='con')
    if IuconS.sum() > 0: ax1.scatter(m[IuconS],n[IuconS],s=4,color=color4,label='ucon')
    if IdconBarS.sum() > 0: ax1.scatter(m[IdconBarS],n[IdconBarS],s=4,color=color5,label='cap')

    # region labels 
    ax1.text(4, 4, 'acon', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.1'))
    ax1.text(0.3, 1.05, 'con', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.1'))
    ax1.text(1, 1.1, 'dcon', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.1'))
    ax1.text(2.1, 1.34, 'ucon', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.1'))
    ax1.set_title('Endogenous grid')
    
    try:
        m_intersect = e_grids_intersect['m']
        n_intersect = e_grids_intersect['n']
        ax1.scatter(m_intersect,n_intersect,s=4,color='black', marker ='x', alpha=0.5, label = 'intersect')
    except:
        pass
    
    #5. Raw RHS grid 
    a = model_RFC.par.grid_a_pd_nd
    b = model_RFC.par.grid_b_pd_nd
    
    m_raw = egrids_raw['m']
    n_raw = egrids_raw['n']
    d_raw = np.array(egrids_raw['d'])
    c_raw = egrids_raw['c']
    nm_raw = np.column_stack((n_raw,m_raw))

    mask_nan = np.isnan(nm_raw).any(axis=1)
    mask_inf = np.isinf(nm_raw).any(axis=1)
    mask_nan1 = mask_nan | mask_inf
    nm_raw = nm_raw[~mask_nan1]
    m_raw = m_raw[~mask_nan1]
    n_raw = n_raw[~mask_nan1]
    d_raw = d_raw[~mask_nan1]
    c_raw = c_raw[~mask_nan1]
    n_clean = egrids_clean['n']
    m_clean = egrids_clean['m']
    o_clean = egrids_clean['o']
    vf_clean = egrids_clean['v']

    c_clean = o_clean[:,0]
    d_clean = o_clean[:,2]
    a_raw = m_raw - c_raw - d_raw
    a_clean = m_clean - c_clean - d_clean

    b_raw = n_raw + d_raw + model_RFC.par.chi*np.log(1+d_raw)
    b_clean = n_clean + d_clean + model_RFC.par.chi*np.log(1+d_clean)

    # do a scatter plot of the raw grid
    # d. indicators
    I = a < 1e-7
    a[I] = 0

    I = (m_raw < fig_max_m) & (n_raw < fig_max_n)
    I = (b_raw < 8) & (c_raw < 2) & (a_raw < 5)
    
    # Indicators for the raw grid points 
    Icon = (a_raw <= 1e-10) & (d_raw == 0) & (I == 1)
    IconBar = (a_raw <= 1e-10) & (d_raw >= par.p_L) & (I == 1)
    Iucon = (a_raw > 0) & (d_raw < par.p_L) & (I == 1)
    Iacon = (a_raw <= 1e-10) & (d_raw < par.p_L) & (I == 1) &(d_raw >0)
    Idcon = (a_raw > 0) & (d_raw == 0) & (I == 1)
    IdBarcon = (a_raw > 0)  & (I == 1)& (d_raw >= par.p_L)

    # do a scatter plot of the clean grid
    # d. indicators
    I = a_clean < 1e-7
    a_clean[I] = 0
    I = (m_clean < fig_max_m) & (n_clean < fig_max_n)
    I = (b_clean < 8) & (c_clean < 2) & (a_clean < 5)

    # Indicators for the clean grid points
    Iconc = (a_clean  <= 1e-10) & (d_clean  <= 1e-10) & (I == 1)
    IconcBar = (a_clean  <= 1e-10) & (d_clean  >= par.p_L) & (I == 1)
    Iuconc = (a_clean > 0) & (d_clean > 0) & (I == 1)
    Iaconc = (a_clean <= 0) & (d_clean < par.p_L) & (I == 1)
    Idconc = (a_clean > 0) & (d_clean  <= 1e-10) & (I == 1)
    IdBarconc = (a_clean > 0)  & (I == 1)& (d_clean  >= par.p_L) 

    ax = fig.add_subplot(122)  # 122 means 1 row, 2 columns, second plot

    if Iacon.sum() > 0: ax.scatter(b_raw[Iacon],c_raw[Iacon],s=6,color='red', marker ='.', alpha=0.5, label = 'sub-optimal')
    if Idcon.sum() > 0: ax.scatter(b_raw[Idcon],c_raw[Idcon],s=4,color='red', marker ='.', alpha=0.5)
    if Iucon.sum() > 0: ax.scatter(b_raw[Iucon],c_raw[Iucon],s=4, color='red', marker ='.', alpha=0.5)
    if Icon.sum() > 0: ax.scatter(b_raw[Icon],c_raw[Icon],s=4,color='red', marker ='.', alpha=0.5)
    if IdBarcon.sum() > 0: ax.scatter(b_raw[IdBarcon],c_raw[IdBarcon],s=4,color='red', marker ='.', alpha=0.5)
    if IconBar.sum() > 0: ax.scatter(b_raw[IconBar],c_raw[IconBar],s=4,color='red', marker ='.', alpha=0.5)
    
    # plot clean exogenous grid
    if Iaconc.sum() > 0: 
        scatter = ax.scatter(b_clean[Iaconc],c_clean[Iaconc],s=4,color=color2, marker = marker2, label = 'acon')
        centroid = (b_clean[Iaconc].mean(), c_clean[Iaconc].mean())

    if Iconc.sum() > 0: 
        scatter = ax.scatter(b_clean[Iconc],c_clean[Iconc],s=4,color=color1, marker = marker2, label = 'dcon')
        centroid = (b_clean[Iconc].mean(), c_clean[Iconc].mean())
        
    if Idconc.sum() > 0: 
        scatter = ax.scatter(b_clean[Idconc],c_clean[Idconc],s=4,color=color3, marker = marker2, label = 'con')
        centroid = (b_clean[Idconc].mean(), c_clean[Idconc].mean())
        
    if Iuconc.sum() > 0: 
        scatter = ax.scatter(b_clean[Iuconc],c_clean[Iuconc],s=4, color=color4, marker = marker2, label = 'ucon')
        centroid = (b_clean[Iuconc].mean(), c_clean[Iuconc].mean())

    if IdBarconc.sum() > 0: 
        scatter = ax.scatter(b_clean[IdBarconc],c_clean[IdBarconc],s=4, color=color5, marker = marker2, label = 'cap')
    
    if IconcBar.sum() > 0: ax.scatter(b_clean[IconcBar],c_clean[IconcBar],s=4,color=color6, alpha=0.5)

    # region lables 
    plt.text(6, 1.75, 'acon', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.1'))
    plt.text(3.35, 1.65, 'sub-opt', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.1'))
    plt.text(2, 1.05, 'dcon', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.1'))
    plt.text(2, 0.3, 'con', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.1'))
    plt.text(2.1, 1.34, 'ucon', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.1'))

    ax.set_yticks([0,1,2])
    ax.set_ylabel('Consumption')
    ax.set_xlabel(r'End of time $t$ pension assets')

    ax.set_title('Exogenous grid')
    #fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)
    plt.savefig('{}/regions_t_{}_6reg.png'.format(plot_path,t))

    return None