import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
from mpl_toolkits.basemap import Basemap, maskoceans
import pygrib, os, sys
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import time
from datetime import date, timedelta
from matplotlib import animation
import matplotlib.animation as animation
import types
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import nclcmaps
import pandas as pd
import xarray as xr
from scipy import interpolate
import operator
import multiprocessing


#Read in with xarray
ds = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_15ms_0500m_90sec.nc')
#ds = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_20ms_0000m_radiation_warmerlake.nc')



#xpos = np.load('xpos_traj_500m.npy')
#ypos = np.load('ypos_traj_500m.npy')
#zpos_terrain = np.load('zpos_traj_500m.npy')
#color = np.load('color_traj_500m.npy')


###############################################################################
###############################################################################   
##########################   Set up Trajectories ##############################
###############################################################################
###############################################################################


#Dimension size variables
num_x = ds.uinterp[0,0,0,:].size
num_y = ds.uinterp[0,0,:,0].size
num_z = ds.uinterp[0,:,0,0].size

x = np.arange(0,num_x,1)
y = np.arange(0,num_y,1)
z = np.arange(0,num_z,1)





###############################################################################
##################### INFO TO CALCULATE SEEDS #################################
#############  These are variables the user changes  ##########################
###############################################################################
#Backward trajectories
num_seeds_z = 51 #Up to 5000m (one seed every grid point so they can be contoured)
num_seeds_x = ds.nx-1000 #About right half of domain (one seed every fgridpoint)
time_steps = 10 #Run trajectories back 100 time steps (all during steady-state)
time = 260 #Start near end of simulation
hor_resolution = 200 #meters
vert_resolution = 100 #meters
time_step_length = 90 #seconds
rdist = 50 #distance from back
ymid = np.int(ds.ny/2)
###############################################################################
###############################################################################

#%%


#Variable
var_name1 = 'th'
#variable1 = np.zeros((time_steps, num_seeds_z, num_seeds_x))

var_name2 = 'prspert'#Coded to be a budget variable
#variable2 = np.zeros((time_steps, num_seeds_z, num_seeds_x))

#Variable
#var_name1 = 'th'
#variable1 = np.zeros((num_seeds_z, num_seeds_x))
#
#var_name2 = 'winterp'#Coded to be a budget variable
#variable2 = np.zeros((num_seeds_z, num_seeds_x))

var1 = getattr(ds,var_name1)[time,:num_seeds_z,ymid,-num_seeds_x-rdist:-rdist].values
var2 = getattr(ds,var_name2)[time-10:time+10,:num_seeds_z,ymid-10:ymid+10,-num_seeds_x-rdist:-rdist].values

var2 = np.mean(var2, axis = 0)
var2 = np.mean(var2, axis = 1)

zh = ds.zh[0,:51,ymid,-num_seeds_x-rdist:-rdist].values/100
#
#var1 = getattr(ds,var_name1)[time,:num_seeds_z,ymid,-num_seeds_x-rdist:-rdist].values
#var2 = getattr(ds,var_name2)[time,:num_seeds_z,ymid,-num_seeds_x-rdist:-rdist].values


#var2 = np.sum(var2, axis = 0)*120
#var1 = np.mean(var1, axis = 0)
#var2 = np.mean(var2, axis = 0)

###############################################################################
############## Set ncl_cmap as the colormap you want ##########################
###############################################################################

### 1) In order for this to work the files "nclcmaps.py" and "__init__.py"
### must be present in the dirctory.
### 2) You must "import nclcmaps"
### 3) The path to nclcmaps.py must be added to tools -> PYTHONPATH manager in SPyder
### 4) Then click "Update module names list" in tolls in Spyder and restart Spyder
                
## The steps above describe the general steps for adding "non-built in" modules to Spyder

###############################################################################
###############################################################################


#Read in colormap and put in proper format
colors1 = np.array(nclcmaps.colors['WhiteBlueGreenYellowRed'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_var = nclcmaps.make_cmap(colors, bit=True)

#Read in colormap and put in proper format
colors1 = np.array(nclcmaps.colors['amwg256'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_dth = nclcmaps.make_cmap(colors, bit=True)



##########  Create Grid ########
### The code below makes the data terrain following 
x1d = np.arange(0,num_seeds_x,1)
y1d = np.arange(0,num_seeds_z,1)*3



xmin = ds.th[0,0,0,:].size-num_seeds_x-rdist
xmax = ds.th[0,0,0,:].size-rdist

xlen = xmax-xmin
xmin = ds.th[0,0,0,:].size-num_seeds_x-rdist
xmax = ds.th[0,0,0,:].size-rdist
xlen = xmax-xmin

zmin = 0
zmax = num_seeds_z


try:
    z = np.array(ds.zs[0,ymid,-num_seeds_x-rdist:-rdist])/1000*30 #Div by 1000 to go to km and mult by 30 to match y dim
except:
    z = np.zeros((xlen))
x2d = np.zeros((num_seeds_z, num_seeds_x))
y2d = np.zeros((num_seeds_z, num_seeds_x))

for i in range(num_seeds_z):
    x2d[i,:] = x1d
for j in range(num_seeds_x):
    y2d[:,j] = y1d+z[j]
        



#%%

##############################   Plot ########################################
    
fig = plt.figure(num=None, figsize=(18,9),  facecolor='w', edgecolor='k')
for j in range(1,3):
    subplot = 210 + j
    ax = plt.subplot(subplot,aspect = 'equal')
    plt.subplots_adjust(left=0.04, bottom=0.1, right=0.9, top=0.95, wspace=0, hspace=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    

    
    ##Levels
    zlmin = 266
    zlmax = 275
    zlevels = np.arange(zlmin,zlmax, 0.05)
    zlevels_ticks = np.arange(zlmin,zlmax,1)
    zlevels_ticks_labels = np.arange(zlmin,zlmax, 1).astype(int)
    
    xlmin = -50
    xlmax = 50.01
    xlevels = np.arange(xlmin,xlmax, 0.05)
    xlevels_ticks = np.arange(xlmin,xlmax,1)
    xlevels_ticks_labels = np.arange(xlmin,xlmax, 1)
    
#    # Calculate Wind Barbs
#    yy = np.arange(0,50,5)
#    yy2 = np.arange(0,150,15)
#    xx = np.arange(0,800,40)
#    points = np.meshgrid(yy, xx)
#    x, y = np.meshgrid(yy2, xx)
#    U = var2
#    V = var1
#    quiv = ax.quiver(y, x, U[points], V[points], zorder = 4, color = 'k', scale = 2000)
    
    
    
        
    #Plot mean vert disp
    if j == 1:
        z_disp_plot = plt.contourf(x2d, zh*3, var1, zlevels,  cmap = cmap_var, alpha = 1, zorder = 3)
        #z_disp_plot = plt.contourf(x1d, y1d, var1, zlevels,  cmap = cmap_var,vmin = -zlmax, alpha = 1, zorder = 3)

    if j == 2:
        #x_disp_plot = plt.contourf(x1d, y1d, var2, xlevels,  cmap = cmap_var,vmin = -xlmax, alpha = 1, zorder = 3)
        x_disp_plot = plt.contourf(x2d, zh*3, var2, xlevels,  cmap = cmap_dth,vmin = -xlmax, alpha = 1, zorder = 3)

    
    #Plot Terrain
    terrain = plt.plot(x1d, z, c = 'slategrey', linewidth = 4, zorder = 4)
    
    #Plot Lake
    lake = np.array(ds.xland[0,ymid,-num_seeds_x-rdist:-rdist])
    lake[lake == 1] = np.nan
    lake_plt = plt.plot(x1d, lake-3, c = 'blue', linewidth = 6, zorder = 5)
    
    
    #Title
    if j == 2:
        sub_title = '[Run: 20 $\mathregular{ms^{-1}}$ and 1500m]'
        ax.text(590, -50, sub_title, fontsize = 20)
    
    
    ### Label and define grid area
    #y-axis
    plt.ylim([-3,np.max(y2d[:,0])+1])
    plt.yticks(np.arange(0, np.max(y2d[:,0]+1),30))
    ax.set_yticklabels(np.arange(0,np.max(y2d[:,0]+1),1).astype(int), fontsize = 15, zorder = 6)
    #x-axis
    plt.xticks(np.arange(12*5,xlen,100))
    ax.set_xticklabels(np.arange(xmin*0.2+12,xmax*0.2+12.1,20).astype(int), fontsize = 15)
    plt.xlim([-1,xlen-1])
    #axis labels
    plt.ylabel('Height (km)', fontsize = 20, labelpad = 8)
    if j == 2:
        plt.xlabel('Distance within Domain (km)', fontsize = 20, labelpad = 9)
                
    #Colorbar
    if j == 1:
        zcbaxes = fig.add_axes([0.92, 0.59, 0.03, 0.3])             
        zcbar = plt.colorbar(z_disp_plot, cax = zcbaxes, ticks = zlevels_ticks)
        zcbar.ax.set_yticklabels(zlevels_ticks_labels)
        zcbar.ax.tick_params(labelsize=15)
        plt.text(0.25, -0.12, 'm', fontsize = 21)
    
    if j == 2:
        xcbaxes = fig.add_axes([0.92, 0.16, 0.03, 0.3])             
        xcbar = plt.colorbar(x_disp_plot, cax = xcbaxes, ticks = xlevels_ticks)
        xcbar.ax.set_yticklabels(xlevels_ticks_labels)
        xcbar.ax.tick_params(labelsize=15)
        plt.text(0.18, -0.15, 'km', fontsize = 21)
    
    #Labels
    if j == 1:
        sub_title = 'Mean Potential Temperature'
    if j == 2:
        sub_title = 'Mean u-wind'
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax.text(10, 130, sub_title, fontsize = 20, bbox = props, zorder = 5)


plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/plots/2panel.png", dpi=350)
plt.close(fig)



