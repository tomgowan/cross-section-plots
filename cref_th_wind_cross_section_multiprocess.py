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
from scipy import signal
import operator
import multiprocessing
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.patheffects as PathEffects
import scipy.ndimage
import matplotlib.patches as patches
import multiprocessing as mp

#Read in with xarray
ds_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_25ms_0000m_90sec.nc')
ds_small = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_0500m_90sec.nc')
ds_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_2000m_90sec.nc')




###############################################################################
###############################################################################   
##########################   Set up Trajectories ##############################
###############################################################################
###############################################################################


#Dimension size variables
num_x = ds_no.nx
num_y = ds_no.ny
num_z = ds_no.nz

x = np.arange(0,num_x,1)
y = np.arange(0,num_y,1)
z = np.arange(0,num_z,1)




###############################################################################
time_steps = np.min([ds_no.dbz[:,0,0,0].size, ds_small.dbz[:,0,0,0].size, ds_tall.dbz[:,0,0,0].size])
start_time_step = 10 #Starting time step
hor_resolution = 150 #[meters]
vert_resolution = 100 #[meters] (resolution away from terrain)
time_step_length = 90.0 #[seconds]
height_bot = 0
height_top = 36
left_grd = 669
z_scale = 12

###############################################################################
###############################################################################







###############################################################################
#############################   PLOTS   #######################################
###############################################################################



#########  Create Grid ########
## The code below makes the data terrain following 
ymid = np.int(ds_no.ny/2)
x1d = np.arange(0,ds_no.nx-left_grd,1)
y1d = np.arange(0,height_top,1)

#Create 2D arrays for plotting data
x2d = np.zeros((3,height_top, ds_no.nx-left_grd))
y2d = np.zeros((3,height_top, ds_no.nx-left_grd))
lake = np.zeros((3, ds_no.nx-left_grd))
zs = np.zeros((3,height_top, ds_no.nx-left_grd))
run = ['ds_no', 'ds_small', 'ds_tall']

for j in range(3):
    model_run = eval(run[j])
    try:
        z = np.array(model_run.zs[0,ymid,left_grd:])/vert_resolution #Convert to gridpoints
    except:
        z = np.zeros((model_run.nx-left_grd))

    for i in range(height_top):
        x2d[j,i,:] = x1d
    for k in range(model_run.nx-left_grd):
        y2d[j,:,k] = y1d+z[k]
        
    #Variables from output to plot
    lake[j,:] = model_run.xland[0,ymid,left_grd:].values

        



#%%
#Read in colormap and put in proper format
colors1 = np.array(nclcmaps.colors['amwg256'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_dth = nclcmaps.make_cmap(colors, bit=True)


#Read in colormap and put in proper format
colors1 = np.array(nclcmaps.colors['WhiteBlueGreenYellowRed'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_th = nclcmaps.make_cmap(colors, bit=True)

colors1 = np.array(nclcmaps.colors['prcp_1'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_precip = nclcmaps.make_cmap(colors, bit=True)


###############################################################################
###############################################################################



###############################################################################
#############################   Plot  #########################################
###############################################################################
#Use multiple processors to create images
def plotting(i): 


#for i in range(time_steps-6):   
#for i in range(0,9): 
    
    #################### Get meteorlogical variables ###############################
    
    #Read in with xarray
    ds_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_25ms_0000m_90sec.nc', lock = True)
    ds_small = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_0500m_90sec.nc', lock = True)
    ds_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_2000m_90sec.nc', lock = True)
    
    #Reflectivity
    dbz_no = ds_no.dbz[i,:height_top,ymid,left_grd:].values
    dbz_small = ds_small.dbz[i,:height_top,ymid,left_grd:].values
    dbz_tall = ds_tall.dbz[i,:height_top,ymid,left_grd:].values
    
    #Theta
    theta_no = ds_no.th[i,:height_top,ymid,left_grd:].values
    theta_small = ds_small.th[i,:height_top,ymid,left_grd:].values
    theta_tall = ds_tall.th[i,:height_top,ymid,left_grd:].values
    
    #Wind Barbs 
    zz = np.arange(2, height_top, 4)
    xx = np.arange(30, ds_no.nx-left_grd, 60)
    points = np.meshgrid(zz, xx)
    x, y = np.meshgrid(zz, xx)
    
    
    U_no = ds_no.uinterp[i,:height_top,ymid,left_grd:].values
    U_small = ds_small.uinterp[i,:height_top,ymid,left_grd:].values
    U_tall = ds_tall.uinterp[i,:height_top,ymid,left_grd:].values
    W_no = ds_no.winterp[i,:height_top,ymid,left_grd:].values
    W_small = ds_small.winterp[i,:height_top,ymid,left_grd:].values
    W_tall = ds_tall.winterp[i,:height_top,ymid,left_grd:].values




    secs = (i*time_step_length)
    fig = plt.figure(num=None, figsize=(13,10), facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.94, top=0.9, wspace=0.12, hspace=0.2)
    
    #Levels for cref
    creflmin = 10
    creflmax = 55.01
    creflevels = np.arange(creflmin,creflmax, 0.05)
    creflevels_ticks = np.arange(creflmin,creflmax,5)
    creflevels_ticks_labels = np.arange(creflmin,creflmax, 5).astype(int)
    
    #Levels for theta
    tlmin = 256
    tlmax = 280
    tlevels = np.arange(tlmin,tlmax, 1.5)
    tlevels_ticks = np.arange(tlmin,tlmax,1.5)
    tlevels_ticks_labels = np.arange(tlmin,tlmax, 1.5).astype(int)
    
    for j in range(1,4):
        subplot = 310 + j
        ax = plt.subplot(subplot,aspect = 'equal')
        if j == 1:
            plt.title("Reflectivity, Potential Temperature, and Wind [elapsed time = %d seconds]"  % secs, fontsize = 18, y = 1.1) 
    
        
        #Plot reflectivity
        if j == 1:
            dbz_plot = plt.contourf(x2d[j-1,:,:], y2d[j-1,:,:]*z_scale, dbz_no[:,:], creflevels, cmap = cmap_precip, extend = 'max', alpha = 1,  zorder = 3)
        if j == 2:
            dbz_plot = plt.contourf(x2d[j-1,:,:], y2d[j-1,:,:]*z_scale, dbz_small[:,:], creflevels, cmap = cmap_precip, extend = 'max', alpha = 1,  zorder = 3)
        if j == 3:
            dbz_plot = plt.contourf(x2d[j-1,:,:], y2d[j-1,:,:]*z_scale, dbz_tall[:,:], creflevels, cmap = cmap_precip, extend = 'max', alpha = 1,  zorder = 3)
        
        
        #Plot Terrain
        terrain = plt.plot(x1d, y2d[j-1,0,:]*z_scale, c = 'slategrey', linewidth = 4, zorder = 10)
    

        #Plot Lake
        lake[lake == 1] = np.nan
        lake_plt = plt.plot(x1d, lake[j-1], c = 'blue', linewidth = 4, zorder = 11)
        
        
        #Plot Theta
        if j == 1:
            theta_mean = theta_no[:,:]
        if j == 2:
            theta_mean = theta_small[:,:]
        if j == 3:
            theta_mean = theta_tall[:,:]
        
        theta_t = np.copy(scipy.ndimage.filters.uniform_filter(theta_mean[:,:], 30))
        theta_plot = plt.contour(x2d[j-1,:,:], y2d[j-1,:,:]*z_scale, theta_t, tlevels, alpha = 1, cmap = cm.coolwarm, zorder = 5, linewidths = 1.5)
        plt.setp(theta_plot.collections, path_effects=[PathEffects.withStroke(linewidth=2.5, foreground="w")])
        
        
        #Plot Winds
        if j == 1:
            U_t = np.copy(U_no[:,:])
            W_t = np.copy(W_no[:,:])
        if j == 2:
            U_t = np.copy(U_small[:,:])
            W_t = np.copy(W_small[:,:])
        if j == 3:
            U_t = np.copy(U_tall[:,:])
            W_t = np.copy(W_tall[:,:])
            
        x = np.copy(x2d[j-1,:,:])
        z = np.copy(y2d[j-1,:,:])
        
        quiv = ax.quiver(x[points], z[points]*z_scale, U_t[points], W_t[points], zorder = 4, color = 'k', width = 0.0017, scale = 750)
        

        
        #Plot Characteristics
        plt.grid(True, color = 'white', )
        plt.xticks(np.arange(0,num_x,20000/hor_resolution))
        ytick = np.arange(0,height_top*z_scale,5*z_scale)
        plt.yticks(ytick)
        ax.set_yticklabels(ytick*vert_resolution/z_scale, fontsize = 13)
        ax.set_xticklabels(np.arange(left_grd*hor_resolution/1000,num_x*hor_resolution/1000,20), fontsize = 13)
        plt.xlim([0,num_x-left_grd])
        plt.ylim([-2,ytick[-2]])
        plt.axvspan(0,num_x,color='gainsboro',lw=0)
        if j == 3:
            plt.xlabel('Distance (km)', fontsize = 16)
        plt.ylabel('Height (m)', fontsize = 16)

        #Labels
        sub_title = ['No Mountain', '500m Mountain', '2000m Mountain']
        props = dict(boxstyle='square', facecolor='white', alpha=1)
        ax.text(17, 320, sub_title[j-1], fontsize = 14, bbox = props, zorder = 6)
    
    #Colorbar
    cbaxes = fig.add_axes([0.92, 0.25, 0.035, 0.5])
    cbar = plt.colorbar(dbz_plot, orientation='vertical', cax = cbaxes, ticks = creflevels_ticks)
    cbar.ax.set_yticklabels(creflevels_ticks_labels)
    cbar.ax.tick_params(labelsize=12)
    plt.text(-0.05, -0.07, 'dBZ', fontsize = 16)

    
  
    #Labels
    sub_title = ['[Run: High Wind]']
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax.text(0.76, -0.29, sub_title[0], fontsize = 18, zorder = 5, transform=ax.transAxes)
    

    #Quiver
    ax.quiverkey(quiv, X=0.14, Y=-0.25, U=20, label='20 $\mathregular{ms^{-1}}$', labelpos='W', fontproperties={'size': '20'})


    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/cref_th_high_wind_cross_section_%03d.png" % i, dpi = 80)
    plt.close(fig)
    print(i)
#
plt.switch_backend('Agg')
  
  
#run function to create images
pool = mp.Pool(processes = 10)
pool.map(plotting, range(time_steps-3))#number of processors
pool.close()
pool.join()
        



