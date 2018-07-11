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
#ds_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_25ms_0000m_90sec.nc')
#ds_small = xr.open_dataset('//uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_0500m_90sec.nc')
#ds_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_2000m_90sec.nc')

ds_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_15ms_0000m_60sec.nc')
ds_small = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_15ms_0500m_90sec.nc')
ds_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_15ms_2000m_90sec.nc')







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
time_step_length = 60.0 #[seconds]
height_bot = 0
height_top = 36
left_grd = 669
z_scale = 12
hor_resolution = 150

ts_60sec = 350
ts = np.int(ts_60sec*60/90)
te= 303
te_60sec = 455


###############################################################################
###############################################################################




#%%


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

        

    
#################### Get meteorlogical variables ###############################


#Reflectivity
dbz_no = ds_no.dbz[ts_60sec:te_60sec,:height_top,ymid,left_grd:].mean(dim = 'time').values
dbz_small = ds_small.dbz[ts:te,:height_top,ymid,left_grd:].mean(dim = 'time').values
dbz_tall = ds_tall.dbz[ts:te,:height_top,ymid,left_grd:].mean(dim = 'time').values

#Theta
theta_no = ds_no.th[ts_60sec:te_60sec,:height_top,ymid,left_grd:].mean(dim = 'time').values
theta_small = ds_small.th[ts:te,:height_top,ymid,left_grd:].mean(dim = 'time').values
theta_tall = ds_tall.th[ts:te,:height_top,ymid,left_grd:].mean(dim = 'time').values

#Wind Barbs 
zz = np.arange(2, height_top, 4)
xx = np.arange(30, ds_no.nx-left_grd, 60)
points = np.meshgrid(zz, xx)
x, y = np.meshgrid(zz, xx)


U_no = ds_no.uinterp[ts_60sec:te_60sec,:height_top,ymid,left_grd:].mean(dim = 'time').values
U_small = ds_small.uinterp[ts:te,:height_top,ymid,left_grd:].mean(dim = 'time').values
U_tall = ds_tall.uinterp[ts:te,:height_top,ymid,left_grd:].mean(dim = 'time').values

W_no = ds_no.winterp[ts_60sec:te_60sec,:height_top,ymid,left_grd:].mean(dim = 'time').values
W_small = ds_small.winterp[ts:te,:height_top,ymid,left_grd:].mean(dim = 'time').values
W_tall = ds_tall.winterp[ts:te,:height_top,ymid,left_grd:].mean(dim = 'time').values




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

#Read in colormap and put in proper format
colors1 = np.array(nclcmaps.colors['BlWhRe'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_dif = nclcmaps.make_cmap(colors, bit=True)


###############################################################################
###############################################################################



###############################################################################
#############################   Plot  #########################################
###############################################################################

secs = (i+ts)*time_step_length
fig = plt.figure(num=None, figsize=(20,10), facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.07, bottom=0.17, right=0.97, top=0.95, wspace=0.12, hspace=0.02)

#Levels for cref
creflmin = 10
creflmax = 55.01
creflevels = np.arange(creflmin,creflmax, 0.5)
creflevels_ticks = np.arange(creflmin,creflmax,5)
creflevels_ticks_labels = np.arange(creflmin,creflmax, 5).astype(int)

#Levels for theta
tlmin = 256
tlmax = 280
tlevels = np.arange(tlmin,tlmax, 1)
tlevels_ticks = np.arange(tlmin,tlmax,1)
tlevels_ticks_labels = np.arange(tlmin,tlmax, 1).astype(int)

#Levels for theta_diff
tlmin_diff = -3
tlmax_diff = 3
tlevels_diff = np.arange(tlmin_diff,tlmax_diff, 0.5)
tlevels_ticks_diff = np.arange(tlmin_diff,tlmax_diff,0.5)
tlevels_ticks_labels_diff = np.arange(tlmin_diff,tlmax_diff, 1).astype(int)

#Levels for precip diff
lmind = -20
lmaxd = 20.01
levelsd = np.arange(lmind,lmaxd, 0.25)
levelsd_ticks = np.arange(lmind,lmaxd,4)
levelsd_ticks_labels = np.arange(lmind,lmaxd, 4).astype(int)
    
    

for j in range(1,7):
    subplot = 320 + j
    ax = plt.subplot(subplot,aspect = 'equal')


    
    #Plot reflectivity
    if j == 1:
        dbz_plot = plt.contourf(x2d[0,:,:], y2d[0,:,:]*z_scale, dbz_no[:,:], creflevels, cmap = cmap_precip, extend = 'max', alpha = 1,  zorder = 3)
    if j == 3:
        dbz_plot = plt.contourf(x2d[1,:,:], y2d[1,:,:]*z_scale, dbz_small[:,:], creflevels, cmap = cmap_precip, extend = 'max', alpha = 1,  zorder = 3)
    if j == 5:
        dbz_plot = plt.contourf(x2d[2,:,:], y2d[2,:,:]*z_scale, dbz_tall[:,:], creflevels, cmap = cmap_precip, extend = 'max', alpha = 1,  zorder = 3)
        
    #Set values below 10 to 0 for differencing
    dbz_no[dbz_no < 10] = 0
    dbz_small[dbz_small < 10] = 0
    dbz_tall[dbz_tall < 10] = 0
    
    no_small = np.copy(dbz_no-dbz_small)
    no_tall = np.copy(dbz_no-dbz_tall)
    small_tall = np.copy(dbz_small-dbz_tall)
    
    no_small[no_small == 0] = np.nan
    no_tall[no_tall == 0] = np.nan
    small_tall[small_tall == 0] = np.nan
    
    
    #Plot Precip Diff
    if j == 2:
        prc_plot = plt.contourf(x2d[0,:,:], y2d[0,:,:]*z_scale, no_small, levelsd, cmap = cmap_dif, extend = 'both', alpha = 1,  zorder = 3)
    if j == 4:
        prc_plot = plt.contourf(x2d[0,:,:], y2d[0,:,:]*z_scale,no_tall, levelsd, cmap = cmap_dif, extend = 'both', alpha = 1,  zorder = 3)
    if j == 6:
        prc_plot = plt.contourf(x2d[0,:,:], y2d[0,:,:]*z_scale, small_tall, levelsd, cmap = cmap_dif, extend = 'both', alpha = 1,  zorder = 3)

    
    
    #Plot Terrain
    if j == 1:
        terrain = plt.plot(x1d, y2d[0,0,:]*z_scale, c = 'slategrey', linewidth = 4, zorder = 10)
    if j == 3:
        terrain = plt.plot(x1d, y2d[1,0,:]*z_scale, c = 'slategrey', linewidth = 4, zorder = 10)
    if j == 5:
        terrain = plt.plot(x1d, y2d[2,0,:]*z_scale, c = 'slategrey', linewidth = 4, zorder = 10)
    if j == 2:
        terrain = plt.plot(x1d, y2d[0,0,:]*z_scale, c = 'slategrey', linewidth = 4, zorder = 10)
    if j == 4:
        terrain = plt.plot(x1d, y2d[0,0,:]*z_scale, c = 'slategrey', linewidth = 4, zorder = 10)
    if j == 6:
        terrain = plt.plot(x1d, y2d[0,0,:]*z_scale, c = 'slategrey', linewidth = 4, zorder = 10)
        


    #Plot Lake
    lake[lake == 1] = np.nan
    lake_plt = plt.plot(x1d, lake[0], c = 'blue', linewidth = 4, zorder = 11)

    
    
    #Plot Theta
    if j == 1:
            theta_mean = theta_no[:,:]
            
            theta_t = np.copy(scipy.ndimage.filters.uniform_filter(theta_mean[:,:], 30))
            theta_plot = plt.contour(x2d[0,:,:], y2d[0,:,:]*z_scale, theta_t, tlevels, alpha = 1, cmap = cm.coolwarm, zorder = 5, linewidths = 2)
            clab = ax.clabel(theta_plot, theta_plot.levels[::2], fontsize=11, inline=1, zorder = 10, fmt='%1.0fK')
            plt.setp(clab, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")], zorder = 10)
            plt.setp(theta_plot.collections, path_effects=[PathEffects.withStroke(linewidth=2.5, foreground="w")])
            
    if j == 3:
            theta_mean = theta_small[:,:]
            
            theta_t = np.copy(scipy.ndimage.filters.uniform_filter(theta_mean[:,:], 30))
            theta_plot = plt.contour(x2d[1,:,:], y2d[1,:,:]*z_scale, theta_t, tlevels, alpha = 1, cmap = cm.coolwarm, zorder = 5, linewidths = 2)
            clab = ax.clabel(theta_plot, theta_plot.levels[::2], fontsize=11, inline=1, zorder = 10, fmt='%1.0fK')
            plt.setp(clab, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")], zorder = 10)
            plt.setp(theta_plot.collections, path_effects=[PathEffects.withStroke(linewidth=2.5, foreground="w")])
            
    if j == 5:
            theta_mean = theta_tall[:,:]
            
            theta_t = np.copy(scipy.ndimage.filters.uniform_filter(theta_mean[:,:], 30))
            theta_plot = plt.contour(x2d[2,:,:], y2d[2,:,:]*z_scale, theta_t, tlevels, alpha = 1, cmap = cm.coolwarm, zorder = 5, linewidths = 2)
            clab = ax.clabel(theta_plot, theta_plot.levels[::2], fontsize=11, inline=1, zorder = 10, fmt='%1.0fK')
            plt.setp(clab, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")], zorder = 10)
            plt.setp(theta_plot.collections, path_effects=[PathEffects.withStroke(linewidth=2.5, foreground="w")])


        
    
    #Plot Winds
    if j == 1:
        U_t = np.copy(U_no)
        W_t = np.copy(W_no)
        x = np.copy(x2d[0,:,:])
        z = np.copy(y2d[0,:,:])
        quiv = ax.quiver(x[points], z[points]*z_scale, U_t[points], W_t[points], zorder = 4, color = 'k', width = 0.0017, scale = 550)

    if j == 3:
        U_t = np.copy(U_small)
        W_t = np.copy(W_small)
        x = np.copy(x2d[1,:,:])
        z = np.copy(y2d[1,:,:])
        quiv = ax.quiver(x[points], z[points]*z_scale, U_t[points], W_t[points], zorder = 4, color = 'k', width = 0.0017, scale = 550)

    if j == 5:
        U_t = np.copy(U_tall)
        W_t = np.copy(W_tall)
        x = np.copy(x2d[2,:,:])
        z = np.copy(y2d[2,:,:])
        quiv = ax.quiver(x[points], z[points]*z_scale, U_t[points], W_t[points], zorder = 4, color = 'k', width = 0.0017, scale = 550)
    

    
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
    if j == 5 or j == 6:
        plt.xlabel('Distance (km)', fontsize = 16)
    if j == 1 or j == 3 or j == 5:
        plt.ylabel('Height (m)', fontsize = 16)

    #Labels
    #Labels
    sub_title = ['No Mountain','0m minus 500m',
                 '500m Mountain','0m minus 2000m',
                 '2000m Mountain','500m minus 2000m']
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax.text(20, 305, sub_title[j-1], fontsize = 15, bbox = props, zorder = 12)



#Colorbars
cbaxes = fig.add_axes([0.13, 0.09, 0.3, 0.035])           
cbar = plt.colorbar(dbz_plot, orientation='horizontal', cax = cbaxes, ticks = creflevels_ticks_labels)
cbar.ax.tick_params(labelsize=15)
plt.text(0.45, -1.8, 'dBZ', fontsize = 22)

cbaxes = fig.add_axes([0.6, 0.09, 0.3, 0.035])           
cbar = plt.colorbar(prc_plot, orientation='horizontal', cax = cbaxes, ticks = levelsd_ticks_labels)
cbar.ax.tick_params(labelsize=15)
plt.text(0.45, -1.8, 'dBZ', fontsize = 22)
  
#Labels
sub_title = ['[Run: Low Wind]']
props = dict(boxstyle='square', facecolor='white', alpha=1)
ax.text(0.76, -0.9, sub_title[0], fontsize = 18, zorder = 5, transform=ax.transAxes)


#Quiver
ax.quiverkey(quiv, X=0.14, Y=-0.9, U=20, label='20 $\mathregular{ms^{-1}}$', labelpos='W', fontproperties={'size': '20'})


plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/plots/cref_th_low_wind_cross_section.png", dpi = 150)
plt.close(fig)
print(i)

        



