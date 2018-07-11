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
import shiftedcolormap
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker




    

#Read in with xarray
dsl_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_15ms_0000m_90sec.nc')
dsl_small = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_15ms_0500m_90sec.nc')
dsl_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_15ms_2000m_90sec.nc')

dsh_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_25ms_0000m_90sec.nc')
dsh_small = xr.open_dataset('//uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_0500m_90sec.nc')
dsh_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_2000m_90sec.nc')


###############################################################################
##################### INFO TO CALCULATE SEEDS #################################
#############  These are variables the user changes  ##########################
###############################################################################
#Backward trajectories
num_seeds_z = 151 #Up to 5000m (3 seeds every vertical grid point)
num_seeds_x = 2208 #One for each x gridpoint
time_steps = 250 #Number of time steps to run trajectories back

### Name variables
#Variable
var_name1 = 'th'
#Budgets
var_name2 = 'ptb_mp'


############## Read in trajectory Data for low wind ###########################
xpos = np.load('xpos_traj_disp_15ms_0000m_250time_steps.npy')
ypos = np.load('ypos_traj_disp_15ms_0000m_250time_steps.npy')
zpos_terrain = np.load('zpos_traj_disp_15ms_0000m_250time_steps.npy')
variable1 = np.load('%s_traj_disp_15ms_0000m_250time_steps.npy' % var_name1)
variable2 = np.load('%s_traj_disp_15ms_0000m_250time_steps.npy' %var_name2)


xpos_small = np.load('xpos_traj_disp_15ms_0500m_250time_steps.npy')
ypos_small = np.load('ypos_traj_disp_15ms_0500m_250time_steps.npy')
zpos_terrain_small = np.load('zpos_traj_disp_15ms_0500m_250time_steps.npy')
variable1_small = np.load('%s_traj_disp_15ms_0500m_250time_steps.npy' % var_name1)
variable2_small = np.load('%s_traj_disp_15ms_0500m_250time_steps.npy' %var_name2)


xpos_tall = np.load('xpos_traj_disp_15ms_2000m_250time_steps.npy')
ypos_tall = np.load('ypos_traj_disp_15ms_2000m_250time_steps.npy')
zpos_terrain_tall = np.load('zpos_traj_disp_15ms_2000m_250time_steps.npy')
variable1_tall = np.load('%s_traj_disp_15ms_2000m_250time_steps.npy' % var_name1)
variable2_tall = np.load('%s_traj_disp_15ms_2000m_250time_steps.npy' %var_name2)




###############################################################################
######################  Calculate displacement  ###############################
###############################################################################

##### ENTER VALUE FOR NUMBER OF TIME STEPS BACK TO CALCULATE TIME-MEAN AND PLOT
ts_plot = 60

z_disp = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
x_disp = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
y_disp = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var1_disp = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var2_disp = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))


z_disp_small = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
x_disp_small = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
y_disp_small = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var1_disp_small = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var2_disp_small = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))


z_disp_tall = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
x_disp_tall = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
y_disp_tall = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var1_disp_tall = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var2_disp_tall = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))



for t in range(time_steps-2):
    z_disp[t,:,:] = zpos_terrain[0,:,:] - zpos_terrain[t+1,:,:]
    x_disp[t,:,:] = xpos[0,:,:] - xpos[t+1,:,:]
    y_disp[t,:,:] = ypos[0,:,:] - ypos[t+1,:,:]
    var1_disp[t,:,:] = variable1[0,:,:] - variable1[t+1,:,:]

    #Budget variables
    var2_disp[t,:,:] =  np.sum(variable2[:t+1,:,:], axis = 0)

    
        
###################   SMALL  ##################################################    
    z_disp_small[t,:,:] = zpos_terrain_small[0,:,:] - zpos_terrain_small[t+1,:,:]
    x_disp_small[t,:,:] = xpos_small[0,:,:] - xpos_small[t+1,:,:]
    y_disp_small[t,:,:] = ypos_small[0,:,:] - ypos_small[t+1,:,:]
    var1_disp_small[t,:,:] = variable1_small[0,:,:] - variable1_small[t+1,:,:]

    #Budget variables
    var2_disp_small[t,:,:] =  np.sum(variable2_small[:t+1,:,:], axis = 0)

    
    
#####################   TALL   ################################################    
    z_disp_tall[t,:,:] = zpos_terrain_tall[0,:,:] - zpos_terrain_tall[t+1,:,:]
    x_disp_tall[t,:,:] = xpos_tall[0,:,:] - xpos_tall[t+1,:,:]
    y_disp_tall[t,:,:] = ypos_tall[0,:,:] - ypos_tall[t+1,:,:]
    var1_disp_tall[t,:,:] = variable1_tall[0,:,:] - variable1_tall[t+1,:,:]

    #Budget variables
    var2_disp_tall[t,:,:] =  np.sum(variable2_tall[:t+1,:,:], axis = 0)



###########  DONT FORGET TO CHANGE WHEN RUNNING 25ms RUNS  ####################


mean_z_disp = np.zeros((num_seeds_z, num_seeds_x))
mean_z_disp = np.mean(z_disp[:ts_plot,:,:], axis = 0)
mean_x_disp = np.zeros((num_seeds_z, num_seeds_x))
mean_x_disp = np.mean(x_disp[:ts_plot,:,:], axis = 0)
mean_y_disp = np.zeros((num_seeds_z, num_seeds_x))
mean_y_disp = np.mean(y_disp[:ts_plot,:,:], axis = 0)

#Variable
mean_var1_disp = np.zeros((num_seeds_z, num_seeds_x))
mean_var1_disp = np.mean(var1_disp[:ts_plot,:,:], axis = 0)

##Budget Variables
mean_var2_disp = np.zeros((num_seeds_z, num_seeds_x))
mean_var2_disp = np.mean(var2_disp[:ts_plot,:,:], axis = 0)


###################   SMALL  ##################################################
mean_z_disp_small = np.zeros((num_seeds_z, num_seeds_x))
mean_z_disp_small = np.mean(z_disp_small[:ts_plot,:,:], axis = 0)
mean_x_disp_small = np.zeros((num_seeds_z, num_seeds_x))
mean_x_disp_small = np.mean(x_disp_small[:ts_plot,:,:], axis = 0)
mean_y_disp_small = np.zeros((num_seeds_z, num_seeds_x))
mean_y_disp_small = np.mean(y_disp_small[:ts_plot,:,:], axis = 0)

#Variable
mean_var1_disp_small = np.zeros((num_seeds_z, num_seeds_x))
mean_var1_disp_small = np.mean(var1_disp_small[:ts_plot,:,:], axis = 0)

##Budget Variables
mean_var2_disp_small = np.zeros((num_seeds_z, num_seeds_x))
mean_var2_disp_small = np.mean(var2_disp_small[:ts_plot,:,:], axis = 0)


####################   TALL  ################################################## 
mean_z_disp_tall = np.zeros((num_seeds_z, num_seeds_x))
mean_z_disp_tall = np.mean(z_disp_tall[:ts_plot,:,:], axis = 0)
mean_x_disp_tall = np.zeros((num_seeds_z, num_seeds_x))
mean_x_disp_tall = np.mean(x_disp_tall[:ts_plot,:,:], axis = 0)
mean_y_disp_tall = np.zeros((num_seeds_z, num_seeds_x))
mean_y_disp_tall = np.mean(y_disp_tall[:ts_plot,:,:], axis = 0)

#Variable
mean_var1_disp_tall = np.zeros((num_seeds_z, num_seeds_x))
mean_var1_disp_tall = np.mean(var1_disp_tall[:ts_plot,:,:], axis = 0)

##Budget Variables
mean_var2_disp_tall = np.zeros((num_seeds_z, num_seeds_x))
mean_var2_disp_tall = np.mean(var2_disp_tall[:ts_plot,:,:], axis = 0)


















############## Read in trajectory Data for high wind ###########################
xpos = np.load('xpos_traj_disp_25ms_0000m_250time_steps.npy')
ypos = np.load('ypos_traj_disp_25ms_0000m_250time_steps.npy')
zpos_terrain = np.load('zpos_traj_disp_25ms_0000m_250time_steps.npy')
variable1 = np.load('%s_traj_disp_25ms_0000m_250time_steps.npy' % var_name1)
variable2 = np.load('%s_traj_disp_25ms_0000m_250time_steps.npy' %var_name2)


xpos_small = np.load('xpos_traj_disp_25ms_0500m_250time_steps.npy')
ypos_small = np.load('ypos_traj_disp_25ms_0500m_250time_steps.npy')
zpos_terrain_small = np.load('zpos_traj_disp_25ms_0500m_250time_steps.npy')
variable1_small = np.load('%s_traj_disp_25ms_0500m_250time_steps.npy' % var_name1)
variable2_small = np.load('%s_traj_disp_25ms_0500m_250time_steps.npy' %var_name2)


xpos_tall = np.load('xpos_traj_disp_25ms_2000m_250time_steps.npy')
ypos_tall = np.load('ypos_traj_disp_25ms_2000m_250time_steps.npy')
zpos_terrain_tall = np.load('zpos_traj_disp_25ms_2000m_250time_steps.npy')
variable1_tall = np.load('%s_traj_disp_25ms_2000m_250time_steps.npy' % var_name1)
variable2_tall = np.load('%s_traj_disp_25ms_2000m_250time_steps.npy' %var_name2)




###############################################################################
######################  Calculate displacement  ###############################
###############################################################################

##### ENTER VALUE FOR NUMBER OF TIME STEPS BACK TO CALCULATE TIME-MEAN AND PLOT
ts_plot = 60

z_disp = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
x_disp = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
y_disp = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var1_disp = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var2_disp = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))


z_disp_small = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
x_disp_small = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
y_disp_small = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var1_disp_small = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var2_disp_small = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))


z_disp_tall = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
x_disp_tall = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
y_disp_tall = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var1_disp_tall = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))
var2_disp_tall = np.zeros((time_steps-2, num_seeds_z, num_seeds_x))



for t in range(time_steps-2):
    z_disp[t,:,:] = zpos_terrain[0,:,:] - zpos_terrain[t+1,:,:]
    x_disp[t,:,:] = xpos[0,:,:] - xpos[t+1,:,:]
    y_disp[t,:,:] = ypos[0,:,:] - ypos[t+1,:,:]
    var1_disp[t,:,:] = variable1[0,:,:] - variable1[t+1,:,:]

    #Budget variables
    var2_disp[t,:,:] =  np.sum(variable2[:t+1,:,:], axis = 0)

    
        
###################   SMALL  ##################################################    
    z_disp_small[t,:,:] = zpos_terrain_small[0,:,:] - zpos_terrain_small[t+1,:,:]
    x_disp_small[t,:,:] = xpos_small[0,:,:] - xpos_small[t+1,:,:]
    y_disp_small[t,:,:] = ypos_small[0,:,:] - ypos_small[t+1,:,:]
    var1_disp_small[t,:,:] = variable1_small[0,:,:] - variable1_small[t+1,:,:]

    #Budget variables
    var2_disp_small[t,:,:] =  np.sum(variable2_small[:t+1,:,:], axis = 0)

    
    
#####################   TALL   ################################################    
    z_disp_tall[t,:,:] = zpos_terrain_tall[0,:,:] - zpos_terrain_tall[t+1,:,:]
    x_disp_tall[t,:,:] = xpos_tall[0,:,:] - xpos_tall[t+1,:,:]
    y_disp_tall[t,:,:] = ypos_tall[0,:,:] - ypos_tall[t+1,:,:]
    var1_disp_tall[t,:,:] = variable1_tall[0,:,:] - variable1_tall[t+1,:,:]

    #Budget variables
    var2_disp_tall[t,:,:] =  np.sum(variable2_tall[:t+1,:,:], axis = 0)



###########  DONT FORGET TO CHANGE WHEN RUNNING 25ms RUNS  ####################


mean_z_disph = np.zeros((num_seeds_z, num_seeds_x))
mean_z_disph = np.mean(z_disp[:ts_plot,:,:], axis = 0)
mean_x_disph = np.zeros((num_seeds_z, num_seeds_x))
mean_x_disph = np.mean(x_disp[:ts_plot,:,:], axis = 0)
mean_y_disph = np.zeros((num_seeds_z, num_seeds_x))
mean_y_disph = np.mean(y_disp[:ts_plot,:,:], axis = 0)

#Variable
mean_var1_disph = np.zeros((num_seeds_z, num_seeds_x))
mean_var1_disph = np.mean(var1_disp[:ts_plot,:,:], axis = 0)

##Budget Variables
mean_var2_disph = np.zeros((num_seeds_z, num_seeds_x))
mean_var2_disph = np.mean(var2_disp[:ts_plot,:,:], axis = 0)


###################   SMALL  ##################################################
mean_z_disp_smallh = np.zeros((num_seeds_z, num_seeds_x))
mean_z_disp_smallh = np.mean(z_disp_small[:ts_plot,:,:], axis = 0)
mean_x_disp_smallh = np.zeros((num_seeds_z, num_seeds_x))
mean_x_disp_smallh = np.mean(x_disp_small[:ts_plot,:,:], axis = 0)
mean_y_disp_smallh = np.zeros((num_seeds_z, num_seeds_x))
mean_y_disp_smallh = np.mean(y_disp_small[:ts_plot,:,:], axis = 0)

#Variable
mean_var1_disp_smallh = np.zeros((num_seeds_z, num_seeds_x))
mean_var1_disp_smallh = np.mean(var1_disp_small[:ts_plot,:,:], axis = 0)

##Budget Variables
mean_var2_disp_smallh = np.zeros((num_seeds_z, num_seeds_x))
mean_var2_disp_smallh = np.mean(var2_disp_small[:ts_plot,:,:], axis = 0)


####################   TALL  ################################################## 
mean_z_disp_tallh = np.zeros((num_seeds_z, num_seeds_x))
mean_z_disp_tallh = np.mean(z_disp_tall[:ts_plot,:,:], axis = 0)
mean_x_disp_tallh = np.zeros((num_seeds_z, num_seeds_x))
mean_x_disp_tallh = np.mean(x_disp_tall[:ts_plot,:,:], axis = 0)
mean_y_disp_tallh = np.zeros((num_seeds_z, num_seeds_x))
mean_y_disp_tallh = np.mean(y_disp_tall[:ts_plot,:,:], axis = 0)

#Variable
mean_var1_disp_tallh = np.zeros((num_seeds_z, num_seeds_x))
mean_var1_disp_tallh = np.mean(var1_disp_tall[:ts_plot,:,:], axis = 0)

##Budget Variables
mean_var2_disp_tallh = np.zeros((num_seeds_z, num_seeds_x))
mean_var2_disp_tallh = np.mean(var2_disp_tall[:ts_plot,:,:], axis = 0)










#%%





###############################################################################
########################## Read in data #######################################
###############################################################################

t = 270
t_60 = np.int(t*90/60)
left = 1100
right = 2150
bottom = 0
top = 35*3
near = 160
far = dsh_no.ny-160
ymid = np.int(dsh_no.ny/2)
t_xy = 1



################################################################################
########################### Set up coordinates ################################
################################################################################

vert_resolution = 100
hor_resolution = 150
z_scale = 5./3.

## The code below makes the data terrain following 
x1d = np.arange(0,right-left,1)
z1d = np.arange(0,top,1)

#Create 2D arrays for plotting data (first demnsion for each run)
x2d = np.zeros((6,top, right-left))
z2d = np.zeros((6,top, right-left))
lake = np.zeros((6, right-left))

run = ['dsl_no', 'dsh_no', 'dsl_small','dsh_small', 'dsl_tall', 'dsh_tall']


for j in range(6):
    model_run = eval(run[j])
    lake[j,:] = model_run.xland[0,ymid,left:right].values
    try:
        z = np.array(model_run.zs[0,ymid,left:right])/vert_resolution*3 #Convert to gridpoints
    except:
        z = np.zeros((right-left))+0.4

    for i in range(top):
        x2d[j,i,:] = x1d
    for k in range(right-left):
        z2d[j,:,k] = z1d+z[k]
        
lake[lake == 1] = np.nan
      
  
    
    

###############################################################################
############################## Plot ###########################################
###############################################################################
    
#Colormap
colors1_t = np.array(nclcmaps.colors['amwg256'])#amwg256'])
colors_int_t = colors1_t.astype(int)
colors_t = list(colors_int_t)
cmap_wind = nclcmaps.make_cmap(colors_t, bit = True)

#cmap_wind = nclcmaps.cmap('ncl_default')

#Levels
lmin = -20
lmax = 80.01
levels = np.arange(lmin,lmax, 1)
levels_ticks = np.arange(lmin,lmax,20)
levels_ticks_labels = np.arange(lmin,lmax, 20).astype(int)

#Levels
wlmin = -5.0
wlmax = 5.01
wlevels = np.arange(wlmin,wlmax, 0.5)
wlevels = np.delete(wlevels, np.where(wlevels == 0))


shifted_cmap = shiftedcolormap.shiftedColorMap(cmap_wind, midpoint=1 - lmax/(lmax + abs(lmin)), name='shifted')

    
#########################  Create Fig  ########################################

    
for t in range(200,201):
    print(t)    

    #Create Fig
    fig = plt.figure(num=None, figsize=(14,6), facecolor='w', edgecolor='k')
    #Loop over subplots
    for run in range(1,3):
        
    
        #loop over runs
        run_name = ['mean_x_disp', 'mean_x_disph']
        model_run = eval(run_name[run-1])
        
            
        #############################  Plot xz ###################################
        ax = plt.subplot(210 + run,aspect = 'equal')
        plt.subplots_adjust(left=0.05, bottom=0.3, right=0.95, top=0.98, wspace=0.02, hspace=0.16)
        
        
        #Plot Wind
        wind_plot_xz = plt.contourf(x2d[run-1,:,:], z2d[run-1,:,:]*z_scale, model_run[bottom:top,left:right]*hor_resolution/1000, levels, cmap = shifted_cmap, extend = 'both', alpha = 1,  zorder = 3)
    

        print(run)
        
        #Plot Terrain
        terrain = plt.plot(x1d, z2d[run-1,0,:]*z_scale+0.5, c = 'grey', linewidth = 4, zorder = 14)
        
        #Plot Lake
        lake_plt = plt.plot(x1d, lake[run-1,:]+1, c = 'blue', linewidth = 4, zorder = 15)
        
        #Plot Characteristics
        plt.grid(True, color = 'white', )
        ax.set_xlim([0,right-left])
        ax.set_ylim([0,top*z_scale-5*3*z_scale+1])

        if run == 1 or run == 2:
            ytick = np.arange(0,top*z_scale,5*z_scale*3)
            ax.set_yticks(ytick)
            yticklabs = ytick*vert_resolution/z_scale/1000./3
            ax.set_yticklabels(yticklabs, fontsize = 13)
            ax.set_ylabel("Height (km)", fontsize = 16)
        else:
            ax.yaxis.set_visible(False)
        
        if run == 1 or run == 2:
            xtick = np.arange(0,right-left,20000/hor_resolution+1)
            ax.set_xticks(xtick)
            ax.set_xticklabels((xtick+left)*hor_resolution/1000, fontsize = 13)
            
        else:
            ax.xaxis.set_visible(False)
        
        if run == 2:
            ax.set_xlabel("Distance (km)", fontsize = 16)



    
        #Draw border
        #ax.add_patch(patches.Rectangle((0.3,0),right-left-2,top*z_scale-8,fill=False, zorder = 20, linewidth = 2.5))
        
        #Titles
        props = dict(boxstyle='square', facecolor='white', alpha=1, linewidth = 2)
        if run == 1:
            #plt.title("Low Wind", fontsize = 23, y = 1.03)
            ax.text(1.01, 1, 'Low Wind'.center(15), transform=ax.transAxes,fontsize = 20, rotation = -90)

        if run == 2: 
            #plt.title("High Wind", fontsize = 23, y = 1.03)
            ax.text(1.01, 0.97, 'High Wind'.center(15), transform=ax.transAxes,fontsize = 20, rotation = -90)
        if run == 4: 
            ax.text(1.01, 0.65, 'High Wind'.center(20), transform=ax.transAxes, fontsize = 20, rotation = -90)
        if run == 2:
            ax.text(0.8, -0.6, '1.5 hour time-mean', transform=ax.transAxes, bbox = props, fontsize = 16)

            
    
        
    #Colorbars
    cbaxes = fig.add_axes([0.25, 0.11, 0.5, 0.05])           
    cbar = plt.colorbar(wind_plot_xz, orientation='horizontal', cax = cbaxes, ticks = levels_ticks)
    cbar.ax.tick_params(labelsize=15)
    plt.text(0.22, -1.8, 'Time-mean x-displacement (m)', fontsize = 18)

    #plt.text(0.2, -1.8, 'Time-mean change in ${\Theta}$ (K)', fontsize = 18)
    
    #Quiver
#    plt.quiverkey(quiv, X=-1.65, Y=-0.2, U=20, linewidth = 0.75, color = 'k', label='20 $\mathregular{ms^{-1}}$', labelpos='W', fontproperties={'size': '28'})
    
    #Save and Close
    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/plots/x_disp_xz.png", dpi = 150)
    plt.close(fig)  
    plt.switch_backend('Agg')
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    