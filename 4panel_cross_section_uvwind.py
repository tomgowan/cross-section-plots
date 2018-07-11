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




#Read in with xarray
ds = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1out_200m_20ms_1500m.nc')



#%%




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
colors1 = np.array(nclcmaps.colors['BlWhRe']) #'prcp_1' for precip
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_precip = nclcmaps.make_cmap(colors, bit=True)

colors1_t = np.array(nclcmaps.colors['MPL_Greys'])
colors_int_t = colors1_t.astype(int)
colors_t = list(colors_int_t)
cmap_terrain = nclcmaps.make_cmap(colors_t, bit=True)





############ Variables for cross section ##########
# Made so this can be plotted dynamically

#########  For u-wind #######
#Choose variable
v = eval('ds.u')

#Locations for cross section
ymid = np.int(v[0,0,:,0].size*0.5)
#Get x from 30% to 95%
xmin = np.int(v[0,0,0,:].size*0.297)
xmax = np.int(v[0,0,0,:].size*0.95)
xlen = xmax-xmin

#########  For v-wind #######
v = eval('ds.v')
ylen = len(v[0,0,:504,0]) #For some reason v has 505 gridpoints, while other variables have 504
                          #so we restrict to 504 gridpoints

zmin = 0
zmax = 51




#%%


##############################   Plots ########################################
    
#for i in range(130,131):
for i in range(0,len(ds.u[:,0,0,0])):
    
    secs = i*120
    
    fig = plt.figure(num=None, figsize=(12,5.5), facecolor='w', edgecolor='k')

        
    #Run
    run = ['ds']
    model_run = eval(run[0])
    
    
    #Set up plot
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=1)
    ax3 = plt.subplot2grid((2, 3), (1, 1), colspan=1)
    ax4 = plt.subplot2grid((2, 3), (1, 2), colspan=1)
    
    plt.subplots_adjust(left=0.08, bottom=0.2, right=0.9, top=0.93, wspace=0.2, hspace=0.6)
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')


    #Levels for variable
    levels_u = np.arange(-40,40.01,0.5)
    levels_ticks_u = np.arange(-40,40.01,10)
    
    levels_v = np.arange(-15,15.01,0.25)
    levels_ticks_v = np.arange(-15,15.01,5)
    
    
    
    ##########################  Create Grids for u-wind  ######################
    ### The code below makes the data terrain following 
    x1d = np.arange(0,xlen,1)
    y1d = np.array(30*(model_run.z[zmin:zmax])) #Multiply by 30 to stretch y
	
    try:
        z = np.array(model_run.zs[0,ymid,xmin:xmax])/1000*30 #Div by 1000 to go to m and mult by 30 to match y dim
    except:
        z = np.zeros((xlen))        

    x2d = np.zeros((zmax,xlen))
    y2d = np.zeros((zmax,xlen))
    
    for ii in range(zmax):
        x2d[ii,:] = x1d
    for jj in range(xlen):
        y2d[:,jj] = y1d+z[jj]
            
    #Plot variable
    u_plot = ax1.contourf(x2d, y2d, model_run.u[i,zmin:zmax,ymid,xmin:xmax], levels_u, cmap = cmap_precip, extend = 'both', alpha = 1)
    #Plot Terrain
    terrain = ax1.plot(x1d, z-0.75, c = 'slategrey', linewidth = 3, zorder = 4)
    #Plot Lake
    lake = np.array(model_run.xland[0,ymid,xmin:xmax])
    lake[lake == 1] = np.nan
    lake_plt = ax1.plot(x1d, lake-3, c = 'blue', linewidth = 4, zorder = 5)
    
    ### Label and define grid area
    #y-axis
    ax1.set_ylim([-3,np.max(y2d[:,0])+1])
    ax1.set_yticks(np.arange(0, np.max(y2d[:,0]),30))
    ax1.set_yticklabels(np.arange(0,np.max(y2d[:,0])/30,1).astype(int), fontsize = 13, zorder = 6)
    #x-axis
    ax1.set_xlim([0,xlen])
    ax1.set_xticks(np.arange(0,xlen,100))
    ax1.set_xticklabels(np.arange(xmin*0.2,xmax*0.2,100*0.2).astype(int), fontsize = 13)
    #axis labels
    ax1.set_ylabel('Height (km)', fontsize = 15, labelpad = 8)
    ax1.set_xlabel('Distance within Domain (km)', fontsize = 15, labelpad = 7)
    
    
    
    
    #####################  Create Grids for v-wind  ###########################
    ### The code below makes the data terrain following 
    x1d_v = np.arange(0,ylen,1)
    y1d_v = np.array(30*(model_run.z[zmin:zmax])) #Multiply by 30 to stretch y
	
    try:
        z_v = np.array(model_run.zs[0,:,1000])/1000*30 #Div by 1000 to go to m and mult by 30 to match y dim
    except:
        z_v = np.zeros((ylen))        

    x2d_v = np.zeros((zmax,ylen))
    y2d_v = np.zeros((zmax,ylen))
    
    for ii in range(zmax):
        x2d_v[ii,:] = x1d_v
    for jj in range(ylen):
        y2d_v[:,jj] = y1d_v+z_v[jj]
    
    
    #Plot variables
    v_plot = ax2.contourf(x2d_v, y2d_v, model_run.v[i,zmin:zmax,:504,xmin+1*xlen/4], levels_v, cmap = cmap_precip, extend = 'both', alpha = 1)
    v_plot = ax3.contourf(x2d_v, y2d_v, model_run.v[i,zmin:zmax,:504,xmin+2*xlen/4], levels_v, cmap = cmap_precip, extend = 'both', alpha = 1)
    v_plot = ax4.contourf(x2d_v, y2d_v, model_run.v[i,zmin:zmax,:504,xmin+3*xlen/4], levels_v, cmap = cmap_precip, extend = 'both', alpha = 1)





#    #Labels
#    sub_title = ['1500 Mountain/Low Wind', '2500m Mountain/Low Wind', '2500m Mountain/High Wind']
#    props = dict(boxstyle='square', facecolor='white', alpha=1)
#    plt.text(20, 115, sub_title, fontsize = 17, bbox = props, zorder = 6)
#
#    #Title
#    plt.title("U-Wind [elapsed time = %d seconds]"  % secs, fontsize = 22, y = 1.21) 
#        
#    
#    ### Label and define grid area
#    #y-axis
#    plt.ylim([-3,np.max(y2d[:,0])+1])
#    plt.yticks(np.arange(0, np.max(y2d[:,0]),30))
#    ax1.set_yticklabels(np.arange(0,np.max(y2d[:,0])/30,1).astype(int), fontsize = 13, zorder = 6)
#    #x-axis
#    plt.xlim([0,xlen])
#    plt.xticks(np.arange(0,xlen,100))
#    ax1.set_xticklabels(np.arange(xmin*0.2,xmax*0.2,100*0.2).astype(int), fontsize = 13)
#    #axis labels
#    plt.ylabel('Height (km)', fontsize = 15, labelpad = 8)
#    plt.xlabel('Distance within Domain (km)', fontsize = 15, labelpad = 7)
#    
#    #Colorbar
#    cbaxes = fig.add_axes([0.92, 0.2, 0.035, 0.55])             
#    cbar = plt.colorbar(u_plot, cax = cbaxes, ticks = levels_ticks_u)
#    cbar.ax.tick_params(labelsize=13)
#    plt.text(-.1, -0.12, 'm/s', fontsize = 19)
    
    
    path = '/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/'
    plt.savefig('%s' % path + 'png_for_gifs/cross_section_uv_%03d.png' % i, dpi = 100)
    plt.close(fig)
    

##Build GIF

os.system('convert -delay 12 -quality 100 /uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/cross_section_uv_*.png /uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/gifs/cross_section_uv.gif')

###Delete PNGs





