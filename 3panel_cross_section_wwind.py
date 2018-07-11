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
#ds_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/output/cm1out_200m_20ms_0500m.nc')
#ds_small = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/output/cm1out_200m_20ms_1500m.nc')
#ds_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/output/cm1out_200m_20ms_2500m.nc')

ds_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19B/run/cm1run_20ms_1500m_tug.nc')
ds_small = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19B/run/cm1run_20ms_1500m_tug.nc')
ds_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19B/run/cm1run_20ms_1500m_tug.nc')



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

#Choose variable
v = eval('ds_no.winterp')

#Locations for cross section
ymid = np.int(v[0,0,:,0].size*0.5)
#Get x from 30% to 95%
xmin = np.int(v[0,0,0,:].size*0.297)
xmax = np.int(v[0,0,0,:].size*0.95)
xlen = xmax-xmin

zmin = 0
zmax = [51,51,51]


#%%


##############################   Plots ########################################
    
#for i in range(150,151):
for i in range(0,np.min([ds_no.dbz[:,0,0,0].size, ds_small.dbz[:,0,0,0].size, ds_tall.dbz[:,0,0,0].size])):
    
    secs = i*120
    
    fig = plt.figure(num=None, figsize=(12,7.7), facecolor='w', edgecolor='k')
    for j in range(1,4):
        subplot = 310 + j
        
        #Label to loop over runs
        run = ['ds_no', 'ds_small', 'ds_tall']
        model_run = eval(run[j-1])
        
        
    
        #Set up plot
        ax = plt.subplot(subplot,aspect = 'equal')
        plt.subplots_adjust(left=0.08, bottom=0.06, right=0.9, top=0.93, wspace=0.4, hspace=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')


        
        ##########  Create Grid ########
        ### The code below makes the data terrain following 
        x1d = np.arange(0,xlen,1)
        y1d = np.array(30*(model_run.z[zmin:zmax[j-1]])) #Multiply by 30 to stretch y
	
        try:
            z = np.array(model_run.zs[0,ymid,xmin:xmax])/1000*30 #Div by 1000 to go to m and mult by 30 to match y dim
        except:
            z = np.zeros((xlen))        


        x2d = np.zeros((zmax[j-1],xlen))
        y2d = np.zeros((zmax[j-1],xlen))
        
        for ii in range(zmax[j-1]):
            x2d[ii,:] = x1d
        for jj in range(xlen):
            y2d[:,jj] = y1d+z[jj]
                
        
        #Levels for variable
        levels = np.arange(-10,10.01,0.1)
        levels_ticks = np.arange(-10,10.01,2)
        
        #Plot variable
        ref_plot = plt.contourf(x2d, y2d, model_run.winterp[i,zmin:zmax[j-1],ymid,xmin:xmax], levels, cmap = cmap_precip, extend = 'both', alpha = 1)
    
        #Plot Terrain
        terrain = plt.plot(x1d, z-0.75, c = 'slategrey', linewidth = 3, zorder = 4)
        
        #Plot Lake
        lake = np.array(model_run.xland[0,ymid,xmin:xmax])
        lake[lake == 1] = np.nan
        lake_plt = plt.plot(x1d, lake-3, c = 'blue', linewidth = 4, zorder = 5)
    
    
        #Labels
        sub_title = ['500m/20ms', '1500m/20ms', '2500m/20m']
        props = dict(boxstyle='square', facecolor='white', alpha=1)
        ax.text(20, 115, sub_title[j-1], fontsize = 17, bbox = props, zorder = 6)
    
        if j == 1:
            #Title
            plt.title("Vertical Velocity [elapsed time = %d seconds]"  % secs, fontsize = 22, y = 1.19) 
            
        
        ### Label and define grid area
        #y-axis
        plt.ylim([-3,np.max(y2d[:,0])+1])
        plt.yticks(np.arange(0, np.max(y2d[:,0]),30))
        ax.set_yticklabels(np.arange(0,np.max(y2d[:,0])/30,1).astype(int), fontsize = 13, zorder = 6)
        #x-axis
        plt.xlim([0,xlen])
        plt.xticks(np.arange(0,xlen,100))
        ax.set_xticklabels(np.arange(0,xmax*0.2,100*0.2).astype(int), fontsize = 13)
        #axis labels
        plt.ylabel('Height (km)', fontsize = 15, labelpad = 8)
        if j == 3:
            plt.xlabel('Distance within Domain (km)', fontsize = 15, labelpad = 7)
    
    #Colorbar
    cbaxes = fig.add_axes([0.92, 0.2, 0.035, 0.55])             
    cbar = plt.colorbar(ref_plot, cax = cbaxes, ticks = levels_ticks)
    cbar.ax.tick_params(labelsize=13)
    plt.text(-.1, -0.12, 'm/s', fontsize = 19)
    
    
    path = '/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/'
    plt.savefig('%s' % path + 'png_for_gifs/3panel_cross_section_20ms_w_%03d.png' % i, dpi = 100)
    plt.close(fig)
    

##Build GIF

os.system('convert -delay 12 -quality 100 /uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/3panel_cross_section_20ms_w_*.png /uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/gifs/3panel_cross_section_20ms_w.gif')

###Delete PNGs





