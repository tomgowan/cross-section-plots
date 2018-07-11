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
ds_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1out_200m_30ms_0000m.nc')
ds_small = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1out_200m_30ms_0500m.nc')
ds_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1out_200m_30ms_1500m.nc')



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
colors1 = np.array(nclcmaps.colors['prcp_1'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_precip = nclcmaps.make_cmap(colors, bit=True)

colors1_t = np.array(nclcmaps.colors['MPL_Greys'])
colors_int_t = colors1_t.astype(int)
colors_t = list(colors_int_t)
cmap_terrain = nclcmaps.make_cmap(colors_t, bit=True)






#%%


##############################   Plots ########################################
    
    
for i in range(0,200):
    
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
        x1d = np.arange(0,900,1)
        y1d = np.array(30*(model_run.z[0:30])) #Multiply by 30 to stretch y
	
        try:
            z = np.array(model_run.zs[0,200,300:1200])/1000*30 #Div by 1000 to go to m and mult by 30 to match y dim
	except:
            z = np.zeros((900))        


        x2d = np.zeros((30,900))
        y2d = np.zeros((30,900))
        
        for ii in range(30):
            x2d[ii,:] = x1d
        for jj in range(900):
            y2d[:,jj] = y1d+z[jj]
                
        
        #Levels for CREF
        levels_cref = np.arange(5,40.01,0.5)
        levels_cref_ticks = np.arange(5,40.01,5)
        
        #Plot reflectivity
        ref_plot = plt.contourf(x2d, y2d, model_run.dbz[i,0:30,200,300:1200], levels_cref, cmap = cmap_precip, extend = 'max', alpha = 1,  zorder = 3)
    
        #Plot Terrain
        terrain = plt.plot(x1d, z-0.75, c = 'slategrey', linewidth = 3, zorder = 4)
        
        #Plot Lake
        lake = np.array(model_run.xland[0,200,300:1200])
        lake_plt = plt.plot(x1d[:540], lake[:540]-3, c = 'blue', linewidth = 4, zorder = 5)
    
    
        #Labels
        sub_title = ['No Mountain/High Wind', '500m Mountain/High Wind', '1500m Mountain/High Wind']
        props = dict(boxstyle='square', facecolor='white', alpha=1)
        ax.text(20, 102, sub_title[j-1], fontsize = 17, bbox = props, zorder = 6)
    
        if j == 1:
            #Title
            plt.title("Reflectivity [elapsed time = %d seconds]"  % secs, fontsize = 22, y = 1.21) 
            
        
        ### Label and define grid area
        #y-axis
        plt.ylim([-3,121])
        plt.yticks(np.arange(0,121,30))
        ax.set_yticklabels(np.arange(0,4.01,1).astype(int), fontsize = 13, zorder = 6)
        #x-axis
        plt.xlim([0,900])
        plt.xticks(np.arange(0,901,100))
        ax.set_xticklabels(np.linspace(60,260,11).astype(int), fontsize = 13)
        #axis labels
        plt.ylabel('Height (km)', fontsize = 15, labelpad = 8)
        if j == 3:
            plt.xlabel('Distance within Domain (km)', fontsize = 15, labelpad = 5)
    
    #Colorbar
    cbaxes = fig.add_axes([0.93, 0.2, 0.035, 0.55])             
    cbar = plt.colorbar(ref_plot, cax = cbaxes, ticks = levels_cref_ticks)
    cbar.ax.tick_params(labelsize=15)
    plt.text(-0.24, -0.10, 'dBZ', fontsize = 21)
    
    

    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/3panel_mtn_200m__cross_section_high_wind%03d.png" % i)
    plt.close(fig)
    

##Build GIF
#os.system('module load imagemagick')
os.system('convert -delay 12 -quality 100 /uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/3panel_mtn_200m__cross_section_high_wind*.png /uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/gifs/3panel_mtn_200m__cross_section_high_wind.gif')

###Delete PNGs





