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



#%%

### Read in with netcdf
#cm1_file = "/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1out_lake_200m_skinny_lake.nc"
#fh = Dataset(cm1_file)
#
#cref = fh.variables['cref'][:,:,:]
#xland  = fh.variables['xland'][:]
##zs  = fh.variables['zs'][:]


#Read in with xarray
ds= xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1out_200m_downstream_mtn_idealized.nc')

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

##########  Create Grid ########
### The code below makes the data terrain following 
x1d = np.arange(0,900,1)
y1d = 30*(ds.z[0:30]) #Multiply by 30 to stretch y
z = np.array(ds.zs[0,200,300:1200])/1000*30 #Div by 1000 to go to m and mult by 30 to match y dim

x2d = np.zeros((30,900))
y2d = np.zeros((30,900))

for i in range(30):
    x2d[i,:] = x1d
for j in range(900):
    y2d[:,j] = y1d+z[j]
        

z = np.array(ds.zs[0,200,300:1200])/1000*30 #Div by 1000 to go to m and mult by 30 to match y dim




##############################   Plots ########################################
    
    
for i in range(0,len(ds.dbz[:,0,0,0])):
    
    secs = i*120
    
    #Set up plot
    fig = plt.figure(num=None, figsize=(12,3), dpi=800, facecolor='w', edgecolor='k')
    ax = plt.subplot(111,aspect = 'equal')
    plt.subplots_adjust(left=0.04, bottom=0.01, right=0.9, top=0.9, wspace=0, hspace=0)
    plt.axis('equal')
    plt.axis('off')
    

    
    #Levels for CREF
    levels_cref = np.arange(5,40.01,0.5)
    levels_cref_ticks = np.arange(5,40.01,5)
    
    #Plot reflectivity
    ref_plot = plt.contourf(x2d, y2d, ds.dbz[i,0:30,200,300:1200], levels_cref, cmap = cmap_precip, extend = 'max', alpha = 1,  zorder = 3)

    #Plot Terrain
    terrain = plt.plot(x1d, z, c = 'slategrey', linewidth = 3)
    
    #Plot Lake
    lake = np.array(ds.xland[0,200,300:1200])
    lake_plt = plt.plot(x1d[:540], lake[:540]-3, c = 'blue', linewidth = 3)

    
    #Title
    plt.title("Reflectivity [dBZ] (elapsed time = %d seconds)"  % secs, fontsize = 20, y = 0.87) 

    #Colorbar    
    cbaxes = fig.add_axes([0.93, 0.08, 0.03, 0.8])             
    cbar = plt.colorbar(ref_plot, cax = cbaxes, ticks = levels_cref_ticks)
    cbar.ax.tick_params(labelsize=14)
    
    #Labels
    sub_title = '2500m Mountain'
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax.text(20, 130, sub_title, fontsize = 18, bbox = props, zorder = 5)
    
    
    

    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/downstream_tall_mtn_200m__cross_section_%03d.png" % i)
    plt.close(fig)
    

##Build GIF
#os.system('module load imagemagick')
#os.system('convert -delay 6 -quality 100 /uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/downstream_tall_mtn_200m_*.png ../gifs/downstream_tall_mtn_cref.gif')

###Delete PNGs





