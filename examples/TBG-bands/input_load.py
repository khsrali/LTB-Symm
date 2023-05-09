import numpy as np
import ltbsymm as ls
import matplotlib.pyplot as plt


#Start a TB object and set/load configuration
mytb = ls.TB()

#mytb.load('out_1.08_1AA', bands='bands_.npz', configuration='configuration_.npz')
mytb.load('out_1.08_1AA', bands='bands_.npz')#, configuration='configuration_.npz')

# Detect if there are ant flatbands
mytb.detect_flat_bands()

# Set Fermi level by shifting E=0 to the avergage energies of flat bands at point e.g. 'K1' 
mytb.shift_2_zero('K1', np.array([0,1,2,3]))


# Plot bands and modify figure as you like
plot = mytb.plotter_bands(color_ ='C0') 
plot.set_ylim([-10,15])
plt.savefig('out_1.08_1AA/'+'Bands_'+ ".png", dpi=150) 

plt.show()
