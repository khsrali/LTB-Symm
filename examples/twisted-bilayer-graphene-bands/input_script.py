#!/usr/bin/env python3
import sys
import numpy as np
#from TightBinding_fucnSP12 import TB
from TightBinding import TB
from symmetry import Symm
import matplotlib
import matplotlib.pyplot as plt
from mpi4py import MPI
from matplotlib.gridspec import GridSpec
from functools import partial

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


#Start TB object and set/load configuration
mytb = TB()
#mytb.set_configuration('1.08_1AA.data', r_cut = 5.7, local_normal=True, nl_method='RC')#'RS') 
#mytb.save('ali')
mytb.load_configuration('out_1.08_1AA', ver_='ali')

# Define Hamiltonian and fix the parameters of the Hamiltonian that are the same for all pairs 
def H_ij(v_ij, ez_i, ez_j, a0 = 1.42039011, d0 = 3.344, V0_sigam = +0.48, V0_pi = -2.7, r0 = 0.184* 1.42039011 * np.sqrt(3) ):
    """
        Args:
            d0: float
                Distance between two layers. Notice d0 <= than minimum interlayer distance, otherwise you are exponentially increasing interaction!
            a0: float
                Equilibrium distance between two neghibouring cites.
            V0_sigam: float
                Slater-Koster parameters
            V0_pi: float
                Slater-Koster parameters
            r0: float
                Decay rate of the exponential
    """
    #print(v_ij, ez_i, ez_j)
    dd = np.linalg.norm(v_ij)
    V_sigam = V0_sigam * np.exp(-(dd-d0) / r0 )
    V_pi    = V0_pi    * np.exp(-(dd-a0) / r0 )
    
    tilt_1 = np.power(np.dot(v_ij, ez_i)/ dd, 2)
    tilt_2 = np.power(np.dot(v_ij, ez_j)/ dd, 2)
    t_ij =  V_sigam * (tilt_1+tilt_2)/2 + V_pi * (1- (tilt_1 + tilt_2)/2) 
    
    return t_ij


# Define MBZ and set K-points
mytb.MBZ()
#mytb.set_Kpoints(['Gamma','K1'] , N=8, saveH=False)
mytb.set_Kpoints(['K1','Gamma','M2', 'K2'] , N=8)

# For twisted bilayer graphene sigma=np.abs(V0_pi-V0_sigam)/2 . An approximate value that flat bands are located
mytb.calculate_bands(H_ij, n_eigns = 4, sigma=np.abs(-2.7-0.48)/2, solver='primme' , return_eigenvectors = False) # 'scipy' 'primme'


mytb.save()

#if rank ==0:
    #plot = mytb.plotter_bands(color_ ='C0')
    
exit()

mytb = TB()
mytb.load('out_1.08_1AA')
mytb.load_configuration('out_1.08_1AA', ver_='ali')

sm = Symm(mytb)
sm.build_map('C2z',['X','Y','Z'])

exit()


if rank ==0:
    # === PLOTTING ===
    save_flag = True #False
    font = {'pdf.fonttype' : 42,
            'font.size' : 20,
            'font.family' : 'Helvetica'}
    plt.rcParams.update(font)

    #fig = plt.figure(figsize=(10,4))
    #gs = GridSpec(1,2, width_ratios=[4,1], wspace=0.1)
    #fig = plt.figure(figsize=(8,4))
    #gs = GridSpec(2,1, height_ratios=[1,4], wspace=0.1)
    #axB = fig.add_subplot(gs[0])
    #axDos = fig.add_subplot(gs[1], sharey=axB)
    #plt.setp(axDos.get_yticklabels(), visible=False)
    #plt.setp(axDos.get_xticklabels(), visible=False)

    #axDos = fig.add_subplot(gs[1])
    #axB = fig.add_subplot(gs[1])
    # --- PLOT DOS ---
    #fig, (axB, axDos) = plt.subplots(1,2, sharey=True)#, figsize=(5,10))
    #plot = mytb.plotter_DOS(axDos)

    # --- PLOT BANDS ---
    # ax = plt.gca()
    #plot = mytb.plotter(axB)
    #plot = mytb.plotter(shift_tozero = -1.9318658113479614) # Unbuckled
    #plot = mytb.plotter(shift_tozero = -2.8900318, color_ ='C0') # 1fold d0 = 3.344 cutoff 4.01
    #plot = mytb.plotter(shift_tozero = -3.9582479578324783, color_ ='C0') # 1fold d0 = 3.3 cutoff 7.0
    mytb.detect_flat_bands()
    mytb.shift_2_zero('K1', np.array([0,1,2,3]))
    plot = mytb.plotter_bands(color_ ='C0') 
    #plot = mytb.plotter( shift_tozero = 9.077795017749024, color_ ='C0') # 1fold cut 3.0, d0 =3.3
    #plot = mytb.plotter(color_ ='C0') # 1fold
    #plot = mytb.plotter(color_ ='C2') # 3fold
    
    #plot.set_ylim([-12,25])
    plot.set_ylim([-10,15])
    plt.savefig(mytb.folder_name+'Bands_'+mytb.save_name + ".png", dpi=300)
    #mytb.build_sym_operation(tol_=0.1)
    plt.show()
    #mytb.embed_flatVec(singlePoint)
    #tol_ = 0.3
    #mytb.point_symmetry_check(singlePoint, 'C2x',  tol_=tol_ )
    #exit()
    #mytb.point_symmetry_check(singlePoint, 'C2y',  tol_=tol_ )
    #mytb.point_symmetry_check(singlePoint, 'C2z',  tol_=tol_ )


    # 3D plot
    #plot = mytb.plotter_3D(shift_tozero = -2.8900318, color_ ='C0') # 1fold
    #plot.set_ylim([-10,15])
    #plt.savefig(mytb.folder_name+'BandsDOS_'+mytb.save_name +'_all'+ ".png", dpi=300)
    


    #=====================

    ## note:
    #angles: #1.08455  #2.1339 #1.050120879794409 # from Jin



    ## for only plotting
    #y_low = -10
    #y_high = +15
    #mytb = TB()
    ##mytb.load('calculation_1.08_3fold_type_c_cut_4.01.npz') # to be written
    ##mytb.load('1.08_1AA_cut_4.01.npz') # to be written
    #mytb.load(sys.argv[1]) # to be written
    ##mytb.load('1.08_buckled_best_wish_Ali_cut_4.01_redundant.npz') # to be written
    #plot = mytb.plotter(color_='C0', save_flag =False) #seagreen 'tomato'
    ##plot.show()
    ##exit()
    #plot.savefig(mytb.folder_name+'Bands_'+mytb.save_name +'_all'+ ".png", dpi=300)
    ##plot.show()
    ##exit()
    #fig = plot.gcf()
    #plot.ylim([y_low,y_high])
    #fig.set_size_inches(16.65, 9.45, forward=True)
    #plot.savefig(mytb.folder_name+'Bands_'+mytb.save_name +'_flat'+ ".png", dpi=300)

    #plot.xlim([20,50])
    #plot.ylim([-5,+5])
    #fig = plot.gcf()
    #fig.set_size_inches(5.0, 9.45, forward=True)
    #plot.savefig(mytb.folder_name+'Bands_'+mytb.save_name +'_gamma'+ ".png", dpi=300)




    #plot.show()

    ### crap
    ##plot.set_color("black")