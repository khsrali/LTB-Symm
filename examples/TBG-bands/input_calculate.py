import numpy as np
import ltbsymm as ls
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# Start a TB object and set/load configuration
mytb = ls.TB()
mytb.set_configuration('1.08_1AA.data', r_cut = 5.7, local_normal=True, nl_method='RS')
mytb.save(configuration =True)

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
mytb.set_Kpoints(['K1','Gamma','M2', 'K2'] , N=32)

# For twisted bilayer graphene sigma=np.abs(V0_pi-V0_sigam)/2 . An approximate value that flat bands are located
mytb.calculate_bands(H_ij, n_eigns = 4, sigma=np.abs(-2.7-0.48)/2, solver='primme', return_eigenvectors = False) 


mytb.save(bands=True)

MPI.Finalize()
