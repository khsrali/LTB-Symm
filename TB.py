from read_data import play_with_lammps as pwl
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
### note:
'''it seems Mattia's structure has the triangle on of the hexagon on top! 
this might be important because I am using his defination of path for G1 and G2 '''


def highsymm_path(symm_points,n_k_points):
    """ Generates equi-distance high symmetry path 
    along a given points."""
    diff_symm_points = np.diff(symm_points, axis=0)
    path = np.array(symm_points[0],ndmin=2)

    total_lenght = 0 # geodesic length
    step_list = np.zeros(diff_symm_points.shape[0]) # number of points between two high symm_points
    for ii in range(diff_symm_points.shape[0]): # calculate total geodesic length
        symmetry_point_linear_displacement = np.linalg.norm(diff_symm_points[ii]) 
        total_lenght += symmetry_point_linear_displacement
    
    for ii in range(diff_symm_points.shape[0]): # find and attach "steps" number of points between two high symm_points to path.
        symmetry_point_linear_displacement = np.max([elm for elm in np.abs(diff_symm_points[ii])])
        steps=int(np.round(symmetry_point_linear_displacement*n_k_points/total_lenght))
        step_list[ii] = steps
        for jj in range(steps):
            path = np.append(path,[path[-1] + diff_symm_points[ii] * 1.0 / steps], axis=0)
            
    print("requested n_k_points=",n_k_points)
    print("actual n_k_points=",path.shape[0])

    return path, step_list



def T_bone_sp(vc_mat, ez=np.array([0,0,1])):
    
    T00 = sp.lil_matrix((N, N), dtype='float')
    if ez.shape == (3,):
        print("\nusing **global** ez=[0,0,1] ...\n")
        flag_ez = False
        ez_ = ez
    elif ez.shape == (N,3):
        print("\nusing **local** ez ...\n")
        flag_ez = True
    else:
        print('Wrong ez!! please provide only in shape (N,3)')
        exit(1)
    
    for ii in range(N):
        neighs = rd_.nl[ii][~(np.isnan(rd_.nl[ii]))].astype('int')
        for jj in neighs:
            
            # calculate the hoping
            v_c = np.array([ vc_mat[0][ii,jj],  vc_mat[1][ii,jj],  vc_mat[2][ii,jj] ])
            dd = np.linalg.norm(v_c)
            
            if flag_ez == True: 
                ez_ = ez[ii]
                
            tilt = np.power(np.dot(v_c, ez_)/ dd, 2) 
            V_sigam = scaling_factor *V0_sigam * np.exp(-(dd-d0) / r0 )
            V_pi    = scaling_factor *V0_pi    * np.exp(-(dd-a0) / r0 )
            
            t_d =  V_sigam * tilt + V_pi * (1-tilt)
            
            T00[ii, jj] = t_d
            #t = t_d * np.exp(-1j * np.dot(K_, v_c))
    T00_copy = T00.copy()
    T00_trans = sp.lil_matrix.transpose(T00, copy=True)
    T00_dagger  = sp.lil_matrix.conjugate(T00_trans, copy=True)
    T00 = sp.lil_matrix(T00_dagger + T00_copy)
    
    return T00

def T_meat_sp(K_, T_0, vc_mat):
    
    modulation_matrix = sp.lil_matrix((N, N), dtype='complex')
    #print('making modulation_matrix..')
    for ii in range(N):
        neighs = rd_.nl[ii][~(np.isnan(rd_.nl[ii]))].astype('int')
        for jj in neighs:
            
            v_c = np.array([ vc_mat[0][ii,jj],  vc_mat[1][ii,jj],  vc_mat[2][ii,jj] ])
            modulation_matrix[ii,jj] = np.exp(-1j * np.dot(v_c, K_))
    #print('multipling modulation_matrix')
    return T_0.multiply(modulation_matrix)




def T_bone(vc_mat, ez=np.array([0,0,1]) ):
    '''
    please provid ez, only in dimention of (N,3)
    '''
    if sparse_flag:
        return_ = T_bone_sp(vc_mat, ez=ez )
       
    else:
        dd_mat = np.linalg.norm(vc_mat, axis=2)

        if ez.shape == (3,):
            print("\nusing **global** ez=[0,0,1] ...\n")
            tilt_mat = np.power(np.dot(vc_mat, ez)/ dd_mat, 2) 
        elif ez.shape == (N,3):
            print("\nusing **local** ez ...\n")
            tilt_mat = np.zeros((N,N))
            for ii in range(N):
                tilt_mat[ii] = np.power(np.dot(vc_mat[ii], ez[ii])/ dd_mat[ii], 2) 
        else:
            print('Wrong ez!! please provide only in shape (N,3)')
            exit(1)
        ##
        ##
        dd_mat_mask = np.zeros(dd_mat.shape, dtype='int')
        dd_mat_mask[dd_mat!=0] = 1

        V_sigam_mat = scaling_factor * V0_sigam * np.exp(-(dd_mat - d0*dd_mat_mask) / r0 ) # the factor of two is from H.C
        V_pi_mat    = scaling_factor * V0_pi    * np.exp(-(dd_mat - a0*dd_mat_mask) / r0 )

        ##
        del dd_mat


        return_ = V_sigam_mat * tilt_mat + V_pi_mat * (1-tilt_mat) ## will not work for a sparse matrix
        return_[dd_mat_mask==0] = 0.0
        np.nan_to_num(return_, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        del V_pi_mat
        del V_sigam_mat
        del tilt_mat
        
        ## H.C
        return_ = return_ + return_.transpose()
        
    print('T_bone is constructed..')
    return return_
    

def T_meat(K_, T_0, vc_mat):
    
    if sparse_flag:
        return_ = T_meat_sp(K_, T_0, vc_mat)
    
    else:    
        modulation_matrix = np.exp(-1j * np.dot(vc_mat, K_))
        return_ = T_0 * modulation_matrix
        del modulation_matrix
        
    return return_


## main parameters from Mattia's thesis, prx, and prb 
''' apparrantly he made a mistake to use 1.3978 as average C-C distance in rebo potential, 
while the true value according to Jin is  1.42039011 '''
d0 = 3.344 #3.4331151895378   #3.4331151895378 #from Jin's data point d_ave #3.344 Mattia  # 3.4 AB  ## Mattia 105: 3.50168 ## Mattia 1.08: 3.50133635
# d0 must be <= than minimum interlayer distance! no one explained this, it was hard find  :( 
a0 = 1.42039011  #1.42039011 #Jin #1.3978 Mattia    # 1.42 AB  ## Matia 105: 1.42353  ## Mattia 1.08: 1.43919
aa = a0 * np.sqrt(3)
r0 = 0.184*aa #  0.3187*a0 and -2.8 ev
V0_sigam = +0.48 #ev
V0_pi    = -2.7#-2.8 #ev
onsite_ = 0
cut_fac = 4.01
r_cut = cut_fac*a0 # cutoff for interlayer hopings
scaling_factor = 0.5 # the famous factor of 2, to be or not to be there!
sigma_shift = scaling_factor*np.abs(V0_pi-V0_sigam)/2 
phi_ = 2.1339/2#1.08455/2  #2.1339/2 #1.050120879794409/2  #1.08455/2 # from Jin

## other arguments
file_name = sys.argv[1]
version_ = file_name[:-5] +'_cut_' + str(cut_fac) #'v_0' #
sparse_flag = True
#sparse_flag = False
n_k_points = 10 # number of K-points in the given path
n_eigns = 100 # number of eigen values to calculate


## Attemps to use G1 and G2, making it easier for multi-moire, no rotation is required..
N_b1 = 1
N_b2 = 1
alpha_ = 4*np.pi*np.sin(np.deg2rad(phi_)) / (np.sqrt(3)*aa) 
G1 = alpha_ * np.array([np.sqrt(3), -1, 0]) * (1/N_b1)
G2 = alpha_ * np.array([0, 2, 0]) * (1/N_b2)
gamma = np.array([0,0,0])
M = G2 / 2
K1 = (G2 - G1) /3
K2 = (2*G2 + G1) / 3
### The working one
#gamma = np.array([0,0,0])
#alpha_ = 4*np.pi*np.sin(np.deg2rad(phi_)) / (np.sqrt(3)*aa) 
#M = -0.5*alpha_*np.array([1, np.sqrt(3), 0])
#K1 =  alpha_*np.array([0, -2/np.sqrt(3), 0])
#K2 = -alpha_*np.array([1,  1/np.sqrt(3), 0])

#rot_angle = np.deg2rad(90)
#rot = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0], [np.sin(rot_angle), np.cos(rot_angle), 0], [0,0,1]])

#K1 = np.dot(rot, K1)
#M  = np.dot(rot, M)
#K2 = np.dot(rot, K2)
#symm_points = np.array([K1,gamma,M,K2])
#symm_label = ['K1','gamma','M','K2']
symm_points = np.array([gamma, M])
symm_label = ['gamma', 'M']





## useful for unrelax to compare with Mattia, it seems he has an extra factor of 2!
#symm_points = np.array([gamma, K1, K2, gamma])
#symm_label = ['gamma','K1', 'K2', 'gamma']
#symm_points = np.array([gamma, K1])
#symm_label = ['gamma','K1']

#### works for AB!! for this _-_ shape of graphene
#M_mono = 2*np.pi/(2*np.sqrt(3)*aa*50) * np.array([np.sqrt(3), -1, 0])
#K_mono = 4*np.pi/(3*np.sqrt(3)*aa*50) * np.array([np.sqrt(3),  0, 0])
#gamma = np.array([0,0,0])

##symm_points = np.array([gamma,M_mono,K_mono,gamma])
##symm_label = ['gamma','M', 'K', 'gamma']
#symm_points = np.array([K_mono,gamma])
#symm_label = ['K', 'gamma']
####



K_points, step_list = highsymm_path(symm_points, n_k_points)
n_k_points = K_points.shape[0] 
Eigns = np.zeros([n_k_points, n_eigns])

##

rd_ = pwl(file_name, sparse_flag) # read_data
'''I implemented two different approach to calculateso so-called neighbors_list of each atom in r_cut range, 
The pupose was, to make sure I am extracting the right thing from lammps, considering lammps using labels atoms with a different indexing methods, also sometime it might skips someatoms if you define bonds! be careful! 
But now, I checked with correct setting of neigh_list_lammps it works in perfect agreement with neigh_list_me_smart'''

#nl_type = rd_.neigh_list_lammps(file_name, cutoff=r_cut, l_width=100) ## use lammps to extract neighbors_list
nl_type = rd_.neigh_list_me_smart(cutoff=r_cut, l_width=200, load_ = True, version_ = version_ ) ## use numpy functions to do the *double for*. It is fast enough and can even get faster if I use a skin method instead of 9 replica.

if nl_type == 'half' : 
    print('half neigh_list_lammps is not supported anymore!')
    exit(1)
    
N = rd_.tot_number
pos_ = rd_.coords 

xlen = rd_.xlen
ylen = rd_.ylen
xy = rd_.xy

xlen_half = xlen/2
ylen_half = ylen/2

neighbors_list = rd_.nl

rd_.vector_connection_matrix()
vector_mat = rd_.dist_matrix
vector_mat.flags["WRITEABLE"] = False

rd_.normal_vec()


T0 = T_bone(vector_mat, ez=rd_.ez_local)


## file description
save_name = "{0}".format(version_)


###
t_start = time.time()
kk =0
for k_ in K_points:
    t_loop = time.time()
    #if sparse_flag:
        #H = sp.lil_matrix((N, N), dtype='complex')
    
    H = T_meat(k_, T0, vector_mat)

    ## just for a check
    if kk == 0:        
        row_,col_ = H.nonzero()
        print('tot_non_zero_elements=',row_.shape[0],"\n")

    
    ######## Engine to find eigenvalues
    #print('solving..')
    eigvals = eigsh(H, k=n_eigns, sigma=sigma_shift, which='LM', return_eigenvectors=False, mode='normal')  ## note: largest eigen values around sigma=0 is crazy faster than smallest eigenvalues!!! I checked they are exact same results. This well known by others as efficency of the algorithem. 
    #print('solved..')
    Eigns[kk] = np.real(eigvals)
    ###
    
    kk += 1
    print("{:.2f} percent completed, {:.2f}s per K-point, ETR: {:.2f}s".format(100*kk/n_k_points, (time.time() - t_loop), (n_k_points-kk)*(time.time() - t_start)/kk ), end = "\r")
    

## done! , saveing ...
np.savez(save_name, Eigns=Eigns, K_points=K_points, step_list = step_list, sigma_shift=sigma_shift, symm_label=symm_label) 
print("Total time: {:.2f} seconds\n".format(time.time() - t_start))

## plot
plt.figure(figsize=(5, 10))
for k_ in range(K_points.shape[0]):
    yy = (Eigns[k_, :]-sigma_shift) *1000
    xx = np.full(n_eigns ,k_)
    plt.plot(xx, yy, 'o', linewidth=5, markersize=3, color='C0')

Jins_dick = step_list
jj = 0
xpos_ = [0]
plt.axvline(jj, color='black')
for shit_ in Jins_dick:
    jj += shit_
    plt.axvline(jj, color='gray')
    xpos_.append(jj)
    
plt.xticks(xpos_, symm_label,fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("E (mev)", fontsize=13)
title_ = ''#save_name
plt.title(title_)
plt.grid(axis='y')
plt.grid(axis='x')
plt.savefig(save_name + ".png")
plt.show()
 
exit()

