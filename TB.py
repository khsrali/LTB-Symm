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



def vector_connection(POS_ii, POS_jj):
    dist_ = POS_jj - POS_ii
    
    ## for debugging
    shit_0 = shit_1 = shit_2 = shit_3 =0
    old_dist_size = np.linalg.norm(dist_)
    ## 
    
    if dist_[1] > ylen_half:
        dist_[1] -= ylen
        dist_[0] -= xy
        shit_0 =1
    elif -1*dist_[1] > ylen_half:
        dist_[1] += ylen
        dist_[0] += xy
        shit_1 =1
               
    if dist_[0] > xlen_half:
        dist_[0] -= xlen
        shit_2 =1
    elif -1*dist_[0] > xlen_half:
        dist_[0] += xlen
        shit_3 =1
    
    ## for debugging
    dist_size = np.linalg.norm(dist_)
    if dist_size > 1.01*r_cut:
        print('something is wrong with PBC')
        print('POS_ii, POS_jj\n', POS_ii,'\n', POS_jj)
        print('New dist =',dist_size)
        print('Old dist=',old_dist_size)
        print(shit_0,shit_1,shit_2,shit_3)
        exit(1)
    ##
    
    return dist_


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


def T_all(K_, POS_i, POS_j):
    v_c = vector_connection(POS_i, POS_j)
    dd = np.linalg.norm(v_c)
    
    tilt = np.power(np.dot(v_c, ez)/ dd, 2) 
    V_sigam = V0_sigam * np.exp(-(dd-d0) / r0 )
    V_pi    = V0_pi    * np.exp(-(dd-a0) / r0 )
    t_d =  V_sigam * tilt + V_pi * (1-tilt)

    t = t_d * np.exp(-1j * np.dot(K_, v_c))
    
    return t


def log_out(str_s): # for logging purposes, I will use it once the code is polished
    out_file = open(log_file_name, mode='a')
    for str_ in str_s:
        str__ = str_ if type(str_) == str else str(str_)
        out_file.write(str__)
        out_file.write('\n')
    out_file.close()





## main parameters from Mattia's thesis, prx, and prb 
''' apparrantly he made a mistake to use 1.3978 as average C-C distance in rebo potential, 
while the true value according to Jin is  1.42039011 '''
ez = np.array([0,0,1])
d0 = 3.4 #3.4331151895378 #from Jin's data point #3.444 Mattia  # 3.4 #AB
a0 = 1.42 #1.42039011  #1.42039011 #Jin #1.3978 Mattia    # 1.42 # AB
aa = a0 * np.sqrt(3)
r0 = 0.3187*a0 #  0.3187*a0 and -2.8 ev
V0_sigam = +0.48 #ev
V0_pi    = -2.7#-2.8 #ev
cut_fac = 4.1
r_cut = cut_fac*a0 # cutoff for interlayer hopings

## other arguments
file_name = sys.argv[1]
version_ = file_name[:-5] +'_cut_' + str(cut_fac) #'v_0' #
phi_ = 1.050120879794409/2  #1.08455/2 # from Jin

n_k_points = 50 # number of K-points in the given path
n_eigns = 50 # number of eigen values to calculate



#G1 = (1/aa)*(4*np.pi*np.sin(theta)/np.sqrt(3))*np.array([1, -np.sqrt(3), 0]) # from Mattia's thesis, Fuck him, he forgot the dimension
#G2 = (1/aa)*(4*np.pi*np.sin(theta)/np.sqrt(3))*np.array([1, +np.sqrt(3), 0])


### rotate G, It might be needed if we have zigzag, and he had armchair. Still I don't know
##phi_ = np.deg2rad(30)
##rot = np.array([[np.cos(phi_), -np.sin(phi_), 0], [np.sin(phi_), np.cos(phi_), 0], [0,0,1]])

##G1 = np.dot(rot, G1)
##G2 = np.dot(rot, G2)

#gamma = np.array([0,0,0])
#K1 = (G1-G2)/3
#M = +G2/2
#K2 = 1/3 * G2 + 2/3 * G1


#G1 = (1/aa)*(8*np.pi*np.sin(theta)/np.sqrt(3))*np.array([np.sqrt(3)/2, -1/2, 0]) # from Mattia's thesis, Fuck him, he forgot the dimension
#G2 = (1/aa)*(8*np.pi*np.sin(theta)/np.sqrt(3))*np.array([np.sqrt(3)/2, +1/2, 0])

#gamma = np.array([0,0,0])
#K1 = (G1+G2)/3
#M = +G1/2
#K2 = (1/aa)*(8*np.pi*np.sin(theta)/3)*np.array([ +1/2, -np.sqrt(3)/2, 0])

##


gamma = np.array([0,0,0])
alpha_ = 4*np.pi*np.sin(np.deg2rad(phi_)) / (np.sqrt(3)*aa) 
M = -0.5*alpha_*np.array([1, np.sqrt(3), 0])
K1 = alpha_*np.array([0, -2/np.sqrt(3), 0])
K2 = -alpha_*np.array([1, 1/np.sqrt(3), 0])


rot_angle = np.deg2rad(90)
rot = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0], [np.sin(rot_angle), np.cos(rot_angle), 0], [0,0,1]])

K1 = np.dot(rot, K1)
M  = np.dot(rot, M)
K2 = np.dot(rot, K2)
#symm_points = np.array([K1,gamma,M,K2])
#symm_label = ['K1','gamma','M','K2']
symm_points = np.array([gamma, K1, K2, gamma])
symm_label = ['gamma','K1', 'K2', 'gamma']

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

rd_ = pwl(file_name) # read_data
'''I implemented two different approach to calculateso so-called neighbors_list of each atom in r_cut range, 
The pupose was, to make sure I am extracting the right thing from lammps, considering lammps using labels atoms with a different indexing methods, also sometime it might skips someatoms if you define bonds! be careful! 
But now, I checked with correct setting of neigh_list_lammps it works in perfect agreement with neigh_list_me_smart'''

#nl_type = rd_.neigh_list_lammps(file_name, cutoff=r_cut, l_width=100) ## use lammps to extract neighbors_list
nl_type = rd_.neigh_list_me_smart(cutoff=r_cut, l_width=100, load_ = True, version_ = version_ ) ## use numpy functions to do the *double for*. It is fast enough and can even get faster if I use a skin method instead of 9 replica.

N = rd_.tot_number
pos_ = rd_.coords 

xlen = rd_.xlen
ylen = rd_.ylen
xy = rd_.xy

xlen_half = xlen/2
ylen_half = ylen/2

neighbors_list = rd_.nl


## file description
save_name = "{0}".format(version_)
log_file_name = 'out_' + save_name + '.txt'
log_out([save_name, "Misfit=",  phi_, " degrees"])


###
t_start = time.time()
kk =0
for k_ in K_points:
    t_loop = time.time()
    H = sp.lil_matrix((N, N), dtype='complex')
    
    ### all atoms at once!
    for i in range(N):
        neighs = neighbors_list[i][~(np.isnan(neighbors_list[i]))].astype('int')
        for j in neighs:
            #print('i,j', i,j)
            value_   = T_all(k_, pos_[i], pos_[j]) 
            H[i, j] = value_
    
    if nl_type == 'half':
        H_copy = H.copy()
        H_trans = sp.lil_matrix.transpose(H, copy=True)
        H_dagger = sp.lil_matrix.conjugate(H_trans, copy=True)
        H = sp.lil_matrix(H_dagger + H_copy)
    
    ## just for a check
    if kk == 0:        
        row_,col_ = H.nonzero()
        print('tot_non_zero_elements=',row_.shape[0],"\n")
        for iii in range(N):
            if H[i, i] != 0:
                print('H[{0}, {0}] is not zero! but= {1}'.format(i,H[i, i] ))
    ##
    
    ######## Engine to find eigenvalues
    #print("eigsh starting..., time: {:.2f} seconds         \r".format(time.time() - t_start))
    log_out(["eigsh starting..., time: {:.2f} seconds\n".format(time.time() - t_start)])
    # eigvals= eigsh(H, k=n_eigns, which='SM', return_eigenvectors=False, mode='normal')
    eigvals = eigsh(H, k=n_eigns, sigma=0.8, which='LM', return_eigenvectors=False, mode='normal')  ## note: largest eigen values around sigma=0 is crazy faster than smallest eigenvalues!!! I checked they are exact same results. This well known by others as efficency of the algorithem. 
    Eigns[kk] = np.real(eigvals)
    ###
    
    ## 
    np.savez("tmp" + save_name, Eigns=Eigns, K_points=K_points, step_list = step_list) #, Eign_vecs=Eign_vecs)
    kk += 1
    #print(int(100*kk/n_k_points), "percent, done: {:.2f} seconds       \r".format(time.time() - t_loop))
    print("{:.2f} percent completed, {:.2f}s per K-point, ETR: {:.2f}s".format(100*kk/n_k_points, (time.time() - t_loop), (n_k_points-kk)*(time.time() - t_start)/kk ), end = "\r")
    
    
    log_out([int(100*kk/n_k_points), "percent, done: {:.2f} seconds\n".format(time.time() - t_loop)])

## done! , saveing ...
np.savez(save_name, Eigns=Eigns, K_points=K_points, step_list = step_list) #, Eign_vecs=Eign_vecs)
print("Total time: {:.2f} seconds\n".format(time.time() - t_start))
log_out(["Total time: {:.2f} seconds\n".format(time.time() - t_start)])

## plot
plt.figure(figsize=(5, 10))
for k_ in range(K_points.shape[0]):
    yy = Eigns[k_, :] *1000
    xx = np.full(n_eigns ,k_)
    plt.plot(xx, yy, 'o', linewidth=5, markersize=3, color='C0')

Jins_dick = step_list
jj = 0
xpos_ = [0]
plt.axvline(jj, color='black')
for shit_ in Jins_dick:
    jj += shit_
    plt.axvline(jj, color='black')
    xpos_.append(jj)
    
plt.xticks(xpos_, symm_label,fontsize=14)
plt.yticks(fontsize=14)
#plt.xlim([-0.35,0.35])
#plt.xlabel("", fontsize=13)
plt.ylabel("E (mev)", fontsize=13)
title_ = ''#save_name
plt.title(title_)
plt.grid(axis='y')
plt.grid(axis='x')
plt.savefig(save_name + ".png")
plt.show()
 
exit()


## Here I put things which I used to debug and I don't need anymore, but one day I might. So I don't delete...

#fnn = rd_.fnn  ##
### first nearest neighbor !in plain!
#for i in range(N):
    #for j in fnn[i]:
        #H[i, j] += T_first(k_, pos_[i], pos_[j])


#def T_first(K_, POS_i, POS_j):
    ### please implement PBC
    ##v_c = vector_connection(POS_i, POS_j)
    ##t = t0 * np.exp(-1j * np.dot(K_, v_c))
    ##rr = np.linalg.norm(v_c)
    ##t *= np.exp(-beta * (float(rr) / r0 - 1))
    #t= V0_pi
    #return t


#def check_hermiticity(G, atol=1e-8):
    ## return np.allclose(G, np.conjugate(np.transpose(G)), rtol=rtol, atol=atol)
    #GT = np.conjugate(np.transpose(H))
    #GGT = np.abs(np.abs(G - GT)) #- rtol * np.abs(b)
    #bool_ = (GGT.max() <= atol)
    #if not bool_:
        #print("\n shit=false")
        #print(G[19,26])
        #print(GT[19,26])
        #print(GT[19,26]==G[19,26])
        #print(np.abs(np.abs(GT[19,26] - G[19,26])))
        #print(GGT.max())
        #print(GGT.getnnz())
        #exit(1)
    #return bool_

#print("hermitian=", check_hermiticity(H))

## rotate G, It might be needed if we have zigzag, and he had armchair. Still I don't know
#phi_ = np.deg2rad(30)
#rot = np.array([[np.cos(phi_), -np.sin(phi_), 0], [np.sin(phi_), np.cos(phi_), 0], [0,0,1]])

#G1 = np.dot(rot, G1)
#G2 = np.dot(rot, G2)
##
