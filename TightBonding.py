from configuration import pwl
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys

class TB:
    def __init__(self):
        pass # we can move some essentials here, if needed
    
    def calculate(self, n_eigns, local_normal_flag=True, load_neigh = True):
        self.n_eigns = n_eigns # number of eigen values to calculate
        self.Eigns = np.zeros([self.n_k_points, self.n_eigns])
        # build neigh_list
        version_ = self.file_name[:-5] +'_cut_' + str(self.cut_fac)
        self.conf_.neigh_list_me_smart(cutoff=self.r_cut, l_width=200, load_ = load_neigh, version_ = version_ ) 
        # build distance matrix
        self.conf_.vector_connection_matrix()
        # build normal vectors
        self.conf_.normal_vec(local_normal_flag)
        
        # build the 'Bone' matrix
        if self.sparse_flag:
            T0 = self.T_bone_sp()
        else:
            T0 = self.T_bone()
        print('T_bone is constructed..')
        
        # make and solve T_meat for each k point
        t_start = time.time()
        for kk in range(self.n_k_points):
            t_loop = time.time()
            
            if self.sparse_flag:
                H = self.T_meat_sp(self.K_path[kk], T0)
            else:
                H = self.T_meat(self.K_path[kk], T0)
        
            eigvals = eigsh(H, k=self.n_eigns, sigma=self.sigma_shift, which='LM', return_eigenvectors=False, mode='normal')
            self.Eigns[kk] = np.real(eigvals) - self.sigma_shift
            print("{:.2f} percent completed, {:.2f}s per K-point, ETR: {:.2f}s".format(100*(kk+1)/self.n_k_points, (time.time() - t_loop), (self.n_k_points-(kk+1))*(time.time() - t_start)/(kk+1) ), end = "\r")
        
        print("Total time: {:.2f} seconds\n".format(time.time() - t_start))
        

    def MBZ(self):
        '''
        not fully implemented yet, I dream of a method which can find the MBZ  and define it all.
        Right now, you have to define G1 and G2 by hand
        !!note!! I don't want G1 and G2 change for larger cells!!
        Maybe later I change my mind
        '''
        vector_b1 = np.array([self.conf_.xlen, 0, 0])
        vector_b2 = np.array([self.conf_.xy, self.conf_.ylen, 0])
        
        alpha_ = 4*np.pi*np.sin(np.deg2rad(self.phi_)) / (3*self.a0) #(np.sqrt(3)*aa) 

        G1 = -1*alpha_ * np.array([np.sqrt(3), -1, 0]) 
        G2 = alpha_ * np.array([0, 2, 0]) 
        
        self.MBZ_gamma = np.array([0,0,0])
        self.MBZ_M = G2 / 2
        self.MBZ_K1 = (G2 + G1) /3
        self.MBZ_K2 = (2*G2 - G1) / 3
        self.MBZ_K2_prime = (2*G1 - G2) /3
        
        self.MBZ_X = (2*G1 - G2) / 4
        self.MBZ_W = G1 / 2
        self.MBZ_Y = G2 / 4        
            
    
    def set_symmetry_path(self, highsymm_points, N):
        
        #N == number of K-points in the given path
        self.symm_label = highsymm_points
        
        ## calculate the Mini BZ        
        self.MBZ()
        
        ## build the path
        self.symm_coords = np.zeros((len(self.symm_label),3))
        ii = 0
        for label_ in self.symm_label:
            if label_ == 'K1':
                self.symm_coords[ii] = self.MBZ_K1
            elif label_ == 'K2':
                self.symm_coords[ii] = self.MBZ_K2
            elif label_ == 'M':
                self.symm_coords[ii] = self.MBZ_M
            elif label_ == 'gamma':
                self.symm_coords[ii] = self.MBZ_gamma
            elif label_ == 'K2_prime':
                self.symm_coords[ii] = self.MBZ_K2_prime
                
            elif label_ == 'X':
                self.symm_coords[ii] = self.MBZ_X
            elif label_ == 'Y':
                self.symm_coords[ii] = self.MBZ_Y
            elif label_ == 'W':
                self.symm_coords[ii] = self.MBZ_W
                
            ii +=1
        
        """ Generates equi-distance high symmetry path 
        along a given points."""
        diff_symm_points = np.diff(self.symm_coords, axis=0)
        self.K_path = np.array(self.symm_coords[0],ndmin=2)

        total_lenght = 0 # geodesic length
        step_list = np.zeros(diff_symm_points.shape[0]+1) # number of points between two high symm_coords # +1 is to include the first point
        for ii in range(diff_symm_points.shape[0]): # calculate total geodesic length
            symmetry_point_linear_displacement = np.linalg.norm(diff_symm_points[ii]) 
            total_lenght += symmetry_point_linear_displacement
        
        for ii in range(diff_symm_points.shape[0]): # find and attach "steps" number of points between two high symm_coords to path.
            symmetry_point_linear_displacement = np.max([elm for elm in np.abs(diff_symm_points[ii])])
            steps=int(np.round(symmetry_point_linear_displacement*N/total_lenght))
            step_list[ii+1] = steps
            for jj in range(steps):
                self.K_path = np.append(self.K_path,[self.K_path[-1] + diff_symm_points[ii] * 1.0 / steps], axis=0)
                
        print("requested n_k_points=", N)
        self.n_k_points = self.K_path.shape[0]
        print("actual n_k_points=",self.n_k_points)
        
        self.K_path_Highsymm_indices = step_list
     
        

    
    def set_configuration(self, file_name, phi_, sparse_flag=True):
        '''
        phi_ is the twist angle
        '''
        self.phi_ = phi_/2
        self.file_name= file_name
        self.sparse_flag=sparse_flag
        
        #phi_ = np.nan # 1.08455/2  #2.1339/2 #1.050120879794409/2  #1.08455/2 # from Jin
        self.conf_ = pwl(self.file_name, self.sparse_flag)
        

    def save(self, str_='', write_param = True):
        if write_param == True:
            self.save_name = self.file_name[:-5] +'_cut_' + str(self.cut_fac) + str_
        else:
            self.save_name = str_
            
        np.savez(self.save_name , Eigns=self.Eigns, K_path=self.K_path, K_path_Highsymm_indices = self.K_path_Highsymm_indices, sigma_shift=self.sigma_shift, symm_label=self.symm_label) 

        
    
    
    def set_parameters(self, d0, a0, V0_sigam, V0_pi, cut_fac):
        self.d0 = d0
        self.a0 = a0
        self.V0_sigam = V0_sigam
        self.V0_pi = V0_pi
        self.cut_fac = cut_fac
    
    #def __init__(self, d0, a0, V0_sigam, V0_pi, cut_fac):
        ''' apparrantly he made a mistake to use 1.3978 as average C-C distance in rebo potential, 
        while the true value according to Jin is  1.42039011 '''
        #d0 = np.nan #3.344 #3.4331151895378   #3.4331151895378 #from Jin's data point d_ave #3.344 Mattia  # 3.4 AB  ## Mattia 105: 3.50168 ## Mattia 1.08: 3.50133635
        # d0 must be <= than minimum interlayer distance! no one explained this, it was hard find  :( 
        #a0 = np.nan  #1.42039011 #Jin #1.3978 Mattia    # 1.42 AB  ## Matia 105: 1.42353  ## Mattia 1.08: 1.43919
        #V0_sigam = np.nan #+0.48 #ev
        #V0_pi    = np.nan #-2.7#-2.8 #ev
        #cut_fac = np.nan #4.01
        
        onsite_ = 0
        self.scaling_factor = 1#0.5 # the famous factor of 2, to be or not to be there!
        self.r_cut = cut_fac*a0 # cutoff for interlayer hopings
        self.r0 = 0.184* a0 * np.sqrt(3) #  0.3187*a0 and -2.8 ev
        self.sigma_shift = self.scaling_factor*np.abs(V0_pi-V0_sigam)/2 

        

    ## plot
    def plotter(self):
        plt.figure(figsize=(5, 10))
        for k_ in range(self.n_k_points):
            yy = self.Eigns[k_, :] *1000
            xx = np.full(self.n_eigns ,k_)
            plt.plot(xx, yy, 'o', linewidth=5, markersize=3, color='C0')

        jj = 0
        xpos_ = []
        for length_ in self.K_path_Highsymm_indices:
            jj += length_
            plt.axvline(jj, color='gray')
            xpos_.append(jj)
            
        plt.xticks(xpos_, self.symm_label,fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel("E (mev)", fontsize=13)
        title_ = ''#save_name
        plt.title(title_)
        plt.grid(axis='y')
        plt.grid(axis='x')
        plt.savefig('Bands_'+self.save_name + ".png")
        #plt.show()

        return  plt
    




############ functions

    def T_bone_sp(self):
        
        vc_mat = self.conf_.dist_matrix
        ez = self.conf_.ez_local
        T00 = sp.lil_matrix((self.conf_.tot_number, self.conf_.tot_number), dtype='float')
        
        if ez.shape == (3,):
            flag_ez = False
            ez_ = ez
        elif ez.shape == (self.conf_.tot_number,3):
            flag_ez = True
        else:
            print('Wrong ez!! there is a bug, please report code: bone_sp')
            exit(1)
        
        for ii in range(self.conf_.tot_number):
            neighs = self.conf_.nl[ii][~(np.isnan(self.conf_.nl[ii]))].astype('int')
            for jj in neighs:
                
                # calculate the hoping
                v_c = np.array([ vc_mat[0][ii,jj],  vc_mat[1][ii,jj],  vc_mat[2][ii,jj] ])
                dd = np.linalg.norm(v_c)
                
                if flag_ez == True: 
                    ez_ = ez[ii]
                    
                tilt = np.power(np.dot(v_c, ez_)/ dd, 2) 
                V_sigam = self.scaling_factor *self.V0_sigam * np.exp(-(dd-self.d0) / self.r0 )
                V_pi    = self.scaling_factor *self.V0_pi    * np.exp(-(dd-self.a0) / self.r0 )
                
                t_d =  V_sigam * tilt + V_pi * (1-tilt)
                
                T00[ii, jj] = t_d
                #t = t_d * np.exp(-1j * np.dot(K_, v_c))
        T00_copy = T00.copy()
        T00_trans = sp.lil_matrix.transpose(T00, copy=True)
        T00_dagger  = sp.lil_matrix.conjugate(T00_trans, copy=True)
        T00 = sp.lil_matrix(T00_dagger + T00_copy)
        
        return T00

    def T_meat_sp(self, K_, T_0):
        
        modulation_matrix = sp.lil_matrix(( self.conf_.tot_number, self.conf_.tot_number), dtype='complex')
        #print('making modulation_matrix..')
        for ii in range(self.conf_.tot_number):
            neighs = self.conf_.nl[ii][~(np.isnan(self.conf_.nl[ii]))].astype('int')
            for jj in neighs:
                
                v_c = np.array([ self.conf_.dist_matrix[0][ii,jj],  self.conf_.dist_matrix[1][ii,jj],  self.conf_.dist_matrix[2][ii,jj] ])
                modulation_matrix[ii,jj] = np.exp(-1j * np.dot(v_c, K_))
        #print('multipling modulation_matrix')
        return T_0.multiply(modulation_matrix)


    def T_meat(self, K_, T_0):
        
        modulation_matrix = np.exp(-1j * np.dot(self.conf_.dist_matrix, K_))
        return_ = T_0 * modulation_matrix
        del modulation_matrix
            
        return return_




    def T_bone(self):
        '''
        please provid ez, only in dimention of (N,3)
        '''
        vc_mat = self.conf_.dist_matrix
        ez = self.conf_.ez_local
            
        dd_mat = np.linalg.norm(vc_mat, axis=2)

        if ez.shape == (3,):
            tilt_mat = np.power(np.dot(vc_mat, ez)/ dd_mat, 2) 
        elif ez.shape == (self.conf_.tot_number, 3):
            tilt_mat = np.zeros((self.conf_.tot_number, self.conf_.tot_number))
            for ii in range(self.conf_.tot_number):
                tilt_mat[ii] = np.power(np.dot(vc_mat[ii], ez[ii])/ dd_mat[ii], 2) 
        else:
            print('Wrong ez!! there is a bug, please report code: bone_full')
            exit(1)
        ##
        ##
        dd_mat_mask = np.zeros(dd_mat.shape, dtype='int')
        dd_mat_mask[dd_mat!=0] = 1

        V_sigam_mat = self.scaling_factor * self.V0_sigam * np.exp(-(dd_mat - self.d0*dd_mat_mask) / self.r0 ) # the factor of two is from H.C
        V_pi_mat    = self.scaling_factor * self.V0_pi    * np.exp(-(dd_mat - self.a0*dd_mat_mask) / self.r0 )

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
            
        return return_
    

