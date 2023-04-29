import time, os, sys
from configuration import pwl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.sparse as sp
#from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from mpi4py import MPI
from misc import get_brillouin_zone_2d, plot_BZ2d#, get_uc_patch2D
import spglib
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.linalg import ishermitian
import primme
import warnings


'''
This code is using MPI
'''


class TB:
    def __init__(self, ):
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        print('TB object created')


    def build_up(self, H_style, local_normal=True, load_neigh=True, nl_method='RS'):
        """
            Build up some basic requirments: normal vectors, neighborlist, and the bone matrix

            Args:
                H_style: str
                    type of the Hamiltonian, 'ave' or '9X'
                local_normal: boolean
                    To use local_normal to orbital direction(True) or consider them all aligned in Z direction (False). (default = True)
                load_neigh: boolean
                    load a previously created neighborlist. (default = True)
                nl_method: str
                    which method to use for creating neighborlist. 'RS' -reduce space implementation- (faster but might have a bug in rhombic cells, to be investigated) or 'RC' -replicating coordinates- (slower, bug free) (default = 'RS')
            Returns: None
        """
        
        # check input args
        try:
            assert H_style == 'ave' or dtype == '9X'
        except AssertionError: raise TypeError('Only ave(averaged) or 9X(full 9-term Hamiltonian) is supported')
            
        # build neigh_list
        version_ = self.file_name[:-5] +'_cut_' + str(self.cut_fac)
        
        if self.rank == 0:
            # calculate neigh_list
            self.conf_.neigh_list(cutoff=self.r_cut,  method=nl_method, load_ = load_neigh, version_ = version_ )
            
            # send to other cups
            signal_ = True
            for ii in range(1,self.size):
                req = self.comm.isend(signal_, dest=ii, tag=11)
                req.wait()
        else:
            req = self.comm.irecv(source=0, tag=11)
            signal_ = req.wait() 
            assert signal_ is True
            self.conf_.neigh_list(cutoff=self.r_cut, load_ = True, version_ = version_ ) 
            
        
        # build distance matrix
        self.conf_.vector_connection_matrix()
        # build normal vectors
        self.conf_.normal_vec(local_normal)

        # build the 'Bone' matrix
        if self.sparse_flag:
            self.T0 = self.T_bone_sp(H_style)
        else:
            self.T0 = self.T_bone(H_style)
        print('T_bone is constructed..')
        

    def engine_mpi(self, kpoints, n_eigns, solver='primme', return_eigenvectors=False):
        """
            Engine is mpi parallel! 
            Soon to be a private method.
            
            Args:
                kpoints: numpy array in the shape of (n,3)
                    K points
                
                n_eigns: 
                    Number of eigen values desired out of Lanczos solver. 
                
                solver: str
                    'primme' (default) or 'scipy'. Caution!  scipy is faster sometimes but it has a error propagation bug in eigenvectros. sometimes returns nonorthonormal. Perhaps a bug in their's GramSchmidt. 
                    For symmetry checking 'primme' is recommended. 
                    While for band structure calculation 'scipy' is recommended.
                
                return_eigenvectors: boolean
                    True (default)
        """
        
        # Check args
        try:
            assert solver == 'primme' or solver == 'scipy'
        except AssertionError:
            raise ValueError("Wrong solver! Available options for solver: 'primme' or 'scipy' ")
        
        if solver=='scipy' and return_eigenvectors==True:
            warnings.warn("Setting not recommended!\n scipy has a bug regarding orthonormality of eigenvectors. It is better to use solver solver= 'primme' if you want right symmetries in your wavefunction. Eigenvalues remain bug free.",category=Warning)
        #
        
        npoints = np.array(kpoints).shape[0]
        Eigns = np.zeros([npoints, n_eigns], dtype=self.dtypeR)
        if return_eigenvectors:
            Eignvecs = np.zeros([npoints, self.conf_.tot_number, n_eigns], dtype=self.dtypeC)
            
        
        
        ## new slicing
        share_ = npoints // self.size 
        share_left = npoints % self.size 
        slice_i = (self.rank)*share_ 
        slice_f = (self.rank+1)*share_ 
        
        kk_range = np.arange(slice_i, slice_f)
        
        rev_rank = self.size - self.rank 
        if rev_rank <= share_left:
            kk_range = np.append(kk_range, -rev_rank)
            
        slice_size = kk_range.shape[0]

        print('Start loop on %i K-points' % npoints)
        t_start = time.time()
        
        if self.rank ==0 or True:
            pbar = tqdm(total=slice_size, unit=' K-point', desc='rank {}'.format(self.rank)) # Initialise
        
        for kk in kk_range:
            t_loop = time.time()

            if self.sparse_flag:
                H = self.T_meat_sp(kpoints[kk], self.T0 )
            else:
                H = self.T_meat(kpoints[kk], self.T0 )
            
            
            if return_eigenvectors:
                if solver == 'primme':
                    eigvals, Eignvecs[kk] = primme.eigsh(H, k=n_eigns, return_eigenvectors=True, which=self.sigma_shift, tol=1e-12)
                else:
                    eigvals, Eignvecs[kk] = eigs(H, k=n_eigns, sigma=self.sigma_shift, which='LM', return_eigenvectors=True, )
                
            else:
                if solver == 'primme':
                    eigvals = primme.eigsh(H, k=n_eigns, return_eigenvectors=False, which=self.sigma_shift, tol=1e-12)
                else:
                    eigvals = eigs(H, k=n_eigns, sigma=self.sigma_shift, which='LM', return_eigenvectors=False,)
            
            Eigns[kk] = np.real(eigvals) - self.sigma_shift
            #self.H = H.todense()
            if self.saveH:
                try:
                    idxH = np.where(self.K_path_Highsymm_indices==kk)[0][0]
                    self.save(str_=self.K_label[idxH], write_param = False, H = H)
                except IndexError:
                    pass
                
            if self.rank ==0 or True:
                pbar.update()
                
        if self.rank ==0 or True:
            pbar.close()
            
        print("Total time: {:.2f} seconds".format(time.time() - t_start))
        
        
        ## collect from all cpus
        if return_eigenvectors:
            sendbufVec = Eignvecs
            recvbufVec = None
            
        sendbuf = Eigns
        recvbuf = None
        if self.rank == 0:
            recvbuf    = np.zeros([self.size, npoints, n_eigns], dtype=self.dtypeR)
            if return_eigenvectors:
                recvbufVec = np.zeros([self.size, npoints, self.conf_.tot_number, n_eigns], dtype=self.dtypeC)

        self.comm.Gather(sendbuf, recvbuf, root=0)
        if return_eigenvectors:
            self.comm.Gather(sendbufVec, recvbufVec, root=0)
            
        if self.rank == 0:
            for ii in range(1,self.size):
                
                slice_i = (ii)*share_
                slice_f = (ii+1)*share_ #if ii != self.size-1 else npoints
                Eigns[slice_i:slice_f] = recvbuf[ii][slice_i:slice_f]     
                
                if return_eigenvectors:
                    Eignvecs[slice_i:slice_f]  = recvbufVec[ii][slice_i:slice_f]
                
                rev_rank = self.size - ii 
                if  rev_rank <= share_left:
                    Eigns[-rev_rank] = recvbuf[ii][-rev_rank]
                    ##
                    if return_eigenvectors:
                        Eignvecs[-rev_rank]  = recvbufVec[ii][-rev_rank]
            
            print('Succesfully collected eigensvalues ')

            if return_eigenvectors: 
                print('Succesfully collected vectors!')
                return Eigns, Eignvecs
            else:
                return Eigns        
        else:
            return None


    def set_configuration(self, file_and_folder, sparse_flag=True, dtype='float', version=''):
        """
            Set the basic configuration

            Args:
                file_and_folder: str
                    address to a txt file containing coordinates and cell info. 
                    This file should be in one of these formats: lammpstrj, XYZ
                sparse_flag: boolean, optional
                    To use sparse matrix. (default = True)
                    Calculation are faster with non-sparse, but eager on ram size. 
                dtype: str, optional
                    precision of Calculation. float(default) or double is supported. 
                version: str, optional
                    Postfix to name the output folder in format of 'calculation_'+filename(directory excluded)+version
                    
            Returns: None
        """
        
        try:
            assert dtype == 'float' or dtype == 'double'
        except AssertionError: raise TypeError('Only float(default) or double is supported for dtype')
    
        self.sparse_flag=sparse_flag
        self.dtypeR = dtype
        self.dtypeC = 'c'+dtype
        self.file_name = file_and_folder.split('/')[-1]
        self.folder_name = '/'.join(file_and_folder.split('/')[:-1]) #+ '/'

        # Creating new folder:
        new_folder = 'calculation_' + self.file_name +version+ '/'
        self.folder_name += new_folder
        if self.rank == 0:  os.makedirs(self.folder_name, exist_ok=True)
        
        # Call on configuration.py
        self.conf_ = pwl(self.folder_name, self.file_name, self.sparse_flag, self.dtypeR)


    def set_parameters(self, d0, a0, V0_sigam, V0_pi, cut_fac):
        """
            Set some parameters of the twisted-bilayer-model
            Args:
                d0: float
                    Distance between two layers. Notice d0 <= than minimum interlayer distance, otherwise you are exponentially increasing interaction!
                a0: float
                    Equilibrium distance between two neghibouring cites.
                V0_sigam: float
                    ... 
                V0_pi: float
                    ...
                cut_fac: float
                    Neighboring cutoff in unit a0. Circular based neighbor detecting method.
                    larger value includes more neighbor for each cite.
            Returns: None
        """
        self.d0 = d0
        self.a0 = a0
        self.V0_sigam = V0_sigam
        self.V0_pi = V0_pi
        self.cut_fac = cut_fac

        onsite_ = 0 # for later development, right now always 0
        self.r_cut = cut_fac*a0 
        self.r0 = 0.184* a0 * np.sqrt(3) #  0.3187*a0 and -2.8 ev
        # approximate guess for sigma value in the eignvalue problem.
        self.sigma_shift = np.abs(V0_pi-V0_sigam)/2



    def MBZ(self, g1=0, g2=0):
        """
            Get mini brillouin zone or calculates or automatically based on unitcell size.
            Note: automatic method is only implemented for two cases that the angle between a and b lattice vectors is either 60 or 90 degrees.
            Args:
                g1: shape (3,) numpy array
                    Second vector of reciprocal space. 
                    Set 0 to define automatically. (default=0)
                g2: shape (3,) numpy array
                    Second vector of reciprocal space. 
                    Set 0 to define automatically. (default=0)
            Returns: None
        """
        if not (g1==0 and g2==0):
            try:
                assert g1.shape == (3,) and g2.shape == (3,) 
            except AssertionError: 
                raise TypeError("g1 & g1 must be numpy array in shape (3,) or simply leave empty for automatic calculation...")
            self.g1 = g1
            self.g2 = g2
            print('vectors: g1, g2',gx,gy)
            
        else:
            if self.conf_.xy ==0:
                gx = 2*np.pi/np.linalg.norm(self.conf_.xlen) *np.array([1,0,0])
                gy = 2*np.pi/np.linalg.norm(self.conf_.ylen) *np.array([0,1,0])
                self.g1 = gx
                self.g2 = gy
                self.MBZ_X = gx/2
                self.MBZ_W = (gx+gy)/2
                self.MBZ_Y = gy/2
                self.MBZ_gamma = np.array([0,0,0])
                print('vectors: g1, g2',gx,gy)
            else:
                angle_ = np.rad2deg(np.arccos(np.dot(vector_b1,vector_b2)/(np.linalg.norm(vector_b1)*np.linalg.norm(vector_b2))))
                
                if not np.isclose(angle_, 60, atol=1):
                    raise ValueError("angle(a,b) is {0} \nNote: g1 & g2 could be automatically calculated only if the angle between a and b lattice vectors is either 60 or 90 degrees. Otherwise you should provide as input.".format(angle_))
                
                print('Box scew is not zero.. g1 & g2 will not be orthogonal')

                vector_b1 = np.array([self.conf_.xlen, 0, 0])
                vector_b2 = np.array([self.conf_.xy, self.conf_.ylen, 0])
                vector_b3 = np.array([0,0,1])
                
                volume = np.abs(np.dot(vector_b3,  np.cross(vector_b1, vector_b2) ) )            
                g1 = 2*np.pi/volume * np.cross(vector_b3, vector_b2)
                g2 = 2*np.pi/volume * np.cross(vector_b3, vector_b1)
                print('vectors: g1, g2',gx,gy)
                self.g1 = g1
                self.g2 = g2
                
                self.MBZ_gamma = np.array([0,0,0])
                self.MBZ_m2 = g2 / 2
                self.MBZ_m1 = g1 / 2
                self.MBZ_k1 = (g2 + g1) /3
                self.MBZ_k2 = (2*g2 - g1) / 3
                
    
    def label_translator(self, label_ ):
        """
            Translate label to coordinate of high symmetry point in reciprocal space.
            
            Args:
                label_: str, 
                    Predefined label of high symmetry point
                    For orthogonal cells: 'Gamma', 'X', 'Y', and 'W'
                    For rhombic cells: 'Gamma', 'M1', 'M2', 'K1', 'K2'
                    
            Returns: 
                numpy array in shape of (3,) dtype='float'
            
        """
        try:
            if label_ == 'K1':
                K= self.MBZ_k1
            elif label_ == 'K2':
                K= self.MBZ_k2
            elif label_ == 'M1':
                K= self.MBZ_m1
            elif label_ == 'M2':
                K= self.MBZ_m2
            elif label_ == 'Gamma':
                K= self.MBZ_gamma
            elif label_ == 'X':
                K= self.MBZ_X
            elif label_ == 'Y':
                K= self.MBZ_Y
            elif label_ == 'W':
                K= self.MBZ_W
            else:
                raise KeyError('Unrecognised high symmetry point. \nPlease make sure: 1) if g1 & g2 has found automatically \n 2) you are using the right Key depending on your rhombic or orthogonal unitcell')
        except NameError:
            raise KeyError('Unrecognised high symmetry point. \nPlease make sure: 1) if g1 & g2 has found automatically \n 2) you are using the right Key depending on your rhombic or orthogonal unitcell')
            
        return K
    
    def set_Kpoints(self, K_label, K_path=None, N=0, saveH = False):
        """
            Get a set of K points to calculte bands along, either a discrete or continues.
            
            Args:
                K_label: list of str, list(str)
                    Label of high symmetry points in reciprocal space.
                    They most correspond to K_path respectivly.
                    
                K_path: numpy array in shape of (n, 3), optional
                    All coordinates that you would like to calculate electronics levels for.
                    Note: None(default) is only acceptable, if g1 & g2 are calculated automatically. In that case, high symmetry coordinates are already predifined.  
                    For orthogonal cells: 'Gamma', 'X', 'Y', and 'W'
                    For rhombic cells: 'Gamma', 'M1', 'M2', 'K1', 'K2'
                    
                    Alternatively, you can provide a list of all/high-symmetry coordinates that you give their names for K_label.
                    
                N: int, optional
                    Number of K-points in the given path. If N=0(default) then calculation is done only for the provided list. If non-zero N, calculation is done for N point, displaced uniformly along the provided path.
                    If N != 0, it must be larger than number of elements in K_label.
                saveH: boolean
                    whether to save Hamiltonian at high symmetry points. False(default)
            Returns: None
        
        """
        
        # check arguments
        self.saveH = saveH
        self.K_label = np.array(K_label)
        try: 
            assert N==0 or N > K_label.shape
        except AssertionError:
            raise ValueError("If N != 0, it must be larger than number of elements in K_label. You can set N=0(default) for a discrete calculation")
        try: 
            assert K_path==None or K_path.shape[1] == 3
        except AssertionError:
            raise ValueError("K_path should be an numpy array in shape of (n, 3)")
        
        
        Kmode = 'discrete' if N == 0 else 'continues'
        
        
        #
        if Kmode == 'discrete':
            print('your path is discrete')
        else:
            print('your path is continues')
            print("requested n_k_points={0}".format(N))
            N = self.size*(N//self.size) if N!=1 else N
        #self.K_label = np.array(K_label)
        
        ## build the path
        if K_path==None:
            self.K_path_discrete = np.zeros((self.K_label.shape[0],3))
            ii = 0
            for label_ in self.K_label:
                self.K_path_discrete[ii] = self.label_translator(label_)
                ii +=1
        else:
            self.K_path_discrete = K_path
        
        if Kmode == 'continues':
            """ Generates equi-distance high symmetry path 
            along a given points."""
            diff_discrete = np.diff(self.K_path_discrete, axis=0)
            self.K_path_continues = np.array(self.K_path_discrete[0],ndmin=2)
            
            linear_displacement = np.linalg.norm(diff_discrete,axis=1)
            total_lenght = np.sum(linear_displacement) # geodesic length
            
            step_list = (linear_displacement*(N-1)/total_lenght).astype('int')
            if N!=1:
                step_list[-1] += np.abs(np.sum(step_list) - (N-1))
            
            for ii in range(diff_discrete.shape[0]): # attach "steps" number of points between two high K_path_discrete to path.
                for jj in range(step_list[ii]):
                    self.K_path_continues = np.append(self.K_path_continues,[self.K_path_continues[-1] + diff_discrete[ii] * 1.0 / step_list[ii]], axis=0)
                    
            self.K_path_Highsymm_indices = np.cumsum(np.insert(step_list,0,0))
            
            self.K_path = self.K_path_continues
        
        elif Kmode == 'discrete':
            self.K_path = self.K_path_discrete
            self.K_path_Highsymm_indices = np.arange(self.K_label.shape[0])
            
        self.n_k_points = self.K_path.shape[0]
        print("actual n_k_points={0}".format(self.n_k_points))
    
    
    def make_Cmat(self, K, pls_check=False,tol_ = 0.1):
        '''
        This method makes true C!
        K is label like : Gamma or float in shape(3,)
        '''
        if self.rank != 0:
            raise RuntimeError('Please plot using **if rank==0** in mpi mode')
        
        if np.array(K).shape == (3,) :
            K = np.array(K)
        elif type(K) == str :
            K = self.label_translator(K)
        else:
            raise KeyError('make_Cmat: Please provide either str or [kx,ky,kz]')
        
        #self.K_path[0]
        #assert K.shape[0] == 3
        
        self.Cop_x = sp.lil_matrix((self.conf_.tot_number, self.conf_.tot_number), dtype='int')
        self.Cop_y = sp.lil_matrix((self.conf_.tot_number, self.conf_.tot_number), dtype='int')
        self.Cop_z = sp.lil_matrix((self.conf_.tot_number, self.conf_.tot_number), dtype='int')
        print('**I am making C2 matrix for this K point:**', K )
        for sh in range(self.conf_.tot_number):
            i = sh
            j = self.new_orders[0][sh]
            k = self.new_orders[1][sh]
            l = self.new_orders[2][sh]
            convention_x = 1 if (self.conf_.atomsAllinfo[ i , 4] //self.conf_.xlen_half) %2 == 0 else  np.exp(+1j *  self.conf_.xlen * K[0] )
            convention_y = 1 if (self.conf_.atomsAllinfo[ i , 5] //self.conf_.ylen_half) %2 == 0 else  np.exp(+1j *  self.conf_.ylen * K[1] )
            #convention_y = 1
            self.Cop_x[i, j] = convention_x#*convention_y
            self.Cop_y[i, k] = convention_y#-1 if i<k else +1
            self.Cop_z[i, l] = convention_x*convention_y
        
        if pls_check:
            
            
            def check_commute(A,B):
                AB = A @ B
                BA = B @ A
                nonzAB = AB.nonzero()
                nonzAB_tot = nonzAB[0].shape[0]
                nonzBA = BA.nonzero()
                nonzBA_tot = nonzBA[0].shape[0]
                if nonzAB_tot != nonzBA_tot:
                    print('*they do not commute* : number of non Z elements are no equal')
                    return 1
                elm0 = np.all(nonzAB[0]==nonzBA[0])
                elm1 = np.all(nonzAB[1]==nonzBA[1])
                if not elm0 or not elm1:
                    print("*they do not commute* : operators don't match ")
                    return 1
                
                count_p = 0
                count_m = 0
                for zz in range(nonzAB_tot):
                    i = nonzAB[0][zz]
                    j = nonzAB[1][zz]
                    #print('\r',org[i, j], H_primeY[i, j],end='')
                    if  np.isclose( AB[i, j], BA[i, j], rtol=0.1, atol=0):
                        count_p +=1
                    elif np.isclose( AB[i, j], -BA[i, j], rtol=0.1, atol=0):
                        count_m +=1
                    else:
                        print(AB[i, j], -BA[i, j])
                        print('*they do not commute* : non sensical element')
                        return 1
                        #break
                
                if count_p == nonzAB_tot:
                    print('*they commute*')
                elif count_m == nonzAB_tot:
                    print('*they anti-commute*')
                elif count_p == count_m:
                    print('*they half commute and half anti-commute* : count_p,count_m',count_p,count_m)
                else:
                    print("*they do not commute* : but +- of each other in a non equal way: count_p,count_m",count_p,count_m)
            
            def check_square(A):
                A2 = A @ A
                A2.eliminate_zeros()
                nonz = A2.nonzero()
                nonz_tot = nonz[0].shape[0]
                
                if self.conf_.tot_number < nonz_tot:
                    print('square has more non-zero elements that identity')
                    return 1
                elif self.conf_.tot_number > nonz_tot:
                    print('square has less non-zero elements that identity')
                    return 1
                
                count_p = 0
                count_m = 0
                for zz in range(nonz_tot):
                    i = nonz[0][zz]
                    j = nonz[1][zz]
                    
                    if i != j:
                        print('*square has non diagonal elements*')
                        return 1
                    
                    if  np.isclose( A2[i, j], +1, rtol=0.1, atol=0):
                        count_p +=1
                    elif np.isclose( A2[i, j],-1, rtol=0.1, atol=0):
                        count_m +=1
                    else:
                        print(A2[i, j])
                        print('*square is not +- identity* : A2[i, j]',A2[i, j])
                        return 1
                
                if count_p == nonz_tot:
                    print('*square = identity*')
                elif count_m == nonz_tot:
                    print('*square = -identity*')
                elif count_p == count_m:
                    print('*square =  half and half +- identity* : count_p,count_m',count_p,count_m)
                else:
                    print("*square is not identity* : +-1 in a non equal way: count_p,count_m",count_p,count_m)
            
            ## check comuting features
            print('[Cx, Cy]')
            check_commute(self.Cop_x, self.Cop_y)
            print('[Cz, Cy]')
            check_commute(self.Cop_z, self.Cop_y)
            print('[Cx, Cz]')
            check_commute(self.Cop_x, self.Cop_z)
            print('Cx @ Cx')
            check_square(self.Cop_x)
            print('Cy @ Cy')
            check_square(self.Cop_y)
            print('Cz @ Cz')
            check_square(self.Cop_z)
            
            
            ##
            org = sp.lil_matrix(self.H, dtype=self.H.dtype, copy=True)
            #org = self.H
            
            H_primeX = sp.lil_matrix(sp.linalg.inv(self.Cop_x) @ org @ self.Cop_x)
            H_primeY = sp.lil_matrix(sp.linalg.inv(self.Cop_y) @ org @ self.Cop_y)
            H_primeZ = sp.lil_matrix(sp.linalg.inv(self.Cop_z) @ org @ self.Cop_z)
            
            nonZ = org.nonzero()
            nonZ_tot = nonZ[0].shape[0]
            mistake_limit = 10000
            
            
            print('checking C2x')
            pbar = tqdm(total=nonZ_tot, unit=' check', desc='C2x ') # Initialise
            case = True
            mistake_buffer = 0
            for zz in range(nonZ_tot):
                i = nonZ[0][zz]
                j = nonZ[1][zz]
                if not np.isclose( org[i, j], H_primeX[i, j], rtol=tol_, atol=0):
                    mistake_buffer += 1
                    if mistake_buffer > mistake_limit:
                        print(org[i, j], H_primeX[i, j])
                        case = False
                        break
                pbar.update()
            pbar.close()
            if case: print('Hamiltoninan is invariance under C2x. Buffer: ',mistake_buffer)
                    
        
            print('checking C2y')
            pbar = tqdm(total=nonZ_tot, unit=' check', desc='C2y ') # Initialise
            case = True
            mistake_buffer = 0
            for zz in range(nonZ_tot):
                i = nonZ[0][zz]
                j = nonZ[1][zz]
                #print('\r',org[i, j], H_primeY[i, j],end='')
                if not np.isclose( org[i, j], H_primeY[i, j], rtol=tol_, atol=0):
                    mistake_buffer += 1
                    if mistake_buffer > mistake_limit:
                        print(org[i, j], H_primeY[i, j])
                        print('i,j=',i,j)
                        case = False
                        break
                pbar.update()
            pbar.close()
            if case: print('Hamiltoninan is invariance under C2y. Buffer: ',mistake_buffer)
            
            print('checking C2z')
            pbar = tqdm(total=nonZ_tot, unit=' check', desc='C2z ') # Initialise
            case = True
            mistake_buffer = 0
            for zz in range(nonZ_tot):
                i = nonZ[0][zz]
                j = nonZ[1][zz]
                if not np.isclose( org[i, j], H_primeZ[i, j], rtol=tol_, atol=0):
                    mistake_buffer += 1
                    if mistake_buffer > mistake_limit:
                        print(org[i, j], H_primeZ[i, j])
                        case = False
                        break
                pbar.update()
            pbar.close()
            if case: print('Hamiltoninan is invariance under C2z. Buffer: ',mistake_buffer)
            
     
    def check_hamiltoninan_symmetry(self,operation, tol = 0.1):
        if self.rank != 0:
            raise RuntimeError('Please plot using **if rank==0** in mpi mode')
        #Operations:: 'C2x', 'C2y', 'C2z'
        oper_mat = sp.lil_matrix((self.conf_.tot_number, self.conf_.tot_number), dtype='int')
        
        whos =  {'C2x':0, 'C2y':1, 'C2z':2}
        op_ = self.new_orders[whos[operation]]
        
        org = sp.lil_matrix(self.H, dtype=self.H.dtype, copy=True) 
        trans_op = sp.lil_matrix(self.H[op_][:,op_], dtype=self.H.dtype, copy=True) 
        
        #print('ishermitian for ',operation,ishermitian(trans_op.todense(), rtol=0.0))
        nonZ = org.nonzero()
        nonZ_tot = nonZ[0].shape[0]
        ##Make the M matrix
        #version_ = self.file_name[:-5] +'_cut_' + str(self.cut_fac) + '_Pf'
        #self.conf_.neigh_list_me_smart(cutoff=self.r_cut, l_width=300, load_ = True, version_ = version_ )
        M = sp.lil_matrix((self.conf_.tot_number, self.conf_.tot_number), dtype='int')
        for ii in range(self.conf_.tot_number):
            neighs = self.nl[ii][~(np.isnan(self.nl[ii]))].astype('int')
            for jj in neighs:
                sign_before_X = 1 if self.B_flag[0][ii,jj]==0 else -1
                sign_after_X  = 1 if self.B_flag[0][op_[ii],op_[jj]]==0 else -1
                #sign_before_Y = 1 if self.B_flag[1][ii,jj]==0 else -1
                #sign_after_Y  = 1 if self.B_flag[1][op_[ii],op_[jj]]==0 else -1
                
                convention = sign_before_X*sign_after_X #* sign_before_Y*sign_after_Y
                
                M[ii,jj] = convention
                #if convention == -1 : print('\r','minu',end='')
                #if convention == +1 : print('\r','plus',end='')
        #print('ishermitian for ',operation,ishermitian(M.todense(), rtol=0.0))
        #print('mmm',M[ii,jj],M[jj,ii])
        print('before:',trans_op.nonzero()[0].shape[0])
        trans = trans_op.multiply(M)
        print('after:',trans.nonzero()[0].shape[0])
        

        
        ###
        case_ = True
        senario = 'exact'
        pbar = tqdm(total=nonZ_tot, unit=' elements', desc='checking '+operation) # Initialise
        for zz in range(nonZ_tot):
            i = nonZ[0][zz]
            j = nonZ[1][zz]
            flag_ = np.isclose(org[i, j], trans[i, j], rtol=tol, atol=0)
            if not flag_ :
                senario = 'conjugate'
                #print(org[i, j], '\n is not \n',trans[i, j],'\n')
                break
            pbar.update()
        pbar.close()
        pbar = tqdm(total=nonZ_tot, unit=' elements', desc='checking '+operation) # Initialise

        conj_count = 0
        ex_count = 0
        if senario == 'conjugate':
            for zz in range(nonZ_tot):
                i = nonZ[0][zz]
                j = nonZ[1][zz]
                #flag_ = np.isclose(org[i, j], np.conjugate(trans[i, j]), rtol=tol, atol=0)
                aA = org[i, j]
                bB = trans[i, j]
                #if np.isclose(org[i, j]/trans[i, j], -1, rtol=tol): print('-1')
                if np.isclose(aA,  np.conjugate(bB), rtol=tol, atol=0):
                    conj_count +=1
                elif np.isclose(aA, bB, rtol=tol, atol=0):
                    ex_count +=1
                else:
                    senario = 'negative'
                    break
                
                pbar.update()
            pbar.close()
            print('conj_count, ex_count' , conj_count, ex_count)
        pbar = tqdm(total=nonZ_tot, unit=' elements', desc='checking '+operation) # Initialise
        
        if senario == 'negative':
            for zz in range(nonZ_tot):
                i = nonZ[0][zz]
                j = nonZ[1][zz]
                flag_ = np.isclose(org[i, j], -trans[i, j], rtol=tol, atol=0)
                if not flag_ :
                    senario = 'absolute'
                    break
                
                pbar.update()
            pbar.close()
        pbar = tqdm(total=nonZ_tot, unit=' elements', desc='checking '+operation) # Initialise
        
        p_count = 0
        m_count = 0
        if senario == 'absolute':
            for zz in range(nonZ_tot):
                i = nonZ[0][zz]
                j = nonZ[1][zz]
                #flag_ = np.isclose(np.abs(org[i, j]), np.abs(trans[i, j]), rtol=tol, atol=0)
                aA = org[i, j]
                bB = trans[i, j]
                #if np.isclose(org[i, j]/trans[i, j], -1, rtol=tol): print('-1')
                if np.isclose(aA, 1*bB, rtol=tol, atol=0):
                    p_count +=1
                elif np.isclose(aA, -1*bB, rtol=tol, atol=0):
                    m_count +=1
                #if not flag_ :
                else:
                    case_ = False
                    senario = 'nothing'
                    print(org[i, j], '\n is not equal to \n',trans[i, j],'\n')
                    break
                
                pbar.update()
            pbar.close()
            print('p_count, m_count' , p_count, m_count)
        
        print('Under',operation,', Hamiltoninan invariance is:',case_, 'and senario is',senario)
        #print('complex count is ',count)
        
    
    def build_sym_operation(self, tol_=0.1):
        if self.rank != 0:
            raise RuntimeError('Please plot using **if rank==0** in mpi mode')
    
        all_X = np.copy(self.conf_.atomsAllinfo[ : , 4])
        all_Y = np.copy(self.conf_.atomsAllinfo[ : , 5])
        all_Z = np.copy(self.conf_.atomsAllinfo[ : , 6])
        wave_info = np.zeros((self.conf_.tot_number, 3), dtype= self.dtypeR)

        wave_info[:, 0] = all_X
        wave_info[:, 1] = all_Y
        wave_info[:, 2] = all_Z
        
        new_orders = np.zeros((3, self.conf_.tot_number), dtype='int') # 'C2x', 'C2y', 'C2z'
        who =0
        for which_operation in ['C2x', 'C2y', 'C2z']:
            print('making the operation for ', which_operation)
            wave_info_trs = np.copy(wave_info)
            if which_operation == 'C2x':
                #x+1/2,-y,-z
                print('doing c2x')
                wave_info_trs[:, 0] += 1/2 * self.conf_.xlen
                wave_info_trs[:, 1] *= -1
                wave_info_trs[:, 2] *= -1
                
                
            elif which_operation == 'C2y':
                #-x,y+1/2,-z
                print('doing c2y')
                wave_info_trs[:, 0] *= -1
                wave_info_trs[:, 1] += 1/2 * self.conf_.ylen
                wave_info_trs[:, 2] *= -1
                
                if '_zxact' not in self.file_name and 'noa0_relaxed' not in self.file_name and '1.08_0fold_no18' not in self.file_name:
                    print('I am doing a0 shit')
                    wave_info_trs[:, 0] -= self.a0 ## # for 1 fold i don't know why it is this way!
                    #wave_info_trs[:, :, :, 0] -= 2*self.a0 ## # for 0 fold i don't know why it is this way!
        
            elif which_operation == 'C2z':     
                #-x+1/2,-y+1/2,z
                print('doing c2z')
                wave_info_trs[:, 0] = -1*wave_info_trs[:, 0] + 1/2 * self.conf_.xlen
                wave_info_trs[:, 1] = -1*wave_info_trs[:, 1] + 1/2 * self.conf_.ylen
                
                if '_zxact' not in self.file_name and 'noa0_relaxed' not in self.file_name and '1.08_0fold_no18' not in self.file_name:
                    wave_info_trs[:, 0] -= self.a0 ## for 1 fold #i don't know why it is this way!
                    #wave_info_trs[:, :, :, 0] -= 2*self.a0 ### for 0 fold #i don't know why it is this way!
            
                            
            ## translate to cell(0,0) 
            x_back = (wave_info_trs[:, 0]//self.conf_.xlen)
            y_back = (wave_info_trs[:, 1]//self.conf_.ylen)
            wave_info_trs[:, 0] -=  x_back*self.conf_.xlen
            wave_info_trs[:, 1] -=  y_back*self.conf_.ylen
            
            ## get the right indices to compare 
            new_order = np.zeros(self.conf_.tot_number, dtype='int')
            
            for nn in range(self.conf_.tot_number):
                
                cond_all = np.isclose( np.linalg.norm(wave_info_trs - wave_info[nn], axis=1) , 0, rtol=0, atol=0.2) 
        
                idx = np.where(cond_all)[0]
                
                if idx.shape[0] == 0:
                    # pbc thing
                    possible = [0,-1,+1]
                    print('I have to do PBC on nn=',nn)
                    for pX in possible:
                        if idx.shape[0] == 0:
                            for pY in possible:
                                desire_coord = np.copy(wave_info[nn])
                                #print('doing ',pX,pY, desire_coord)
                                desire_coord[0] += pX * self.conf_.xlen 
                                desire_coord[1] += pY * self.conf_.ylen 
                        
                                cond_all = np.isclose( np.linalg.norm(wave_info_trs - desire_coord, axis=1) , 0, rtol=0, atol=0.2) 
                                idx = np.where(cond_all)[0]
                                
                                if idx.shape[0] >0 :
                                    print('yup!, fixed!')
                                    break
                    
                
                if idx.shape[0] != 1:
                    print('idx=',idx,'    pos',wave_info[nn,0:3])
                    plt.show()
                    raise RuntimeError('Cannot match the patterns... ')
                new_order[nn] = idx[0]
                
            new_orders[who] = new_order
            who += 1
        self.new_orders = new_orders
        np.savez(self.folder_name + 'Operations_' +self.save_name, new_orders=self.new_orders)
        
    
    def point_symmetry_check(self, which_K, diagno, mix_pairs = 2, block=False, tol_=0.1 ,skip_diag = False):
        """
        param: which_operation: C2x, C2...
        """
        full_vec = True
        #full_vec = False
        if self.rank != 0:
            raise RuntimeError('Please plot using **if rank==0** in mpi mode')
            
        
        if full_vec:
            
            xpos_ = np.cumsum(self.K_path_Highsymm_indices)
            id_ = xpos_[np.where(self.K_label==which_K)[0][0]]
            #print('checking symmetry for point='+which_K)
            print('id_={0}'.format(id_))
            
            
            all_X = np.copy(self.conf_.atomsAllinfo[ : , 4])
            all_Y = np.copy(self.conf_.atomsAllinfo[ : , 5])
            all_Z = np.copy(self.conf_.atomsAllinfo[ : , 6])
            all_xyz = np.copy(self.conf_.atomsAllinfo[ : , 4:7])
            
            flat_range= 8 #self.N_flat
            wave_info = np.zeros((flat_range, self.conf_.tot_number, 7), self.dtypeR)

            for ii in range(flat_range):
                wave_info[ii, :, 0] = all_X
                wave_info[ii, :, 1] = all_Y
                wave_info[ii, :, 2] = all_Z
                
                wave_info[ii, :, 3] = np.real(self.bandsVector[id_, :, ii])
                wave_info[ii, :, 4] = np.imag(self.bandsVector[id_, :, ii])
                wave_info[ii, :, 5] = np.absolute(self.bandsVector[id_, :, ii])
                wave_info[ii, :, 6] = np.angle(self.bandsVector[id_, :, ii], deg=True)
            
            which_operations =  ['C2x', 'C2y', 'C2z']
            whos =  {'C2x':0, 'C2y':1, 'C2z':2}
            #who = 1
            Cmats = [self.Cop_x, self.Cop_y, self.Cop_z]
            #Cmats = [sp.lil_matrix(sp.linalg.inv(self.Cop_x)), sp.lil_matrix(sp.linalg.inv(self.Cop_y)), sp.lil_matrix(sp.linalg.inv(self.Cop_z))]
            #opp = 0 # 'C2x'
            #opp = 1 # 'C2y'
            #opp = 2 # 'C2z'
            opp = whos[diagno]
            #mix_pairs  = 4
            #skip_diag = True
            
            only_itself = True 
            #only_itself = False
            #for opp in range(2,3): #range(3):
            #for opp in range(0,1): #range(3):
            #for opp in range(1,2): #range(3):
            #for which_operation in ['C2x', 'C2y', 'C2z']:
            #for which_operation in ['C2x']:
            #who = whos[which_operation]
            #invv = sp.lil_matrix(sp.linalg.inv(Cmats[opp]))
            ### take away the phase
            #print("self.K_path[0] ",self.K_path[0] )
            #phase_1 = np.exp(-1j*np.dot((all_xyz ), self.K_path[0] ) ) ## fix the id of self.K_path
            phase_1 = 1
            phase_2 = 1 #self.phaseSigns[who] #1
            #print('shapepee',self.phaseSigns[who].shape)
            #phase_2 = np.exp(1j*np.dot((all_xyz[self.new_orders[who]]), self.K_path[0]) ) ## fix the id of self.K_path
            ###
            
            new_bases = np.zeros((flat_range, self.conf_.tot_number), dtype=self.dtypeC)
            old_vecs = np.zeros((flat_range, self.conf_.tot_number), dtype=self.dtypeC)
            old_vecs_op = np.zeros((flat_range, self.conf_.tot_number), dtype=self.dtypeC)
            very_new_bases = np.zeros((flat_range, self.conf_.tot_number), dtype=self.dtypeC)
            #eignvals_neu = np.zeros(flat_range, dtype='f' if self.dtype==None else self.dtype)
            for ii in range(0, flat_range, mix_pairs):
                S = np.zeros((mix_pairs,mix_pairs), dtype=self.dtypeC)
                
                print('\n\n**I am mixing these energies: ')
                for jj in range(mix_pairs):
                    old_vecs[ii+jj] =    (wave_info[jj+ii, :, 3] + 1j*wave_info[jj+ii, :, 4])*phase_1
                    print(self.bandsEigns[id_, ii+jj] )
                    #old_vecs_op[ii+jj] =   sp.lil_matrix(self.Cop_x @  sp.lil_matrix(old_vecs[ii+jj]) )
                    #old_vecs_op[ii+jj] = (wave_info[jj+ii, self.new_orders[who], 3] + 1j*wave_info[jj+ii, self.new_orders[who], 4])*phase_2
                    
                
                for fl_i in range(mix_pairs):
                    for fl_f in range(mix_pairs):
                        #S[fl_i, fl_f] = np.dot(np.conjugate( old_vecs[ii+fl_i].T ), old_vecs_op[ii+fl_f] ) 

                        element = (sp.lil_matrix(np.conjugate(old_vecs[ii+fl_i])) @ Cmats[opp]  @  sp.lil_matrix.transpose(sp.lil_matrix(old_vecs[ii+fl_f]), copy=True))
                        assert element.get_shape() == (1,1)
                        #exit()
                        S[fl_i, fl_f] =  element[0,0]
                
                w, v = np.linalg.eig(S)
                #eignvals_neu[ii:ii+mix_pairs] = w
                #print('<psi| '+which_operation+' |psi>')
                print('<psi| '+which_operations[opp]+' |psi>')
                #np.set_printoptions(suppress=True)
                print(np.array2string(S, separator=",", precision=4))
                #for zz in range(mix_pairs):
                    #print(S[zz,:])
                print('eignvalues: ',w)
                print('sum (w**2)=', np.sum(np.power(w,2)),'\n')
                #print(eignvals_neu[ii:ii+mix_pairs].shape)
                #print(w.shape)
                #new_bases = np.zeros((mix_pairs, self.conf_.tot_number), dtype='complex' if self.dtype==None else 'c'+self.dtype)
                
                #continue
                
                if skip_diag:
                    for kk in range(mix_pairs):
                        new_bases[ii+kk] = old_vecs[ii+kk]
                else:
                    for kk in range(mix_pairs):
                        for qq in range(mix_pairs):
                            new_bases[ii+kk] += old_vecs[ii+qq]*v[qq, kk]
                    
                    
                    
                if block:
                    #simultaneous diagonalization
                    for se_al in range(3):
                        if se_al == 0:
                        #if se_al == 1:
                            #continue
                            sdot = np.zeros((mix_pairs,mix_pairs), dtype=self.dtypeC)
                            print('\n second diagonalizing respect to ', se_al)
                            for sei in range(mix_pairs):
                                for sef in range(mix_pairs):
                                    element = (sp.lil_matrix(np.conjugate(new_bases[ii+sei])) @ Cmats[se_al]  @  sp.lil_matrix.transpose(sp.lil_matrix(new_bases[ii+sef]),copy=True))
                                    assert element.get_shape() == (1,1)
                                    sdot[sei, sef] = element[0,0]
                            #####
                            #print('redigonalize\n', sdot)
                            #w, v = np.linalg.eig(sdot)
                            #print('eignvalues: ',w)
                            #for kk in range(mix_pairs):
                                #for qq in range(mix_pairs):
                                    #very_new_bases[ii+kk] += new_bases[ii+qq]*v[qq, kk]
                            
                            #####
                            upper_block = sdot[:2, :2]
                            #upper_block = sdot[:2, 2:]
                            print('upper_block is\n', upper_block)
                            w, v = np.linalg.eig(upper_block)
                            print('eignvalues: ',w)
                            for kk in range(2):
                                for qq in range(2):
                                    very_new_bases[ii+kk] += new_bases[ii+qq]*v[qq, kk]
                            ###
                            lower_block = sdot[2:, 2:]
                            #lower_block = sdot[2:, :2]
                            print('lower_block is\n', lower_block)
                            w, v = np.linalg.eig(lower_block)
                            print('eignvalues: ',w)
                            for kk in range(2):
                                for qq in range(2):
                                    very_new_bases[ii+kk+2] += new_bases[ii+qq+2]*v[qq, kk]       
                            
                            
                            new_bases[ii:ii+mix_pairs] = very_new_bases[ii:ii+mix_pairs]
                            
                #for se_al in which_operations:
                for se_al in range(3):
                    sdot = np.zeros((mix_pairs,mix_pairs), dtype=self.dtypeC)
                    print('\nfinal check if diagonalized respect to ', se_al)
                    for sei in range(mix_pairs):
                        for sef in range(mix_pairs):
                            #sdot[sei, sef]  = np.dot(np.conjugate(new_bases[ii+sei, :].T),  new_bases[ii+sef, self.new_orders[whos[se_al]]]  )
                                
                            element = (sp.lil_matrix(np.conjugate(new_bases[ii+sei])) @ Cmats[se_al]  @  sp.lil_matrix.transpose(sp.lil_matrix(new_bases[ii+sef]),copy=True))
                            assert element.get_shape() == (1,1)
                            sdot[sei, sef] = element[0,0]
                            
                    print(np.array2string(sdot, separator=",", precision=4))
                        
            

                
            ### Operations Phoenix
            #new_bases_guaged = np.zeros(new_bases.shape, dtype=new_bases.dtype)
            #for ch in range(0,flat_range):
            ##ch = 0
                #angle_i = np.angle(new_bases[ch], deg=False)
                #angle_f = np.angle(new_bases[ch][self.new_orders[2]], deg=False)
                #delta_phase = angle_f - angle_i #-np.pi
                #delta_phase[ delta_phase>+np.pi ] -= (2*np.pi)
                #delta_phase[ delta_phase<-np.pi ] += (2*np.pi)
                
                #rot = np.exp(1j*(delta_phase/2))
                ##rot2 = np.exp(1j*(-delta_phase/2))
                
                #new_bases_guaged[ch] =rot*new_bases[ch] 
                ##new_bases_guaged[ch+1] =rot2*new_bases[ch+1] 
            #new_bases = new_bases_guaged
            
            #sdot  = np.dot(new_bases_guaged, np.conjugate(new_bases_guaged.T))
            #print('orthonormality of Operations Phoenix', sdot)
            
            #new_b_guaged_orth = np.zeros(new_bases.shape, dtype=new_bases.dtype)
            #for ii in range(0, flat_range, mix_pairs):
                #S = np.zeros((mix_pairs,mix_pairs), dtype='complex' if self.dtype==None else 'c'+self.dtype)
                #for fl_i in range(mix_pairs):
                    #for fl_f in range(mix_pairs):
                        #S[fl_i, fl_f] = np.dot(np.conjugate(new_bases_guaged[ii+fl_i].T), new_bases_guaged[ii+fl_f][self.new_orders[2]])
            
                #for kk in range(mix_pairs):
                    #for qq in range(mix_pairs):
                        #new_b_guaged_orth[ii+kk] += new_bases_guaged[ii+qq]*v[qq, kk]
                
                #sdot  = np.dot(new_b_guaged_orth[ii:ii+mix_pairs], np.conjugate(new_b_guaged_orth[ii:ii+mix_pairs].T))
                #print('orthonormality of Operations Phoenix', sdot)
            ##new_bases = new_b_guaged_orth
            #print('resolving')
            #H = sp.lil_matrix(self.H, dtype='complex' if self.dtype==None else 'c'+self.dtype)
            ##eigvals, new_bases = primme.eigsh(H, k=8, which=self.sigma_shift, tol=1e-10, v0=new_bases_guaged.T)
            #eigvals, new_bases = eigs(H, k=8, sigma=self.sigma_shift, v0=new_bases_guaged[0,:].T)
            #new_bases = new_bases.T
            #print('eigvals', (eigvals-self.sigma_shift)*1000*0.5)
            ###    
            #exit()
            
            print('\n')
            #print("diagonalized in ",which_operation)
            for wihh in range(3):
                #only_itself = True if wihh ==2 else False
                 
                print('checking ', which_operations[wihh])
                fig, axs = plt.subplots(4,2,figsize=(10, 10))
                plt.title(which_operations[wihh])
                axs = axs.flatten()
                only_amp_str = ''
                for kk in range(flat_range):
                    transed = ((Cmats[wihh]  @  sp.lil_matrix.transpose(sp.lil_matrix(new_bases[kk]), copy=True)).todense())#.flatten()[0,:]
                    transed = np.ravel(transed)
                    assert new_bases[kk].shape == transed.shape
                    #print(type(transed),transed.shape)
                    #print(new_bases[kk].shape)
                    #print('************')
                    for qq in range(kk, flat_range):
                        #check_ = np.isclose(np.absolute(new_bases[qq]), np.absolute(new_bases[kk][self.new_orders[wihh]] ), rtol=0.2, atol=0.0)
                        check_ = np.isclose(np.absolute(new_bases[qq]), np.absolute(transed), rtol=0.2,  atol=0.0)
                        flag_  = np.isclose( np.count_nonzero(check_), self.conf_.tot_number, rtol=tol_, atol=0) 
                        #print('{0} dot {1} is '.format(ii+qq,ii+kk), np.dot(np.conjugate(new_bases[qq].T), new_bases[kk][new_orders[wihh]]))
                        #print(np.count_nonzero(check_))
                        condintion = (kk==qq) if only_itself else True
                        if flag_ and condintion:
                            print('{0} to {1} **symmetry Holds!** '.format(qq,kk), np.count_nonzero(check_), ' of ', self.conf_.tot_number)
                            #print('instance: ',np.angle(new_bases[qq][10], deg=True), np.angle(new_bases[kk][new_orders[wihh]][10], deg=True))
                            #print('instance: ',np.angle(new_bases[qq][1356], deg=True), np.angle(new_bases[kk][new_orders[wihh]][1356], deg=True))
                            #print('dot product is ', np.dot(np.conjugate(new_bases[qq].T), new_bases[kk][new_orders[wihh]]))
                            #print('old product is ', np.dot(np.conjugate(new_bases[qq].T), new_bases[kk]))
                            #if which_operation == 'C2z':
                            #delta_phase = np.angle(new_bases[qq], deg=True) - np.angle(new_bases[kk][self.new_orders[wihh]], deg=True)  # transformed respect to org
                            delta_phase = np.angle(new_bases[qq], deg=True) - np.angle(transed, deg=True)  # transformed respect to org
                            delta_phase[ delta_phase>+180 ] -= 360
                            delta_phase[ delta_phase<-180 ] += 360
                            delta_phase = np.abs(delta_phase)
                            
                            #delta_phase=np.rad2deg(np.arccos((np.real(new_bases[qq])*np.real(transed) + np.imag(new_bases[qq])*np.imag(transed) )/(np.absolute(new_bases[qq])*np.absolute(transed))))
                            
                            #axs[(kk)].hist(delta_phase, weights=np.absolute(new_bases[qq]),bins=360)
                            np.set_printoptions(precision=4)
                            if np.std(delta_phase) <20:
                                print('   phase is ', np.round(np.mean(delta_phase),5), np.round(np.std(delta_phase),5) )
                                axs[kk].hist(delta_phase, bins=45, density=True) #, weights=np.absolute(new_bases[qq])
                                #axs[(kk)].set_xlim([-5,185])
                            else:
                                # even+i*odd thing
                                vec_0a = (+np.real(new_bases[qq]) + np.imag(new_bases[qq])) + 1j*(+np.real(new_bases[qq]) - np.imag(new_bases[qq])) # THe working one!
                                vec_1a = (+np.real(new_bases[qq]) + np.imag(new_bases[qq])) + 1j*(+np.real(new_bases[qq]) + np.imag(new_bases[qq])) #/np.sqrt(2)
                                vec_2a = (+np.real(new_bases[qq]) - np.imag(new_bases[qq])) + 1j*(+np.real(new_bases[qq]) + np.imag(new_bases[qq])) #/np.sqrt(2)
                                vec_3a = (+np.real(new_bases[qq]) - np.imag(new_bases[qq])) + 1j*(+np.real(new_bases[qq]) - np.imag(new_bases[qq])) #/np.sqrt(2)
                                #vec_b = new_bases[kk][self.new_orders[wihh]] *np.sqrt(2) 
                                vec_b = transed *np.sqrt(2) 
                                case_n = 0
                                for vec_a in [vec_0a,vec_1a, vec_2a, vec_3a]:
                                    #vec_a *= vec_a*phase_1
                                    #vec_b *= vec_b*phase_2
                                    check_m = np.isclose(np.absolute(vec_a), np.absolute(vec_b), rtol=0.2, atol=0.0)
                                    
                                    if  np.isclose( np.count_nonzero(check_m), self.conf_.tot_number, rtol=tol_, atol=0):
                                        delta_phase=np.rad2deg(np.arccos((np.real(vec_a)*np.real(vec_b) + np.imag(vec_a)*np.imag(vec_b) )/(np.absolute(vec_a)*np.absolute(vec_b))))
                                        print('\tmagnetic #{0} check: '.format(case_n), np.count_nonzero(check_m),'   phase is ', np.mean(delta_phase), np.std(delta_phase))
                                    case_n += 1
                                    #delta_phase = np.angle(vec_a, deg=True) - np.angle(vec_b, deg=True)  # transformed respect to org
                                    #delta_phase[ delta_phase>+180 ] -= 360
                                    #delta_phase[ delta_phase<-180 ] += 360
                                    #delta_phase = np.abs(delta_phase)
                                    #axs[kk].hist(delta_phase, bins=45, density=True) #, weights=np.absolute(new_bases[qq])
                                    #axs[(kk)].set_xlim([-5,185])
                                    #axs[(kk)].set_ylim([0,1])
                            
                        elif flag_:
                            #print('   Only in amplitude {0} to {1} '.format(qq,kk), np.count_nonzero(check_), ' of ', self.conf_.tot_number)
                            only_amp_str += '   Only in amplitude {0} to {1}  {2} of {3} \n'.format(qq,kk, np.count_nonzero(check_), self.conf_.tot_number)
                print(only_amp_str)
            print('\n')
            #who += 1
            self.new_bases = new_bases
            
            #check if bases are True bases of H
            Hdense = self.H.todense() ## please improve maybe using sp multiply
            for ch in range(flat_range):
                psi =  np.expand_dims(new_bases[ch], axis=1)
                #left = np.matmul( np.conjugate(psi.T),  self.H)
                left = np.matmul( np.conjugate(psi.T),  Hdense ) ## please improve maybe using sp multiply
                value_ = np.matmul( left,  psi)
                
                #phi =  np.expand_dims(old_vecs[ch], axis=1)
                #left = np.matmul( np.conjugate(phi.T),  self.H)
                #value_1 = np.matmul( left,  phi)
                
                #print((self.bandsEigns[id_, ch]), ' is ', (value_-self.sigma_shift)*1000*0.5, (value_1-self.sigma_shift)*1000*0.5 )
                assert value_.shape == (1,1)
                print((np.real(value_[0,0])-self.sigma_shift)*1000*0.5)
            #print(eignvals_neu)
            
            #fig, axs = plt.subplots(1,3,figsize=(7, 10))
            #cm = plt.cm.get_cmap('RdYlBu')
            #lev_ref = 6
            #lev_trs = 6
            #type_dic = {0:'a', 1:'b', 2:'d', 3:'c'} # for 1 fold
            #sc0 = axs[0].scatter(wave_info[0, :, 0], wave_info[0, :, 1], 
                                #c=np.angle(new_bases[lev_ref], deg=True),  cmap=cm, 
                                #s= 4000*np.real(new_bases[lev_ref])) 
            #sc1 = axs[1].scatter(wave_info[0, self.new_orders[2], 0], wave_info[0, self.new_orders[2], 1], 
                                #c=np.angle(new_bases[lev_trs][self.new_orders[2]], deg=True),
                                #cmap=cm, s= 4000*np.real(new_bases[lev_trs][self.new_orders[2]]))
            
            #fig.colorbar(sc0,orientation='vertical')
            #fig.colorbar(sc1,orientation='vertical')
            ##axs[2].plot(wave_info[0, :, 0], wave_info[0, :, 1], 'o', color='tomato', markersize = 4)
            ##axs[2].plot(wave_info[0, new_order, 0], wave_info[0, new_order, 1], 'o', color='seagreen', markersize = 3)

            
            ##plt.colorbar(sc)
            ##axs[0].set_title('{0}{1}'.format(lev_ref, type_dic[ref_tp]))
            ##axs[1].set_title('{0}{1}'.format(lev_trs, type_dic[partner_tp]))
            #axs[0].set_aspect('equal', 'box')
            #axs[1].set_aspect('equal', 'box')
            #axs[2].set_aspect('equal', 'box')
            
            #fig.tight_layout()
            #plt.show()
            
        #############
        else:
            
            if self.rank ==0:
                #if orientation != '1_fold' :
                    #raise RuntimeError('Non rectangular boxes are not supported yet, for checking symmetry')
                if self.conf_.xy !=0:
                    #raise RuntimeError('Non rectangular boxes are not supported yet, for checking symmetry')
                    all_X = np.copy(self.conf_.atomsAllinfo[ : , 4]) 
                    all_Y = np.copy(self.conf_.atomsAllinfo[ : , 5])
                    all_X -= (all_X//self.conf_.xhi)*self.conf_.xlen
                else:
                    all_X = np.copy(self.conf_.atomsAllinfo[ : , 4])
                    all_Y = np.copy(self.conf_.atomsAllinfo[ : , 5])

                
                xpos_ = np.cumsum(self.K_path_Highsymm_indices)
                id_ = xpos_[np.where(self.K_label==which_K)[0][0]]
                print('checking symmetry for point='+which_K)
                print('id_={0}'.format(id_))
                
                #if self.orientation  != '1_fold':
                ## temporary: define symmetry operations here...
                ##
                ref_tp = 0
                flat_range= 8 #self.N_flat
                #type_range = 4
                wave_info = np.zeros((flat_range, 4, self.conf_.tot_number//4, 6), dtype=self.dtypeR)
                ##wave_info ITEM:  x y  real img  amp angle

                
                for ii in range(flat_range):
                    for type_ in range(4):
                        abcd = self.conf_.sub_type==10*(type_+1)
                        #print(self.conf_.sub_type)
                        wave_info[ii, type_, :, 0] = all_X[abcd]
                        wave_info[ii, type_, :, 1] = all_Y[abcd]
                        
                        wave_info[ii, type_, :, 2] = np.real(self.bandsVector[id_, abcd, ii])
                        wave_info[ii, type_, :, 3] = np.imag(self.bandsVector[id_, abcd, ii])
                        wave_info[ii, type_, :, 4] = np.absolute(self.bandsVector[id_, abcd, ii])
                        wave_info[ii, type_, :, 5] = np.angle(self.bandsVector[id_, abcd, ii], deg=True)
                
                # Set Fabrizio's cell to (0,xlen) and (0,ylen)
                # and put everything in the box 
                #wave_info[:, :, :, 0] -= (self.conf_.xlo - 1/4 * self.conf_.xlen) # for 1 fold
                #wave_info[:, :, :, 1] -= (self.conf_.ylo - 1/4 * self.conf_.ylen) # for 1 fold
                
                #wave_info[:, :, :, 0] -=  (wave_info[:, :, :, 0]//self.conf_.xlen)*self.conf_.xlen
                #wave_info[:, :, :, 1] -=  (wave_info[:, :, :, 1]//self.conf_.ylen)*self.conf_.ylen
                
                # apply transformation
                wave_info_trs = np.copy(wave_info)
                if which_operation == 'C2x':
                    #x+1/2,-y,-z
                    print('doing c2x')
                    wave_info_trs[:, :, :, 0] += 1/2 * self.conf_.xlen
                    wave_info_trs[:, :, :, 1] *= -1
                    
                    # type is the same, layer changes :: 0,3 (a,c) and (1,2) b,d have the same type! ...
                    # ...specifically for this system!! To be fixed later
                    partner_tp = 3 # for 1 fold
                    #partner_tp = 2  # for 0 fold
                    
                elif which_operation == 'C2y':
                    #-x,y+1/2,-z
                    print('doing c2y')
                    wave_info_trs[:, :, :, 0] *= -1
                    wave_info_trs[:, :, :, 1] += 1/2 * self.conf_.ylen
                    
                    if '_zxact' not in self.file_name and 'noa0_relaxed' not in self.file_name and '1.08_0fold_no18' not in self.file_name:
                        wave_info_trs[:, :, :, 0] -= self.a0 ## # for 1 fold i don't know why it is this way!
                        #wave_info_trs[:, :, :, 0] -= 2*self.a0 ## # for 0 fold i don't know why it is this way!
                    
                    # type and layer changes
                    partner_tp = 2 # for 1 fold
                    #partner_tp = 3  # for 0 fold
            
                elif which_operation == 'C2z':     
                    #-x+1/2,-y+1/2,z
                    print('doing c2z')
                    wave_info_trs[:, :, :, 0] = -1*wave_info_trs[:, :, :, 0] + 1/2 * self.conf_.xlen
                    wave_info_trs[:, :, :, 1] = -1*wave_info_trs[:, :, :, 1] + 1/2 * self.conf_.ylen
                    
                    if '_zxact' not in self.file_name and 'noa0_relaxed' not in self.file_name  and '1.08_0fold_no18' not in self.file_name:
                        wave_info_trs[:, :, :, 0] -= self.a0 ## for 1 fold #i don't know why it is this way!
                        #wave_info_trs[:, :, :, 0] -= 2*self.a0 ### for 0 fold #i don't know why it is this way!
                    
                    # layer is the same, type changes
                    partner_tp = 1
                
                elif which_operation == 'sigma_yz' :
                    wave_info_trs[:, :, :, 0] = -1*wave_info_trs[:, :, :, 0] + 1/2 * self.conf_.xlen
                    partner_tp = 2
                
                elif which_operation == 'sigma_xz' :
                    wave_info_trs[:, :, :, 1] = -1*wave_info_trs[:, :, :, 1] + 1/2 * self.conf_.ylen
                    partner_tp = 0
                    
                #elif which_operation == 'mz' :
                    #wave_info_trs[:, :, :, 0] += 1/2 * self.conf_.xlen
                    #wave_info_trs[:, :, :, 1] -= 1/2 * self.conf_.ylen
                
                
                ## translate to cell(0,0) 
                wave_info_trs[:, :, :, 0] -=  (wave_info_trs[:, :, :, 0]//self.conf_.xlen)*self.conf_.xlen
                wave_info_trs[:, :, :, 1] -=  (wave_info_trs[:, :, :, 1]//self.conf_.ylen)*self.conf_.ylen
                
                ## get the right indices to compare 
                new_order = np.zeros(self.conf_.tot_number//4, dtype='int')

                
                #plt.figure()
                #fig, axs = plt.subplots(2, 2)
                fig, axs = plt.subplots(1,3,figsize=(7, 10))
                cm = plt.cm.get_cmap('RdYlBu')
                lev_ref = 6
                lev_trs = 6
                #type_dic = {0:'a', 1:'b', 2:'d', 3:'c'} # for 1 fold
                type_dic = {0:'a', 1:'b', 3:'d', 2:'c'} # for 0 fold
                sc = axs[0].scatter(wave_info[lev_ref, ref_tp, :, 0], wave_info[lev_ref, ref_tp, :, 1], c=wave_info[lev_ref, ref_tp, :, 5],  cmap=cm, s= 4000*wave_info[lev_ref, ref_tp, :, 4]) 
                sc = axs[1].scatter(wave_info_trs[lev_trs, partner_tp, :, 0], wave_info_trs[lev_trs, partner_tp, :, 1], c=wave_info_trs[lev_trs, partner_tp, :, 5],  cmap=cm, s= 4000*wave_info_trs[lev_trs, partner_tp, :, 4])
                
                
                axs[2].plot(wave_info[0, ref_tp, :, 0], wave_info[0, ref_tp, :, 1], 'o', color='tomato', markersize = 4)
                axs[2].plot(wave_info_trs[0, partner_tp, :, 0], wave_info_trs[0, partner_tp, :, 1], 'o', color='seagreen', markersize = 3)

                
                #plt.colorbar(sc)
                #axs[0].set_title('{0}{1}'.format(lev_ref, type_dic[ref_tp]))
                #axs[1].set_title('{0}{1}'.format(lev_trs, type_dic[partner_tp]))
                axs[0].set_aspect('equal', 'box')
                axs[1].set_aspect('equal', 'box')
                axs[2].set_aspect('equal', 'box')
                #fig.tight_layout()
                #plt.show()
                #exit()
                
                for nn in range(self.conf_.tot_number//4):
                    
                    cond_all = np.isclose( np.linalg.norm(wave_info_trs[0, partner_tp, :, 0:2] - wave_info[0, ref_tp, nn, 0:2], axis=1) , 0, rtol=0, atol=0.9) 
                    
                    idx = np.where(cond_all)[0]
                    
                    
                    if idx.shape[0] == 0:
                        # pbc thing
                        possible = [0,-1,+1]
                        print('I have to do PBC on nn=',nn)
                        for pX in possible:
                            if idx.shape[0] == 0:
                                for pY in possible:
                                    desire_coord = np.copy(wave_info[0, ref_tp, nn, 0:2])
                                    #print('doing ',pX,pY, desire_coord)
                                    desire_coord[0] += pX * self.conf_.xlen 
                                    desire_coord[1] += pY * self.conf_.ylen 
                            
                                    cond_all = np.isclose( np.linalg.norm(wave_info_trs[0, partner_tp, :, 0:2] - desire_coord, axis=1) , 0, rtol=0, atol=0.5) 
                                    idx = np.where(cond_all)[0]
                                    
                                    if idx.shape[0] >0 :
                                        print('yup!, fixed!')
                                        break
                        
                    
                    if idx.shape[0] != 1:
                        print('idx=',idx,'    pos',wave_info[0, ref_tp, nn, 0:2])
                        plt.show()
                        raise RuntimeError('Cannot match the patterns... ')
                    new_order[nn] = idx[0]
                
                #mode_ = 'mixed'
                mode_ = 'isolated'
                ##compare! isolated!
                aml=4
                ph_c =5
                fig, axs = plt.subplots(4,2,figsize=(10, 10))
                plt.title(which_operation)
                axs = axs.flatten()
                if mode_ == 'isolated':
                    for ii in range(flat_range):
                        likelihood = np.array([])
                        likelevels = np.array([])
                        Dot = np.array([])
                        found = False
                        for jj in range(flat_range):
                            check_ = np.isclose(wave_info[ii, ref_tp, :, aml], wave_info_trs[jj, partner_tp, new_order, aml], rtol=0.3, atol=0.0)
                            the_ratio = wave_info[ii, ref_tp, :, aml] / wave_info_trs[jj, partner_tp, new_order, aml]
                            flag_ = np.isclose( np.count_nonzero(check_), self.conf_.tot_number//4, rtol=tol_, atol=0) 
                            
                            ## approach dot product:
                            vec2 = vec1 = np.zeros(self.conf_.tot_number//4 , dtype=self.dtypeC)
                            vec1 = wave_info[ii, ref_tp, :, 2] + 1j*wave_info[ii, ref_tp, :, 3]
                            vec2 = wave_info_trs[jj, partner_tp, new_order, 2] + 1j*wave_info_trs[jj, partner_tp, new_order, 3]
                            Dot = np.append(Dot, np.dot(np.conjugate(vec1.T), vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) ) )
                            #Dot =  np.dot(np.conjugate(vec1.T), vec2)
                            
                            # even+i*odd
                            #vec1 = (-wave_info[ii, ref_tp, :, 2] + wave_info[ii, ref_tp, :, 3]) /np.sqrt(2)
                            #vec2 = wave_info_trs[jj, partner_tp, new_order, 2]# + 1j*wave_info_trs[jj, partner_tp, new_order, 3]
                            #check_m = np.isclose(vec1,vec2, rtol=0.3, atol=0.0)
                            #magnetic = vec2 / vec1
                            #check_m1 = np.isclose(magnetic, (+1+1j)/np.sqrt(2) , rtol=0.3, atol=0.0)
                            #check_m2 = np.isclose(magnetic, (+1-1j)/np.sqrt(2) , rtol=0.3, atol=0.0)
                            #check_m3 = np.isclose(magnetic, (-1-1j)/np.sqrt(2) , rtol=0.3, atol=0.0)
                            #check_m4 = np.isclose(magnetic, (-1+1j)/np.sqrt(2) , rtol=0.3, atol=0.0)
                            vec1_p1 =  vec1 + 1j*vec2
                            vec1_p2 =  vec1 - 1j*vec2
                            ang_vec1 = np.angle(vec1, deg=True)
                            ang_vec2 = np.angle(vec2, deg=True)
                            ang_p1 = np.angle(vec1_p1, deg=True)
                            ang_p2 = np.angle(vec1_p2, deg=True)
                            
                            
                            ##
                            likelihood = np.append(likelihood, np.count_nonzero(check_))
                            likelevels = np.append(likelevels, jj)
                            
                            if flag_:
                                
                                delta_phase = wave_info_trs[jj, partner_tp, new_order, ph_c] - wave_info[ii, ref_tp, :, ph_c]  # transformed respect to org
                                
                                delta_phase[ delta_phase>+180 ] -= 360
                                delta_phase[ delta_phase<-180 ] += 360
                                delta_phase = np.abs(delta_phase)
                                #axs[ii].hist(delta_phase,bins=360)
                                
                                #mydelta = ang_vec1-ang_p1
                                #mydelta[ mydelta>+180 ] -= 360
                                #mydelta[ mydelta<-180 ] += 360
                                #axs[ii].hist(ang_vec1,bins=180)#,alpha=0.5)
                                #axs[ii].hist(ang_vec2,bins=180)#,alpha=0.5)
                                #axs[ii].hist(ang_p1,bins=180)
                                #axs[ii].hist(ang_p2,bins=180)
                                
                                for shit in [ang_vec1]:#, ang_p1, ang_p2]:
                                    mydelta = ang_vec2 - shit
                                    mydelta[ mydelta>+180 ] -= 360
                                    mydelta[ mydelta<-180 ] += 360
                                    axs[ii].hist(mydelta,bins=90)
                                
                                
                                phase_shift = np.mean(delta_phase)
                                std = np.std(delta_phase)
                                
                                #print('delta_phase=',delta_phase)
                                #print('symmetry Holds!')
                                print('symmetry Holds! ',np.count_nonzero(check_), ' of ', self.conf_.tot_number//4)
                                print('level {0} -> {1}. phase_shift is '.format(ii, jj), np.round(phase_shift,3),' with std=', np.round(std,3))
                                print('energy of level {0} is {1}'.format(ii, self.bandsEigns[id_, ii]-self.shift_tozero) )
                                #print(np.count_nonzero(check_), ' of ', self.conf_.tot_number//4)
                                print('the_ratio, mean, std:',the_ratio, np.mean(the_ratio), np.std(the_ratio) )
                                #print('magnetic check', np.count_nonzero(check_m1),  np.count_nonzero(check_m2), np.count_nonzero(check_m3), np.count_nonzero(check_m4) )
                                #print('Dot',Dot)
                                #print('\n')
                                #break
                                found = True
                            #else:
                                #likelihood = np.append(likelihood, np.count_nonzero(check_))
                                #likelevels = np.append(likelevels, jj)
                            if jj == flat_range-1 and found == False:
                                #continue
                                #print('symmetry sucks!')
                                #else
                                print('symmetry sucks!')
                                print('the_ratio, mean, std:',the_ratio, np.mean(the_ratio), np.std(the_ratio) )
                                np.set_printoptions(precision=1)
                                likelihood=100* likelihood/ (self.conf_.tot_number//4)
                                print("'level {0} looks ".format(ii), likelihood, " % like levels",  likelevels, 'total of {:.2f} \n'.format(np.sum(likelihood)))
                            if jj == flat_range-1:# and found == True:
                                print('Dot',Dot)
                                print("'level {0} looks ".format(ii), likelihood, " % like levels",  likelevels, 'total of {:.2f} \n'.format(np.sum(likelihood)))
                                print('\n')
                            
                elif mode_ == 'mixed':
                    ### compare linear combination
                    np.set_printoptions(precision=2)
                    def func(X, a0, a1):
                    #def func(X, a0, a1, a2, a3, a4, a5, a6, a7):
                    #def func(X, a0, a1, a2, a3):#, a4, a5, a6, a7):
                        #np.expand_dims(co, axis=0)
                        #sum_ = np.zeros(X.shape[1], dtype='float')
                        #for i in range(X.shape[0]):
                            #sum_ += coeffs[0]*X[0]
                        #return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] #+ a4*X[4] + a5*X[5] +a6*X[6] + a7*X[7] 
                        #return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5] +a6*X[6] + a7*X[7] 
                        return a0*X[0] + a1*X[1] 
                    
                    
                    for ii in range(flat_range): 
                        fl_i = ii if ii%2==0 else ii-1
                        fl_f = ii+1 if ii%2==0 else ii
                        #print('fl_i,fl_f', fl_i,fl_f)
                        X  = np.copy(wave_info[fl_i:fl_f+1, ref_tp, :, aml])
                        #X  = np.copy(wave_info[0:8, ref_tp, :, aml])
                        #if ii < 4:
                            #X  = np.copy(wave_info[0:4, ref_tp, :, aml])
                        #else:
                            #X  = np.copy(wave_info[4:8, ref_tp, :, aml])
                        
                        
                        zz = np.copy(wave_info_trs[ii, partner_tp, new_order, aml])
                        
                        popt, pcov = curve_fit(func, X, zz)
                        print('your parameters for level {0} is '.format(ii), popt)
                        
                        construction_ = func(X, *popt)
                        check_ = np.isclose(construction_, zz, rtol=0.2, atol=0.0)
                        #check_ = np.isclose(wave_info[ii, ref_tp, :, aml], zz, rtol=0.2, atol=0.0)
                        print(np.count_nonzero(check_))
                        
                        flag_ = np.isclose( np.count_nonzero(check_), self.conf_.tot_number//4, rtol=tol_, atol=0) 
                        
                        
                        #sc = axs[0].scatter(wave_info[ii, ref_tp, :, 0],
                                            #wave_info[ii, ref_tp, :, 1],
                                            #c=0*popt[0]*wave_info[fl_i, ref_tp, :, 5]+
                                            #wave_info[fl_f, ref_tp, :, 5],  cmap=cm,
                                            #s= 4000*(
                                                #0*popt[0]*wave_info[fl_i, ref_tp, :, 4] +
                                                #1*wave_info[fl_f, ref_tp, :, 4]))
                        
                        #sc = axs[1].scatter(wave_info_trs[ii, partner_tp, :, 0], wave_info_trs[ii, partner_tp, :, 1], c=wave_info_trs[ii, partner_tp, :, 5],  cmap=cm, s= 4000*wave_info_trs[ii, partner_tp, :, 4])
                        
                        #plt.show()
                        #exit()
                            
                        if flag_:
                            
                            #delta_phase = wave_info_trs[jj, partner_tp, new_order, ph_c] - wave_info[ii, ref_tp, :, ph_c]  # transformed respect to org
                            
                            #delta_phase[ delta_phase>+180 ] -= 360
                            #delta_phase[ delta_phase<-180 ] += 360
                            #delta_phase = np.abs(delta_phase)
                            #phase_shift = np.mean(delta_phase)
                            #std = np.std(delta_phase)
                            
                            ##print('delta_phase=',delta_phase)
                            ##print('symmetry Holds!')
                            print('symmetry Holds! ',np.count_nonzero(check_), ' of ', self.conf_.tot_number//4)
                            #print('level {0} -> {1}. phase_shift is '.format(ii, jj), np.round(phase_shift,3),' with std=', np.round(std,3))
                            #print(np.count_nonzero(check_), ' of ', self.conf_.tot_number//4)
                            print('\n')
                            #break
                        else:

                            print('symmetry sucks!')
                            #np.set_printoptions(precision=1)
                            likelihood=100* np.count_nonzero(check_)/ (self.conf_.tot_number//4)
                            print("level {:.0f} has {:.2f} symmetry \n".format(ii, likelihood))
                    

    def embed_flatVec(self, which_K, vers='', d_phase=False,  vec_='lanczos'):
        '''
        which_K   !!string!!
         vec_='lanczos' or 'ortho' or 'phasesign'
        '''
        if self.rank ==0:
            ## find the index
            xpos_ = np.cumsum(self.K_path_Highsymm_indices)
            print("K_label",self.K_label)
            print("which_K",which_K)
            print("[np.where(self.K_label==which_K)",np.where(self.K_label==which_K))
            print("xpos_",xpos_)
            id_ = xpos_[np.where(self.K_label==which_K)[0][0]]
            print('Embedding eigen vectors for point='+which_K)
            print('id_={0}'.format(id_))
            
            if not hasattr(self, 'N_flat'):
                raise AttributeError('MAKE sure you sorted eign vectors before this!!')
            ## write the thing
            #fname = self.folder_name + self.file_name[:-5] + '_cut_' + str(self.cut_fac) + '_'+which_K + '_{0}.lammpstrj'
            dph = '_delta_' if d_phase else ''
            fname = self.folder_name +  vers+'_'+vec_+'_'+ dph+ self.save_name + '_'+which_K + '_{0}'+'.lammpstrj'
            print('fname::',fname)
            header_ = ''
                        
            
            if self.conf_.xy ==0:
                header_ += "ITEM: TIMESTEP \n{0} \nITEM: NUMBER OF ATOMS \n"+"{0} \nITEM: BOX BOUNDS pp pp ss \n".format(self.conf_.tot_number)
                header_ +="{0} {1} \n{2} {3} \n{4} {5} \n".format(                                                                                                                                                                            self.conf_.xlo, self.conf_.xhi, 
                self.conf_.ylo, self.conf_.yhi,
                self.conf_.zlo, self.conf_.zhi)
            else:
                header_ += "ITEM: TIMESTEP \n{0} \nITEM: NUMBER OF ATOMS \n"+"{0} \nITEM: BOX BOUNDS xy xz yz pp pp ss \n".format(self.conf_.tot_number)
                header_ +="{0} {1} {6} \n{2} {3} 0 \n{4} {5} 0 \n".format(                                                                                                                                                                            self.conf_.xlo, self.conf_.xhi+self.conf_.xy, 
                self.conf_.ylo, self.conf_.yhi,
                self.conf_.zlo, self.conf_.zhi,
                self.conf_.xy)
                
                
            header_ += "ITEM: ATOMS id type x y z  "
            #append_ = " "
            #for ii in range(self.N_flat):
                 #header_ += " abs{:.0f} ".format(ii+1)#, self.bandsEigns[id_, ii]-self.shift_tozero )
                 #append_ +=  " phase{:.0f}::{:.5f} ".format(ii+1, self.bandsEigns[id_, ii]-self.shift_tozero)
            #header_ += append_
            #fmt_ = ['%.0f']*2 + ['%.12f']*(3+2*self.N_flat)
            out_range= 8 #self.N_flat

            header_ += " abs phase "#.format(ii+1)#, self.bandsEigns[id_, ii]-self.shift_tozero )

            fmt_ = ['%.0f']*2 + ['%.12f']*(3+2*1)

            if vec_ == 'lanczos':
                vecs = self.bandsVector[id_]
            elif vec_ ==  'ortho':
                vecs = self.new_bases.T
            #elif vec_ ==  'phasesign':
                #vecs = self.new_bases.T
                ##for shit in range(self.conf_.tot_number):
                    ##assert self.phaseSigns[2][self.new_orders[2]][shit] == self.phaseSigns[2][shit]
                #for ii in range(out_range):
                    #vecs[:, ii] =  vecs[self.new_orders[2], ii] * self.phaseSigns[2]
            else:
                raise TypeError('unrecognized vector type')
            #print(vecs.shape)
            
            for ii in range(out_range):
                angle_ = np.angle(vecs[:, ii], deg=True)
                if d_phase:
                    angle_t = np.angle(vecs[self.new_orders[2], ii], deg=True)
                    delta_phase = angle_ - angle_t
                    delta_phase[ delta_phase>+180 ] -= 360
                    delta_phase[ delta_phase<-180 ] += 360
                    delta_phase = np.abs(delta_phase)
                    angle_ = delta_phase
                
                
                XX = np.concatenate((self.conf_.atomsAllinfo[:,np.r_[0]],  np.expand_dims(self.conf_.sub_type, axis=1), 
                self.conf_.atomsAllinfo[:,np.r_[4:7]],
                np.expand_dims(np.absolute(vecs[:, ii]), axis=1), 
                np.expand_dims(angle_, axis=1),
                ), axis = 1)

                np.savetxt(fname.format(ii), XX ,header=header_.format(ii), comments='', fmt=fmt_) #fmt='%.18e'
            
            print('levels:\n', self.bandsEigns[id_, :out_range]-self.shift_tozero)
            np.savetxt(fname+'__eign_values__', self.bandsEigns[id_, :out_range]-self.shift_tozero) 

            #for ii in range(self.N_flat):
                #XX = np.concatenate((self.conf_.atomsAllinfo[:,np.r_[0]],  np.expand_dims(self.conf_.sub_type, axis=1), self.conf_.atomsAllinfo[:,np.r_[4:7]],
                                    #np.absolute(self.bandsVector[id_, :, :self.N_flat]), 
                                    #np.angle(self.bandsVector[id_, :, :self.N_flat]),
                                    #), axis = 1) #

                #np.savetxt(fname, XX ,header=header_, comments='', fmt=fmt_) #fmt='%.18e'
            
            
            
            #file_ = open(fname, 'w')
            #file_.write(header_)
            #for ii in range(self.conf_.tot_number):
                #file_.write(('{:.0f} '*4 + '{:.12f} '*3 + ' 0 0 0 ' + '{:.2e} '*self.N_flat +'  \n').format(*self.conf_.atomsAllinfo[ii], *self.bandsVector[id_, ii, :self.N_flat] ))
                
            #file_.close()
    

    def save(self, str_='', write_param = True, H = None):
        if write_param == True:
            self.save_name =  self.file_name[:-5]+'_d0_' + str(self.d0) +'_cut_' + str(self.cut_fac) + '_' + str_
        else:
            self.save_name =  str_
        if H!= None:
            #np.savez(self.folder_name + 'HH_' +self.save_name, H=self.H)
            sp.save_npz(self.folder_name + 'HH_' +self.save_name, sp.csr_matrix(H, copy=True))
            return 0
        
        if self.rank ==0 :
            if hasattr(self, 'bandsEigns'):
                np.savez(self.folder_name + 'bands_' +self.save_name ,
                        bandsEigns=self.bandsEigns, K_path=self.K_path, 
                        K_path_Highsymm_indices = self.K_path_Highsymm_indices, 
                        K_label=self.K_label, K_path_discrete= self.K_path_discrete,
                        g1=self.g1, g2=self.g2, orientation=self.orientation, 
                        bandsVector = self.bandsVector,
                        file_name = self.file_name,
                        sub_type= self.conf_.sub_type)
                

                np.savez(self.folder_name + 'conf_' +self.save_name, B_flag=self.conf_.B_flag, nl = self.conf_.nl)
                
            if hasattr(self, 'dosEigns'):
                np.savez(self.folder_name + 'DOS_'+self.save_name,
                        dosEigns=self.dosEigns,
                        K_grid=self.K_grid, K_mapping=self.K_mapping)
            
            #if hasattr(self, 'new_orders'):
                #np.savez(self.folder_name + 'Operations_' +self.save_name, new_orders=self.new_orders)
                
            if hasattr(self, 'eigns_3D'):
                np.savez(self.folder_name + '3Dband_'+self.save_name, gsize_v=self.gsize_v, gsize_h=self.gsize_h, flat_grid=self.flat_grid, eigns_3D=self.eigns_3D, eigns_3D_reduced=self.eigns_3D_reduced)

    def load(self, folder_='', ver_ ='',HH=''):
        if self.rank ==0 :
            data_name_dos = None
            data_name_band = None
            data_name_3Dbands = None
            data_name_HH = None
            data_name_operation = None
            #data_name_phaseSigns = None
            data_name_conf = None
            self.folder_name = folder_  if folder_[-1] == '/' else folder_+'/'
            for lis in os.listdir(self.folder_name):
                if lis[-(4+len(ver_)):] == ver_+'.npz':
                    try:
                        if float(lis[:-4].split('_cut_')[1].split('_')[0]) == self.cut_fac:
                            if  float(lis[:-4].split('_d0_')[1].split('_')[0]) == self.d0:
                                #print(lis)
                                if 'bands_' in lis:
                                    data_name_band = lis
                                elif 'DOS_' in lis:
                                    data_name_dos = lis
                                elif '3Dband_' in lis:
                                    data_name_3Dbands = lis
                                elif 'HH_' in lis and HH in lis and HH != '':
                                    data_name_HH = lis
                                elif 'Operations_' in lis:
                                    data_name_operation = lis
                                #elif 'phaseSigns_' in lis:
                                    #data_name_phaseSigns = lis
                                elif 'conf_' in lis:
                                    data_name_conf = lis
                                    
                    except IndexError: pass
                    
            if data_name_band != None :
            
                print('loading band structure: '+data_name_band)
        
                self.save_name =  data_name_band.split('bands_')[1].split('.npz')[0]

                data_band = np.load(self.folder_name + data_name_band, allow_pickle=True)
                self.bandsEigns = data_band['bandsEigns']
                self.K_path = data_band['K_path']
                self.K_path_Highsymm_indices = data_band['K_path_Highsymm_indices']
                self.K_label = data_band['K_label']
                self.K_path_discrete = data_band['K_path_discrete']
                self.g1 = data_band['g1']
                self.g2 = data_band['g2']
                self.orientation = data_band['orientation']
                
                try:
                    self.bandsVector = data_band['bandsVector']
                    if self.bandsVector.all() == None:
                        self.bandsVector_exist = False
                    else:
                        self.bandsVector_exist = True
                except KeyError: 
                    self.bandsVector = None
                    self.bandsVector_exist = False
                try:
                    self.conf_.sub_type = data_band['sub_type']
                except KeyError: 
                    self.conf_.sub_type = None
                                
            else:
                raise FileNotFoundError('cannot find bands file')
            
            if data_name_HH != None:# and load_H:
                print('loading HH')
                #self.H = np.load(self.folder_name + data_name_HH, allow_pickle=True)['H']
                self.H = sp.lil_matrix(sp.load_npz(self.folder_name + data_name_HH))
                #print(type(self.H), self.H.shape)
                #exit()
            if data_name_operation != None:
                print('loading Operations_')
                self.new_orders = np.load(self.folder_name + data_name_operation, allow_pickle=True)['new_orders']
            
            #if data_name_phaseSigns != None:
                #print('loading phaseSigns_')
                #self.phaseSigns = np.load(self.folder_name + data_name_phaseSigns, allow_pickle=True)['phaseSigns']
            
            if data_name_conf != None:
                print('loading conf_')
                conf_shit = np.load(self.folder_name + data_name_conf, allow_pickle=True)
                self.nl = conf_shit['nl']
                self.B_flag = conf_shit['B_flag']
            
            if data_name_dos != None :
                print('loading density of states')
                data_dos = np.load(self.folder_name + data_name_dos)
                self.dosEigns = data_dos['dosEigns']
                self.K_grid = data_dos['K_grid']
                self.K_mapping = data_dos['K_mapping']
            else:
                print('DOS file not found, so not loaded.')
                self.dosEigns = None
            
            if data_name_3Dbands != None :
                print('loading 3D bands')
                data_3Dbands = np.load(self.folder_name + data_name_3Dbands)
                self.gsize_v = data_3Dbands['gsize_v']
                self.gsize_h = data_3Dbands['gsize_h']
                self.flat_grid = data_3Dbands['flat_grid']
                self.eigns_3D = data_3Dbands['eigns_3D']
                self.eigns_3D_reduced = data_3Dbands['eigns_3D_reduced']
                
            else:
                print('3D bands file not found, so not loaded.')



    def T_bone_sp(self, H_style):
        """
            Build the sparse skeleton of Hamiltonian matrix. Please turn to private!
            
            Args:
                H_style: 'str'
                    type of Hamiltonina formula in use
            
            Returns:
                T00 matrix
        """
        
        vc_mat = self.conf_.dist_matrix
        ez = self.conf_.ez
        T00 = sp.lil_matrix((self.conf_.tot_number, self.conf_.tot_number), dtype=self.dtypeR)

        if ez.shape == (3,):
            flag_ez = False
            ez_ = ez
        elif ez.shape == (self.conf_.tot_number,3):
            flag_ez = True
        else:
            raise RuntimeError('Wrong ez!! unexpected.')

        for ii in range(self.conf_.tot_number):
            neighs = self.conf_.nl[ii][~(np.isnan(self.conf_.nl[ii]))].astype('int')
            for jj in neighs:
                
                # calculate the hoping
                v_c = np.array([ vc_mat[0][ii,jj],  vc_mat[1][ii,jj],  vc_mat[2][ii,jj] ])
                dd = np.linalg.norm(v_c)

                V_sigam = self.V0_sigam * np.exp(-(dd-self.d0) / self.r0 )
                V_pi    = self.V0_pi    * np.exp(-(dd-self.a0) / self.r0 )
                
                if H_style == 'ave':
                    #Kolmogorov-Crespi-like approximate
                    tilt_1 = np.power(np.dot(v_c, ez[ii])/ dd, 2)
                    tilt_2 = np.power(np.dot(v_c, ez[jj])/ dd, 2)
                    t_d =   V_sigam * (tilt_1+tilt_2)/2 + V_pi * (1- (tilt_1 + tilt_2)/2 )
                
                elif H_style == '9X':
                    t_d = 0 
                    # x,x y,y z,z
                    lmn = v_c / dd
                    for pp in range(3):
                        t_d += ez[ii][pp]*ez[jj][pp] *( (lmn[pp]**2)*V_sigam + (1-lmn[pp]**2)*V_pi )
                        
                        t_d += ez[ii][pp]*ez[jj][(pp+1)%3] * lmn[pp]*lmn[(pp+1)%3] *(V_sigam - V_pi)
                        t_d += ez[ii][pp]*ez[jj][(pp+2)%3] * lmn[pp]*lmn[(pp+2)%3] *(V_sigam - V_pi)
                
                
                T00[ii, jj] = t_d *0.5 # *0.5  because of H.C.

        T00_copy = T00.copy()
        T00_trans = sp.lil_matrix.transpose(T00, copy=True)
        T00_dagger  = sp.lil_matrix.conjugate(T00_trans, copy=True)
        T00 = sp.lil_matrix(T00_dagger + T00_copy)
        print('T00 ishermitian ',ishermitian(T00.todense(), rtol=0.0))
        return T00

    def T_meat_sp(self, K_, T_0):
        """
            Adds the sparse modulation phase at each K_ to T_0. Turn to private.
        """

        modulation_matrix = sp.lil_matrix(( self.conf_.tot_number, self.conf_.tot_number), dtype=self.dtypeC)
        #print('making modulation_matrix..')
        for ii in range(self.conf_.tot_number):
            neighs = self.conf_.nl[ii][~(np.isnan(self.conf_.nl[ii]))].astype('int')
            for jj in neighs:
                ## tight binding type 1, with a phase
                #v_c = np.array([ self.conf_.dist_matrix[0][ii,jj],  self.conf_.dist_matrix[1][ii,jj],  self.conf_.dist_matrix[2][ii,jj] ])
                #modulation_matrix[ii,jj] = np.exp(-1j * np.dot(v_c, K_))#*2
                ###
                
                ## tight binding type 2, no extra phase
                thing = [1*self.conf_.B_flag[0][ii,jj] * self.conf_.xlen, 1*self.conf_.B_flag[1][ii,jj] * self.conf_.ylen, 0]
                modulation_matrix[ii,jj] = np.exp(+1j * np.dot(thing, K_ )) 
        
        # for 9X it makes more sense to do it here...
        #M1 = modulation_matrix.copy()
        #MT = sp.lil_matrix.transpose(modulation_matrix, copy=True)
        #MD  = sp.lil_matrix.conjugate(MT, copy=True)
        #modulation_matrix = sp.lil_matrix(MD + M1)
        print('modulation_matrix ishermitian ',ishermitian(T_0.multiply(modulation_matrix).todense()))
        
        return T_0.multiply(modulation_matrix)


    def T_meat(self, K_, T_0):
        """
            Adds the non-sparse modulation phase at each K_ to T_0. Turn to private.
        """

        ## at the moment implemented only for tight binding type 1
        ## maybe I should change this
        modulation_matrix = np.exp(-1j * np.dot(self.conf_.dist_matrix, K_))
        return_ = T_0 * modulation_matrix
        del modulation_matrix

        return return_

    def T_bone(self):
        """
            Build the non-sparse skeleton of Hamiltonian matrix. Please turn to private!
            
            Args:
                H_style: 'str' !! Not available yet!!
                    type of Hamiltonina formula in use
            
            Returns:
                T00 matrix
        """
        raise NotImplementedError('update the Hamiltonian!!')

        vc_mat = self.conf_.dist_matrix
        ez = self.conf_.ez

        dd_mat = np.linalg.norm(vc_mat, axis=2)

        if ez.shape == (3,):
            tilt_mat = np.power(np.dot(vc_mat, ez)/ dd_mat, 2)
        elif ez.shape == (self.conf_.tot_number, 3):
            tilt_mat = np.zeros((self.conf_.tot_number, self.conf_.tot_number))
            for ii in range(self.conf_.tot_number):
                tilt_mat[ii] = np.power(np.dot(vc_mat[ii], ez[ii])/ dd_mat[ii], 2)
        else:
            raise RuntimeError('Wrong ez!! there is a bug, please report code: bone_full')
        ##
        ##
        dd_mat_mask = np.zeros(dd_mat.shape, dtype='int')
        dd_mat_mask[dd_mat!=0] = 1

        V_sigam_mat = self.V0_sigam * np.exp(-(dd_mat - self.d0*dd_mat_mask) / self.r0 ) 
        V_pi_mat    = self.V0_pi    * np.exp(-(dd_mat - self.a0*dd_mat_mask) / self.r0 )

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


    def check_orthonormality(self, ii):
        """
            Checks if Vector are correctly orthonormal. (See the notes about solver.)
            Args:
                ii: int
                    Index of Kpoint to check. 
        """
        np.set_printoptions(suppress=True)
        print('DOT', np.dot(np.conjugate(self.bandsVector[k_].T), self.bandsVector[k_]) )

    ## plotting tools
    def detect_flat_bands(self, E_range=0.03):
        """
            Detect flat bands, assuming they are around E=0.
            All levels are sorted respect to E=0
                        
            Args:
                E_range: float
                    A positive value in unit [eV], setting a window of energy around E=0(sigma) to search for flat bands. (default = 30)
            
            Returns: None
        """
        if self.rank == 0:
            ## sort to have flat bands at the beggining of the bandsEigns
            N_flat = 0
            N_flat_old =0
            for k_ in range(self.bandsEigns.shape[0]):
                eigs_now = self.bandsEigns[k_, :]

                arg_sort = np.argsort(np.abs( eigs_now - 0.02 ))
                self.bandsEigns[k_, :] = eigs_now[arg_sort] 
                
                if self.bandsVector_exist == True:
                    self.bandsVector[k_] = self.bandsVector[k_, :, arg_sort].T
                    
                
                # find the flatbands
                N_flat = np.all([-1*E_range<eigs_now, eigs_now<E_range],axis=0).sum()
                try:
                    assert N_flat_old == N_flat or k_==0
                except AssertionError:
                    raise("Could not detect flat bands at k_={2}, N_flat_old != N_flat, {0}!={1} \n maybe try to change E_range".format(N_flat_old, N_flat, k_))
                        
                N_flat_old = N_flat

            self.n_flat = N_flat

    def shift_2_zero(self, k_label, idx_s):
        """
            A precise shift of Fermi level to zero, at given K point.
            Only useful if flat bands already exist.
            
            Note: this function resorts also flatbands based on the new shift.
            
            Args:
                k_label: str 
                    A specific k_label point to use of shifting. Must be an element of K_label
                
                idx_s: a list of indices (numpy)
                    Indices of flat bands to use for centering the Fermi-level. At least two Indices are required
            
            Hint for twisted bilayer graphene: 'K1' or 'K2'
        """
        if self.rank == 0:
            # check arguments
            try:
                assert  idx_s.shape >= (2,)
            except AssertionError:
                raise ValueError("Cannot shift, minimum two indices of flat bands are required")

            xpos_ = self.K_path_Highsymm_indices
            
            # find zero
            shift_tozero = 0
            try:
                idx = np.where(self.K_label==k_label)[0][0]
                shift_tozero = np.average( self.bandsEigns[int(xpos_[idx]), idx_s])
                print("I'm shifting to zero")
            except IndexError:
                raise ValueError("Wrong idx_s of flat bands")
                
            print("shift_tozero={0}".format(shift_tozero))
            
            self.bandsEigns -= shift_tozero
            # I am not sure why below is useful:
            ##resort flatbands after shift
            #for k_ in range(self.bandsEigns.shape[0]):
                #eigs_now = self.bandsEigns[k_, :N_flat] - shift_tozero
                
                #arg_sort = np.argsort(eigs_now )
                #arg_sort = np.flip(arg_sort)
                ##self.bandsEigns[k_, :N_flat] = eigs_now[arg_sort]
                #self.bandsEigns[k_, :N_flat] = self.bandsEigns[k_, arg_sort]
                ##print('sorted', eigs_now[arg_sort])
                ##print('sorted', self.bandsEigns[k_, :N_flat])
                #if self.bandsVector_exist == True:
                    #self.bandsVector[k_, :, :N_flat] = self.bandsVector[k_, :, arg_sort].T
                    #print('Vectors re-sorted after shift_2_zero..')
                            
    def plotter_bands(self, ax=None, color_='black',  y_shift=0):
        """
            Plot band structure
                        
            Args:
                ax: matplotlib object, optional
                    axis to plot, if not provided will create an ply.figure()
                color_: str 
                    color of lines, passed to matplotlib
                y_shift: float
                    an arbitraty vertical shift (default=0)
            
            Returns:
                ax: matplotlib object
        """

        if self.rank == 0:
            
            if ax==None:
                fig, ax = plt.subplots(figsize=(7, 10))
                #fig = plt.figure(figsize=(5, 10))
                #plt.figure(figsize=(5, 10))
                mpl.rcParams['pdf.fonttype'] = 42
                fontsize_ =20
                plt.rcParams['font.family'] = 'Helvetica'

            
            # plot far-bands
            for k_ in range(self.n_k_points):
                yy = self.bandsEigns[k_, :]*1000
                xx = np.full(n_eigns ,k_)

                ax.plot(xx[self.n_flat:], yy[self.n_flat:], '.', color=color_, linewidth=5, markersize=1)

            # plot flat-bands
            for flt in range(self.n_flat):
                xx = np.arange(self.n_k_points)
                yy = self.bandsEigns[:, flt]*1000
                ax.plot(xx, yy, '.', linewidth=3, markersize=5,color=color_)
                #ax.plot(xx, yy, '-o', linewidth=3, markersize=6, color='C{0}'.format(flt))

            ## plot vertical lines
            #for jj in range(self.n_Hsym_points):
                #plt.axvline(xpos_[jj], color='gray')
                
            xlabels = np.char.replace(self.K_label, 'Gamma',r'$\Gamma$')
            
            xpos_ = self.K_path_Highsymm_indices
            ax.set_xticks(xpos_, xlabels)
            #ax.set_yticks(fontsize=fontsize_)
            ax.tick_params(axis='both', which='major', labelsize=fontsize_+5)
            
            if xpos_.shape[0] >1:
                ax.set_xlim([xpos_[0],xpos_[-1]])
            ax.set_ylabel("E (meV)",fontsize=fontsize_)
            title_ = ''#save_name
            ax.set_title(title_+ 'Total number of flat bands= '+str(self.n_flat))#,fontsize=fontsize_)
            ax.grid(axis='y', c='gray',alpha=0.5)
            ax.grid(axis='x', c='gray',alpha=0.5)
            plt.gcf().subplots_adjust(left=0.15)
                        
            #fig.tight_layout()
            return  ax


    def plot_BZ(self, ax=None, ws_params={'ls': '--', 'color': 'tab:gray', 'lw': 1, 'fill': False}):
        if self.rank != 0:
            raise RuntimeError('Please plot using **if rank==0** in mpi mode')     
        if ax==None:
            fig, ax = plt.subplots(1,1)
        
        lab_offset = 0#0.01
        ax.set_title(r' $\to$ '.join(self.K_label))
        for K, Klab in zip(self.K_path_discrete, self.K_label):
            ax.annotate(Klab, [K[0], K[1]], [K[0]+lab_offset, K[1]+lab_offset])
        if hasattr(self, 'K_grid'):
            #ax.scatter(self.K_grid[:,0], self.K_grid[:,1], marker='.', c='pink', alpha=0.5, zorder=-10)
            ax.quiver(0, 0, *self.g1[:2], angles='xy', scale_units='xy', scale=1, color='k')
            ax.quiver(0, 0, *self.g2[:2], angles='xy', scale_units='xy', scale=1, color='k')

            u_rec = np.array([self.g1, self.g2, [0,0,1]])
            ax.scatter(self.K_grid[:,0], self.K_grid[:,1], c=self.K_mapping, cmap='jet')

            ax.scatter(self.K_grid[np.unique(self.K_mapping),0], self.K_grid[np.unique(self.K_mapping),1], c='w', marker='.', alpha=0.2)

            uc_style = {'closed': True,
                        'fill': False,
                        'ec' : "black",
                        'ls': "--",
                        'lw': 0.4}
            #uc_patch = get_uc_patch2D(u_rec, shift=-(u_rec[0]+u_rec[1])[:2]/2, plt_arg=uc_style)
            #ax.add_patch(uc_patch)

        BZ_2d = get_brillouin_zone_2d(np.array([self.g1[:2], self.g2[:2]]))
        plot_BZ2d(ax, BZ_2d, ws_params)
        ax.scatter(self.K_path[:,0], self.K_path[:,1], c=range(self.K_path.shape[0]), cmap='viridis', marker='.')
        ax.set_xlabel(r'$k_x$ [$\AA^{-1}$]')
        ax.set_ylabel(r'$k_y$ [$\AA^{-1}$]')
        return ax

    
    def plotter_DOS(self, ax):
        if self.rank != 0:
            raise RuntimeError('Please plot using **if rank==0** in mpi mode')
        if self.dosEigns != None:
            # --- Integrate ---
            # Integral extremal
            E_flat = self.dosEigns.flatten()
            #print(self.dosEigns.shape, self.dosEigns.shape[0]*self.dosEigns.shape[1], E_flat.shape)
            Emin, Emax = np.min(E_flat), np.max(E_flat)
            #print(Emin, Emax)
            nbins = 100
            cnts, bins = np.histogram(E_flat, bins=nbins)
            bin_center = bins[:-1] + np.diff(bins)/2
            self.DOS = cnts
            self.Ebin = bins
            self.Ebin_center = bin_center


            #plt.bar(self.Ebin_center, self.DOS, np.diff(self.Ebin), alpha=0.9, color='tab:green')
            #plt.plot(self.Ebin_center, self.DOS, '.-', alpha=0.9, color='tab:green')
            #plt.plot(self.Ebin_center, self.DOS, '-', color='tab:blue')
            # remember factor 0.5 from h.C.
            ax.fill_betweenx(self.Ebin_center*1e3*0.5, self.DOS, 0, alpha=0.5, color='tab:blue') # meV
            #ax.fill_betweenx(self., 0, self.Ebin_center, alpha=0.5, color='tab:blue')


    def plotter_3D(self, ax=None, color_='black', shift_tozero=None):
        if self.rank != 0:
            raise RuntimeError('Please plot using **if rank==0** in mpi mode')
        ## plot
        if ax==None:
            fig, ax = plt.subplots(figsize=(7, 10))
            ax = plt.axes(projection='3d')
        
        x = np.linspace(-1, 1, self.eigns_3D.shape[0])
        y = np.linspace( 0, 1, self.eigns_3D.shape[1])
        X, Y = np.meshgrid(x, y)

        print( self.eigns_3D.shape)
        #print( self.gsize_v)
        #print( self.eigns_3D.shape)
        data = (self.eigns_3D*1000*0.5 - shift_tozero).flatten()
        print(data.shape)
        np.savetxt("3d_data.txt",data,delimiter=',')
        exit()
        #X, Y = np.meshgrid(x, y)
        for ii in range(self.eigns_3D.shape[2]):
            Z = self.eigns_3D[:,:,ii]*1000*0.5 - shift_tozero
            ax.scatter3D(X, Y, Z, color='black', linewidth=0.1)
        #ax.set_ylim()
        ax.set_zlim([-10,15])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('E (meV)')
        
        return  ax




### DOS staff


    def MP_grid(self, Na, Nb):
        '''
        written by Andrea, I don't know what it does
        '''
        if not hasattr(self, 'g1'):
            self.MBZ()
            
        if Na%2 or Nb%2: print('It should be more efficient to use even divisions')
        #iK = 0
        #for ia in np.linspace(0,1,Na):
        #    for ib in np.linspace(0,1,Nb):
        #        self.K_grid[iK] = ia*self.g1+ib*self.g2
        #        iK+=1
        #self.nKgrid = self.K_grid.shape[0]
        u_rec = np.array([self.g1, self.g2, [0,0,1]]).T
        #print(u_rec)
        lattice = np.linalg.inv(u_rec)
        #print(lattice)
        shift = np.array([0,0,0])
        positions = np.array([[0,0,0]])
        numbers = [1]
        
        cell = (lattice, positions, numbers)
        sg = spglib.get_spacegroup(cell, symprec=1e-3)
        mesh = [Na, Nb, 1]
        print('Space group detected (spglib) %s' % str(sg))
        self.K_mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift)
        #plt.scatter(grid[:,0], grid[:,1])
        #plt.show()
        self.K_grid = np.dot(u_rec, ((grid + shift/2) / mesh).T).T
        #plt.scatter(self.K_grid[:,0], self.K_grid[:,1])
        #plt.show()
        self.nKgrid = self.K_grid.shape[0]
        print('Created %i x %i K grid (nKgrid=%i)' % (Na, Nb, self.nKgrid))
        print("Number of ir-kpoints: %d" % np.unique(self.K_mapping).shape[0] )
        self.nKgrid_uniq = np.unique(self.K_mapping).shape[0]


    def calculate_bands(self, n_eigns, solver, return_eigenvectors):    
        """
            Calculates band structure and vectors if requested.
            
            Args:             
                n_eigns: 
                    Number of eigen values desired out of Lanczos solver. 
                
                solver: str
                    'primme' (default) or 'scipy'. Caution!  scipy is faster sometimes but it has a error propagation bug in eigenvectros. sometimes returns nonorthonormal. Perhaps a bug in their's GramSchmidt. 
                    For symmetry checking 'primme' is recommended. 
                    While for band structure calculation 'scipy' is recommended.
                
                return_eigenvectors: boolean
                    True (default)
        """
        
        if return_eigenvectors:
            self.bandsEigns, self.bandsVector = self.engine_mpi(self.K_path, n_eigns, solver, return_eigenvectors=True)
            self.bandsVector_exist = True
        else:
            self.bandsEigns = self.engine_mpi(self.K_path, n_eigns, solver)
            self.bandsVector = None
            self.bandsVector_exist = False


    def calculate_DOS(self, n_eigns, solver):
        '''
        :param int n_dos: representing density for DOS
        '''
        
        self.dosEigns = self.engine_mpi(self.K_grid, n_eigns, solver)

        #uniq_map = np.unique(self.K_mapping)
        ##print('dfdfd=',uniq_map.shape)
        ##print('dfdfd=',self.K_mapping.shape)
        #ir_Eigns = self.engine_mpi(self.K_grid[uniq_map], n_eigns)
        
        #if self.rank == 0:
            #self.dosEigns = np.zeros([self.nKgrid, n_eigns]) 
            ## AS: on full grid (should actually store only needed once... more memory efficicient . 
            #for ikk, kk in enumerate(uniq_map):
                #self.dosEigns[self.K_mapping == kk] = ir_Eigns[ikk]

    
    def set_grid(self, n_points):
        if not hasattr(self, 'g1'):
            self.MBZ()
        #if self.rank == 0:
        if self.orientation  == '1_fold':
            #Ali: because it is D2, I only calculate upper half of BZ
            h_edge = 2*np.abs(np.linalg.norm(self.MBZ_X)) 
            v_edge =   np.abs(np.linalg.norm(self.MBZ_Y))
            
            ratio_ = h_edge/v_edge if h_edge > v_edge else v_edge/h_edge # ratio_ > 1 by defination
            
            discretes = int(np.sqrt(n_points/ratio_))
            
            discretes_h = discretes*ratio_ if h_edge > v_edge else discretes
            discretes_v = discretes if h_edge > v_edge else discretes*ratio_
            
            h_left  = np.linspace(self.MBZ_X,   self.MBZ_gamma,  num=int(discretes_h/2),  endpoint=False)
            h_right = np.linspace(self.MBZ_gamma, -1*self.MBZ_X, num=int(discretes_h/2)+1)
            h_ = np.concatenate((h_left, h_right))
            
            ## just to have minimum idle cores # dont worry about it
            gsize_h = h_.shape[0]
            gsize_v = int(discretes_v) #v_up.shape[0]
            #N = self.size*(gsize_h*gsize_v//self.size) 
            #gsize_v = N//gsize_h
            
            v_up  = np.linspace(self.MBZ_gamma, self.MBZ_Y, num=gsize_v)
            
            self.flat_grid = np.zeros((gsize_h*gsize_v, 3))
            
            for ii in range(gsize_v):
                self.flat_grid[ii*gsize_h : (ii+1)*gsize_h] = h_ + v_up[ii]
            
            self.gsize_h = gsize_h
            self.gsize_v = gsize_v
                    
        else:
            raise KeyError('I am sorry, at the moment uniform_grid is only developed for 1-fold')
    
    def reduced2full(self):
        if self.rank == 0:
            if self.orientation  == '1_fold':
                # apply D2 symmetry to fill the BZ
                eign_mat = self.eigns_3D_reduced.reshape((self.gsize_h, self.gsize_v, -1))
                eign_mat_lower = np.rot90(eign_mat[1:,:,:], 2, (0,1)) 
                #np.flip(nn,axis=0) # this one mirros
                self.eigns_3D = np.concatenate((eign_mat_lower, eign_mat))
                
                del eign_mat
                del eign_mat_lower
                
                #grid_mat = self.flat_grid.reshape((self.gsize_h, self.gsize_v, -1))
                #grid_mat_lower = np.rot90(grid_mat[1:,:,:], 2, (0,1)) 
                #self.grid_mat = np.concatenate((grid_mat_lower, grid_mat))
                
            else:
                raise KeyError('I am sorry, at the moment uniform_grid is only developed for 1-fold')
        
    
    def calculate_3Dbands(self, n_eigns, solver):
        #if self.rank == 0: 
            #self.eigns_3D = np.zeros((self.gsize_h*self.gsize_v, n_eigens), dtype='f')
            
        self.eigns_3D_reduced  = self.engine_mpi(self.flat_grid, n_eigns, solver)
        
        self.reduced2full()
            
        
