import time, os
from .configuration import Pwl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from mpi4py import MPI
from tqdm import tqdm
from scipy.linalg import ishermitian
import primme
import warnings

from .misc import get_brillouin_zone_2d, plot_BZ2d#, get_uc_patch2D

'''
This code is using MPI
'''


class TB:
    def __init__(self):
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.print0('TB object created')
                

    def print0(self, *argv):
        if self.rank == 0:
            print(*argv)

    def engine_mpi(self, T_M, kpoints, n_eigns, solver='primme', return_eigenvectors=False):
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
            Eignvecs = np.zeros([npoints, self.conf.tot_number, n_eigns], dtype=self.dtypeC)
            
        
        
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

        #print('Start loop on %i K-points' % npoints)
        t_start = time.time()
        
        if self.rank ==0:# or True:
            #pbar = tqdm(total=slice_size, unit=' K-point', desc='rank {}'.format(self.rank)) # Initialise
            pbar = tqdm(total=npoints, unit=' K-point', desc='Estimated') # Initialise
        
        for kk in kk_range:
            t_loop = time.time()

            H = T_M(kpoints[kk], self.T0 )
            
            
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
                    self.save(version=self.K_label[idxH], H = H)
                except IndexError:
                    pass
                
            if self.rank ==0:# or True:
                pbar.update(self.size)
                
        if self.rank ==0:# or True:
            pbar.close()
            
        print("Total time: {:.2f} seconds ".format(time.time() - t_start) + "on rank {0}".format(self.rank))
        
        
        ## collect from all cpus
        if return_eigenvectors:
            sendbufVec = Eignvecs
            recvbufVec = None
            
        sendbuf = Eigns
        recvbuf = None
        if self.rank == 0:
            recvbuf    = np.zeros([self.size, npoints, n_eigns], dtype=self.dtypeR)
            if return_eigenvectors:
                recvbufVec = np.zeros([self.size, npoints, self.conf.tot_number, n_eigns], dtype=self.dtypeC)

        self.comm.Gather(sendbuf, recvbuf, root=0)
        if return_eigenvectors:
            self.comm.Gather(sendbufVec, recvbufVec, root=0)
            
        if self.rank == 0:
            for ii in range(1,self.size):
                
                slice_i = (ii)*share_
                slice_f = (ii+1)*share_ 
                Eigns[slice_i:slice_f] = recvbuf[ii][slice_i:slice_f]     
                
                if return_eigenvectors:
                    Eignvecs[slice_i:slice_f]  = recvbufVec[ii][slice_i:slice_f]
                
                rev_rank = self.size - ii 
                if  rev_rank <= share_left:
                    Eigns[-rev_rank] = recvbuf[ii][-rev_rank]
                    ##
                    if return_eigenvectors:
                        Eignvecs[-rev_rank]  = recvbufVec[ii][-rev_rank]
            
            Eigns *= 0.5 # 0.5 because of H.C. more efficent this way

            if return_eigenvectors: 
                print('Succesfully collected eigensvalues and vectors')
                return Eigns, Eignvecs
            else:
                print('Succesfully collected eigensvalues ')
                return Eigns        
        else:
            return None, None if return_eigenvectors else None


    def set_configuration(self, file_and_folder, r_cut, local_normal=True, nl_method='RS', sparse_flag=True, dtype='float', version=''):
        """
            Set the basic configuration. And build up some basic requirments: normal vectors, neighborlist,

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
                r_cut: float
                    Neighboring cutoff in the same distance unit as the coordinate file. Circular based neighbor detecting method.
                    Larger value includes more neighbor for each cite.
                local_normal: boolean
                    To use local_normal to orbital direction(True) or consider them all aligned in Z direction (False). (default = True)
                nl_method: str
                    which method to use for creating neighborlist. 'RS' -reduce space implementation- (faster but might have a bug in rhombic cells, to be investigated) or 'RC' -replicating coordinates- (slower, bug free) (default = 'RS')
                    nl_method is ignored if load_neigh=True and a previously calculated neighborlist exist.
        """
        
        # check input
        try:
            assert dtype == 'float' or dtype == 'double'
        except AssertionError: raise TypeError('Only float(default) or double is supported for dtype')
        
        self.r_cut = r_cut 
        self.sparse_flag=sparse_flag
        self.dtypeR = dtype
        self.dtypeC = 'c'+dtype
        self.file_name = file_and_folder.split('/')[-1]
        self.folder_name = '/'.join(file_and_folder.split('/')[:-1]) #+ '/'

        # Creating new folder:
        new_folder = 'out_' + '.'.join(self.file_name.split('.')[:-1]) +version+ '/'
        self.folder_name += new_folder
        if self.rank == 0:  os.makedirs(self.folder_name, exist_ok=True)
        
        # Call on configuration.py
        self.conf = Pwl(self.folder_name, self.sparse_flag, self.dtypeR)
        
        self.conf.read_coords(self.file_name)

        # build neigh_list
        version_ = '_cutoff_' + str(self.r_cut)
        
        if self.rank == 0:  
            # calculate neigh_list
            self.conf.neigh_list(cutoff=self.r_cut,  nl_method=nl_method, load_ = False, version_ = version_ )
            
            # send to other cups
            signal_ = True
            for ii in range(1,self.size):
                req = self.comm.isend(signal_, dest=ii, tag=11)
                req.wait()
        else:
            req = self.comm.irecv(source=0, tag=11)
            signal_ = req.wait() 
            assert signal_ is True
            self.conf.neigh_list(cutoff=self.r_cut, load_ = True, version_ = version_ ) 
            
        
        # build distance matrix
        self.conf.vector_connection_matrix()
        # build normal vectors
        self.conf.normal_vec(local_normal)



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
            
        else:
            if self.conf.xy ==0:
                gx = 2*np.pi/np.linalg.norm(self.conf.xlen) *np.array([1,0,0])
                gy = 2*np.pi/np.linalg.norm(self.conf.ylen) *np.array([0,1,0])
                self.g1 = gx
                self.g2 = gy
                self.MBZ_X = gx/2
                self.MBZ_W = (gx+gy)/2
                self.MBZ_Y = gy/2
                self.MBZ_gamma = np.array([0,0,0])
            else:
                vector_b1 = np.array([self.conf.xlen, 0, 0])
                vector_b2 = np.array([self.conf.xy, self.conf.ylen, 0])
                vector_b3 = np.array([0,0,1])
                
                angle_ = np.rad2deg(np.arccos(np.dot(vector_b1,vector_b2)/(np.linalg.norm(vector_b1)*np.linalg.norm(vector_b2))))
                
                if not np.isclose(angle_, 60, atol=1):
                    raise ValueError("angle(a,b) is {0} \nNote: g1 & g2 could be automatically calculated only if the angle between a and b lattice vectors is either 60 or 90 degrees. Otherwise you should provide as input.".format(angle_))
                
                #self.print0('Box scew is not zero.. g1 & g2 will not be orthogonal')

                
                volume = np.abs(np.dot(vector_b3,  np.cross(vector_b1, vector_b2) ) )            
                g1 = 2*np.pi/volume * np.cross(vector_b3, vector_b2)
                g2 = 2*np.pi/volume * np.cross(vector_b3, vector_b1)
                self.g1 = g1
                self.g2 = g2
                
                self.MBZ_gamma = np.array([0,0,0])
                self.MBZ_m2 = g2 / 2
                self.MBZ_m1 = g1 / 2
                self.MBZ_k1 = (g2 + g1) /3
                self.MBZ_k2 = (2*g2 - g1) / 3
                
        self.print0('K vectors: \ng1={0}, \ng2={1}'.format(self.g1,self.g2)) 
    
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
            assert N==0 or N > self.K_label.shape[0]
        except AssertionError:
            raise ValueError("If N != 0, it must be larger than number of elements in K_label. You can set N=0(default) for a discrete calculation")
        try: 
            assert K_path==None or K_path.shape[1] == 3
        except AssertionError:
            raise ValueError("K_path should be an numpy array in shape of (n, 3)")
        
        
        Kmode = 'discrete' if N == 0 else 'continues'
        
        
        #
        if Kmode == 'discrete':
            self.print0('your path is discrete')
        else:
            self.print0('your path is continues')
            self.print0("requested n_k_points={0}".format(N))
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
            
        n_k_points = self.K_path.shape[0]
        self.print0("actual n_k_points={0}".format(n_k_points))
 

    def save(self, version='', bands=False, dos=False, _3Dbands=False, configuration = False, H = None):

        if H!= None:
            #np.savez(self.folder_name + 'HH_' +version, H=self.H)
            sp.save_npz(self.folder_name + 'HH_' +version, sp.csr_matrix(H, copy=True))
            return 0
        
        
        if self.rank ==0 :
            #if hasattr(self, 'conf'):
            if configuration:
                np.savez(self.folder_name + 'configuration_' +version,
                        nl = self.conf.nl,
                        dtypeR = self.conf.dtypeR,
                        sparse_flag = self.conf.sparse_flag,
                        xy = self.conf.xy,
                        xlen = self.conf.xlen,
                        ylen = self.conf.ylen,
                        zlen = self.conf.zlen,
                        tot_number= self.conf.tot_number,
                        xlen_half = self.conf.xlen_half,
                        ylen_half = self.conf.ylen_half,
                        coords = self.conf.coords,
                        atomsAllinfo = self.conf.atomsAllinfo,
                        fnn_id = self.conf.fnn_id,
                        B_flag = self.conf.B_flag,
                        dist_matrix = self.conf.dist_matrix,
                        fnn_vec = self.conf.fnn_vec,
                        ez = self.conf.ez,
                        cutoff = self.conf.cutoff,
                        local_flag = self.conf.local_flag,
                        file_name = self.conf.file_name)
            
            #if hasattr(self, 'bandsEigns'):
            if bands:
                np.savez(self.folder_name + 'bands_' +version ,
                        bandsEigns=self.bandsEigns, K_path=self.K_path, 
                        K_path_Highsymm_indices = self.K_path_Highsymm_indices, 
                        K_label=self.K_label, K_path_discrete= self.K_path_discrete,
                        g1=self.g1, g2=self.g2,  
                        bandsVector = self.bandsVector,
                        file_name = self.file_name)
            
            #if hasattr(self, 'dosEigns'):
            if dos:
                np.savez(self.folder_name + 'DOS_'+version,
                        dosEigns=self.dosEigns,
                        K_grid=self.K_grid, K_mapping=self.K_mapping)
            

            #if hasattr(self, 'eigns_3D'):
            if _3Dbands:
                np.savez(self.folder_name + '3Dband_'+version, gsize_v=self.gsize_v, gsize_h=self.gsize_h, flat_grid=self.flat_grid, eigns_3D=self.eigns_3D, eigns_3D_reduced=self.eigns_3D_reduced)


   
       
    def load(self, folder_='', bands=None, HH=None, dos=None, _3Dbands=None, configuration = None):
        
        if self.rank ==0 :
            
            self.folder_name = folder_  if folder_[-1] == '/' else folder_+'/'
                        
            if configuration != None:
                
                self.print0('loading ', self.folder_name+ configuration)
                conf_file = np.load(self.folder_name + configuration, allow_pickle=True)
                self.conf = Pwl(self.folder_name)
                
                self.dtypeR = self.conf.dtypeR = str(conf_file['dtypeR'])
                self.dtypeC = 'c'+ self.dtypeR
                self.r_cut = self.conf.cutoff = conf_file['cutoff']
                self.conf.nl = conf_file['nl']
                self.sparse_flag = self.conf.sparse_flag = conf_file['sparse_flag']
                self.conf.xy = conf_file['xy']
                self.conf.xlen = conf_file['xlen']
                self.conf.ylen = conf_file['ylen']
                self.conf.zlen = conf_file['zlen']
                self.conf.tot_number = conf_file['tot_number']
                self.conf.xlen_half = conf_file['xlen_half']
                self.conf.ylen_half = conf_file['ylen_half']
                self.conf.coords = conf_file['coords']
                self.conf.atomsAllinfo = conf_file['atomsAllinfo']
                self.conf.fnn_id = conf_file['fnn_id']
                self.conf.B_flag = conf_file['B_flag']
                self.conf.dist_matrix = conf_file['dist_matrix']
                self.conf.fnn_vec = conf_file['fnn_vec']
                self.conf.ez = conf_file['ez']
                self.conf.local_flag = conf_file['local_flag']
                self.conf.file_name = self.file_name = conf_file['file_name']
   
            
                    
            if bands != None :
                
                print('loading ', self.folder_name+ bands)
                data_band = np.load(self.folder_name + bands, allow_pickle=True)
                
                self.bandsEigns = data_band['bandsEigns']
                self.K_path = data_band['K_path']
                self.K_path_Highsymm_indices = data_band['K_path_Highsymm_indices']
                self.K_label = data_band['K_label']
                self.K_path_discrete = data_band['K_path_discrete']
                self.g1 = data_band['g1']
                self.g2 = data_band['g2']
                self.file_name = data_band['file_name']

                
                try:
                    self.bandsVector = data_band['bandsVector']
                    if self.bandsVector.all() == None:
                        self.bandsVector_exist = False
                    else:
                        self.bandsVector_exist = True
                except KeyError: 
                    self.bandsVector = None
                    self.bandsVector_exist = False

            if dos != None :
                print('loading ', self.folder_name+ dos)
                data_dos = np.load(self.folder_name + dos)
                self.dosEigns = data_dos['dosEigns']
                self.K_grid = data_dos['K_grid']
                self.K_mapping = data_dos['K_mapping']
            else:
                self.dosEigns = None
            
            if _3Dbands != None :
                print('loading ', self.folder_name+ _3Dbands)
                data_3Dbands = np.load(self.folder_name + _3Dbands)
                self.gsize_v = data_3Dbands['gsize_v']
                self.gsize_h = data_3Dbands['gsize_h']
                self.flat_grid = data_3Dbands['flat_grid']
                self.eigns_3D = data_3Dbands['eigns_3D']
                self.eigns_3D_reduced = data_3Dbands['eigns_3D_reduced']
                
            if HH != None:# and load_H:
                print('loading ', self.folder_name+ HH)
                
                #H = np.load(self.folder_name + HH, allow_pickle=True)['H']
                H = sp.lil_matrix(sp.load_npz(self.folder_name + HH))

                return H

    


    def T_bone_sp(self, H_style):
        """
            Build the sparse skeleton of Hamiltonian matrix. Please turn to private!
            
            Args:
                H_style: function
                    Pairwise Hamiltonina formula in use. It should get the following arguments in order:
                    H_ij(vector_ij, normal_i, normal_j)
                    Where vector_ij is the vector connecing cite "i" to cite "j". normal_i (normal_j) is the normal vector to surface at cite "i" ("j").
            Returns:
                T00 matrix
        """
        
        T00 = sp.lil_matrix((self.conf.tot_number, self.conf.tot_number), dtype=self.dtypeR)

        for ii in range(self.conf.tot_number):
            neighs = self.conf.nl[ii][~(np.isnan(self.conf.nl[ii]))].astype('int')
            for jj in neighs:
                
                # calculate the hoping
                v_ij = np.array([ self.conf.dist_matrix[0][ii,jj],  self.conf.dist_matrix[1][ii,jj],  self.conf.dist_matrix[2][ii,jj] ])
                
                T00[ii, jj] = H_style(v_ij, self.conf.ez[ii], self.conf.ez[jj]) 

        T00_copy = T00.copy()
        T00_trans = sp.lil_matrix.transpose(T00, copy=True)
        T00_dagger  = sp.lil_matrix.conjugate(T00_trans, copy=True)
        T00 = sp.lil_matrix(T00_dagger + T00_copy)

        self.print0('T_bone ishermitian ',ishermitian(T00.todense(), rtol=0.0))
        return T00

    def T_meat_sp1(self, K_, T_0):
        """
            Adds the sparse modulation phase at each K_ to T_0. Turn to private.
            Tight binding type 1, with a phase
        """

        modulation_matrix = sp.lil_matrix(( self.conf.tot_number, self.conf.tot_number), dtype=self.dtypeC)
        #print('making modulation_matrix..')
        for ii in range(self.conf.tot_number):
            neighs = self.conf.nl[ii][~(np.isnan(self.conf.nl[ii]))].astype('int')
            for jj in neighs:
                ## tight binding type 1, with a phase
                v_c = np.array([ self.conf.dist_matrix[0][ii,jj],  self.conf.dist_matrix[1][ii,jj],  self.conf.dist_matrix[2][ii,jj] ])
                modulation_matrix[ii,jj] = np.exp(-1j * np.dot(v_c, K_))
        
        #print('modulation_matrix ishermitian ',ishermitian(T_0.multiply(modulation_matrix).todense()))
        
        return T_0.multiply(modulation_matrix)


    def T_meat_sp2(self, K_, T_0):
        """
            Adds the sparse modulation phase at each K_ to T_0. Turn to private.
            Tight binding type 2, no extra phase
        """

        modulation_matrix = sp.lil_matrix(( self.conf.tot_number, self.conf.tot_number), dtype=self.dtypeC)
        #print('making modulation_matrix..')
        for ii in range(self.conf.tot_number):
            neighs = self.conf.nl[ii][~(np.isnan(self.conf.nl[ii]))].astype('int')
            for jj in neighs:
                ## tight binding type 2, no extra phase
                thing = [1*self.conf.B_flag[0][ii,jj] * self.conf.xlen, 1*self.conf.B_flag[1][ii,jj] * self.conf.ylen, 0]
                modulation_matrix[ii,jj] = np.exp(+1j * np.dot(thing, K_ )) 
        
        #print('modulation_matrix ishermitian ',ishermitian(T_0.multiply(modulation_matrix).todense()))
        
        return T_0.multiply(modulation_matrix)


    def T_meat(self, K_, T_0):
        """
            Adds the non-sparse modulation phase at each K_ to T_0. Turn to private.
        """

        ## at the moment implemented only for tight binding type 1
        ## maybe I should change this
        modulation_matrix = np.exp(-1j * np.dot(self.conf.dist_matrix, K_))
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

        vc_mat = self.conf.dist_matrix
        ez = self.conf.ez

        dd_mat = np.linalg.norm(vc_mat, axis=2)

        if ez.shape == (3,):
            tilt_mat = np.power(np.dot(vc_mat, ez)/ dd_mat, 2)
        elif ez.shape == (self.conf.tot_number, 3):
            tilt_mat = np.zeros((self.conf.tot_number, self.conf.tot_number))
            for ii in range(self.conf.tot_number):
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
            
            print(N_flat," flat bands detected")
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
                #print("I'm shifting to zero")
            except IndexError:
                raise ValueError("Wrong idx_s of flat bands")
                
            print("shifting to zero by {0}".format(shift_tozero))
            
            self.bandsEigns -= shift_tozero

        

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
                fontsize_ =16
                plt.rcParams['font.family'] = 'Helvetica'
                mpl.rcParams.update({'figure.autolayout': True})

            n_eigns = self.bandsEigns.shape[1]
            n_k_points = self.bandsEigns.shape[0]
            
            
            if hasattr(self, 'n_flat'):
                # plot far-bands
                for k_ in range(n_k_points):
                    yy = self.bandsEigns[k_, :]*1000
                    xx = np.full(n_eigns ,k_)

                    ax.plot(xx[self.n_flat:], yy[self.n_flat:], '.', color=color_, linewidth=5, markersize=1)

                # plot flat-bands
                for flt in range(self.n_flat):
                    xx = np.arange(n_k_points)
                    yy = self.bandsEigns[:, flt]*1000
                    ax.plot(xx, yy, '.', linewidth=3, markersize=5,color=color_)
                    #ax.plot(xx, yy, '-o', linewidth=3, markersize=6, color='C{0}'.format(flt))
            else:
                # plot all bands
                for k_ in range(n_k_points):
                    yy = self.bandsEigns[k_, :]*1000
                    xx = np.full(n_eigns ,k_)
                    
                    ax.plot(xx, yy, '.', color=color_, linewidth=5, markersize=5)


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
            title_ = ''
            if hasattr(self, 'n_flat'):
                ax.set_title(title_+ 'Total number of flat bands= '+str(self.n_flat), fontsize=fontsize_)
            ax.grid(axis='y', c='gray',alpha=0.5)
            ax.grid(axis='x', c='gray',alpha=0.5)
            plt.gcf().subplots_adjust(left=0.2)
            #mpl.rcParams.update({'figure.autolayout': True})

            #fig.tight_layout()
            return  ax



    def calculate_bands(self, H_style, n_eigns, sigma, solver, tbt='type1' , return_eigenvectors=False):    
        """
            Calculates band structure and vectors if requested.
            
            Args:
                H_style: function
                    Pairwise Hamiltonina formula in use. It should get the following arguments in order:
                    H_ij(vector_ij, normal_i, normal_j)
                    Where vector_ij is the vector connecing cite "i" to cite "j". normal_i (normal_j) is the normal vector to surface at cite "i" ("j").
                n_eigns: 
                    Number of eigen values desired out of Lanczos solver. 
                
                solver: str
                    'primme' (default) or 'scipy'. Caution!  scipy is faster sometimes but it has a error propagation bug in eigenvectros. sometimes returns nonorthonormal. Perhaps a bug in their's GramSchmidt. 
                    For symmetry checking 'primme' is recommended. 
                    While for band structure calculation 'scipy' is recommended.
                
                tbt: str
                    type of tight binding of H(K)
                    'type1'(default) is the usual modulation t_ij(k) = t_ij * np.exp(-1j * np.dot(v_c, k))
                    'type2': t_ij(k) = t_ij * np.exp(-1j * np.dot(R, k)) where R is the lattice vector pointing to neighboring cells, only if j is in outside the cell. 
                    'type2' is useful for symmetry operation, but at the moment it is only implemented for rectangular lattice.
                
                return_eigenvectors: boolean
                    False (default)
                    
                sigma: float
                    Find the n_eigns eigenvalues around this value
        """
        
        # Check args
        try:
            assert tbt=='type1' or tbt=='type2'
        except AssertionError:
            raise ValueError("Wrong tbt! Available options: 'type1' or 'type2' ")
        if tbt=='type2' and self.conf.xy !=0:
             raise NotImplementedError("tbt='type2' is not implemented for non orthogonal lattice.")
        
        
        self.sigma_shift = sigma
        
        # build the 'Bone' matrix
        if self.sparse_flag:
            self.T0 = self.T_bone_sp(H_style)
            if tbt=='type1':
                T_M = self.T_meat_sp1
            if tbt=='type2':
                T_M = self.T_meat_sp2
        else:
            self.T0 = self.T_bone(H_style)

        self.print0('T_bone is constructed..')
        
        # calculate levels
        if return_eigenvectors:
            self.bandsEigns, self.bandsVector = self.engine_mpi(T_M, self.K_path, n_eigns, solver, return_eigenvectors=True)
            self.bandsVector_exist = True
        else:
            self.bandsEigns = self.engine_mpi(T_M, self.K_path, n_eigns, solver)
            self.bandsVector = None
            self.bandsVector_exist = False


### DOS + BZ plot

    def calculate_DOS(self, n_eigns, solver):
        '''
        :param int n_dos: representing density for DOS
        '''
        
        self.dosEigns = self.engine_mpi(self.K_grid, n_eigns, solver)


    
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
        
