import time
import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree
from tqdm import tqdm
'''
This code is not using MPI
'''

'''
I am sure this code is well commented! AS
'''

class Pwl:

    def __init__(self, folder_name, sparse_flag=True, dtype=None):
        
        """
            Define the namespace, useful in case of loading
        """
        self.dtypeR = dtype
        self.folder_name = folder_name
        self.sparse_flag = sparse_flag
        
        self.xy = None
        self.xlen = None
        self.ylen = None
        self.zlen = None
        #self.xhi = None
        #self.yhi = None
        #self.zhi = None
        self.tot_number= None
        self.xlen_half = None
        self.ylen_half = None
        self.coords = None
        self.atomsAllinfo = None
        self.fnn_id = None
        self.B_flag = None
        self.dist_matrix = None
        self.fnn_vec = None
        self.ez = None
        self.cutoff = None
        self.local_flag = None
        self.file_name = ''
        
        
        
    def read_coords(self, file_name):
        """
            Read coordinates from files.
            
            Args:
                file_name: str
                    Correctly onlt LAMMPS format is accepted.
        """
        
        self.file_name = file_name
        file_ = open(file_name)
        l_ = file_.readlines()
        self.xy = 0
        header_ = ''
        
        ii = 0
        end_flag = False
        while ii<20:
            if 'atoms' in l_[ii]:
                self.tot_number = int(l_[ii].split()[0])
            elif 'atom types' in l_[ii]:
                self.tot_type = int(l_[ii].split()[0])
            elif 'xlo' in l_[ii]:
                self.xlo = float(l_[ii].split()[0])
                self.xhi = float(l_[ii].split()[1])
                self.xlen = self.xhi  - self.xlo # positive by defination
            elif 'ylo' in l_[ii]:
                self.ylo = float(l_[ii].split()[0])
                self.yhi = float(l_[ii].split()[1])
                self.ylen = self.yhi  - self.ylo # positive by defination
            elif 'zlo' in l_[ii]:
                self.zlo = float(l_[ii].split()[0])
                self.zhi = float(l_[ii].split()[1])
                self.zlen = self.zhi  - self.zlo # positive by defination
            elif 'xy xz yz' in l_[ii]:
                self.xy = float(l_[ii].split()[0])
            elif 'Atoms' in l_[ii]:
                skiplines = ii + 1 +1
                end_flag = True
                header_ += l_[ii] 

            header_ +=l_[ii] if not end_flag else ''

            ii += 1

        self.ylen_half = self.ylen/2
        self.xlen_half = self.xlen/2

        file_.close()
        ### get coordinates 
        self.atomsAllinfo = np.loadtxt(file_name, skiprows =skiplines, max_rows= self.tot_number, dtype=self.dtypeR)
        self.coords = self.atomsAllinfo[:,4:7]
    

    def vector_connection_matrix(self, fnn_cutoff=1.55):
        """
            Create geometrical vectros connecting each two neighbors.
            Args:
                fnn_cutoff: float 
                    Maximum cut off to detect first nearest neghbours (default =1.55)
            Returns: None
        """
        print('creating vector_connection_matrix...')

        if self.sparse_flag:
            dist_matrix_X = sp.lil_matrix((self.tot_number, self.tot_number), dtype=self.dtypeR)
            dist_matrix_Y = sp.lil_matrix((self.tot_number, self.tot_number), dtype=self.dtypeR)
            dist_matrix_Z = sp.lil_matrix((self.tot_number, self.tot_number), dtype=self.dtypeR)
            boundary_flag_X = sp.lil_matrix((self.tot_number, self.tot_number), dtype='int')
            boundary_flag_Y = sp.lil_matrix((self.tot_number, self.tot_number), dtype='int')
            #self.dist_matrix_norm = sp.lil_matrix((self.tot_number, self.tot_number), dtype='float')
            self.dist_matrix = np.array([dist_matrix_X, dist_matrix_Y, dist_matrix_Z]) # being 3,N,N # no otherway
            self.B_flag = np.array([boundary_flag_X, boundary_flag_Y]) # being 3,N,N # no otherway
        else:
            self.dist_matrix = np.zeros((self.tot_number, self.tot_number, 3), self.dtypeR)

        self.fnn_vec = np.full((self.tot_number,3,3), np.nan)
        self.fnn_id  = np.zeros((self.tot_number,3), dtype='int' )

        for ii in range(self.tot_number):
            neighs = self.nl[ii][~(np.isnan(self.nl[ii]))].astype('int') 
            # number of neighbors are not known, so we use this approach
            zz = 0
            for jj in neighs:

                dist_ = self.coords[jj] - self.coords[ii]

                yout =0
                xout =0
                old_dist_size = np.linalg.norm(dist_)

                if dist_[1] > self.ylen_half:
                    dist_[1] -= self.ylen
                    dist_[0] -= self.xy
                    yout = -1
                elif -1*dist_[1] > self.ylen_half:
                    dist_[1] += self.ylen
                    dist_[0] += self.xy
                    yout = +1

                if dist_[0] > self.xlen_half:
                    dist_[0] -= self.xlen
                    xout = -1
                elif -1*dist_[0] > self.xlen_half:
                    dist_[0] += self.xlen
                    xout = +1
                
                dist_size = np.linalg.norm(dist_)
                if dist_size < fnn_cutoff:
                    self.fnn_vec[ii, zz] = dist_
                    self.fnn_id[ii, zz] = jj
                    zz += 1

                ## for debugging
                if dist_size > 1.01*self.cutoff:
                    print('POS_ii, POS_jj\n {0} \n {1}'.format(self.coords[ii], self.coords[jj]))
                    print('New dist = {0}'.format(dist_size))
                    print('Old dist = {0}'.format(old_dist_size))
                    raise RuntimeError('something is wrong with PBC')

                if self.sparse_flag:
                    self.dist_matrix[0][ii, jj] = dist_[0]
                    self.dist_matrix[1][ii, jj] = dist_[1]
                    self.dist_matrix[2][ii, jj] = dist_[2]
                    
                    self.B_flag[0][ii, jj] = xout
                    self.B_flag[1][ii, jj] = yout

                else:
                    self.dist_matrix[ii, jj] = dist_

            if zz!=3:
                raise RuntimeError("there is an error in finding first nearest neighbors:  please tune *fnn_cutoff*")

    
    def normal_vec(self, local = False):
        """
            To create local normal vector.
            Note: On the choice of axis, it is assumed that the two main dimension of structure is on average perpendicular to Z axis.
            Warning!! chirality is not computed. Always the positive normal_vec (pointing upward) is returned.
        
            Args:
                local: boolean
                    If to calculated vertical normals:

            Returns: None
        """
        self.local_flag = local
        self.ez = np.full((self.tot_number,3), np.nan)
        
        if local:
            print("calculating local normal vectors ...")
            for ii in range(self.tot_number):
                neighs = self.fnn_vec[ii]
                aa = np.cross(neighs[0], neighs[1])
                bb = np.cross(neighs[1], neighs[2])
                cc = np.cross(neighs[2], neighs[0])

                norm_ = aa + bb + cc
                norm_ = norm_/np.linalg.norm(norm_)
                
                if norm_[2] < 0:
                    norm_ *= -1

                self.ez[ii] = norm_
        else:
            print("ez=[0,0,1] for all orbitals ...")
            self.ez = np.full((self.tot_number,3), [0,0,1])

    def neigh_list(self, cutoff, nl_method='RS', l_width = 500, load_=True, version_=''):
        """
            Neighbor detect all neighbours of all cites within a cutoff range.
            
            Args:
                cutoff: float
                    Used to detect neighbors within a circular range around each individual cites.
                method: str, optional
                    which method to use for creating neighborlist. 'RS' -reduce space implementation- (faster but might have a bug in rhombic cells, to be investigated) or 'RC' -replicating coordinates- (slower, bug free) (default = 'RS')
                l_width: int, optional
                    Maximum number of allowed neghbours. For memory efficency reasons it is better to keep number small (default = 500)
                    In rare senarios if you use very large cutoff, you might need to increase this quantity. You get a MemoryError in this case.
                load_: boolean, optional
                    Load a previously created neighborlist. (default = True)
                version_: str, optional
                    A postfix for save name
            Returns: None
        """
        
        # check inputs
        try:
            assert nl_method == 'RC' or nl_method == 'RS'
        except AssertionError: raise TypeError("Wrong nl_method. Only 'RC' or 'RS'")
 
        self.cutoff = cutoff

        if load_:
            try:
                data_ = np.load(self.folder_name + 'neigh_list_{0}.npz'.format(version_))
                self.nl     = data_['np_nl']
                tot_neigh = data_['tot_neigh']
                ave_neigh = data_['ave_neigh']
                #print('neigh_list is loaded from the existing file: neigh_list_{0}.npz'.format(version_))
                print('ave_neigh={0} \ntot_neigh={1}'.format(ave_neigh, tot_neigh))
                #print("if want to rebuild the neigh_list, use: build_up(..,load_neigh=False)")
            except FileNotFoundError:   
                load_ = False
                print('A neighlist file was not found, building one... ')

        if not load_:
            # Init array with maximum neighbours l_width
            np_nl = np.full((self.tot_number, l_width), np.nan)
            
            if nl_method == 'RC':
                
                coords_9x = np.concatenate((self.coords,
                                            self.coords+np.array([+self.xlen,0,0]),
                                            self.coords+np.array([-self.xlen,0,0]),
                                            self.coords+np.array([+self.xy,+self.ylen,0]),
                                            self.coords+np.array([-self.xy,-self.ylen,0]),
                                            self.coords+np.array([+self.xlen+self.xy,+self.ylen,0]),
                                            self.coords+np.array([-self.xlen+self.xy,+self.ylen,0]),
                                            self.coords+np.array([-self.xlen-self.xy,-self.ylen,0]),
                                            self.coords+np.array([+self.xlen-self.xy,-self.ylen,0])) , axis=0)

                print('Start loop on %i atoms' % self.tot_number)
                t0 = time.time()
                pbar = tqdm(total=self.tot_number, unit='neigh list', desc='creating neigh') # Initialise
                tot_neigh = 0
                for i in range(self.tot_number):
                    cond_R = np.linalg.norm(coords_9x-coords_9x[i],axis=1)  < cutoff
                    possible = np.where(cond_R == True)[0]
                    possible = possible % self.tot_number
                    possible = np.unique(possible)
                    possible = np.delete(possible, np.argwhere(possible==i))
                    k = possible.shape[0]
                    try:
                        np_nl[i][:k] = possible
                    except ValueError:
                        if l_width <= k : 
                            raise MemoryError('please increase *l_width* in neigh_list_me_smart')
                        else:
                            raise ValueError('np_nl[i][:k] = possible')

                    tot_neigh += k
                    pbar.update()
                pbar.close()
        
                del coords_9x
                
            elif nl_method == 'RS':   
                ''' Andrea reduce space implementation '''
                # !!! ASSUMES CELL CAN BE OBATINED LIKE THIS !!!
                z_off = 30 # assume z is not well defined...
                A1 = np.array([self.xlen, 0, 0 ])
                A2 = np.array([self.xy, self.ylen, 0])
                A3 = np.array([0,0,z_off])
                u, u_inv = calc_matrices_bvect(A1, A2, A3)
                
                print('Ask KDTree for neighbours d0=%.3f (it may take a while)' % self.cutoff)
                # List containing neighbour pairs
                neighs = np.array(list(pbc_neigh(self.coords, u, self.cutoff))) # convert from set to numpy array
                ## Ali: I don't understand. pbc_neigh returns a numpy array, whu do you convert to list then numpy array? converting from numpy modules to numpy is strongly discouraged. 
                
                # Now we just need to re-order: get all the entries relative to each atom
                print('Start loop on %i atoms' % self.tot_number)
                pbar = tqdm(total=self.tot_number, unit='neigh list', desc='creating neigh') # Initialise
                tot_neigh = 0
                for i in range(self.tot_number):
                    # Where in neigh list (j->k) the current atom i appears?
                    mask_left = np.where(neighs[:,0] == i)[0] # is it index on the left?
                    mask_right = np.where(neighs[:,1] == i)[0]  # is it index on the right?
                    mask = np.concatenate([mask_left, mask_right]) # more or less a logical orlogical OR
                    # All index pairs where this atom is present
                    c_neighs = neighs[mask].flatten()
                    c_neighs = c_neighs[c_neighs != i] # get only the indices differnt from considered atom i
                    k = len(c_neighs) # number of neighbours 
                    # Ali why  len? pls use shape[0]
                    np_nl[i][:k] = c_neighs # save the indices of the k neighbours of atom i
                    tot_neigh += k
                    pbar.update()
                pbar.close()
                
  
            
            ave_neigh= tot_neigh/self.tot_number
                
            ## decrease the array size accordignly
            max_n = np.max(np.count_nonzero(~np.isnan(np_nl),axis=1))
            np_nl = np_nl[:,:max_n]
            
            np.savez(self.folder_name + 'neigh_list_{0}'.format(version_), np_nl=np_nl, tot_neigh=tot_neigh, ave_neigh=ave_neigh )

            self.nl = np_nl

            print('ave_neigh={0} \ntot_neigh={1}'.format(ave_neigh, tot_neigh))




def pbc_neigh(pos, u, d0, sskin=10):

    u_inv = np.linalg.inv(u) # Get matrix back to real space
    S = u_inv.T # Lattice matrix

    print('Search in unit cell')
    pos_tree = KDTree(pos)
    nn_uc = np.array(list(pos_tree.query_pairs(d0))) # Nn in unit cell

    # Go to reduced space: lattice vectors are (1,0), (0,1)
    posp = np.dot(u, (pos).T).T # Fast numpy dot, but weird convention on row/cols
    # Define a skin: no need to sarch in whole unit cell, only close to PBC
    skinp = (1/2-d0/np.sqrt(np.linalg.det(S))*sskin)
    skinmask = np.logical_or(np.abs(posp-0.5)[:,0]>skinp,
                             np.abs(posp-0.5)[:,1]>skinp)
    posp_ind = np.array(range(posp.shape[0]))
    skin_ind = posp_ind[skinmask]
    pospm = posp[skinmask]
    print('N=%i Nskin=%i (skin=%.4g)' % (posp.shape[0], pospm.shape[0], skinp))
    # Wrap cell to center
    tol = 0
    ## Ali: using python objects mixed with numpy heavily slows your code. Slower than not using numpy!
    nn_pbc = []
    for shift in [          # look for repatitions:
            [-0.5, 0, 0],   # along a1
            [0, -0.5, 0],   # along a2
            [-0.5, -0.5,0]  # along both
    ]:
        print('Search in pbc shift %s' % str(shift))
        posp2 = pospm - np.floor(pospm + shift)
        #posp2 = posp - np.floor(posp + shift)
        # Map back to real space
        posp2p = np.dot(u_inv, posp2.T).T
        # Compute distances in real space
        pos2_tree = KDTree(posp2p)
        # Record the indices of full position, not just skin
        nn_pbc += list(skin_ind[list(pos2_tree.query_pairs(d0))])
        #nn_pbc += list(pos2_tree.query_pairs(d0))
    nn_pbc = np.array(nn_pbc)

    # Merge the nearest neighbours, deleting duplicates
    return np.unique(np.concatenate([nn_uc, nn_pbc]), axis=0)

def calc_matrices_bvect(b1, b2, b3):
    """Metric matrices from primitive lattice vectors b1, b2.

    Return matrix to map to unit cell and inverse, back to real space."""
    St = np.array([b1, b2, b3])
    u = np.linalg.inv(St).T
    u_inv = St.T
    return u, u_inv
