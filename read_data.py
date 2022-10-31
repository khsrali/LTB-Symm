import numpy as np
import sys
import scipy.sparse as sp

try: 
    from lammps import lammps
except ModuleNotFoundError:
    pass

'''
I am sure this code has no bug!
'''

class play_with_lammps:

    def __init__(self, file_name, sparse_flag):
        file_ = open(file_name)           
        l_ = file_.readlines()
        self.sparse_flag = sparse_flag
        self.xy = 0
        
        ii = 0
        while ii<20:
            if 'atoms' in l_[ii]:
                self.tot_number = int(l_[ii].split()[0])
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
            
            ii += 1
                
        self.ylen_half = self.ylen/2
        self.xlen_half = self.xlen/2
        
        file_.close()

        ### get coordinates and bonds info
        self.coords = np.loadtxt(file_name, skiprows =skiplines, max_rows= self.tot_number)[:,4:7]
    
    
    def vector_connection_matrix(self, fnn_cutoff=1.55):        
        print('creating vector_connection_matrix...')
        
        if self.sparse_flag:           
            dist_matrix_X = sp.lil_matrix((self.tot_number, self.tot_number), dtype='float')
            dist_matrix_Y = sp.lil_matrix((self.tot_number, self.tot_number), dtype='float')
            dist_matrix_Z = sp.lil_matrix((self.tot_number, self.tot_number), dtype='float')
            #self.dist_matrix_norm = sp.lil_matrix((self.tot_number, self.tot_number), dtype='float')
            self.dist_matrix = np.array([dist_matrix_X, dist_matrix_Y, dist_matrix_Z]) # being 3,N,N # no otherway
        else:
            self.dist_matrix = np.zeros((self.tot_number, self.tot_number, 3), dtype='float')
        
        self.fnn_vec = np.full((self.tot_number,3,3), np.nan)
        
        for ii in range(self.tot_number):
            neighs = self.nl[ii][~(np.isnan(self.nl[ii]))].astype('int') # number of neighbors are not known, so you put this super 
            zz = 0
            for jj in neighs:
                
                dist_ = self.coords[jj] - self.coords[ii]
        
                ## for debugging
                shit_0 = shit_1 = shit_2 = shit_3 =0
                old_dist_size = np.linalg.norm(dist_)
                ## 
                
                if dist_[1] > self.ylen_half:
                    dist_[1] -= self.ylen
                    dist_[0] -= self.xy
                    shit_0 =1
                elif -1*dist_[1] > self.ylen_half:
                    dist_[1] += self.ylen
                    dist_[0] += self.xy
                    shit_1 =1
                        
                if dist_[0] > self.xlen_half:
                    dist_[0] -= self.xlen
                    shit_2 =1
                elif -1*dist_[0] > self.xlen_half:
                    dist_[0] += self.xlen
                    shit_3 =1
                
                
                dist_size = np.linalg.norm(dist_)
                if dist_size < fnn_cutoff:
                    self.fnn_vec[ii, zz] = dist_
                    zz += 1
                
                ## for debugging
                if dist_size > 1.01*self.cutoff:
                    print('something is wrong with PBC')
                    print('POS_ii, POS_jj\n', self.coords[ii],'\n', self.coords[jj])
                    print('New dist =',dist_size)
                    print('Old dist=',old_dist_size)
                    print(shit_0,shit_1,shit_2,shit_3)
                    exit(1)
                
                if self.sparse_flag:
                    self.dist_matrix[0][ii, jj] = dist_[0]
                    self.dist_matrix[1][ii, jj] = dist_[1]
                    self.dist_matrix[2][ii, jj] = dist_[2]
                    #self.dist_matrix_norm[ii, jj] = dist_size
                    
                else:
                    self.dist_matrix[ii, jj] = dist_
                
                
            if zz!=3:
                print("there is an error in finding first nearest neighbors\n  please tune *fnn_cutoff*")
                exit(1)
        
        #if not self.sparse_flag: 
            #self.dist_matrix_norm = np.linalg.norm(self.dist_matrix, axis=2)
    
    
    def normal_vec(self):
        
        self.ez_local = np.full((self.tot_number,3), np.nan)
        
        for ii in range(self.tot_number):
            neighs = self.fnn_vec[ii]
            aa = np.cross(neighs[0], neighs[1])
            bb = np.cross(neighs[1], neighs[2])
            cc = np.cross(neighs[2], neighs[0])
            
            norm_ = aa + bb + cc 
            norm_ = norm_/np.linalg.norm(norm_)
            
            # note!! this code does not know about chirality, so better to always return the positive normal_vec
            if norm_[2] < 0:
                norm_ *= -1
            
            self.ez_local[ii] = norm_
        
    
    
    def neigh_list_me_smart(self, cutoff, l_width = 100, load_=True, version_=None):
        
        ''' returns **full** neigh_list '''
        self.cutoff = cutoff
        self.l_width = l_width
        
        if load_ == True:
            try:
                data_ = np.load('neigh_list_{0}.npz'.format(version_))
                np_nl     = data_['np_nl']
                tot_neigh = data_['tot_neigh']
                ave_neigh = data_['ave_neigh']
                print('\nWarning! \nneigh_list is loaded from the existing file: \n', 'neigh_list_{0}.npz'.format(version_))
                print("if you prefer to rebuild the neigh_list, please use: ", 'neigh_list_me_smart(..., load_=False)\n')
            except FileNotFoundError:   load_ = False
        
        if load_ == False:
            
            np_nl = np.full((self.tot_number, l_width), np.nan) 
            dd = self.coords+np.array([+self.xlen,0,0])
            coords_9x = np.concatenate((self.coords,
                                    self.coords+np.array([+self.xlen,0,0]),
                                    self.coords+[-self.xlen,0,0],
                                    self.coords+[+self.xy,+self.ylen,0],
                                    self.coords+[-self.xy,-self.ylen,0],
                                    self.coords+[+self.xlen+self.xy,+self.ylen,0],
                                    self.coords+[-self.xlen+self.xy,+self.ylen,0],
                                    self.coords+[-self.xlen-self.xy,-self.ylen,0],
                                    self.coords+[+self.xlen-self.xy,-self.ylen,0]) , axis=0)
            
            tot_neigh = 0
            for i in range(self.tot_number):
                print(' creating neigh_list {} of {}'.format(i+1, self.tot_number), end = "\r")
                #print(xcoords9)
                cond_R = np.linalg.norm(coords_9x-coords_9x[i],axis=1)  < cutoff
                possible = np.where(cond_R == True)[0]
                possible = possible % self.tot_number
                possible = np.unique(possible)
                possible = np.delete(possible, np.argwhere(possible==i))
                k = possible.shape[0]
                np_nl[i][:k] = possible
                # something like this will improve the code if needed..
                #cond1x = coords_9x[:, 0] < (coords_9x[j][0] + cutoff)
                #cond2x = coords_9x[:, 0] < (coords_9x[j][0] + cutoff)
                #cond1y = coords_9x[:, 1] > (coords_9x[j][1] - cutoff)
                #cond2y = coords_9x[:, 1] < (coords_9x[j][1] + cutoff)
                #cc = np.all([cond1x, cond2x, cond1y, cond2y], axis=0)
                #dd = np.where(cc == True)[0]
                tot_neigh += k
            
            del coords_9x
            ave_neigh= tot_neigh/self.tot_number
            np.savez('neigh_list_{0}'.format(version_), np_nl=np_nl, tot_neigh=tot_neigh, ave_neigh=ave_neigh ) 

            
        self.nl = np_nl
        self.ave = ave_neigh
        
        print('\nave_neigh=',self.ave)
        print('tot_neigh=',tot_neigh)
        
        return 'full'
    
    
    
    def neigh_list_lammps(self, file_name, cutoff, l_width = 100):
        
        ''' returns **half** neigh_list 
        # remember lammps tags runs from [1 to N) '''
        lmp = lammps()
        lmp.commands_string("""
        units         metal
        atom_style    full
        boundary      p  p  s
        box tilt large
        read_data     {0}
        mass 1 1.0
        #mass 2 1.0
        neighbor 0.0 bin
        #pair_style       airebo 1.4
        #pair_coeff * *   CH.airebo  C C
        pair_style lj/cut {1}
        pair_coeff * * 0.0 0.0
        #pair_style hybrid lj/cut {1} airebo 0.1 airebo 0.1
        #pair_coeff * * airebo 1 CH.airebo C NULL
        #pair_coeff * * airebo 2 CH.airebo NULL C
        #pair_coeff 1 2 lj/cut 1.0 1.0
        run 0 post no""".format(file_name, cutoff))
        
        # look up the neighbor list
        pair_lists = lmp.find_pair_neighlist('lj/cut') #('lj/cut')
        lmp_nl = lmp.numpy.get_neighlist(pair_lists)
        tags = lmp.extract_atom('id')
        print("half neighbor list with {} entries".format(lmp_nl.size))

        np_nl = np.full((lmp_nl.size, l_width), np.nan) #c[~(np.isnan(c))]

        tot_neigh = 0
        for i in range(0,lmp_nl.size):
            idx, nlist  = lmp_nl.get(i)
            #print("\natom {} with ID {} has {} neighbors:".format(idx,tags[idx],nlist.size))
            tot_neigh += nlist.size
            for j in range(nlist.size):
                np_nl[ tags[idx]-1 ][j] = tags[nlist[j]]-1
                #print("  atom {} with ID {}".format(nlist[j], tags[nlist[j]]))
        
        ave_neigh= tot_neigh*2/lmp_nl.size
        
        self.nl = np_nl
        self.ave = ave_neigh

        print('ave_neigh=',self.ave)
        print('tot_neigh=',tot_neigh)
        
        return 'half'
## for debugging, you may delete later.
#read_ = play_with_lammps(sys.argv[1])
#print(read_.fnn)
#print(read_.coords.shape)
#print(read_.bonds.shape)
#read_.neigh_list(sys.argv[1], cutoff=4*1.39)
#print(read_.nl)
#print('ave=', read_.ave)






#delete later
#bond_flag = False
#if bond_flag == True:
    #self.tot_bonds  = int(l_[2].split()[0])
    #self.bonds  = np.loadtxt(file_name, skiprows =11+self.tot_number+3, max_rows= self.tot_bonds, dtype=int) 

    #fnn = np.full((self.tot_number,3), -1)# first nearest neighbor 

    #for bond in self.bonds:
        #bn_i = bond[2]
        #bn_f = bond[3]
        #if  fnn[bn_i-1][0] == -1:
            #fnn[bn_i-1][0] = bn_f-1
        #elif  fnn[bn_i-1][1] == -1:
            #fnn[bn_i-1][1] = bn_f-1
        #elif  fnn[bn_i-1][2] == -1:
            #fnn[bn_i-1][2] = bn_f-1
        #else:
            #print('Problem on Bonds..')
            #exit()
            
        #if  fnn[bn_f-1][0] == -1:
            #fnn[bn_f-1][0] = bn_i-1
        #elif  fnn[bn_f-1][1] == -1:
            #fnn[bn_f-1][1] = bn_i-1
        #elif  fnn[bn_f-1][2] == -1:
            #fnn[bn_f-1][2] = bn_i-1
        #else:
            #print('Problem on Bonds..')
            #exit()
    #self.fnn = fnn
    
    

    #def __init__(self, file_name):
        #file_ = open(file_name)           
        #l_ = file_.readlines()

        #self.tot_number = int(l_[1].split()[0])
        
        #### get box
        #self.xlo = float(l_[5].split()[0]) 
        #self.xhi = float(l_[5].split()[1])
        #self.xlen = self.xhi  - self.xlo # positive by defination

        #self.ylo = float(l_[6].split()[0]) 
        #self.yhi = float(l_[6].split()[1])
        #self.ylen = self.yhi  - self.ylo
        
        #self.zlo = float(l_[7].split()[0]) 
        #self.zhi = float(l_[7].split()[1])
        #self.zlen = self.zhi  - self.zlo
        
        #self.ylen_half = self.ylen/2
        #self.xlen_half = self.xlen/2
        
        #self.xy = 0 if 'xy xz yz' not in l_[8] else float(l_[8].split()[0])
                    
        #file_.close()

        #### get coordinates and bonds info
        #self.coords = np.loadtxt(file_name, skiprows =11, max_rows= self.tot_number)[:,4:7]
        
        #bond_flag = False
        #if bond_flag == True:
            #self.tot_bonds  = int(l_[2].split()[0])
            #self.bonds  = np.loadtxt(file_name, skiprows =11+self.tot_number+3, max_rows= self.tot_bonds, dtype=int) 

            #fnn = np.full((self.tot_number,3), -1)# first nearest neighbor 

            #for bond in self.bonds:
                #bn_i = bond[2]
                #bn_f = bond[3]
                #if  fnn[bn_i-1][0] == -1:
                    #fnn[bn_i-1][0] = bn_f-1
                #elif  fnn[bn_i-1][1] == -1:
                    #fnn[bn_i-1][1] = bn_f-1
                #elif  fnn[bn_i-1][2] == -1:
                    #fnn[bn_i-1][2] = bn_f-1
                #else:
                    #print('Problem on Bonds..')
                    #exit()
                    
                #if  fnn[bn_f-1][0] == -1:
                    #fnn[bn_f-1][0] = bn_i-1
                #elif  fnn[bn_f-1][1] == -1:
                    #fnn[bn_f-1][1] = bn_i-1
                #elif  fnn[bn_f-1][2] == -1:
                    #fnn[bn_f-1][2] = bn_i-1
                #else:
                    #print('Problem on Bonds..')
                    #exit()
            #self.fnn = fnn
    
    
