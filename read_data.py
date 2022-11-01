import numpy as np
import sys
import scipy.sparse as sp

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
    

