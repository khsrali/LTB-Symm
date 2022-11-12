import sys, logging, time
import numpy as np
import scipy.sparse as sp

'''
I am sure this code has no bug!
'''

class pwl:

    def __init__(self, folder_name, file_name, sparse_flag, debug=False):
        name = 'pwl'
        self.debug = debug
        # -------- SET UP LOGGER -------------
        self.log_out = logging.getLogger(name) # Set name identifying the logger.
        # Adopted format: level - current function name - message. Width is fixed as visual aid.
        logging.basicConfig(format='[%(levelname)7s - %(name)5s: %(funcName)20s] %(message)s')
        self.log_out.setLevel(logging.INFO)
        if debug: self.log_out.setLevel(logging.DEBUG)
        self.log_out.debug('Created pwl object')

        self.folder_name = folder_name
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
        self.log_out.info('creating vector_connection_matrix...')

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
                    # AS: don't know exactly what types these are, so I brute force string convert
                    self.log_out.error('POS_ii, POS_jj\n %s \n %s' % (str(self.coords[ii]), str(self.coords[jj])))
                    self.log_out.error('New dist = %s' % str(dist_size))
                    self.log_out.error('Old dist = %s' % str(old_dist_size))
                    self.log_out.error('shits: %i %i %i' % (shit_0,shit_1,shit_2,shit_3))
                    err_msg = 'something is wrong with PBC'
                    raise RuntimeError(err_msg)

                if self.sparse_flag:
                    self.dist_matrix[0][ii, jj] = dist_[0]
                    self.dist_matrix[1][ii, jj] = dist_[1]
                    self.dist_matrix[2][ii, jj] = dist_[2]
                    #self.dist_matrix_norm[ii, jj] = dist_size

                else:
                    self.dist_matrix[ii, jj] = dist_


            if zz!=3:
                raise RuntimeError("there is an error in finding first nearest neighbors:  please tune *fnn_cutoff*")

        #if not self.sparse_flag:
            #self.dist_matrix_norm = np.linalg.norm(self.dist_matrix, axis=2)


    def normal_vec(self, local_normal_flag=True):
        '''
        change ez_local to ez latter..
        '''
        if local_normal_flag:
            self.log_out.info("using **local** ez ...")
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
        else:
            self.log_out.info("using **global** ez=[0,0,1] ...")
            self.ez_local = np.array([0,0,1])

    def neigh_list_me_smart(self, cutoff, l_width = 100, load_=True, version_=None):

        ''' returns **full** neigh_list '''
        self.cutoff = cutoff
        self.l_width = l_width

        if load_ == True:
            try:
                #print('trying to load=','neigh_list_{0}.npz'.format(version_))
                data_ = np.load(self.folder_name + 'neigh_list_{0}.npz'.format(version_))
                np_nl     = data_['np_nl']
                tot_neigh = data_['tot_neigh']
                ave_neigh = data_['ave_neigh']
                self.log_out.warning('neigh_list is loaded from the existing file: neigh_list_{0}.npz'.format(version_))
                self.log_out.warning("if want to rebuild the neigh_list, please use: calculate(..., load_neigh=False)")
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

            self.log_out.info('Start loop on %i atoms' % self.tot_number)
            t0 = time.time()
            tot_neigh = 0
            for i in range(self.tot_number):
                # AS: printing at each atom position makes it very slow!
                #self.log_out.info(' creating neigh_list {} of {}'.format(i+1, self.tot_number))
                self.progress_bar(' creating neigh_list {} of {}'.format(i+1, self.tot_number),
                                  frac=i/self.tot_number, out=sys.stderr
                                  )
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
            texec = time.time() - t0
            print('\nLoop on atoms finished in %is (%.2fmin)' % (texec, texec/60), file=sys.stderr) # leave the completed progress bar on screen
            del coords_9x
            ave_neigh= tot_neigh/self.tot_number
            np.savez(self.folder_name + 'neigh_list_{0}'.format(version_), np_nl=np_nl, tot_neigh=tot_neigh, ave_neigh=ave_neigh )


        self.nl = np_nl
        self.ave = ave_neigh

        self.log_out.info('ave_neigh=%f' % self.ave)
        self.log_out.info('tot_neigh=%f' % tot_neigh)

        return 'full'

    def progress_bar(self, message, frac=None, width=50, out=sys.stderr):
        if frac == None: print('\r', message, file=out, end='')
        else: print('\r', '['+'='*int(frac*width)+' '*int((1-frac)*width)+']', message, file=out, end='')


    def neigh_list_AS(self, cutoff, l_width = 100, load_=True, version_=None):

        ''' returns **full** neigh_list '''
        self.cutoff = cutoff
        self.l_width = l_width

        if False:

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

            self.log_out.info('Start loop on %i atoms' % self.tot_number)
            t0 = time.time()
            tot_neigh = 0
            for i in range(self.tot_number):
                # AS: printing at each atom position makes it very slow!
                #self.log_out.info(' creating neigh_list {} of {}'.format(i+1, self.tot_number))
                self.progress_bar(' creating neigh_list {} of {}'.format(i+1, self.tot_number),
                                  frac=i/self.tot_number, out=sys.stderr
                                  )
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
            texec = time.time() - t0
            print('\nLoop on atoms finished in %is (%.2fmin)' % (texec, texec/60), file=sys.stderr) # leave the completed progress bar on screen
            del coords_9x
            ave_neigh= tot_neigh/self.tot_number
            np.savez(self.folder_name + 'neigh_list_{0}'.format(version_), np_nl=np_nl, tot_neigh=tot_neigh, ave_neigh=ave_neigh )


        self.nl = np_nl
        self.ave = ave_neigh

        self.log_out.info('ave_neigh=%f' % self.ave)
        self.log_out.info('tot_neigh=%f' % tot_neigh)

        return 'full'
