import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import warnings



class Symm:
    
    """
        Defines an object usefull to investigate and diagonalize symmetry operation
    """
    
    def __init__(self, tb):
        
        self.conf = tb.conf
        self.tb = tb
        self.new_orders = {}
        self.symmKey = np.array([], dtype=str)
        self.Cmat = {}
        self.new_bases = {}
        
        self.symmorphicity = True
        """
        symmorphicity: boolean
                    If the group is symmorphic. True(default)
                    **Important notice in the case of non-symmorphic groups: at the moment Cmat is only implemented for C2 symmetries, C2x, C2y, and C2z. **
        """
        
        print('Symm object created')
        

    def plotter_coords(self, image1, image2, title=''):
        fig, axs = plt.subplots(1,1,figsize=(7, 10))
        axs = [axs]
        axs[0].plot(image1[:, 0], image1[:, 1], 'o', color='seagreen', markersize = 4, alpha=0.5)
        axs[0].plot(image2[:, 0], image2[:, 1], 'o', color='tomato', markersize = 4, alpha=0.5)
        axs[0].set_title(title)
        axs[0].set_aspect('equal', 'box')
        fig.tight_layout()
        
        return plt
        
        
    def save(self, version=''):
        """
            save symm object
        """
        
        np.savez(self.tb.folder_name + 'Symm_' + version, 
                 new_orders=self.new_orders,
                 symmorphicity=self.symmorphicity,
                 symmKey = self.symmKey,
                 Cmat = self.Cmat,
                 new_bases = self.new_bases)
    
    def load(self, folder_, Symmdata):
        """
            load symm object
        """
        
        folder_ = folder_  if folder_[-1] == '/' else folder_+'/'
        
        print('loading ', folder_+Symmdata)
        data = np.load(folder_ + Symmdata, allow_pickle=True)
        self.symmKey = data['symmKey']
        self.symmorphicity = data['symmorphicity']
        
        self.Cmat = data['Cmat'].item()
        
        self.new_orders = data['new_orders'].item()
        
        try: 
            self.new_bases =  data['new_bases'].item()
        except KeyError:
            pass
 
      
      
    def build_map(self, name, operation, atol=0.2, plot = False):
        """
            Map the indecis transformation under a given symmetry operation.
            Origin always considered at (0,0,0)
                        
            Args:
                name: str
                    A name for this operation
                    e.g.: 'C2x', 'C2y', 'Inv', 'C3z', ..
                    
                operation: A list of three string
                    Defining the the operation in the following order: [operation for X, operation for Y, operation for Z]
                    These arithmetics and namespace are acceptable:
                    +, -, /, *, X, Y, Z, Rx, Ry, and Rz. 
                    Rx, Ry, and Rz are lattice vectors along their directions.
                    X, Y, and Z are coordinates of cites inside unitcell.
                    Examples: 
                    ['X','Y','Z']   is identity
                    ['-X','-Y','Z'] is C2z
                    ['X+1/2*Rx','-Y','-Z'] is C2x with a non-symmorphic translation.
                    
                atol: float
                    absolute tolerance to find the matching id.
                
                plot: boolean
                    show the coordinates in case of success. plot=False (default)
                    in case of failure plot is always True
        """
        self.symmKey = np.append(self.symmKey, name)
        
        txt_ = ''
        if any([('Rx' in op or 'Ry' in op or 'Rz' in op ) for op in operation]):
            self.symmorphicity = False
            txt_ = ' non-symmorphic '
            
        print("Operation "+name+ " is defined as "+txt_+', '.join(operation))
        
        wave_info_trs = np.copy(self.conf.coords)
        dicval = {"X": wave_info_trs[:, 0], "Y": wave_info_trs[:, 1], "Z": wave_info_trs[:, 2],
                  "Rx": self.conf.xlen, "Ry": self.conf.ylen, "Rz": self.conf.zlen} 
        wave_info_trs[:,0] = eval(operation[0], {}, dicval)
        wave_info_trs[:,1] = eval(operation[1], {}, dicval)
        wave_info_trs[:,2] = eval(operation[2], {}, dicval)


        # translate back to cell(0,0) 
        x_back = (wave_info_trs[:, 0]//self.conf.xlen)
        y_back = (wave_info_trs[:, 1]//self.conf.ylen)
        wave_info_trs[:, 0] -=  x_back*self.conf.xlen
        wave_info_trs[:, 1] -=  y_back*self.conf.ylen
        
        
        ## get the right indices after transformation 
        new_order = np.zeros(self.conf.tot_number, dtype='int')
        
        border_items = 0
        for nn in range(self.conf.tot_number):
            
            cond_all = np.isclose( np.linalg.norm(wave_info_trs - self.conf.coords[nn], axis=1) , 0, rtol=0, atol=atol) 
    
            idx = np.where(cond_all)[0]
            
            if idx.shape[0] == 0:
                # pbc thing
                possible = [0,-1,+1]
                #print('I have to do PBC on nn=',nn)
                border_items += 1
                for pX in possible:
                    if idx.shape[0] == 0:
                        for pY in possible:
                            desire_coord = np.copy(self.conf.coords[nn])
                            #print('doing ',pX,pY, desire_coord)
                            desire_coord[0] += pX * self.conf.xlen 
                            desire_coord[1] += pY * self.conf.ylen 
                    
                            cond_all = np.isclose( np.linalg.norm(wave_info_trs - desire_coord, axis=1) , 0, rtol=0, atol=atol) 
                            idx = np.where(cond_all)[0]
                            
                            if idx.shape[0] >0 :
                                #print('yup!, fixed!')
                                break
            
            if idx.shape[0] != 1:
                print('idx=',idx,'   pos',self.conf.coords[nn])
                #plt.show()
                message = "Couldn't match the patterns... Are you sure your lattice has "+name+" symmetry in the real space with origin (0,0,0)? If so try to increase tolerance, structure is not perfect. Check the plot."
                
                p = self.plotter_coords(self.conf.coords, wave_info_trs, 'Red(operated) points should have sit on Green(original) points within atol={0}'.format(atol))
                p.show() 
                
                raise RuntimeError(message)
            
            new_order[nn] = idx[0]
            
        print("Symmetry map for "+name+" has built successfully. ", border_items, " where marginal at boundaries.")
        self.new_orders[name] = new_order
        
        if plot:
            p = self.plotter_coords(self.conf.coords, wave_info_trs, 'Successfully match for '+name +' atol={0}'.format(atol))
            p.show()
            
        
    def make_Cmat(self, name, k_label):
        """
            Makes operation matrix.
            
            Arg:
                name: str
                    name of this matrix operation. must be same as those in build_map like:
                    'C2x'
                
                k_label: str 
                    The high symmetry point to make Cmat for. Must be an element of K_label
                    Str like, e.g. 'Gamma' or 'K1'
                
                symmorphicity: boolean
                    If the group is symmorphic. True(default)
                    **Important notice in the case of non-symmorphic groups: at the moment Cmat is only implemented for C2 symmetries, C2x, C2y, and C2z. **
                
        """
        
        
        xpos_ = self.tb.K_path_Highsymm_indices
            
        try:
            idx = np.where(self.tb.K_label==k_label)[0][0]
            K = self.tb.K_path[int(xpos_[idx])]
        except IndexError:
            message = "High symmetry point " + k_label + " not defined"
            raise ValueError(message)
        
        print('**Making {0} matrix at {1}:{2} **'.format(name, k_label, K))
        
        Cop = sp.lil_matrix((self.conf.tot_number, self.conf.tot_number), dtype='int') #dtype=self.tb.dtypeC)#
        
        for sh in range(self.conf.tot_number):
            i = sh
            j = self.new_orders[name][sh]
            
            if self.symmorphicity:
                Cop[i,j] = 1
            else:
                if 'C2x' in name:
                    convention_x = 1 if (self.conf.coords[ i , 0] //self.conf.xlen_half) %2 == 0 else  np.exp(+1j *  self.conf.xlen * K[0] )
                    Cop[i,j] = convention_x
                if 'C2y' in name:
                    convention_y = 1 if (self.conf.coords[ i , 1] //self.conf.ylen_half) %2 == 0 else  np.exp(+1j *  self.conf.ylen * K[1] )
                    Cop[i,j] = convention_y
                if 'C2z' in name:
                    convention_x = 1 if (self.conf.coords[ i , 0] //self.conf.xlen_half) %2 == 0 else  np.exp(+1j *  self.conf.xlen * K[0] )
                    convention_y = 1 if (self.conf.coords[ i , 1] //self.conf.ylen_half) %2 == 0 else  np.exp(+1j *  self.conf.ylen * K[1] )
                    Cop[i,j] = convention_x*convention_y
        try:            
            self.Cmat[k_label][name] = Cop
        except KeyError:
            self.Cmat[k_label] = {name: Cop}
            
    
    def check_square(self, name, k_label, rtol = 0.1, ftol=0):
        """
            Checks if square of Cmat is identity
                Args:     
                    
                    name: str
                        name of this matrix operation. must be same as those in build_map like: 'C2x'
                    
                    rtol: float between 0-1
                        Relative tolerance. rtol=0.1(default)
                    
                    ftol: integer
                    Fault tolerance. Number of acceptable unmached number. ftol=0(default)
                    Warning: might falsly return always True. If you choose a large ftol comparable with your Hamiltonian size.
        """
        A = self.Cmat[k_label][name]
        A2 = A @ A
        A2.eliminate_zeros()
        nonz = A2.nonzero()
        nonz_tot = nonz[0].shape[0]
        
        
        if self.conf.tot_number < nonz_tot:
            print('{0} @ {0} has more non-zero elements that identity at {1}'.format(name, k_label))
            return 
        elif self.conf.tot_number > nonz_tot:
            print('{0} @ {0} has less non-zero elements that identity at {1}'.format(name, k_label))
            return 
        
        count_p = 0
        count_m = 0
        for zz in range(nonz_tot):
            i = nonz[0][zz]
            j = nonz[1][zz]
            
            if i != j:
                print('{0} @ {0} has non diagonal elements at {1}'.format(name, k_label))
                return 
            
            if  np.isclose( A2[i, j], +1, rtol=rtol, atol=0):
                count_p +=1
            elif np.isclose( A2[i, j],-1, rtol=rtol, atol=0):
                count_m +=1
            else:
                print(A2[i, j])
                print('{0} @ {0} is not +- identity at {1} : {0}[i, j]'.format(name, k_label), A2[i, j])
                return 
        
        if np.isclose(count_p, nonz_tot, rtol=0, atol=ftol):
            print('{0} @ {0} = identity at {1}'.format(name, k_label))
        elif np.isclose(count_m, nonz_tot, rtol=0, atol=ftol):
            print('{0} @ {0} = -identity at {1}'.format(name, k_label))
        elif np.isclose(count_p, count_m, rtol=0, atol=ftol):
            print('{0} @ {0} =  half and half +- identity at {1} : count_p,count_m'.format(name, k_label), count_p, count_m)
        else:
            print("{0} @ {0} is not identity  at {1} : +-1 in a non equal way: count_p,count_m".format(name, k_label), count_p, count_m)
        
    
    def check_commute(self, name1, name2, k_label, rtol=0.1, ftol=0):
        """
            Check if two operations commute
            
            Args:     
                name1 & name2: str
                    name of the operation. must be same as those in build_map like: 'C2x'
                
                rtol: float between 0-1
                    Relative tolerance. rtol=0.1(default)
                
                ftol: integer
                    Fault tolerance. Number of acceptable unmached number. ftol=0(default)
                    Warning: might falsly return always True. If you choose a large ftol comparable with your Hamiltonian size.
        """
        A = self.Cmat[k_label][name1]
        B = self.Cmat[k_label][name2]
        AB = A @ B
        BA = B @ A
        nonzAB = AB.nonzero()
        nonzAB_tot = nonzAB[0].shape[0]
        nonzBA = BA.nonzero()
        nonzBA_tot = nonzBA[0].shape[0]
        if nonzAB_tot != nonzBA_tot:
            print('[{0}, {1}] do not commute at {2}: number of non-zero elements are no equal'.format(name1, name2, k_label))
            return 
        elm0 = np.all(nonzAB[0]==nonzBA[0])
        elm1 = np.all(nonzAB[1]==nonzBA[1])
        if not elm0 or not elm1:
            print("[{0}, {1}] do not commute at {2}: operators don't match ".format(name1, name2, k_label))
            return 
        
        count_p = 0
        count_m = 0
        for zz in range(nonzAB_tot):
            i = nonzAB[0][zz]
            j = nonzAB[1][zz]
            #print('\r',org[i, j], H_primeY[i, j],end='')
            if  np.isclose( AB[i, j], BA[i, j], rtol=rtol, atol=0):
                count_p +=1
            elif np.isclose( AB[i, j], -BA[i, j], rtol=rtol, atol=0):
                count_m +=1
            else:
                print(AB[i, j], -BA[i, j])
                print('[{0}, {1}] do not commute at {2}: non comparable elements'.format(name1, name2, k_label))
                return 1
                #break
        
        if np.isclose(count_p, nonzAB_tot, rtol=0, atol=ftol):
            print('[{0}, {1}] do commute at {2}'.format(name1, name2, k_label))
        elif np.isclose(count_m, nonzAB_tot, rtol=0, atol=ftol):
            print('[{0}, {1}] do anti-commute at {2}'.format(name1, name2, k_label))
        elif np.isclose(count_p, count_m, rtol=0, atol=ftol):
            print('[{0}, {1}] do half commute and half anti-commute at {2}: count_p,count_m'.format(name1, name2, k_label), count_p, count_m)
        else:
            print('[{0}, {1}] do not commute at {2}: but +- of each other in a non equal way: count_p,count_m'.format(name1, name2, k_label), count_p, count_m)
    
    
  
    
    def vector_diag(self, k_label, name1, name2=None, subSize = 2, rtol=0.1, skip_diag = False):
        """
            Arg:
                k_label: str 
                    The high symmetry point to make Cmat for. Must be an element of K_label
                    Str like, e.g. 'Gamma' or 'K1'
                    
                rtol: float between 0-1
                    Relative tolerance. rtol=0.1(default)

                name: str
                    name of this matrix operation. must be same as those in build_map like:
                    'C2x'
                
                subSize: int < n_flat
                    Size of the subspace to diagonalize
        """
        print("\n\n"+"="*22 + "\n** vector_diag at "+k_label+" **\n"+"="*22+"\n")
        xpos_ = self.tb.K_path_Highsymm_indices
            
        try:
            id_ = np.where(self.tb.K_label==k_label)[0][0]
            K = self.tb.K_path[int(xpos_[id_])]
        except IndexError:
            message = "High symmetry point " + k_label +" not defined"
            raise ValueError(message)
        
        flat_range= self.tb.n_flat

        new_bases = np.zeros((flat_range, self.conf.tot_number), dtype=self.tb.dtypeC)
        old_vecs  = np.zeros((flat_range, self.conf.tot_number), dtype=self.tb.dtypeC)
        very_new_bases = np.zeros((flat_range, self.conf.tot_number), dtype=self.tb.dtypeC)
        #eignvals_neu = np.zeros(flat_range, dtype='f' if self.dtype==None else self.dtype)
        
        for ii in range(0, flat_range, subSize):
            S = np.zeros((subSize,subSize), dtype=self.tb.dtypeC)
            
            if not skip_diag:
                print('\nDiagonalizing flat bands subspace '+ str(ii/4+1) +' with energies:')
            else:
                print('\nSubspace '+ str(ii/4+1) +' with energies:')
                
            for jj in range(subSize):
                #old_vecs[ii+jj] =    (wave_info[jj+ii, :, 3] + 1j*wave_info[jj+ii, :, 4])#*phase_1
                old_vecs[ii+jj] =  self.tb.bandsVector[id_, :, jj+ii]
                print(self.tb.bandsEigns[id_, ii+jj] )

            
            for fl_i in range(subSize):
                for fl_f in range(subSize):
                    #S[fl_i, fl_f] = np.dot(np.conjugate( old_vecs[ii+fl_i].T ), old_vecs_op[ii+fl_f] ) 

                    element = (sp.lil_matrix(np.conjugate(old_vecs[ii+fl_i])) @ self.Cmat[k_label][name1]  @  sp.lil_matrix.transpose(sp.lil_matrix(old_vecs[ii+fl_f]), copy=True))
                    assert element.get_shape() == (1,1)

                    S[fl_i, fl_f] =  element[0,0]
            
            
            if not skip_diag:
                
                print('<psi| '+name1+' |psi>')
                print(np.array2string(S, separator=",", precision=1, suppress_small=True))
            
                print('Diagonalizing respect to ',name1)
                w, v = np.linalg.eig(S)
                #print('eignvalues: ',w, '\n')
                print('eignvalues: ', np.array2string(w, separator=",", precision=1, suppress_small=True))
                
                for kk in range(subSize):
                    for qq in range(subSize):
                        new_bases[ii+kk] += old_vecs[ii+qq]*v[qq, kk]
                
                
                if name2 != None:
                    #simultaneous diagonalization
                    #continue
                    sdot = np.zeros((subSize,subSize), dtype=self.tb.dtypeC)
                    print('\n Second off-diagonalizing respect to ', name2)
                    for sei in range(subSize):
                        for sef in range(subSize):
                            element = (sp.lil_matrix(np.conjugate(new_bases[ii+sei])) @ self.Cmat[k_label][name2]  @  sp.lil_matrix.transpose(sp.lil_matrix(new_bases[ii+sef]),copy=True))
                            assert element.get_shape() == (1,1)
                            sdot[sei, sef] = element[0,0]
                    #####
                    #print('redigonalize\n', sdot)
                    #w, v = np.linalg.eig(sdot)
                    #print('eignvalues: ',w)
                    #for kk in range(subSize):
                        #for qq in range(subSize):
                            #very_new_bases[ii+kk] += new_bases[ii+qq]*v[qq, kk]
                    
                    #####
                    upper_block = sdot[:2, :2]
                    w, v = np.linalg.eig(upper_block)
                    
                    print('upper_block is\n', np.array2string(upper_block, separator=",", precision=1, suppress_small=True))
                    print('eignvalues: ', np.array2string(w, separator=",", precision=1, suppress_small=True))
                    for kk in range(2):
                        for qq in range(2):
                            very_new_bases[ii+kk] += new_bases[ii+qq]*v[qq, kk]
                    ###
                    lower_block = sdot[2:, 2:]
                    w, v = np.linalg.eig(lower_block)
                    
                    print('lower_block is\n', np.array2string(lower_block, separator=",", precision=1, suppress_small=True))
                    print('eignvalues: ', np.array2string(w, separator=",", precision=1, suppress_small=True))
                    for kk in range(2):
                        for qq in range(2):
                            very_new_bases[ii+kk+2] += new_bases[ii+qq+2]*v[qq, kk]       
                    
                    
                    new_bases[ii:ii+subSize] = very_new_bases[ii:ii+subSize]
                
            else:
                for kk in range(subSize):
                    new_bases[ii+kk] = old_vecs[ii+kk]
                        
            ## Check if diagonal respect to all syymetries defined
            for se_al in self.symmKey:
                sdot = np.zeros((subSize,subSize), dtype=self.tb.dtypeC)
                
                if not skip_diag:
                    print('\nFinal check if diagonalized respect to ', se_al)
                else:
                    print('<psi| '+se_al+' |psi>')
                
                for sei in range(subSize):
                    for sef in range(subSize):
                            
                        element = (sp.lil_matrix(np.conjugate(new_bases[ii+sei])) @ self.Cmat[k_label][se_al]  @  sp.lil_matrix.transpose(sp.lil_matrix(new_bases[ii+sef]),copy=True))
                        assert element.get_shape() == (1,1)
                        sdot[sei, sef] = element[0,0]
                        
                print(np.array2string(sdot, separator=",", precision=1, suppress_small=True))
                   

        self.new_bases[k_label] = new_bases
        
    
