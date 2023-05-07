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
#from TightBinding import TB



class Symm:
    def __init__(self, tb):
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.conf = tb.conf
        self.tb = tb
        self.new_orders = {}
        self.symmKey = []
        self.Cmat = {}
        print('Symm object created')
        
      
    def build_map(self, name, operation, atol=0.2):
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
        """
        self.symmKey.append(name)
        
        print("Operation "+name+ " is defined as "+', '.join(operation))
        wave_info_trs = np.copy(self.conf.coords)
        dicval = {"X": wave_info_trs[:, 0], "Y": wave_info_trs[:, 1], "Z": wave_info_trs[:, 2],
                  "Rx": self.conf.xlen, "Ry": self.conf.ylen, "Rz": self.conf.zlen} 
        wave_info_trs[:,0] = eval(operation[0], {}, dicval)
        wave_info_trs[:,1] = eval(operation[1], {}, dicval)
        wave_info_trs[:,2] = eval(operation[2], {}, dicval)
                                  

        ## translate back to cell(0,0) 
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
                print('idx=',idx,'    pos',self.conf.coords[nn])
                #plt.show()
                raise RuntimeError("Couldn't match the patterns... Are you sure your lattice has "+name+" symmetry in the real space with origin (0,0,0)? If so try to increase tolerance, structure is not perfect.")
            new_order[nn] = idx[0]
            
        print("Symmetry map for "+name+" has built successfully. ", border_items, " where marginal at boundaries.")
        self.new_orders[name] = new_order
        #np.savez(self.conf.folder_name + 'Operations_' +self.conf.save_name, new_orders=self.new_orders)
        
        
    def make_Cmat(self, name, k_label, symmorphicity = True):
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
                    **Notice in the case of non-symmorphic groups: 
                    At the moment Cmat is only implemented for C2 symmetries, C2x, C2y, and C2z. **
                
        """
        print('**Making {0} matrix for point {1} **'.format(name, k_label))
        
        
        xpos_ = self.tb.K_path_Highsymm_indices
            
        try:
            idx = np.where(self.tb.K_label==k_label)[0][0]
            K = self.tb.K_path[int(xpos_[idx])]
        except IndexError:
            message = "High symmetry point " + k_label " not defined"
            raise ValueError(message)
        
        
        Cop = sp.lil_matrix((self.conf.tot_number, self.conf.tot_number), dtype='int')
        
        for sh in range(self.conf.tot_number):
            i = sh
            j = self.new_orders[name][sh]
            
            if symmorphicity:
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
                    
        self.Cmat[name] = Cop
        
    
    def check_square(self, name, rtol = 0.1):
        """
            Checks if square of Cmat is identity
                Args:     
                    
                    name: str
                        name of this matrix operation. must be same as those in build_map like:
                        'C2x'
                    
                    rtol: float between 0-1
                        Relative tolerance. rtol=0.1(default)
        """
        A = self.Cmat[name]
        A2 = A @ A
        A2.eliminate_zeros()
        nonz = A2.nonzero()
        nonz_tot = nonz[0].shape[0]
        
        
        if self.conf.tot_number < nonz_tot:
            print('{0} square has more non-zero elements that identity'.format(name))
            return 
        elif self.conf.tot_number > nonz_tot:
            print('{0} square has less non-zero elements that identity'.format(name))
            return 
        
        count_p = 0
        count_m = 0
        for zz in range(nonz_tot):
            i = nonz[0][zz]
            j = nonz[1][zz]
            
            if i != j:
                print('{0} square has non diagonal elements'.format(name))
                return 
            
            if  np.isclose( A2[i, j], +1, rtol=rtol, atol=0):
                count_p +=1
            elif np.isclose( A2[i, j],-1, rtol=rtol, atol=0):
                count_m +=1
            else:
                print(A2[i, j])
                print('{0} square is not +- identity : {0}[i, j]'.format(name), A2[i, j])
                return 
        
        if count_p == nonz_tot:
            print('{0} square = identity'.format(name))
        elif count_m == nonz_tot:
            print('{0} square = -identity'.format(name))
        elif count_p == count_m:
            print('{0} square =  half and half +- identity : count_p,count_m'.format(name), count_p, count_m)
        else:
            print("{0} square is not identity : +-1 in a non equal way: count_p,count_m".format(name), count_p, count_m)
        
    
    def check_commute(self, name1, name2, rtol, ftol):
        """
            Check if two operations commute
            
            Args:     
                name1 & name2: str
                    name of the operation. must be same as those in build_map like: 'C2x'
                
                rtol: float between 0-1
                    Relative tolerance. rtol=0.1(default)
        """
        A = self.Cmat[name1]
        B = self.Cmat[name2]
        AB = A @ B
        BA = B @ A
        nonzAB = AB.nonzero()
        nonzAB_tot = nonzAB[0].shape[0]
        nonzBA = BA.nonzero()
        nonzBA_tot = nonzBA[0].shape[0]
        if nonzAB_tot != nonzBA_tot:
            print('{0} & {1} do not commute : number of non-zero elements are no equal'.format(name1, name2))
            return 
        elm0 = np.all(nonzAB[0]==nonzBA[0])
        elm1 = np.all(nonzAB[1]==nonzBA[1])
        if not elm0 or not elm1:
            print("{0} & {1} do not commute : operators don't match ".format(name1, name2))
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
                print('{0} & {1} do not commute : non comparable elements'.format(name1, name2))
                return 1
                #break
        
        if count_p == nonzAB_tot:
            print('{0} & {1} commute'.format(name1, name2))
        elif count_m == nonzAB_tot:
            print('{0} & {1} anti-commute'.format(name1, name2))
        elif count_p == count_m:
            print('{0} & {1} half commute and half anti-commute : count_p,count_m'.format(name1, name2), count_p, count_m)
        else:
            print("{0} & {1} do not commute : but +- of each other in a non equal way: count_p,count_m".format(name1, name2), count_p, count_m)
    
    
    def check_Hsymm(self, H, name, rtol=0.1, vtol=0):
        """
            Check if Hamiltonian matrix (at point K) is invariance under symmetry operation
            
            Args:
                H: sparse lil_matrix
                    Hamiltonian matrix.
                
                name: str
                    Name of the operation. must be same as those in build_map like: 'C2x'
                    
                rtol: float between 0-1
                    Relative tolerance. rtol=0.1(default)
                        
                vtol: integer
                    Fault tolerance. Number of acceptable unmached number. vtol=0(default)
                    Warning: check_Hsymm might falsly return always True. If you choose a large vtol comparable with your Hamiltonian size.
        """
        
        org = sp.lil_matrix(H, dtype=H.dtype, copy=True)
        
        H_prime = sp.lil_matrix(sp.linalg.inv(self.Cmat[name]) @ org @ self.Cmat[name])
        
        nonZ = org.nonzero()
        nonZ_tot = nonZ[0].shape[0]
        vtol = 10000
        
        
        print('checking H invariance under ', name)
        pbar = tqdm(total=nonZ_tot, unit=' check', desc=name) # Initialise
        case = True
        v_buffer = 0
        for zz in range(nonZ_tot):
            i = nonZ[0][zz]
            j = nonZ[1][zz]
            if not np.isclose( org[i, j], H_primeX[i, j], rtol=rtol, atol=0):
                v_buffer += 1
                if v_buffer > vtol:
                    print(org[i, j], H_primeX[i, j])
                    case = False
                    break
            pbar.update()
        pbar.close()
        
        if case: 
            print('H is invariance under '+name+'. Buffer: ', v_buffer)
        else:
            print('Buffer exceeded: ', v_buffer, ' H is **not** invariance under '+name)
            
     
     
    
    def vector_diag(self, k_label, name1, name2=None, subSize = 2, block=False, rtol=0.1, skip_diag = False):
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
        
        xpos_ = self.tb.K_path_Highsymm_indices
            
        try:
            id_ = np.where(self.tb.K_label==k_label)[0][0]
            K = self.tb.K_path[int(xpos_[id_])]
        except IndexError:
            message = "High symmetry point " + k_label " not defined"
            raise ValueError(message)
        
        flat_range= self.n_flat
        wave_info = np.zeros((flat_range, self.conf.tot_number, 7), self.tb.dtypeR)

        for ii in range(flat_range):
            wave_info[ii, :, 0:3] = np.copy(self.conf.coords)
            wave_info[ii, :, 3] = np.real(self.tb.bandsVector[id_, :, ii])
            wave_info[ii, :, 4] = np.imag(self.tb.bandsVector[id_, :, ii])
            wave_info[ii, :, 5] = np.absolute(self.tb.bandsVector[id_, :, ii])
            wave_info[ii, :, 6] = np.angle(self.tb.bandsVector[id_, :, ii], deg=True)
        
        
        new_bases = np.zeros((flat_range, self.conf.tot_number), dtype=self.tb.dtypeC)
        old_vecs  = np.zeros((flat_range, self.conf.tot_number), dtype=self.tb.dtypeC)
        very_new_bases = np.zeros((flat_range, self.conf.tot_number), dtype=self.tb.dtypeC)
        #eignvals_neu = np.zeros(flat_range, dtype='f' if self.dtype==None else self.dtype)
        
        for ii in range(0, flat_range, subSize):
            S = np.zeros((subSize,subSize), dtype=self.tb.dtypeC)
            
            print('\n**I am mixing these energies: ')
            for jj in range(subSize):
                old_vecs[ii+jj] =    (wave_info[jj+ii, :, 3] + 1j*wave_info[jj+ii, :, 4])#*phase_1
                print(self.bandsEigns[id_, ii+jj] )
                #old_vecs_op[ii+jj] =   sp.lil_matrix(self.Cop_x @  sp.lil_matrix(old_vecs[ii+jj]) )
                #old_vecs_op[ii+jj] = (wave_info[jj+ii, self.new_orders[who], 3] + 1j*wave_info[jj+ii, self.new_orders[who], 4])*phase_2
                
            
            for fl_i in range(subSize):
                for fl_f in range(subSize):
                    #S[fl_i, fl_f] = np.dot(np.conjugate( old_vecs[ii+fl_i].T ), old_vecs_op[ii+fl_f] ) 

                    element = (sp.lil_matrix(np.conjugate(old_vecs[ii+fl_i])) @ self.Cmats[name1]  @  sp.lil_matrix.transpose(sp.lil_matrix(old_vecs[ii+fl_f]), copy=True))
                    assert element.get_shape() == (1,1)

                    S[fl_i, fl_f] =  element[0,0]
            
            #np.set_printoptions(suppress=True)
            print('<psi| '+name1+' |psi>')
            print(np.array2string(S, separator=",", precision=4))
            
            
            if not skip_diag:
                print('Diagonalizing respect to ',name1)
                w, v = np.linalg.eig(S)
                print('eignvalues: ',w, '\n')
                
                for kk in range(subSize):
                    for qq in range(subSize):
                        new_bases[ii+kk] += old_vecs[ii+qq]*v[qq, kk]
            else:
                for kk in range(subSize):
                    new_bases[ii+kk] = old_vecs[ii+kk]
                
                
                
            if name2 != None:
                #simultaneous diagonalization
                #continue
                sdot = np.zeros((subSize,subSize), dtype=self.tb.dtypeC)
                print('\n Second off-diagonalizing respect to ', name2)
                for sei in range(subSize):
                    for sef in range(subSize):
                        element = (sp.lil_matrix(np.conjugate(new_bases[ii+sei])) @ Cmats[name2]  @  sp.lil_matrix.transpose(sp.lil_matrix(new_bases[ii+sef]),copy=True))
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
                
                
                new_bases[ii:ii+subSize] = very_new_bases[ii:ii+subSize]
                        
            ## Check if diagonal respect to all syymetries defined
            for se_al in self.symmKey:
                sdot = np.zeros((subSize,subSize), dtype=self.tb.dtypeC)
                print('\nFinal check if diagonalized respect to ', se_al)
                for sei in range(subSize):
                    for sef in range(subSize):
                        #sdot[sei, sef]  = np.dot(np.conjugate(new_bases[ii+sei, :].T),  new_bases[ii+sef, self.new_orders[whos[se_al]]]  )
                            
                        element = (sp.lil_matrix(np.conjugate(new_bases[ii+sei])) @ Cmats[se_al]  @  sp.lil_matrix.transpose(sp.lil_matrix(new_bases[ii+sef]),copy=True))
                        assert element.get_shape() == (1,1)
                        sdot[sei, sef] = element[0,0]
                        
                print(np.array2string(sdot, separator=",", precision=4))
                    

        
    #def vector_parity(self,):
        #print('\n')
        ##print("diagonalized in ",which_operation)
        #for wihh in range(3):
            ##only_itself = True if wihh ==2 else False
                
            #print('checking ', which_operations[wihh])
            #fig, axs = plt.subplots(4,2,figsize=(10, 10))
            #plt.title(which_operations[wihh])
            #axs = axs.flatten()
            #only_amp_str = ''
            #for kk in range(flat_range):
                #transed = ((Cmats[wihh]  @  sp.lil_matrix.transpose(sp.lil_matrix(new_bases[kk]), copy=True)).todense())#.flatten()[0,:]
                #transed = np.ravel(transed)
                #assert new_bases[kk].shape == transed.shape
                ##print(type(transed),transed.shape)
                ##print(new_bases[kk].shape)
                ##print('************')
                #for qq in range(kk, flat_range):
                    ##check_ = np.isclose(np.absolute(new_bases[qq]), np.absolute(new_bases[kk][self.new_orders[wihh]] ), rtol=0.2, atol=0.0)
                    #check_ = np.isclose(np.absolute(new_bases[qq]), np.absolute(transed), rtol=0.2,  atol=0.0)
                    #flag_  = np.isclose( np.count_nonzero(check_), self.conf.tot_number, rtol=tol_, atol=0) 
                    ##print('{0} dot {1} is '.format(ii+qq,ii+kk), np.dot(np.conjugate(new_bases[qq].T), new_bases[kk][new_orders[wihh]]))
                    ##print(np.count_nonzero(check_))
                    #condintion = (kk==qq) if only_itself else True
                    #if flag_ and condintion:
                        #print('{0} to {1} **symmetry Holds!** '.format(qq,kk), np.count_nonzero(check_), ' of ', self.conf.tot_number)
                        ##print('instance: ',np.angle(new_bases[qq][10], deg=True), np.angle(new_bases[kk][new_orders[wihh]][10], deg=True))
                        ##print('instance: ',np.angle(new_bases[qq][1356], deg=True), np.angle(new_bases[kk][new_orders[wihh]][1356], deg=True))
                        ##print('dot product is ', np.dot(np.conjugate(new_bases[qq].T), new_bases[kk][new_orders[wihh]]))
                        ##print('old product is ', np.dot(np.conjugate(new_bases[qq].T), new_bases[kk]))
                        ##if which_operation == 'C2z':
                        ##delta_phase = np.angle(new_bases[qq], deg=True) - np.angle(new_bases[kk][self.new_orders[wihh]], deg=True)  # transformed respect to org
                        #delta_phase = np.angle(new_bases[qq], deg=True) - np.angle(transed, deg=True)  # transformed respect to org
                        #delta_phase[ delta_phase>+180 ] -= 360
                        #delta_phase[ delta_phase<-180 ] += 360
                        #delta_phase = np.abs(delta_phase)
                        
                        ##delta_phase=np.rad2deg(np.arccos((np.real(new_bases[qq])*np.real(transed) + np.imag(new_bases[qq])*np.imag(transed) )/(np.absolute(new_bases[qq])*np.absolute(transed))))
                        
                        ##axs[(kk)].hist(delta_phase, weights=np.absolute(new_bases[qq]),bins=360)
                        #np.set_printoptions(precision=4)
                        #if np.std(delta_phase) <20:
                            #print('   phase is ', np.round(np.mean(delta_phase),5), np.round(np.std(delta_phase),5) )
                            #axs[kk].hist(delta_phase, bins=45, density=True) #, weights=np.absolute(new_bases[qq])
                            ##axs[(kk)].set_xlim([-5,185])
                        #else:
                            ## even+i*odd thing
                            #vec_0a = (+np.real(new_bases[qq]) + np.imag(new_bases[qq])) + 1j*(+np.real(new_bases[qq]) - np.imag(new_bases[qq])) # THe working one!
                            #vec_1a = (+np.real(new_bases[qq]) + np.imag(new_bases[qq])) + 1j*(+np.real(new_bases[qq]) + np.imag(new_bases[qq])) #/np.sqrt(2)
                            #vec_2a = (+np.real(new_bases[qq]) - np.imag(new_bases[qq])) + 1j*(+np.real(new_bases[qq]) + np.imag(new_bases[qq])) #/np.sqrt(2)
                            #vec_3a = (+np.real(new_bases[qq]) - np.imag(new_bases[qq])) + 1j*(+np.real(new_bases[qq]) - np.imag(new_bases[qq])) #/np.sqrt(2)
                            ##vec_b = new_bases[kk][self.new_orders[wihh]] *np.sqrt(2) 
                            #vec_b = transed *np.sqrt(2) 
                            #case_n = 0
                            #for vec_a in [vec_0a,vec_1a, vec_2a, vec_3a]:
                                ##vec_a *= vec_a*phase_1
                                ##vec_b *= vec_b*phase_2
                                #check_m = np.isclose(np.absolute(vec_a), np.absolute(vec_b), rtol=0.2, atol=0.0)
                                
                                #if  np.isclose( np.count_nonzero(check_m), self.conf.tot_number, rtol=tol_, atol=0):
                                    #delta_phase=np.rad2deg(np.arccos((np.real(vec_a)*np.real(vec_b) + np.imag(vec_a)*np.imag(vec_b) )/(np.absolute(vec_a)*np.absolute(vec_b))))
                                    #print('\tmagnetic #{0} check: '.format(case_n), np.count_nonzero(check_m),'   phase is ', np.mean(delta_phase), np.std(delta_phase))
                                #case_n += 1
                                ##delta_phase = np.angle(vec_a, deg=True) - np.angle(vec_b, deg=True)  # transformed respect to org
                                ##delta_phase[ delta_phase>+180 ] -= 360
                                ##delta_phase[ delta_phase<-180 ] += 360
                                ##delta_phase = np.abs(delta_phase)
                                ##axs[kk].hist(delta_phase, bins=45, density=True) #, weights=np.absolute(new_bases[qq])
                                ##axs[(kk)].set_xlim([-5,185])
                                ##axs[(kk)].set_ylim([0,1])
                        
                    #elif flag_:
                        ##print('   Only in amplitude {0} to {1} '.format(qq,kk), np.count_nonzero(check_), ' of ', self.conf.tot_number)
                        #only_amp_str += '   Only in amplitude {0} to {1}  {2} of {3} \n'.format(qq,kk, np.count_nonzero(check_), self.conf.tot_number)
            #print(only_amp_str)
        #print('\n')
        ##who += 1
        #self.new_bases = new_bases
        
        ##check if bases are True bases of H
        #Hdense = self.H.todense() ## please improve maybe using sp multiply
        #for ch in range(flat_range):
            #psi =  np.expand_dims(new_bases[ch], axis=1)
            ##left = np.matmul( np.conjugate(psi.T),  self.H)
            #left = np.matmul( np.conjugate(psi.T),  Hdense ) ## please improve maybe using sp multiply
            #value_ = np.matmul( left,  psi)
            
            ##phi =  np.expand_dims(old_vecs[ch], axis=1)
            ##left = np.matmul( np.conjugate(phi.T),  self.H)
            ##value_1 = np.matmul( left,  phi)
            
            ##print((self.bandsEigns[id_, ch]), ' is ', (value_-self.sigma_shift)*1000*0.5, (value_1-self.sigma_shift)*1000*0.5 )
            #assert value_.shape == (1,1)
            #print((np.real(value_[0,0])-self.sigma_shift)*1000*0.5)
        ##print(eignvals_neu)
        
        ##fig, axs = plt.subplots(1,3,figsize=(7, 10))
        ##cm = plt.cm.get_cmap('RdYlBu')
        ##lev_ref = 6
        ##lev_trs = 6
        ##type_dic = {0:'a', 1:'b', 2:'d', 3:'c'} # for 1 fold
        ##sc0 = axs[0].scatter(wave_info[0, :, 0], wave_info[0, :, 1], 
                            ##c=np.angle(new_bases[lev_ref], deg=True),  cmap=cm, 
                            ##s= 4000*np.real(new_bases[lev_ref])) 
        ##sc1 = axs[1].scatter(wave_info[0, self.new_orders[2], 0], wave_info[0, self.new_orders[2], 1], 
                            ##c=np.angle(new_bases[lev_trs][self.new_orders[2]], deg=True),
                            ##cmap=cm, s= 4000*np.real(new_bases[lev_trs][self.new_orders[2]]))
        
        ##fig.colorbar(sc0,orientation='vertical')
        ##fig.colorbar(sc1,orientation='vertical')
        ###axs[2].plot(wave_info[0, :, 0], wave_info[0, :, 1], 'o', color='tomato', markersize = 4)
        ###axs[2].plot(wave_info[0, new_order, 0], wave_info[0, new_order, 1], 'o', color='seagreen', markersize = 3)

        
        ###plt.colorbar(sc)
        ###axs[0].set_title('{0}{1}'.format(lev_ref, type_dic[ref_tp]))
        ###axs[1].set_title('{0}{1}'.format(lev_trs, type_dic[partner_tp]))
        ##axs[0].set_aspect('equal', 'box')
        ##axs[1].set_aspect('equal', 'box')
        ##axs[2].set_aspect('equal', 'box')
        
        ##fig.tight_layout()
        ##plt.show()
            
    #def embed_flatVec(self, which_K, vers='', d_phase=False,  vec_='lanczos'):
        #'''
        #which_K   !!string!!
         #vec_='lanczos' or 'ortho' or 'phasesign'
        #'''
        #if self.rank ==0:
            ### find the index
            #xpos_ = np.cumsum(self.K_path_Highsymm_indices)
            #print("K_label",self.K_label)
            #print("which_K",which_K)
            #print("[np.where(self.K_label==which_K)",np.where(self.K_label==which_K))
            #print("xpos_",xpos_)
            #id_ = xpos_[np.where(self.K_label==which_K)[0][0]]
            #print('Embedding eigen vectors for point='+which_K)
            #print('id_={0}'.format(id_))
            
            #if not hasattr(self, 'N_flat'):
                #raise AttributeError('MAKE sure you sorted eign vectors before this!!')
            ### write the thing
            ##fname = self.folder_name + self.file_name[:-5] + '_cut_' + str(self.cut_fac) + '_'+which_K + '_{0}.lammpstrj'
            #dph = '_delta_' if d_phase else ''
            #fname = self.folder_name +  vers+'_'+vec_+'_'+ dph+ self.save_name + '_'+which_K + '_{0}'+'.lammpstrj'
            #print('fname::',fname)
            #header_ = ''
                        
            
            #if self.conf.xy ==0:
                #header_ += "ITEM: TIMESTEP \n{0} \nITEM: NUMBER OF ATOMS \n"+"{0} \nITEM: BOX BOUNDS pp pp ss \n".format(self.conf.tot_number)
                #header_ +="{0} {1} \n{2} {3} \n{4} {5} \n".format(                                                                                                                                                                            self.conf.xlo, self.conf.xhi, 
                #self.conf.ylo, self.conf.yhi,
                #self.conf.zlo, self.conf.zhi)
            #else:
                #header_ += "ITEM: TIMESTEP \n{0} \nITEM: NUMBER OF ATOMS \n"+"{0} \nITEM: BOX BOUNDS xy xz yz pp pp ss \n".format(self.conf.tot_number)
                #header_ +="{0} {1} {6} \n{2} {3} 0 \n{4} {5} 0 \n".format(                                                                                                                                                                            self.conf.xlo, self.conf.xhi+self.conf.xy, 
                #self.conf.ylo, self.conf.yhi,
                #self.conf.zlo, self.conf.zhi,
                #self.conf.xy)
                
                
            #header_ += "ITEM: ATOMS id type x y z  "
            ##append_ = " "
            ##for ii in range(self.N_flat):
                 ##header_ += " abs{:.0f} ".format(ii+1)#, self.bandsEigns[id_, ii]-self.shift_tozero )
                 ##append_ +=  " phase{:.0f}::{:.5f} ".format(ii+1, self.bandsEigns[id_, ii]-self.shift_tozero)
            ##header_ += append_
            ##fmt_ = ['%.0f']*2 + ['%.12f']*(3+2*self.N_flat)
            #out_range= 8 #self.N_flat

            #header_ += " abs phase "#.format(ii+1)#, self.bandsEigns[id_, ii]-self.shift_tozero )

            #fmt_ = ['%.0f']*2 + ['%.12f']*(3+2*1)

            #if vec_ == 'lanczos':
                #vecs = self.bandsVector[id_]
            #elif vec_ ==  'ortho':
                #vecs = self.new_bases.T
            ##elif vec_ ==  'phasesign':
                ##vecs = self.new_bases.T
                ###for shit in range(self.conf.tot_number):
                    ###assert self.phaseSigns[2][self.new_orders[2]][shit] == self.phaseSigns[2][shit]
                ##for ii in range(out_range):
                    ##vecs[:, ii] =  vecs[self.new_orders[2], ii] * self.phaseSigns[2]
            #else:
                #raise TypeError('unrecognized vector type')
            ##print(vecs.shape)
            
            #for ii in range(out_range):
                #angle_ = np.angle(vecs[:, ii], deg=True)
                #if d_phase:
                    #angle_t = np.angle(vecs[self.new_orders[2], ii], deg=True)
                    #delta_phase = angle_ - angle_t
                    #delta_phase[ delta_phase>+180 ] -= 360
                    #delta_phase[ delta_phase<-180 ] += 360
                    #delta_phase = np.abs(delta_phase)
                    #angle_ = delta_phase
                
                
                #XX = np.concatenate((self.conf.atomsAllinfo[:,np.r_[0]],  np.expand_dims(self.conf.sub_type, axis=1), 
                #self.conf.atomsAllinfo[:,np.r_[4:7]],
                #np.expand_dims(np.absolute(vecs[:, ii]), axis=1), 
                #np.expand_dims(angle_, axis=1),
                #), axis = 1)

                #np.savetxt(fname.format(ii), XX ,header=header_.format(ii), comments='', fmt=fmt_) #fmt='%.18e'
            
            #print('levels:\n', self.bandsEigns[id_, :out_range]-self.shift_tozero)
            #np.savetxt(fname+'__eign_values__', self.bandsEigns[id_, :out_range]-self.shift_tozero) 

            ##for ii in range(self.N_flat):
                ##XX = np.concatenate((self.conf.atomsAllinfo[:,np.r_[0]],  np.expand_dims(self.conf.sub_type, axis=1), self.conf.atomsAllinfo[:,np.r_[4:7]],
                                    ##np.absolute(self.bandsVector[id_, :, :self.N_flat]), 
                                    ##np.angle(self.bandsVector[id_, :, :self.N_flat]),
                                    ##), axis = 1) #

                ##np.savetxt(fname, XX ,header=header_, comments='', fmt=fmt_) #fmt='%.18e'
            
            
            
            ##file_ = open(fname, 'w')
            ##file_.write(header_)
            ##for ii in range(self.conf.tot_number):
                ##file_.write(('{:.0f} '*4 + '{:.12f} '*3 + ' 0 0 0 ' + '{:.2e} '*self.N_flat +'  \n').format(*self.conf.atomsAllinfo[ii], *self.bandsVector[id_, ii, :self.N_flat] ))
                
            ##file_.close()
     
