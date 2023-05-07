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
        
        
    def make_Cmat(self, name, k_label, symmorphicity = False, verify_check=False, tol_ = 0.1):
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
                    If the group is symmorphic. False(default)
                    **Notice in the case of non-symmorphic groups: Cmat is only implemented for C2 symmetries, at the moment. **
                
                verify_check: boolean
                    False(default)
                    
                tol_: float between 0-1
                    Relative tolerance. tol_=0.1(default)
        """
        print('**Making {0} matrix for point {1} **'.format(name, k_label))
        self.Cmat[name] = Cop
        
        
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
            convention_x = 1 if (self.conf.coords[ i , 0] //self.conf.xlen_half) %2 == 0 else  np.exp(+1j *  self.conf.xlen * K[0] )
            convention_y = 1 if (self.conf.coords[ i , 1] //self.conf.ylen_half) %2 == 0 else  np.exp(+1j *  self.conf.ylen * K[1] )
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
                
                if self.conf.tot_number < nonz_tot:
                    print('square has more non-zero elements that identity')
                    return 1
                elif self.conf.tot_number > nonz_tot:
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
        oper_mat = sp.lil_matrix((self.conf.tot_number, self.conf.tot_number), dtype='int')
        
        whos =  {'C2x':0, 'C2y':1, 'C2z':2}
        op_ = self.new_orders[whos[operation]]
        
        org = sp.lil_matrix(self.H, dtype=self.H.dtype, copy=True) 
        trans_op = sp.lil_matrix(self.H[op_][:,op_], dtype=self.H.dtype, copy=True) 
        
        #print('ishermitian for ',operation,ishermitian(trans_op.todense(), rtol=0.0))
        nonZ = org.nonzero()
        nonZ_tot = nonZ[0].shape[0]
        ##Make the M matrix
        #version_ = self.file_name[:-5] +'_cut_' + str(self.cut_fac) + '_Pf'
        #self.conf.neigh_list_me_smart(cutoff=self.r_cut, l_width=300, load_ = True, version_ = version_ )
        M = sp.lil_matrix((self.conf.tot_number, self.conf.tot_number), dtype='int')
        for ii in range(self.conf.tot_number):
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
            
            
            all_X = np.copy(self.conf.atomsAllinfo[ : , 4])
            all_Y = np.copy(self.conf.atomsAllinfo[ : , 5])
            all_Z = np.copy(self.conf.atomsAllinfo[ : , 6])
            all_xyz = np.copy(self.conf.atomsAllinfo[ : , 4:7])
            
            flat_range= 8 #self.N_flat
            wave_info = np.zeros((flat_range, self.conf.tot_number, 7), self.dtypeR)

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
            
            new_bases = np.zeros((flat_range, self.conf.tot_number), dtype=self.dtypeC)
            old_vecs = np.zeros((flat_range, self.conf.tot_number), dtype=self.dtypeC)
            old_vecs_op = np.zeros((flat_range, self.conf.tot_number), dtype=self.dtypeC)
            very_new_bases = np.zeros((flat_range, self.conf.tot_number), dtype=self.dtypeC)
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
                #new_bases = np.zeros((mix_pairs, self.conf.tot_number), dtype='complex' if self.dtype==None else 'c'+self.dtype)
                
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
                        flag_  = np.isclose( np.count_nonzero(check_), self.conf.tot_number, rtol=tol_, atol=0) 
                        #print('{0} dot {1} is '.format(ii+qq,ii+kk), np.dot(np.conjugate(new_bases[qq].T), new_bases[kk][new_orders[wihh]]))
                        #print(np.count_nonzero(check_))
                        condintion = (kk==qq) if only_itself else True
                        if flag_ and condintion:
                            print('{0} to {1} **symmetry Holds!** '.format(qq,kk), np.count_nonzero(check_), ' of ', self.conf.tot_number)
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
                                    
                                    if  np.isclose( np.count_nonzero(check_m), self.conf.tot_number, rtol=tol_, atol=0):
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
                            #print('   Only in amplitude {0} to {1} '.format(qq,kk), np.count_nonzero(check_), ' of ', self.conf.tot_number)
                            only_amp_str += '   Only in amplitude {0} to {1}  {2} of {3} \n'.format(qq,kk, np.count_nonzero(check_), self.conf.tot_number)
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
                if self.conf.xy !=0:
                    #raise RuntimeError('Non rectangular boxes are not supported yet, for checking symmetry')
                    all_X = np.copy(self.conf.atomsAllinfo[ : , 4]) 
                    all_Y = np.copy(self.conf.atomsAllinfo[ : , 5])
                    all_X -= (all_X//self.conf.xhi)*self.conf.xlen
                else:
                    all_X = np.copy(self.conf.atomsAllinfo[ : , 4])
                    all_Y = np.copy(self.conf.atomsAllinfo[ : , 5])

                
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
                wave_info = np.zeros((flat_range, 4, self.conf.tot_number//4, 6), dtype=self.dtypeR)
                ##wave_info ITEM:  x y  real img  amp angle

                
                for ii in range(flat_range):
                    for type_ in range(4):
                        abcd = self.conf.sub_type==10*(type_+1)
                        #print(self.conf.sub_type)
                        wave_info[ii, type_, :, 0] = all_X[abcd]
                        wave_info[ii, type_, :, 1] = all_Y[abcd]
                        
                        wave_info[ii, type_, :, 2] = np.real(self.bandsVector[id_, abcd, ii])
                        wave_info[ii, type_, :, 3] = np.imag(self.bandsVector[id_, abcd, ii])
                        wave_info[ii, type_, :, 4] = np.absolute(self.bandsVector[id_, abcd, ii])
                        wave_info[ii, type_, :, 5] = np.angle(self.bandsVector[id_, abcd, ii], deg=True)
                
                # Set Fabrizio's cell to (0,xlen) and (0,ylen)
                # and put everything in the box 
                #wave_info[:, :, :, 0] -= (self.conf.xlo - 1/4 * self.conf.xlen) # for 1 fold
                #wave_info[:, :, :, 1] -= (self.conf.ylo - 1/4 * self.conf.ylen) # for 1 fold
                
                #wave_info[:, :, :, 0] -=  (wave_info[:, :, :, 0]//self.conf.xlen)*self.conf.xlen
                #wave_info[:, :, :, 1] -=  (wave_info[:, :, :, 1]//self.conf.ylen)*self.conf.ylen
                
                # apply transformation
                wave_info_trs = np.copy(wave_info)
                if which_operation == 'C2x':
                    #x+1/2,-y,-z
                    print('doing c2x')
                    wave_info_trs[:, :, :, 0] += 1/2 * self.conf.xlen
                    wave_info_trs[:, :, :, 1] *= -1
                    
                    # type is the same, layer changes :: 0,3 (a,c) and (1,2) b,d have the same type! ...
                    # ...specifically for this system!! To be fixed later
                    partner_tp = 3 # for 1 fold
                    #partner_tp = 2  # for 0 fold
                    
                elif which_operation == 'C2y':
                    #-x,y+1/2,-z
                    print('doing c2y')
                    wave_info_trs[:, :, :, 0] *= -1
                    wave_info_trs[:, :, :, 1] += 1/2 * self.conf.ylen
                    
                    if '_zxact' not in self.file_name and 'noa0_relaxed' not in self.file_name and '1.08_0fold_no18' not in self.file_name:
                        wave_info_trs[:, :, :, 0] -= self.a0 ## # for 1 fold i don't know why it is this way!
                        #wave_info_trs[:, :, :, 0] -= 2*self.a0 ## # for 0 fold i don't know why it is this way!
                    
                    # type and layer changes
                    partner_tp = 2 # for 1 fold
                    #partner_tp = 3  # for 0 fold
            
                elif which_operation == 'C2z':     
                    #-x+1/2,-y+1/2,z
                    print('doing c2z')
                    wave_info_trs[:, :, :, 0] = -1*wave_info_trs[:, :, :, 0] + 1/2 * self.conf.xlen
                    wave_info_trs[:, :, :, 1] = -1*wave_info_trs[:, :, :, 1] + 1/2 * self.conf.ylen
                    
                    if '_zxact' not in self.file_name and 'noa0_relaxed' not in self.file_name  and '1.08_0fold_no18' not in self.file_name:
                        wave_info_trs[:, :, :, 0] -= self.a0 ## for 1 fold #i don't know why it is this way!
                        #wave_info_trs[:, :, :, 0] -= 2*self.a0 ### for 0 fold #i don't know why it is this way!
                    
                    # layer is the same, type changes
                    partner_tp = 1
                
                elif which_operation == 'sigma_yz' :
                    wave_info_trs[:, :, :, 0] = -1*wave_info_trs[:, :, :, 0] + 1/2 * self.conf.xlen
                    partner_tp = 2
                
                elif which_operation == 'sigma_xz' :
                    wave_info_trs[:, :, :, 1] = -1*wave_info_trs[:, :, :, 1] + 1/2 * self.conf.ylen
                    partner_tp = 0
                    
                #elif which_operation == 'mz' :
                    #wave_info_trs[:, :, :, 0] += 1/2 * self.conf.xlen
                    #wave_info_trs[:, :, :, 1] -= 1/2 * self.conf.ylen
                
                
                ## translate to cell(0,0) 
                wave_info_trs[:, :, :, 0] -=  (wave_info_trs[:, :, :, 0]//self.conf.xlen)*self.conf.xlen
                wave_info_trs[:, :, :, 1] -=  (wave_info_trs[:, :, :, 1]//self.conf.ylen)*self.conf.ylen
                
                ## get the right indices to compare 
                new_order = np.zeros(self.conf.tot_number//4, dtype='int')

                
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
                
                for nn in range(self.conf.tot_number//4):
                    
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
                                    desire_coord[0] += pX * self.conf.xlen 
                                    desire_coord[1] += pY * self.conf.ylen 
                            
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
                            flag_ = np.isclose( np.count_nonzero(check_), self.conf.tot_number//4, rtol=tol_, atol=0) 
                            
                            ## approach dot product:
                            vec2 = vec1 = np.zeros(self.conf.tot_number//4 , dtype=self.dtypeC)
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
                                print('symmetry Holds! ',np.count_nonzero(check_), ' of ', self.conf.tot_number//4)
                                print('level {0} -> {1}. phase_shift is '.format(ii, jj), np.round(phase_shift,3),' with std=', np.round(std,3))
                                print('energy of level {0} is {1}'.format(ii, self.bandsEigns[id_, ii]-self.shift_tozero) )
                                #print(np.count_nonzero(check_), ' of ', self.conf.tot_number//4)
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
                                likelihood=100* likelihood/ (self.conf.tot_number//4)
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
                        
                        flag_ = np.isclose( np.count_nonzero(check_), self.conf.tot_number//4, rtol=tol_, atol=0) 
                        
                        
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
                            print('symmetry Holds! ',np.count_nonzero(check_), ' of ', self.conf.tot_number//4)
                            #print('level {0} -> {1}. phase_shift is '.format(ii, jj), np.round(phase_shift,3),' with std=', np.round(std,3))
                            #print(np.count_nonzero(check_), ' of ', self.conf.tot_number//4)
                            print('\n')
                            #break
                        else:

                            print('symmetry sucks!')
                            #np.set_printoptions(precision=1)
                            likelihood=100* np.count_nonzero(check_)/ (self.conf.tot_number//4)
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
                        
            
            if self.conf.xy ==0:
                header_ += "ITEM: TIMESTEP \n{0} \nITEM: NUMBER OF ATOMS \n"+"{0} \nITEM: BOX BOUNDS pp pp ss \n".format(self.conf.tot_number)
                header_ +="{0} {1} \n{2} {3} \n{4} {5} \n".format(                                                                                                                                                                            self.conf.xlo, self.conf.xhi, 
                self.conf.ylo, self.conf.yhi,
                self.conf.zlo, self.conf.zhi)
            else:
                header_ += "ITEM: TIMESTEP \n{0} \nITEM: NUMBER OF ATOMS \n"+"{0} \nITEM: BOX BOUNDS xy xz yz pp pp ss \n".format(self.conf.tot_number)
                header_ +="{0} {1} {6} \n{2} {3} 0 \n{4} {5} 0 \n".format(                                                                                                                                                                            self.conf.xlo, self.conf.xhi+self.conf.xy, 
                self.conf.ylo, self.conf.yhi,
                self.conf.zlo, self.conf.zhi,
                self.conf.xy)
                
                
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
                ##for shit in range(self.conf.tot_number):
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
                
                
                XX = np.concatenate((self.conf.atomsAllinfo[:,np.r_[0]],  np.expand_dims(self.conf.sub_type, axis=1), 
                self.conf.atomsAllinfo[:,np.r_[4:7]],
                np.expand_dims(np.absolute(vecs[:, ii]), axis=1), 
                np.expand_dims(angle_, axis=1),
                ), axis = 1)

                np.savetxt(fname.format(ii), XX ,header=header_.format(ii), comments='', fmt=fmt_) #fmt='%.18e'
            
            print('levels:\n', self.bandsEigns[id_, :out_range]-self.shift_tozero)
            np.savetxt(fname+'__eign_values__', self.bandsEigns[id_, :out_range]-self.shift_tozero) 

            #for ii in range(self.N_flat):
                #XX = np.concatenate((self.conf.atomsAllinfo[:,np.r_[0]],  np.expand_dims(self.conf.sub_type, axis=1), self.conf.atomsAllinfo[:,np.r_[4:7]],
                                    #np.absolute(self.bandsVector[id_, :, :self.N_flat]), 
                                    #np.angle(self.bandsVector[id_, :, :self.N_flat]),
                                    #), axis = 1) #

                #np.savetxt(fname, XX ,header=header_, comments='', fmt=fmt_) #fmt='%.18e'
            
            
            
            #file_ = open(fname, 'w')
            #file_.write(header_)
            #for ii in range(self.conf.tot_number):
                #file_.write(('{:.0f} '*4 + '{:.12f} '*3 + ' 0 0 0 ' + '{:.2e} '*self.N_flat +'  \n').format(*self.conf.atomsAllinfo[ii], *self.bandsVector[id_, ii, :self.N_flat] ))
                
            #file_.close()
     
