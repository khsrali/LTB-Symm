import sys
import numpy as np
import ltbsymm as ls

# Start a TB object and set/load configuration
mytb = ls.TB()
mytb.load('out_1.08_2AA', bands='bands_.npz', configuration='configuration_.npz')


# Start a Symm object and set/load configuration
sm = ls.Symm(mytb)

# Define all symmetry operations of the space group
sm.build_map('C2z',['-X+1/2*Rx','-Y+1/2*Ry','Z'], atol=0.3, plot = True)
sm.build_map('C2y',['-X','Y+1/2*Ry','-Z'], atol=0.3)
sm.build_map('C2x',['X+1/2*Rx','-Y','-Z'], atol=0.3)

# Make the operation Matrix at a given point of receiprocal space
sm.make_Cmat('C2x', 'Gamma')
sm.make_Cmat('C2y', 'Gamma')
sm.make_Cmat('C2z', 'Gamma')


# Check operations square and how they commute 
sm.check_square('C2x', 'Gamma', ftol = 30)
sm.check_square('C2y', 'Gamma', ftol = 30)
sm.check_square('C2z', 'Gamma', ftol = 30)
sm.check_commute('C2x', 'C2y', 'Gamma', ftol=30) 
sm.check_commute('C2z', 'C2y', 'Gamma', ftol=30) 
sm.check_commute('C2x', 'C2z', 'Gamma', ftol=30) 

# You can also load a previusly calculated Hamiltonian matrix, if you want
H = mytb.load('out_1.08_2AA', HH='HH_Gamma.npz')

# Detect and identify flat bands
mytb.detect_flat_bands()

# Check if wave functions are diagonal respect to symmetries
sm.vector_diag('Gamma', name1='C2x', subSize = 4, skip_diag = True)
sm.vector_diag('Gamma', name1='C2y', subSize = 4, skip_diag = True)
sm.vector_diag('Gamma', name1='C2z', subSize = 4, skip_diag = True)

# Diagonalize wave vectors respect to a given symmetry 
sm.vector_diag('Gamma', name1='C2z', name2= 'C2x', subSize = 4, rtol=0.1, skip_diag = False)

# !!  Output is flat bands Parity !! 


# You can save sm object 
sm.save()

# And load later to skip redundancy
#sm.load('out_1.08_AA', 'Symm_.npz')

