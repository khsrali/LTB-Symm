import sys
from TightBonding import TB

mytb = TB()

mytb.set_configuration(file_name = sys.argv[1], phi_ = 1.08455, sparse_flag = True)

mytb.set_parameters(a0 = 1.42039011, d0 = 3.344, V0_sigam = +0.48, V0_pi = -2.7, cut_fac = 4.01)

#mytb.set_symmetry_path(['gamma', "K2_prime", 'K1','gamma','M','K2'] , N=80)
mytb.set_symmetry_path(['K1','gamma'] , N=10)

mytb.calculate(n_eigns = 5)

#mytb.load() # to be written
mytb.save('_redundant')

plot = mytb.plotter()

plot.show()

## some angles
#1.08455  #2.1339 #1.050120879794409 # from Jin
