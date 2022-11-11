from TightBinding import TB
import sys


mytb = TB()
mytb.set_configuration('1.08_1AA.data', phi_ = 1.08455, orientation = '1_fold' , sparse_flag = False)
mytb.set_parameters(a0 = 1.42039011, d0 = 3.344, V0_sigam = +0.48, V0_pi = -2.7, cut_fac = 4.01)

mytb.set_symmetry_path(['K1','gamma','M','K2'] , N=1) #[]
mytb.calculate(n_eigns = 20)
mytb.save('')



plot = mytb.plotter()
#plot.show()




## note:
#angles: #1.08455  #2.1339 #1.050120879794409 # from Jin



## for only plotting
#y_low = -10
#y_high = +15
#mytb = TB()
##mytb.load('calculation_1.08_3fold_type_c_cut_4.01.npz') # to be written
##mytb.load('1.08_1AA_cut_4.01.npz') # to be written
#mytb.load(sys.argv[1]) # to be written
##mytb.load('1.08_buckled_best_wish_Ali_cut_4.01_redundant.npz') # to be written
#plot = mytb.plotter(color_='C0', save_flag =False) #seagreen 'tomato'
##plot.show()
##exit()
#plot.savefig(mytb.folder_name+'Bands_'+mytb.save_name +'_all'+ ".png", dpi=300)
##plot.show()
##exit()
#fig = plot.gcf()
#plot.ylim([y_low,y_high])
#fig.set_size_inches(16.65, 9.45, forward=True)
#plot.savefig(mytb.folder_name+'Bands_'+mytb.save_name +'_flat'+ ".png", dpi=300)

#plot.xlim([20,50])
#plot.ylim([-5,+5])
#fig = plot.gcf()
#fig.set_size_inches(5.0, 9.45, forward=True)
#plot.savefig(mytb.folder_name+'Bands_'+mytb.save_name +'_gamma'+ ".png", dpi=300)




#plot.show()

### crap
##plot.set_color("black")
