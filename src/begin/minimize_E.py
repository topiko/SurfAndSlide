'''
Created on 8.8.2014

@author: tohekorh
'''

from scipy.optimize import fmin_l_bfgs_b as bfgs_b
import datetime

class minimize():
    
    def __init__(self, VirtAtoms, quess_wrap, bounds_wrap, energies):
        self.VirtAtoms  =   VirtAtoms
        self.wrap       =   quess_wrap
        self.energies   =   energies
        self.bounds     =   bounds_wrap
    
        
    def minimize_energy(self, plot_interval = None):
        
        self.count = 0.
        
        def energy(wrap):
            
            self.VirtAtoms.set_positions_wrap_noPoly(wrap)
            Es  =   self.energies.slide_E()
            Eb  =   self.energies.bend_E()
            Est =   self.energies.strech_E()
            
            self.printProgress(Es, Eb, Est)
                        
            if plot_interval != None and self.count % plot_interval == 0:
                #e, dist = self.energies.slide_E(True)[1:]
                self.VirtAtoms.plotAtoms(iter_count = self.count, show = False)
            
            self.count      +=   1
            
            return Es + Eb + Est


        opm_wrap, E     =   bfgs_b(energy, self.wrap, epsilon=1e-1, \
                                   bounds=self.bounds, approx_grad = True)[:2]

        return E, opm_wrap
        
    
    def printProgress(self, Es, Eb, Est):
        
        print 'time: ' + str(datetime.datetime.now().strftime('%H:%M:%S'))
        print 'Iter count = '   + str(self.count)
        print 'Energy strech     = ' + str(Est)
        print 'Energy bend       = ' + str(Eb)
        print 'Energy slide      = ' + str(Es) 
        print 'Total energy      = ' + str(Es + Eb + Est)
        print        
        
        