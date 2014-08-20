'''
Created on 8.8.2014

@author: tohekorh
'''

import numpy as np


class Energies():
    
    def __init__(self, VirtAtoms):
        self.VirtAtoms                  =   VirtAtoms
        self.layers, self.atomsPerLayer =   self.VirtAtoms.PosAtoms.shape
        # Sliding barrier
        self.E_c                        =   1.0
        self.bend_module                =   5.
        self.strech_module              =   10.
        
    def strech_E(self):
        
        s_sum               =   0.
        
        for ilayer in range(self.layers):
            xset, yset  =   self.VirtAtoms.get_positions(ilayer)    
            for i in range(len(xset) - 1): 
                
                bl          =   np.sqrt((xset[i + 1] - xset[i])**2 + (yset[i + 1] - yset[i])**2)
                
                e           =   (bl - self.VirtAtoms.lattice_const)/self.VirtAtoms.lattice_const
                
                s_sum      +=   .5*self.strech_module*e**2
        
        return s_sum

    def bend_E(self):
        
        b_sum               =   0.
        
        for ilayer in range(self.layers):    
            for x in self.VirtAtoms.get_positions(ilayer)[0]: 
            
                fpx, fppx   =   self.VirtAtoms.get_derivatives(ilayer, x)      
                fprime      =   np.array([1., 0., fpx])
                fpprime     =   np.array([0., 0., fppx])   
                normal      =   np.array([fpx, 0., -1.]/np.sqrt(fpx**2 + 1))
                
                
                C11         =   np.dot(normal, fpprime)/np.linalg.norm(fprime)
               
                #print C11, np.dot(normal, fpprime)
                b_sum      +=   self.bend_module*C11**2   
        
        
        return b_sum   
    
    def slide_E(self, plot_pot = False):
        
        # Sliding energy opposes the tendency for the layers to slide wrt to each other.
        # If sliding occurs, it tends to stop when the 'atoms' are aligned on top of each other.
        
        # Ts is the distances to 'optimal' positions in each layer. These are used to determine
        # the wavelength for the oscillating potential that opposes the sliding.    
        Ts      =   self.VirtAtoms.Ts
        sumE    =   0
        
        def find_Barrier_E(dist, layer):
            
            interval    =   [0.,0.]
            popped_full =   False
            last        =   False
            bE          =   0. 
            
            for i, d in enumerate(Ts[layer]):
                if d <= dist and i != len(Ts[layer]) - 1:   interval = [d, Ts[layer][i + 1]]
                
            if interval == [0.,0.]:
                AvL         =   Ts[layer][-1] - Ts[layer][-2]
                interval    =   [Ts[layer][-1], Ts[layer][-1] + AvL/2.]
                # If the atom is near the edge but close to the last atom of the above surface 
                # it is called last.
                last        =   True
                
                if dist > Ts[layer][-1] + AvL/2.:
                    # If the atom is no longer under the surface on top it is popped full.
                    popped_full =   True    
            
            # Li    =   length of the interval on from optimal point to the next. 
            # phase =   the phase in which the given virtual atom is on the oscillating potential.
            Li      =   interval[1] - interval[0]
            phase   =   (dist - interval[0])/Li*np.pi
                
            
            
            # Here the energy is calculated.    
            if not popped_full and not last:
                bE       =   self.E_c*np.sin(phase)**2

            elif last and not popped_full:
                bE       =   self.E_c/2.*np.sin(phase/2.)**2
            
            elif popped_full:
                bE       =   self.E_c/2.
            
            return bE

        # Loop over all atoms except the top most layer. 
        if not plot_pot:
            for layer in range(1, self.layers):
                for dist in self.VirtAtoms.distances[layer]: 
                    sumE   +=   find_Barrier_E(dist, layer)
                
            return sumE
        # This is for plotting the potential
        if plot_pot:
            acc =   20*self.atomsPerLayer 
            e   =   np.zeros((self.layers - 1, acc))
            ph  =   np.zeros((self.layers - 1, acc))
            for layer in range(1, self.layers):
                for i, dist in enumerate(np.linspace(0., np.amax(self.VirtAtoms.distances[layer]) + 1., acc)): 
                    e[layer - 1, i] =   find_Barrier_E(dist, layer)
                    ph[layer - 1, i]=   dist  
            return sumE, e, ph
    
