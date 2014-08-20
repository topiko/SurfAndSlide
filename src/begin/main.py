'''
Created on 11.8.2014

@author: tohekorh
'''
from surf_bend import VirtualAtoms
from energies import Energies
from minimize_E import minimize


layers, n       =   2, 12
surfAtoms       =   VirtualAtoms(layers, n)


q_wrap, wrap2, bounds \
                =   surfAtoms.get_quess(-0.00001, 'kinky')
slide_e         =   Energies(surfAtoms)
minim           =   minimize(surfAtoms, wrap2, bounds, slide_e)

E, opm_wrap     =   minim.minimize_energy(40)

surfAtoms.set_positions_wrap_noPoly(opm_wrap)

surfAtoms.plotAtoms(iter_count = minim.count, show = True)

print opm_wrap, E
