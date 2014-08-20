'''
Created on 7.8.2014

@author: tohekorh
'''

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy import integrate
import pylab


class VirtualAtoms():
    def __init__(self, layers, n):
        
        self.PosAtoms           =   np.empty((layers,n), dtype='object')
        self.PolConsts          =   np.empty(layers, dtype = 'object')
        self.distances          =   np.empty(layers, dtype = 'object')
        self.Ts                 =   np.empty(layers, dtype = 'object')
        self.layers             =   layers
        self.n                  =   n
        self.lattice_const      =   1.
        self.InterplaneDistance =   1.
        self.polOrder           =   self.n - 1  #self.n - 1   
        
        self.helpx              =   np.empty(layers - 1, dtype='object')
        self.helpy              =   np.empty(layers - 1, dtype='object')
        
        
    def set_positions(self, polyConsts, distances):
        # Here we set the positions for the 'virtual' atoms. 
        # We first set the top most layer as it fixes the curves for the rest of the layers.
        # The curves are set so that the interplane distance is constant; InterplaneDistance.
        # This is achieved by fitting high enough polynomial to the 'equidistant points' that we obtain by:
        # (x,y) = (x0, y0) + InterplaneDistance*normal vector of the above curve in (x0,y0).
        
        self.PolConsts[0]       =   polyConsts
        self.Ts[0]              =   distances[0]
          
        for layer in range(self.layers):
            self.distances[layer]   =   distances[layer]
            
            for i, dist in enumerate(distances[layer]):
                self.PosAtoms[layer][i]    =   self.xtyt(layer, dist)
                
            if layer != self.layers - 1:
                self.set_pol_consts(layer + 1)
                
            #print self.PolConsts
            
    """    
    def set_positions_wrap(self, wrap):
        # The wrapped set does not need the first distance i.e., zero in each layer as this is 
        # constant. Also the x=0 constant in poly is unnecessary as it is not modified by the 
        # optimization routine. Hence we here find polyConsts and distances and the
        # pass these to set_positions.
        
        PolConsts       =   np.zeros(self.polOrder + 1)
        distances       =   np.empty(self.layers, dtype = 'object') 
        
        PolConsts[0]    =   0.0
        PolConsts[1:]   =   wrap[:self.polOrder]
        
        for i in range(self.layers):
            distances[i]       =   np.zeros(self.n)
        
            distances[i][0]    =   0.
            distances[i][1:]   =   wrap[(self.n - 1)*(i + 1):(self.n - 1)*(i+2)]
        
        #print PolConsts
        #print distances
        
        self.set_positions(PolConsts, distances)
    """
    
    def set_positions_wrap_noPoly(self, wrap):
        # Same as before, however, now we do not give the poly constants but the positions of 
        # the atoms in top most layer. We then generate the polynomial constants and and pass the on.
         
        distances       =   np.empty(self.layers, dtype = 'object') 
        
        x_set           =   wrap[:self.n - 2]
        y_set           =   wrap[(self.n - 2):2*(self.n - 2)]
        
        x_set           =   np.concatenate(([0.], x_set, [self.get_positions(0)[0][-1]]))
        y_set           =   np.concatenate(([0.], y_set, [self.get_positions(0)[1][-1]]))
        
        #PolConsts       =   np.polyfit(x_set, y_set, self.polOrder)[::-1]
        
        #
        #guess_consts    =   np.ones(self.polOrder)
        self.layerUnderStudy    =   0
        PolConsts       =   curve_fit(self.fx, x_set, y_set)[0]
        #
        distances[0]    =   np.zeros(self.n)
            
        for i in range(self.n):
            distances[0][i]  =   integrate.quad(lambda x: \
                                 np.sqrt(1 + self.CurveFunc(0, x)[1]**2), 0, x_set[i])[0]
        
        for i in range(1, self.layers):
            distances[i]       =   np.zeros(self.n)
            distances[i][0]    =   0.
            dist_start         =   2*(self.n - 2) 
            distances[i][1:]   =   wrap[dist_start + (self.n - 1)*(i - 1):dist_start + (self.n - 1)*i]
        
        
        self.set_positions(PolConsts, distances)

        
    def get_quess(self, delY_pros, key):
        
        L                   =   (self.n - 1)*self.lattice_const
        # We approximate with circle.
        if key == 'circle':
            deltaY          =   delY_pros*self.InterplaneDistance
            R               =   L**2/(2*deltaY)
            polConsts       =   np.zeros(self.polOrder)
            polConsts[1]    =   1./(2*R) 
            polConsts[3]    =   1./(8*R**3) 
        elif key == 'kinky':
            x_set           =   np.zeros(self.n)
            y_set           =   np.zeros(self.n)
            self.theta      =   np.arcsin(delY_pros*self.InterplaneDistance/(L/2.))
            
            #self.theta      =   -2*np.arcsin(self.lattice_const/(2*self.InterplaneDistance))
            
            
            l               =   self.lattice_const/np.sqrt(2*(1 - np.cos(np.pi - self.theta)))
            self.kinkyPoint =   self.n/2*self.lattice_const + l #+ self.lattice_const
            
            for i in range(self.n):
                if self.lattice_const*i < self.kinkyPoint:
                    x_set[i]    =   self.lattice_const*i
                    y_set[i]    =   0.0
                    self.lastOfFirst    =   i
                if self.lattice_const*i >= self.kinkyPoint:
                    x_set[i]    =   self.kinkyPoint + l*np.cos(self.theta) \
                                    + (i - self.lastOfFirst - 1)*self.lattice_const*np.cos(self.theta)
                    y_set[i]    =   l*np.sin(self.theta) \
                                    + (i - self.lastOfFirst - 1)*self.lattice_const*np.sin(self.theta)
            
            self.layerUnderStudy=   0
            polConsts           =   curve_fit(self.fx, x_set, y_set)[0]
            
            #
            #self.initx, self.inity  =   x_set, y_set
            #self.PolConsts[0]       =   polConsts
            #self.plotAtoms()
            #
            
        distances       =   [np.linspace(0, L, self.n) for i in range(self.layers)]

        self.set_positions(polConsts, distances)
        
        #self.set_positions(np.copy(self.PolConsts[0]), self.Ts)
        
        wrap            =   np.copy(self.PolConsts[0])[1:]
        
        
        for dist in self.Ts:
            wrap        =   np.concatenate((wrap, dist[1:]))
         
            
        #wrap2
        x_set   =   np.zeros(self.n - 2)        
        y_set   =   np.zeros(self.n - 2)        
        x_bounds=   np.empty(self.n - 2, dtype = 'object')        
        y_bounds=   np.empty(self.n - 2, dtype = 'object')        


        for ixy, xy in enumerate(self.PosAtoms[0][1:self.n - 1]): 
            x_set[ixy]  =   xy[0]
            y_set[ixy]  =   xy[1]
            x_bounds[ixy]   =   (0.0001*ixy, None)
            y_bounds[ixy]   =   (None, -0.0001*ixy)
            
        wrap2           =   np.concatenate((x_set, y_set))
        bounds          =   np.concatenate((x_bounds, y_bounds))
        
        for dist in self.Ts[1:]:
            wrap2       =   np.concatenate((wrap2, dist[1:]))
            bhelp       =   np.empty(len(dist) - 1, dtype = 'object')   
            
            for i in range(len(dist[1:])):
                bhelp[i]=   (0.0001*(i + 1.), None)
            
            bounds      =   np.concatenate((bounds, bhelp))
        #
        
        return wrap, wrap2, bounds
        
    def get_positions(self, layer=None):
        #Return the position of atoms in given layer. 
        if layer == None:
            
            xpos    =   np.zeros((self.layers,self.n))
            ypos    =   np.zeros((self.layers,self.n))
            for layer in range(self.layers):
                for ir, r in enumerate(self.PosAtoms[layer]):
                    xpos[layer, ir]    =   r[0]
                    ypos[layer, ir]    =   r[1]
            return xpos, ypos
        else:
            x, y    = np.zeros(self.n), np.zeros(self.n)
            for i, pos in enumerate(self.PosAtoms[layer]): 
                x[i], y[i]  =   pos[0], pos[1]  
            return x,y
    
    def get_derivatives(self, layer, x):
        
        fprime, fpprime =   self.CurveFunc(layer, x)[1:]
        
        #for ic, pol_c in enumerate(self.PolConsts[layer][1:]):
        #    fprime +=   (ic + 1)*pol_c*x**(ic)               

        #for ic, pol_c in enumerate(self.PolConsts[layer][2:]):
        #    fpprime +=   (ic + 2)*(ic + 1)*pol_c*x**(ic)
            
        return fprime, fpprime               

    def get_u(self, layer, i):
        
        orig_pos    =   np.array([i*self.lattice_const, -layer*self.InterplaneDistance]) 
        
        deformed_pos=   self.PosAtoms[layer][i]
        
        return np.array([deformed_pos[0] - orig_pos[0], deformed_pos[1] - orig_pos[1]])
        
    
    
    def fx(self, xset, c0, c2, c3, c4, c5, a2, a3, a4, a5):
        # Fit two polynoms t data set: first to first half and second to last half
        # mats somehow in the midlle. => give posiibility for kinking.

        yset        =   np.zeros(len(xset))
        xBeg        =   xset[:self.lastOfFirst]
        xEnd        =   xset[self.lastOfFirst:]
        csBeg       =   np.array([c0, c2, c3, c4, c5])
        y, dy, ddy  =   0., 0., 0.
        
        # Mats the values and derivatives at the intersection =>
        # P_b(xe) = P_e(xe) & P_b'(xe) = P_e'(xe)
        

        x0          =   self.kinkyPoint - np.cos(self.theta/2.)*self.layerUnderStudy*self.InterplaneDistance
        a1          =   2*(c2 - a2)*x0      + 3*(c3 - a3)*x0**2 \
                      + 4*(c4 - a4)*x0**3   + 5*(c5 - a5)*x0**4
                      
        a0          =   c0 - a1*x0 + (c2 - a2)*x0**2 + (c3 - a3)*x0**3 \
                                   + (c4 - a4)*x0**4 + (c5 - a5)*x0**5
                                   
        
        csEnd       =   np.array([a0, a1, a2, a3, a4, a5])
        
        
        if len(xset) != 1:
            for ix, x in enumerate(xBeg): 
                for ic, c in enumerate(csBeg): 
                    # y       =    c0 + c1*x**2 +   c2*x**3 +   c3*x**4 + ...
                    if ic == 0:
                        yset[ix]   +=   c
                    elif ic > 0:
                        yset[ix]   +=   c*x**(ic + 1.)
    
            for ix, x in enumerate(xEnd): 
                for ia, a in enumerate(csEnd): 
                    # y       =    a0 + a1*x + a2*x**2 + a3*x**3 + ...
                    yset[len(xBeg) + ix]   +=   a*x**ia
            
            return yset
    
        
        if len(xset) == 1:
            x   =   xset[0]
            
            if x < x0:
                for ic, c in enumerate([c0, c2, c3, c4, c5]): 
                    if ic == 0:
                        y  +=   c
                    elif ic > 0:
                        y   +=   c*x**(ic + 1.)
                        dy  +=   (ic + 1.)*c*x**ic
                        ddy +=   (ic + 1.)*ic*c*x**(ic - 1)
            elif x > x0:
                for ia, a in enumerate([a0, a1, a2, a3, a4, a5]): 
                    y       +=   a*x**ia
                    dy      +=   ia*a*x**(ia - 1)
                    ddy     +=   ia*(ia - 1)*a*x**(ia - 2)
            
                if self.layerUnderStudy > 0:
                    upLim   =   self.PosAtoms[self.layerUnderStudy - 1][-1][0] \
                              + self.InterplaneDistance*np.sin(self.theta)
                    if x > upLim:
                        yhelp, dyHelp   =   0., 0.
                        for ia, a in enumerate([a0, a1, a2, a3, a4, a5]): 
                            yhelp   +=   a*upLim**ia
                            dyHelp  +=   ia*a*upLim**(ia - 1)
                        
                        b1  =   dyHelp   
                        b0  =   yhelp - upLim*b1
                        
                        y   =   b0 + b1*x
                        dy  =   b1
                        ddy =   0.
                    
            return y, dy, ddy   
        
    
        
    
    """    
    def fx(self, xset, c0, c1, c2, c3, c4, c5):
        
        yset        =   np.zeros(len(xset))
        constants   =   np.array([c0, c1, c2, c3, c4, c5])
        
        for ix, x in enumerate(xset): 
            for ic, c in enumerate(constants): 
                # y       =    c0 + c1*x**2 +   c2*x**3 +   c3*x**4 + ...
                # y'      =       2*c1*x    + 3*c2*x**2 + 4*c3*x**3 + ...
                if ic == 0:
                    yset[ix]   +=   c
                elif ic > 0:
                    yset[ix]   +=   c*x**(ic + 1.)
        
        return yset
    """      
    def CurveFunc(self, layer, x):
        "This is function defining the curve of the bended plate, it needs \
        the polynomial coefficients of the top most plane. All the rest of the layers are set \
        equidistant from each other."
        # TEESST
        """
        print layer, self.PolConsts[layer]
        
        dx      =   0.0001
        y       =   self.fx([x], c0, c2, c3, c4, c5, a2, a3, a4, a5)
        yPdx    =   self.fx([x + dx], c0, c2, c3, c4, c5, a2, a3, a4, a5)
        yP2dx   =   self.fx([x+2*dx], c0, c2, c3, c4, c5, a2, a3, a4, a5)   
        
        dy      =   (yPdx - y)/dx
        ddy     =   ((yP2dx - yPdx)/dx - dy)/dx  
        """
        
        c0, c2, c3, c4, c5, a2, a3, a4, a5 =   self.PolConsts[layer]
        
        self.layerUnderStudy    =   layer
        y,dy, ddy   =   self.fx([x], c0, c2, c3, c4, c5, a2, a3, a4, a5)

        
        # TEESST
        
        """
        y   =   0.
        dy  =   0.
        ddy =   0.
        
        for ic, c in enumerate(self.PolConsts[layer]): 
            # y       =    c0 + c1*x**2 +   c2*x**3 +   c3*x**4 + ...
            # y'      =       2*c1*x    + 3*c2*x**2 + 4*c3*x**3 + ...
            if ic == 0:
                y   +=   c
            elif ic > 0:
                y   +=   c*x**(ic + 1)
                dy  +=   (ic + 1)*c*x**ic
                
                if ic >= 1:
                    ddy +=   (ic + 1)*ic*c*x**(ic - 1)
        """         
        #y, dy       =   0,0
        #for i in range(self.polOrder + 1):
        #    y      +=  self.PolConsts[layer][i]*x**i  
        #    if i   != 0:
        #        dy +=  i*x**(i-1)*self.PolConsts[layer][i]
        
        return y, dy, ddy
    
    
    def xtyt(self, layer, t):
        
        # Here we make the curve and set the virtual atoms on that curve.        
        
        def dev_from_dist(x):
            dev     = (integrate.quad(lambda x: np.sqrt(1 + self.CurveFunc(layer, x)[1]**2), 0, x)[0] - t)**2
            return dev 
        
        x0  =   minimize(dev_from_dist, t)['x'][0]
        
        return (x0, self.CurveFunc(layer, x0)[0])
        
        
        
    
    def set_pol_consts(self, layer):
        
        #self.PolConsts[layer]   =   self.get_layer_consts(layer)
    
        x,y,t           =   np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)
        for i in range(self.n):
            x0, y0      =   self.PosAtoms[layer - 1][i]
            
            fprime      =   self.CurveFunc(layer - 1, x0)[1]
            norm        =   np.sqrt(1 + fprime**2)
            norm_vec    =   [fprime/norm, -1./norm]
            
            if i != 0:
                x[i]    =   x0 + self.InterplaneDistance*norm_vec[0]             
                y[i]    =   y0 + self.InterplaneDistance*norm_vec[1]    
            elif i == 0:
                x[i]    =   0.0             
                y[i]    =   -layer*self.InterplaneDistance
        
            self.helpx[layer - 1]   =   x
            self.helpy[layer - 1]   =   y
        
            
        #layer_consts            =   np.polyfit(x, y, self.polOrder)
        #self.PolConsts[layer]   =   layer_consts[::-1]             
        
        #guess_consts            =   np.ones(self.polOrder)
        
        self.layerUnderStudy    =   layer
        self.PolConsts[layer]    =   curve_fit(self.fx, x, y)[0]
        
        for i in range(self.n):
            t[i]        =   integrate.quad(lambda x: np.sqrt(1 + self.CurveFunc(layer, x)[1]**2), 0, x[i])[0]
            
        self.Ts[layer]  =   t
        
        
        
    def plotAtoms(self, dist = None, e = None, iter_count = None, show = False):
        #PLOT
        def polynom(xs, layer):
            y   =   np.zeros(len(xs))
            for ix, x in enumerate(xs):
                for ip, pol in enumerate(self.PolConsts[layer]):
                    y[ix]   +=  x**ip*pol
            
            return y
        
        
        
        '''
        #
        xset    =   np.linspace(0, self.n*self.lattice_const, 30)
        yset    =   np.zeros(len(xset))
        
        for ix, x in enumerate(xset):
            yset[ix]    =   self.CurveFunc(0, x)[0]
            
        pylab.plot(xset, yset) #, '-', color = 'black')
        pylab.scatter(self.initx, self.inity)
        pylab.scatter(self.kinkyPoint, 0., color='green')
        print x, yset[ix]
        
        print xset
        print yset
        pylab.axis('equal')
        pylab.show()
        #
        '''
        
        for layer in range(self.layers):
            xset    =   np.linspace(0, np.amax(self.get_positions(layer)[0]), 30)
            yset    =   np.zeros(len(xset))
        
            
            for ix, x in enumerate(xset):
                yset[ix]    =   self.CurveFunc(layer, x)[0]

            
            if layer > 0 and dist != None:
                ampf        =   self.InterplaneDistance/(3*np.amax(e))
                epx         =   np.zeros(len(dist[layer - 1]))
                epy         =   np.zeros(len(dist[layer - 1]))
                for ind, d in enumerate(dist[layer - 1]):
                    r       =   np.array(self.xtyt(layer, d))
                    fprime  =   self.get_derivatives(layer, r[0])[0]
                    normal  =   np.array([-fprime, 1]/np.sqrt(fprime**2 + 1))*ampf
                    epx[ind]=   r[0] + normal[0]*(e[layer - 1, ind] + .33/ampf*self.InterplaneDistance)
                    epy[ind]=   r[1] + normal[1]*(e[layer - 1, ind] + .33/ampf*self.InterplaneDistance)
                
                pylab.plot(epx, epy, '-', color = 'red')
            
            x1, y1 = self.get_positions(layer)
            pylab.plot(xset, yset, '--', color = 'black')
            pylab.scatter(x1, y1)
            
            if layer != self.layers - 1:
                pylab.plot(self.helpx[layer], self.helpy[layer], 'D', color = 'green')
            pylab.scatter(self.kinkyPoint, 0.)
        
        pylab.axis('equal')
        if iter_count != None:
            pylab.savefig('/space/tohekorh/SurfAndSlide/pictures/virt_atoms_iter=%i.png' %iter_count)
        if show: pylab.show()
        pylab.clf()

# The point shall be to optimize the poly constants and distanses in each layer.


