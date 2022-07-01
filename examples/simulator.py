# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)



def params_dict():
    """Information of cosmological parameters that include the labels and physical limits: [label, limit_min, limit_max]
    
    The label is used to plot figures. The physical limits are used to ensure that the simulated parameters have physical meaning.
    
    Note
    ----
    If the physical limits of parameters is unknown or there is no physical limits, it should be set to np.nan.
    """
    return {'omm'     : [r'$\Omega_{\rm m}$', 0.0, 1.0], #the matter density parameter
            'w'       : [r'$w$', np.nan, np.nan], #parameter of wCDM model            
            'muc'     : [r'$\mu_c$', np.nan, np.nan], #5*log10(c/H0/Mpc) + MB + 25
            }


class Simulate_mb(object):
    def __init__(self, z):
        self.z = z
        self.c = 2.99792458e5
    
    def fwCDM_E(self, x, w, omm):
        return 1./np.sqrt( omm*(1+x)**3 + (1-omm)*(1+x)**(3*(1+w)) )
    
    def fwCDM_dl(self, z, w, omm, H0=70):
        def dl_i(z_i, w, omm, H0):
            dll = integrate.quad(self.fwCDM_E, 0, z_i, args=(w, omm))[0]
            dl_i = (1+z_i)*self.c *dll/H0
            return dl_i
        dl = np.vectorize(dl_i)(z, w, omm, H0)
        return dl
    
    def fwCDM_mb(self, params):
        w, omm, mu0 = params
        dl = self.fwCDM_dl(self.z, w, omm, H0=70)
        dl_equ = dl*70. / (self.c/1.0e3)
        return 5*np.log10(dl_equ) + mu0
        
    def simulate(self, sim_params):
        return self.z, self.fwCDM_mb(sim_params)
    
    # def load_params(self, local_sample):
    #     return np.load('../../sim_data/'+local_sample+'/parameters.npy')

    # def load_params_space(self, local_sample):
    #     return np.load('../../sim_data/'+local_sample+'/params_space.npy')
    
    # def load_sample(self, local_sample):
    #     return np.load('../../sim_data/'+local_sample+'/y.npy')

