# -*- coding: utf-8 -*-
import numpy as np
from scipy import integrate
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)


class Simulate_mb(object):
    def __init__(self, z, model_type='fwCDM'):
        self.z = z
        self.model_type = model_type
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
        dl_equ = dl*70. / self.c
        return 5*np.log10(dl_equ) + mu0
        
    def simulate(self, sim_params):
        if self.model_type=='fLCDM':
            return self.z, self.fLCDM_mb(sim_params)
        elif self.model_type=='fwCDM':
            return self.z, self.fwCDM_mb(sim_params)

    def load_params(self, local_sample):
        return np.load('../../sim_data/'+local_sample+'/parameters.npy')

    def load_params_space(self, local_sample):
        return np.load('../../sim_data/'+local_sample+'/params_space.npy')
    
    #new
    def load_sample(self, local_sample):
        return np.load('../../sim_data/'+local_sample+'/y.npy')

