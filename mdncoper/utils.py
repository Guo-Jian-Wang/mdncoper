# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas


def remove_nan(spectra, params):
    """Remove the 'nan' in the numpy array, used for the simulated spectra.

    Parameters
    ----------
    spectra : array-like
        The simulated spectra, Numpy array with one or multi dimension.

    params : array-like
        The simulated parameters, Numpy array with one or multi dimension.
        
    Returns
    -------
    spectra_new : array-like
        The new spectra that do not contain nan.

    params_new : array-like
        The new parameters that do not contain nan.
        
    """
    idx_nan = np.where(np.isnan(spectra))[0]
    if len(idx_nan)==0:
        print("There are no 'nan' in the mock data.")
        return spectra, params
    idx_good = np.where(~np.isnan(spectra))[0]
    idx_nan = np.unique(idx_nan)
    idx_good = np.unique(idx_good)
    
    idx_nan_pandas = pandas.Index(idx_nan)
    idx_good_pandas = pandas.Index(idx_good)
    idx_good_pandas = idx_good_pandas.difference(idx_nan_pandas, sort=False)
    idx_good = idx_good_pandas.to_numpy()
    
    spectra_new = spectra[idx_good]
    params_new = params[idx_good]
    return spectra_new, params_new

