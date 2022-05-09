# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
sys.path.append('../../../../..')

# import pycode.pytorchML.ecopann_master.ecopann.cosmic_params as cosmic_params
# import pycode.pytorchML.mdncoper_master._mdncoper_frozen.mdncoper.mdn as mdn
# import pycode.coplot_master.coplot.plot_contours as plc

import ecopann.cosmic_params as cosmic_params
import mdncoper.mdn as mdn
import coplot.plot_contours as plc

import simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import os
start_time = time.time()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


covariance = False
samples = None

#%% obs data
SNe = np.loadtxt('data/Pantheon_SNe_NoName.txt')
obs_z, obs_mb, err_mb = SNe[:,0], SNe[:,3], SNe[:,4]
pantheon = np.c_[obs_z, obs_mb, err_mb]
if covariance:
    pass
else:
    cov_matrix = None


#%% cosmic model & prior parameter
#fwCDM
param_names = ['w', 'omm', 'muc']
init_params = np.array([[-3, 2], [0, 1], [22, 25]])# biased params
model = simulator.Simulate_mb(obs_z, model_type='fwCDM')


#%%
if covariance:
    pass
else:
    chain_mcmc = np.load('data/chain_fwCDM_3params.npy')

init_chain = None


#%%
stepStop_n = 1

num_train = 1000

spaceSigma = 5


space_type = 'hypercube'


comp_type = 'Gaussian'


comp_n = 3
print('comp_n: %s'%comp_n)


epoch = 1000
print('epoch: %s'%epoch)


hidden_layer = 3
print('hidden_layer: %s'%hidden_layer)


activation_func = 'rrelu'
print('activation_func: %s'%activation_func)


norm_type = 'z_score'
print('norm_type: %s'%norm_type)


noise_type = 'singleNormal'; factor_sigma = 1
print('noise_type/factor_sigma: %s/%s'%(noise_type, factor_sigma))


scale_spectra = False; norm_inputs = True #
print('scale_spectra/norm_inputs: %s/%s'%(scale_spectra,norm_inputs))


scale_params = True; independent_norm = True ##
print('scale_params/independent_norm: %s/%s'%(scale_params, independent_norm))

predictor = mdn.MDN(pantheon, model, param_names, params_dict=None,
                    cov_matrix=cov_matrix, init_chain=init_chain, init_params_space=init_params, 
                    comp_type=comp_type, comp_n=comp_n, hidden_layer=hidden_layer, epoch=epoch, num_train=num_train, 
                    spaceSigma=spaceSigma, space_type=space_type, local_samples=samples, stepStop_n=stepStop_n)

predictor.auto_epoch = False
predictor.auto_batchSize = False
predictor.batch_size = 1250
predictor.noise_type = noise_type
predictor.factor_sigma = factor_sigma
predictor.norm_type = norm_type
predictor.multi_noise = 5
predictor.activation_func = activation_func
predictor.scale_spectra = scale_spectra
predictor.scale_params = scale_params
predictor.norm_inputs = norm_inputs
predictor.independent_norm = independent_norm
predictor.print_info = True

predictor.train(path='net_pantheon_steps', save_items=False, showIter_n=100)


predictor.plot_steps()
# predictor.plot_contours(fill_contours=False, show_titles=True)
# predictor.save_steps()
# predictor.save_contours()


#% MCMC chain
chain_mdn = predictor.chain_mdn

chain_all = [chain_mdn, chain_mcmc]
plc.Contours(chain_all).plot(labels=cosmic_params.ParamsProperty(param_names).labels,smooth=5,
                              fill_contours=False,show_titles=True,line_width=2,layout_adjust=[0.0,0.0],
                              lims=None,legend=True,legend_labels=['MDN', 'MCMC'])


#%%
print("\nTime elapsed: %.3f mins"%((time.time()-start_time)/60))
plt.show()

