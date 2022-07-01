# -*- coding: utf-8 -*-

# import sys
# sys.path.append('..')
# sys.path.append('../../../..')
# import pycode.pytorchML.ecopann_master.ecopann.cosmic_params as cosmic_params
# import pycode.pytorchML.ecopann_master.ecopann.utils as utils
# import pycode.pytorchML.mdncoper_master.mdncoper.mdn as mdn
# import pycode.coplot_master.coplot.plot_contours as plc

import ecopann.cosmic_params as cosmic_params
import ecopann.utils as utils
import mdncoper.mdn as mdn
import coplot.plot_contours as plc

import simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import os
start_time = time.time()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# covariance = True
covariance = False


#%% local data sets
samples = None

#%% obs data
SNe = np.loadtxt('data/Pantheon_SNe_NoName.txt')
obs_z, obs_mb, err_mb = SNe[:,0], SNe[:,3], SNe[:,4]
pantheon = np.c_[obs_z, obs_mb, err_mb]
if covariance:
    err_sys = np.loadtxt('data/Pantheon_Systematic_Error_Matrix.txt')
    err_sys_matrix = np.matrix(err_sys).reshape(1048,1048)
    err_stat_matrix = np.matrix(np.diag(err_mb**2))
    cov_matrix = err_stat_matrix + err_sys_matrix
else:
    cov_matrix = None


#%% cosmic model & initial parameters
param_names = ['w', 'omm', 'muc']
init_params = np.array([[-3, 2], [0, 1], [22, 25]])

model = simulator.Simulate_mb(obs_z)


#%%
if covariance:
    chain_mcmc = np.load('data/chain_fwCDM_3params_cov.npy')
else:
    chain_mcmc = np.load('data/chain_fwCDM_3params.npy')


# init_chain = chain_mcmc
init_chain = None


#%%
#the number of chains
stepStop_n = 3 

#the number of samples in the training set
num_train = 3000

#the number of samples in the validation set
num_vali = 500

spaceSigma = 5

space_type = 'hypercube'

comp_type = 'Gaussian'


comp_n = 3
# print('comp_n: %s'%comp_n)


epoch = 2000
# print('epoch: %s'%epoch)


hidden_layer = 3 #0, 1, 2, 3, 4, 5 ---> 1
# print('hidden_layer: %s'%hidden_layer)


# activation_func = 'relu'
# activation_func = 'elu'
activation_func = 'rrelu'
# activation_func = 'prelu'
# activation_func = 'softplus' # with minmax ? ---> softplus
# activation_func = 'sigmoid'
# activation_func = 'softsign'
# activation_func = 'tanh'
# activation_func = 'logsigmoid'
print('activation_func: %s'%activation_func)


multi_noise = 5
# print('multi_noise: %s'%multi_noise)


norm_type = 'z_score'
# print('norm_type: %s'%norm_type)


# scale_spectra = True; norm_inputs = True 
# scale_spectra = True; norm_inputs = False
scale_spectra = False; norm_inputs = True #
# scale_spectra = False; norm_inputs = False
print('scale_spectra/norm_inputs: %s/%s'%(scale_spectra,norm_inputs))


scale_params = True; independent_norm = True ##
# scale_params = True; independent_norm = False 
# scale_params = False; independent_norm = True #
# scale_params = False; independent_norm = False
print('scale_params/independent_norm: %s/%s'%(scale_params, independent_norm))

predictor = mdn.MDN(pantheon, model, param_names, params_dict=simulator.params_dict(),
                    cov_matrix=cov_matrix, init_chain=init_chain, init_params_space=init_params, 
                    comp_type=comp_type, comp_n=comp_n, hidden_layer=hidden_layer, epoch=epoch, 
                    num_train=num_train, num_vali=num_vali, spaceSigma=spaceSigma, space_type=space_type, 
                    local_samples=samples, stepStop_n=stepStop_n)

predictor.auto_epoch = False
predictor.auto_batchSize = False
predictor.batch_size = 1250
predictor.norm_type = norm_type
predictor.multi_noise = multi_noise
predictor.activation_func = activation_func
predictor.scale_spectra = scale_spectra
predictor.scale_params = scale_params
predictor.norm_inputs = norm_inputs
predictor.independent_norm = independent_norm
predictor.print_info = True

predictor.train(path='net_pantheon_steps', save_items=True, showIter_n=100)


#% chain
chain_ann = predictor.chain_ann
utils.savenpy('net_pantheon_steps/chain_ann', 'chain_%s'%predictor.randn_num, chain_ann)


predictor.eco.plot_loss()

chain_all = [chain_ann, chain_mcmc]
plc.Contours(chain_all).plot(labels=cosmic_params.ParamsProperty(param_names).labels,smooth=5,
                              fill_contours=True,show_titles=True,line_width=2,layout_adjust=[0.0,0.0],
                              lims=None,legend=True,legend_labels=['MDN', 'MCMC'])


#%%
print("\nTime elapsed: %.3f mins"%((time.time()-start_time)/60))
plt.show()

