# -*- coding: utf-8 -*-
import ecopann.data_simulator as ds
import ecopann.space_updater as su
import ecopann.ann as ann

from . import models
import numpy as np


class MDN(ann.ANN):
    def __init__(self, obs_data, model, param_names, params_dict=None, cov_matrix=None, init_chain=None, init_params_space=None,
                 comp_type='Gaussian', comp_n=3, hidden_layer=3, branch_hiddenLayer=2, trunk_hiddenLayer=1, epoch=2000, epoch_branch=2000,
                 num_train=3000, num_vali=500, spaceSigma=5, space_type='hypercube', local_samples=None, stepStop_n=3):
        #observational data & cosmological model
        self.obs_data = obs_data
        self.model = model
        self.param_names = param_names
        self.params_dict = params_dict
        self.cov_matrix = cov_matrix
        self.init_chain = init_chain
        self.init_params_space = self._init_params_space(init_params_space)
        #MDN model
        self.comp_type = comp_type
        self.comp_n = comp_n
        self.hidden_layer = hidden_layer
        self.activation_func = 'softplus'
        self.branch_n = self._branch_n()
        self.branch_hiddenLayer = branch_hiddenLayer
        self.trunk_hiddenLayer = trunk_hiddenLayer
        self.lr = 1e-2
        self.lr_min = 1e-8
        self.batch_size = 750
        self.auto_batchSize = True
        self.epoch = epoch
        self.epoch_branch = epoch_branch
        self.auto_epoch = True
        #training set
        self.num_train = num_train
        self.num_vali = num_vali
        self.spaceSigma = spaceSigma
        self.space_type = space_type
        self.base_N_max = 1500
        self.auto_N = True
        self.local_samples = local_samples
        #data preprocessing
        self.noise_type = 'singleNormal'
        self.factor_sigma = 1
        self.multi_noise = 5
        self.scale_spectra = True
        self.scale_params = True
        self.norm_inputs = True
        self.norm_target = True
        self.independent_norm = False
        self.norm_type = 'z_score'
        #training
        self.set_numpySeed = False #remove?
        self.set_torchSeed = False #remove?
        self.train_branch = False
        self.repeat_n = 3
        #updating
        self.stepStop_n = stepStop_n
        self.expectedBurnIn_step = 10
        self.chain_leng = 10000
    
    #test, using self.space_type before burn-in, to be removed and using the one in ann.py?
    def simulate(self, step=1, burn_in=False, burnIn_step=None, space_type_all=[], prev_space=None,
                 chain_all=[], sim_spectra=None, sim_params=None):
        """Simulate data and update parameter space.
        """
        if step==1:
            # set training number
            training_n = self.base_N
            # set space_type
            if self.init_chain is None:
                if self.space_type=='hypersphere' or self.space_type=='hyperellipsoid' or self.space_type=='posterior_hyperellipsoid':
                    s_type = 'hypercube'
                    # s_type = 'LHS' #test
                else:
                    s_type = self.space_type
            else:
                s_type = self.space_type
            space_type_all.append(s_type)
            print('\n'+'#'*25+' step {} '.format(step)+'#'*25)
            if self.branch_n==1:
                simor = ds.SimSpectra(training_n, self.model, self.param_names, chain=self.init_chain, params_space=self.init_params_space, 
                                      spaceSigma=self.spaceSigma, params_dict=self.params_dict, space_type=s_type, 
                                      cut_crossedLimit=True, local_samples=self.local_samples, prevStep_data=None)
            else:
                simor = ds.SimMultiSpectra(self.branch_n, training_n, self.model, self.param_names, chain=self.init_chain, params_space=self.init_params_space, 
                                           spaceSigma=self.spaceSigma, params_dict=self.params_dict, space_type=s_type, 
                                           cut_crossedLimit=True, local_samples=self.local_samples, prevStep_data=None)
            sim_spectra, sim_params = simor.simulate()
            prev_space = simor.params_space #used for next step
        else:
            if step==2:
                chain_0 = self.init_chain
            elif step>=3:
                chain_0 = chain_all[-2]
            updater = su.UpdateParameterSpace(step,self.param_names,chain_all[-1],chain_0=chain_0,init_params_space=self.init_params_space,spaceSigma=self.spaceSigma,params_dict=self.params_dict)
            if updater.small_dev(limit_dev=0.001):
                #to be improved to get chain_ann after exit()???, or remove these two lines???
                exit()
            #this is based on experiments, update this??? eg. max(updater.param_devs)<0.5?
            # if burnIn_step is None and max(updater.param_devs)<1 and max(updater.error_devs)<0.5:
            # if burnIn_step is None and max(updater.param_devs)<0.5 and max(updater.error_devs)<0.5: #test!!!
            if burnIn_step is None and max(updater.param_devs)<=0.25 and max(updater.error_devs)<=0.25: #test!!!
                burn_in = True
                burnIn_step = step - 1 #let good chain contain burn-in step chain, should be step-2? the good chain will not contain burn-in step chain!
                print('\n\n'+'='*73)
                if burnIn_step>=10:
                    print('*'*5+' '*24+'Burn-In step: %s'%(burnIn_step)+' '*23+'*'*5)
                else:
                    print('*'*5+' '*24+'Burn-In step: %s'%(burnIn_step)+' '*24+'*'*5)
                print('*'*5+' '*11+'The parameters have reached stable values'+' '*11+'*'*5)
                print('*'*5+' '*1+'The chains of later steps can be used for parameter inference'+' '*1+'*'*5)
                print('='*73+'\n')
            
            if burn_in:
                print('\n'+'#'*25+' step {}/{} '.format(step, burnIn_step+self.stepStop_n)+'#'*25)
            else:
                print('\n'+'#'*25+' step {} '.format(step)+'#'*25)
            self.spaceSigma_min = updater.spaceSigma_min
            updater.print_learningRange()
            
            # set training number, should this (spaceSigma_max>=10) be updated??? 
            # if not burn_in and updater.spaceSigma_max>10:
            #     training_n = self.base_N
            # else:
            #     training_n = self.num_train
            #test
            if burn_in:
                training_n = self.num_train + self.num_vali
            else:
                if max(updater.param_devs)<=0.5 and max(updater.error_devs)<=0.25:
                    training_n = self.num_train
                else:
                    training_n = self.base_N
                    
            # set space_type
            if burn_in:
                s_type = self.space_type
            else:
                # s_type = 'hypercube'
                s_type = self.space_type #test!!!
                # # s_type = 'LHS' #test !!!
                
                # # if max(updater.param_devs)<1 and max(updater.error_devs)<0.5: #test
                # if max(updater.param_devs)<0.5 and max(updater.error_devs)<0.5: #test
                #     s_type = self.space_type #test!!!
                # else:
                #     s_type = 'hypercube' #test
                    
            space_type_all.append(s_type)
            if space_type_all[-1]==space_type_all[-2]:
                prevStep_data = [sim_spectra, sim_params]
            else:
                prevStep_data = None
            # #test
            # if burn_in:
            #     prevStep_data = None ##test!!!
            # else:
            #     prevStep_data = [sim_spectra, sim_params] #test !!!
            
            rel_dev_limit = 0.1 #0.1 #test
            
            cut_crossedLimit = True
            # cut_crossedLimit = False #test !!!
            
            
            # check whether it has problems when using previous_data???
            if self.branch_n==1:
                simor = ds.SimSpectra(training_n, self.model, self.param_names, chain=chain_all[-1], params_space=None, spaceSigma=updater.spaceSigma_all,
                                      params_dict=self.params_dict, space_type=s_type, cut_crossedLimit=cut_crossedLimit, local_samples=None, prevStep_data=prevStep_data, rel_dev_limit=rel_dev_limit) #reset local_samples & previous_data???
            else:
                simor = ds.SimMultiSpectra(self.branch_n, training_n, self.model, self.param_names, chain=chain_all[-1], params_space=None, spaceSigma=updater.spaceSigma_all,
                                           params_dict=self.params_dict, space_type=s_type, cut_crossedLimit=cut_crossedLimit, local_samples=None, prevStep_data=prevStep_data, rel_dev_limit=rel_dev_limit) #reset local_samples & previous_data???
            simor.prev_space = prev_space
            sim_spectra, sim_params = simor.simulate()
            prev_space = simor.params_space #used for next step
        
        
        #test, to be added to the code???
        # good_index = np.where(~np.isnan(sim_spectra[:,0])) #test
        # sim_spectra = sim_spectra[good_index] #test
        # sim_params = sim_params[good_index] #test
        return sim_spectra, sim_params, burn_in, burnIn_step, space_type_all, prev_space
    
    def _train(self, sim_spectra, sim_params, step=1, burn_in=False, burnIn_step=None, 
               randn_num=0.123, sample=None, save_items=True, 
               showIter_n=100):
        if burn_in:
            idx = np.random.choice(self.num_train+self.num_vali, self.num_train+self.num_vali, replace=False)
            if self.branch_n==1:
                train_set = [sim_spectra[idx[:self.num_train]], sim_params[idx[:self.num_train]]]
                vali_set = [sim_spectra[idx[self.num_train:]], sim_params[idx[self.num_train:]]]
            else:
                sim_spectra_train = [sim_spectra[i][idx[:self.num_train]] for i in range(self.branch_n)]
                sim_params_train = sim_params[idx[:self.num_train]]
                sim_spectra_vali = [sim_spectra[i][idx[self.num_train:]] for i in range(self.branch_n)]
                sim_params_vali = sim_params[idx[self.num_train:]]
                train_set = [sim_spectra_train, sim_params_train]
                vali_set = [sim_spectra_vali, sim_params_vali]
        else:
            train_set = [sim_spectra, sim_params]
            vali_set = [None, None]
        if self.branch_n==1:
            self.eco = models.OneBranchMDN(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, cov_matrix=self.cov_copy, params_dict=self.params_dict,
                                           comp_type=self.comp_type, comp_n=self.comp_n, hidden_layer=self.hidden_layer, activation_func=self.activation_func,
                                           noise_type=self.noise_type, factor_sigma=self.factor_sigma, multi_noise=self.multi_noise)
        else:
            self.eco = models.MultiBranchMDN(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, cov_matrix=self.cov_copy, params_dict=self.params_dict,
                                             comp_type=self.comp_type, comp_n=self.comp_n, branch_hiddenLayer=self.branch_hiddenLayer, trunk_hiddenLayer=self.trunk_hiddenLayer, activation_func=self.activation_func,
                                             noise_type=self.noise_type, factor_sigma=self.factor_sigma, multi_noise=self.multi_noise)
        self.eco.lr = self.lr
        self.eco.lr_min = self.lr_min
        self.eco.batch_size = self.batch_size
        self.eco.auto_batchSize = self.auto_batchSize
        self.eco.epoch = self.epoch
        self.eco.base_epoch = self.base_epoch
        self.eco.auto_epoch = self.auto_epoch
        if step==1:
            self.eco.print_info = True
        self.eco.scale_spectra = self.scale_spectra
        self.eco.scale_params = self.scale_params
        self.eco.norm_inputs = self.norm_inputs
        self.eco.norm_target = self.norm_target
        self.eco.independent_norm = self.independent_norm
        self.eco.norm_type = self.norm_type
        if step>=2:
            self.eco.spaceSigma_min = self.spaceSigma_min
        self.eco.auto_repeat_n = False
        self.eco.burn_in = burn_in
        self.eco.burnIn_step = burnIn_step
        self.eco.transfer_learning = False
        self.eco.randn_num = randn_num
        
        if self.branch_n==1:
            self.eco.train(repeat_n=self.repeat_n, showIter_n=showIter_n)
        else:
            self.eco.lr_branch = self.lr
            self.eco.epoch_branch = self.epoch_branch
            self.eco.train(repeat_n=self.repeat_n, train_branch=self.train_branch, parallel=False, showIter_n=showIter_n) #reset parallel???
        
        #predict chain
        #Note: here use self.cov_copy is to avoid data type error in "eco"
        chain_1 = self.eco.predict_chain(self.obs_data, chain_leng=self.chain_leng)
        
        #save results
        if save_items:
            sample_i = '%s_step%s'%(sample, step) if sample is not None else None
            self.eco.save_net(path=self.path, sample=sample_i)
            self.eco.save_loss(path=self.path, sample=sample_i)
            self.eco.save_chain(path=self.path, sample=sample_i)            
            self.eco.save_hparams(path=self.path, sample=sample_i)
        return chain_1

