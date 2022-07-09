# -*- coding: utf-8 -*-
from ecopann.models import OneBranchMLP, MultiBranchMLP, LoadNet, LoadLoss, LoadChain
import ecopann.data_processor as dp
import ecopann.data_simulator as ds
import ecopann.optimize as optimize
import ecopann.evaluate as evaluate
import ecopann.utils as utils
import ecopann.cosmic_params as cosmic_params

from . import fcnet
import numpy as np
import torch
from torch.autograd import Variable


class OneBranchMDN(OneBranchMLP):
    def __init__(self, train_set, param_names, vali_set=[None,None], obs_errors=None, cov_matrix=None, params_dict=None,
                 comp_type='Gaussian', comp_n=3, hidden_layer=3, activation_func='softplus',
                 noise_type='singleNormal', factor_sigma=1, multi_noise=5, sigma_scale_factor=1.0):
        #data
        self.spectra, self.params = train_set
        self.spectra_base = np.mean(self.spectra, axis=0)
        self.params_base = np.mean(self.params, axis=0)
        self.param_names = param_names
        self.params_n = len(param_names)
        self.spectra_vali, self.params_vali = vali_set
        self.obs_errors = obs_errors
        self.cholesky_factor = self._cholesky_factor(cov_matrix)
        self.params_dict = params_dict
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict)
        self.params_limit = p_property.params_limit
        #MDN model
        self.comp_type = comp_type
        self.comp_n = comp_n
        self.hidden_layer = hidden_layer
        self.activation_func = activation_func
        self.lr = 1e-2
        self.lr_min = 1e-8
        self.batch_size = 750
        self.auto_batchSize = True
        self.epoch = 2000
        self.base_epoch = 1000
        self.auto_epoch = True
        self.fix_initialize = False
        self.print_info = False
        #data preprocessing
        self.noise_type = noise_type
        self.factor_sigma = factor_sigma
        self.multi_noise = multi_noise
        self.scale_spectra = True
        self.scale_params = True
        self.norm_inputs = True
        self.norm_target = True
        self.independent_norm = False
        self.norm_type = 'z_score'
        #training
        self.spaceSigma_min = 5
        self.auto_repeat_n = False
        self.burn_in = False
        self.burnIn_step = None
        self.transfer_learning = False
        self.randn_num = round(abs(np.random.randn()/5.), 5)
    
    def _net(self):
        if self.fix_initialize:
            torch.manual_seed(1000) #Fixed parameter initialization
        self.node_in = self.spectra.shape[1]
        self.node_out = self.params.shape[1]
        if self.comp_type=='Gaussian':
            if self.params_n==1:
                self.net = fcnet.GaussianMDN(node_in=self.node_in, node_out=self.node_out, hidden_layer=self.hidden_layer,
                                             comp_n=self.comp_n, nodes=None, activation_func=self.activation_func)
            else:
                self.net = fcnet.MultivariateGaussianMDN(node_in=self.node_in, node_out=self.node_out, hidden_layer=self.hidden_layer,
                                                          comp_n=self.comp_n, nodes=None, activation_func=self.activation_func)
                # print('test __net, ??????')
                # self.net = fcnet.GaussianMDN(node_in=self.node_in, node_out=self.node_out, hidden_layer=self.hidden_layer,
                #                               comp_n=self.comp_n, nodes=None, activation_func=self.activation_func)
            
        elif self.comp_type=='Beta':
            if self.params_n==1:
                self.net = fcnet.BetaMDN(node_in=self.node_in, node_out=self.node_out, hidden_layer=self.hidden_layer,
                                         comp_n=self.comp_n, nodes=None, activation_func=self.activation_func)
            else:
                #test, should be change to MultivariateBeta?
                self.net = fcnet.BetaMDN(node_in=self.node_in, node_out=self.node_out, hidden_layer=self.hidden_layer,
                                         comp_n=self.comp_n, nodes=None, activation_func=self.activation_func)
        elif self.comp_type=='Kum':
            if self.params_n==1:
                self.net = fcnet.KumMDN(node_in=self.node_in, node_out=self.node_out, hidden_layer=self.hidden_layer,
                                        comp_n=self.comp_n, nodes=None, activation_func=self.activation_func)
            else:
                pass
        if self.print_info:
            print(self.net)

    @property
    def loss_func(self):
        if self.comp_type=='Gaussian':
            if self.params_n==1:
                return fcnet.gaussian_loss
            else:
                return fcnet.multivariateGaussian_loss
                # print('test __net, ??????')
                # return fcnet.gaussian_loss
        elif self.comp_type=='Beta':
            if self.params_n==1:
                return fcnet.beta_loss
            else:
                return fcnet.beta_loss #test, should change to multivariateBeta?
        elif self.comp_type=='Kum':
            if self.params_n==1:
                return fcnet.kum_loss
            else:
                pass
            
    def check_normType(self):
        if self.comp_type=='Gaussian':
            self.a = 0
            self.b = 1
        elif self.comp_type=='Beta' or self.comp_type=='Kum':
            self.a = 1e-6 #1e-6
            self.b = 0.999999 #0.999999
            if not self.norm_target:
                self.norm_target = True
                print("'norm_target' is set to True because of using '%s' components."%self.comp_type)
            if self.norm_type!='minmax':
                self.norm_type = 'minmax'
                print("'norm_type' is set to 'minmax' because of using '%s' components."%self.comp_type)
                
    def train(self, repeat_n=3, showIter_n=100):
        self._net()
        if self.transfer_learning:
            self.copyLayers_fromTrainedNet()
        self.transfer_net()
        
        self.optimizer = self._optimizer(name='Adam')
        
        if self.auto_batchSize:
            self._auto_batchSize()
        self._check_batchSize()
        # print('batch size: %s'%self.batch_size)
        if self.auto_epoch:
            self._auto_epoch()
        if self.auto_repeat_n:
            repeat_n = self._auto_repeat_n(repeat_n)
        self.iteration = self.multi_noise*len(self.spectra)//self.batch_size * repeat_n
        
        self.statistic()
        self.transfer_data()
        self.check_normType()
                
        self.train_loss = []
        self.vali_loss = []
        # np.random.seed(1000)#
        print('randn_num: %s'%self.randn_num)
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.spectra,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            self.inputs = self.preprocessing_input(self.inputs, self.spectra_base_torch, a=self.a, b=self.b)
            self.target = self.preprocessing_target(self.target, a=self.a, b=self.b)
            
            running_loss = []
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.inputs), self.batch_size, replace=False)
                # batch_index = np.random.choice(len(self.inputs), self.batch_size, replace=True) #test
                xx = self.inputs[batch_index]
                yy = self.target[batch_index]
                xx = Variable(xx)
                yy = Variable(yy, requires_grad=False)
                self.omega, self.mu_alpha, self.sigma_cholesky_beta = self.net(xx)
                _loss = self.loss_func(self.omega, self.mu_alpha, self.sigma_cholesky_beta, yy, logsumexp=True)
                # self.loss.append(_loss.item())
                self.net.zero_grad()
                _loss.backward()
                self.optimizer.step()
                running_loss.append(_loss.item())
            loss_mean = np.mean(running_loss)
            self.train_loss.append(loss_mean)
            
            #vali_loss
            if self.spectra_vali is not None:
                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.spectra_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
                self.inputs_vali = self.preprocessing_input(self.inputs_vali, self.spectra_base_torch, a=self.a, b=self.b)
                self.target_vali = self.preprocessing_target(self.target_vali, a=self.a, b=self.b)
                self.net.eval()
                omega_vali, mu_alpha_vali, sigma_cholesky_beta_vali = self.net(Variable(self.inputs_vali))
                _vali_loss = self.loss_func(omega_vali, mu_alpha_vali, sigma_cholesky_beta_vali, Variable(self.target_vali, requires_grad=False), logsumexp=True)
                self.vali_loss.append(_vali_loss.item())
                self.net.train()
            
            if subsample_num%showIter_n==0:
                if self.spectra_vali is None:
                    print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.optimizer.param_groups[0]['lr']))
                else:
                    print('(epoch:%s/%s; train_loss/vali_loss:%.5f/%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.vali_loss[-1], self.optimizer.param_groups[0]['lr']))
            lrdc = optimize.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[0]['lr'] = lrdc.exp()
            
        if self.use_multiGPU:
            self.net = self.net.module.cpu()
        else:
            self.net = self.net.cpu()
        self.train_loss = np.array(self.train_loss)
        self.vali_loss = np.array(self.vali_loss)
        return self.net, self.train_loss, self.vali_loss
    
    @property
    def sampler(self):
        if self.comp_type=='Gaussian':
            if self.params_n==1:
                return fcnet.gaussian_sampler
            else:
                return fcnet.multivariateGaussian_sampler
                # print('test __net, ??????')
                # return fcnet.gaussian_sampler
        elif self.comp_type=='Beta':
            if self.params_n==1:
                return fcnet.beta_sampler
            else:
                return fcnet.beta_sampler #test, should change to multivariateBeta?
        elif self.comp_type=='Kum':
            if self.params_n==1:
                return fcnet.kum_sampler
            else:
                pass

    def _predict(self, spectra, use_GPU=False, in_type='numpy'):
        """Make predictions using a well-trained network.
        
        Parameters
        ----------
        spectra : numpy array or torch tensor
            The inputs of the network.
        use_GPU : bool
            If True, calculate using GPU, otherwise, calculate using CPU.
        in_type : str
            The data type of the spectra, 'numpy' or 'torch'.
        """
        if use_GPU:
            self.net = self.net.cuda()
            if in_type=='numpy':
                spectra = dp.numpy2cuda(spectra)
            elif in_type=='torch':
                spectra = dp.torch2cuda(spectra)
        else:
            if in_type=='numpy':
                spectra = dp.numpy2torch(spectra)
        self.net = self.net.eval() #this works for the batch normalization layers
        self.omega, self.mu_alpha, self.sigma_cholesky_beta = self.net(Variable(spectra))
        pred = self.sampler(self.omega, self.mu_alpha, self.sigma_cholesky_beta)
        if use_GPU:
            pred = dp.cuda2numpy(pred.data)
        else:
            pred = dp.torch2numpy(pred.data)
        #reshape chain
        if len(pred.shape)==1:
            pred = pred.reshape(-1, 1)
        return pred

    def predict_chain(self, obs_spectra, chain_leng=10000):
        # limit parameters using params_limit???, do not set limit here, set limit in chain_ann
        # torch.manual_seed(1000)#
        obs_spectra = dp.numpy2torch(obs_spectra)
        obs_best = obs_spectra[:,1]
        self.obs_best_multi = torch.ones((chain_leng, len(obs_best))) * obs_best
        self.obs_best_multi = self.preprocessing_input(self.obs_best_multi, dp.numpy2torch(self.spectra_base), a=self.a, b=self.b)
        self.chain = self._predict(self.obs_best_multi, in_type='torch')
        self.chain = self.preprocessing_target_inv(self.chain, a=self.a, b=self.b)
        self.chain = self.cut_params(self.chain) #remove non-physical parameters
        # self.scale_chain() #test
        return self.chain
    
    def save_net(self, path='mdn', sample=None):
        if sample is None:
            fileName = 'net_train%s_batch%s_epoch%s_comp%s_%s.pt'%(len(self.params),self.batch_size,self.epoch,self.comp_n,self.randn_num)
        else:
            fileName = 'net-%s_train%s_batch%s_epoch%s_comp%s_%s.pt'%(sample,len(self.params),self.batch_size,self.epoch,self.comp_n,self.randn_num)
        utils.saveTorchPt(path+'/net', fileName, self.net)
    
    def save_loss(self, path='mdn', sample=None):
        if sample is None:
            fileName = 'loss_train%s_batch%s_epoch%s_comp%s_%s'%(len(self.params),self.batch_size,self.epoch,self.comp_n,self.randn_num)
        else:
            fileName = 'loss-%s_train%s_batch%s_epoch%s_comp%s_%s'%(sample,len(self.params),self.batch_size,self.epoch,self.comp_n,self.randn_num)
        utils.savenpy(path+'/loss', fileName, [self.train_loss, self.vali_loss])
    
    def save_chain(self, path='mdn', sample=None):
        if sample is None:
            fileName = 'chain_train%s_batch%s_epoch%s_comp%s_%s'%(len(self.params),self.batch_size,self.epoch,self.comp_n,self.randn_num)
        else:
            fileName = 'chain-%s_train%s_batch%s_epoch%s_comp%s_%s'%(sample,len(self.params),self.batch_size,self.epoch,self.comp_n,self.randn_num)
        utils.savenpy(path+'/chains', fileName, self.chain)
    
    def save_hparams(self, path='mdn', sample=None):
        if sample is None:
            fileName = 'hparams_train%s_batch%s_epoch%s_comp%s_%s'%(len(self.params),self.batch_size,self.epoch,self.comp_n,self.randn_num)
        else:
            fileName = 'hparams-%s_train%s_batch%s_epoch%s_comp%s_%s'%(sample,len(self.params),self.batch_size,self.epoch,self.comp_n,self.randn_num)
        utils.savenpy(path+'/hparams', fileName, [self.spectra_statistic, self.params_statistic, self.spectra_base, self.params_base, self.param_names, self.params_dict, self.comp_type,
                                                  self.scale_spectra, self.scale_params, self.norm_inputs, self.norm_target, self.independent_norm, self.norm_type, self.a, self.b, 
                                                  self.burnIn_step])

class LoadHparams(object):
    def __init__(self, path='ann', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)

    def load_hparams(self):
        file_path = evaluate.FilePath(filedir=self.path+'/hparams', randn_num=self.randn_num, suffix='.npy').filePath()
        self.spectra_statistic, self.params_statistic, self.spectra_base, self.params_base, self.param_names, self.params_dict, self.comp_type, self.scale_spectra, self.scale_params, self.norm_inputs, self.norm_target, self.independent_norm, self.norm_type, self.a, self.b, self.burnIn_step = np.load(file_path, allow_pickle=True)
        self.params_n = len(self.param_names)
        p_property = cosmic_params.ParamsProperty(self.param_names, params_dict=self.params_dict)
        self.params_limit = p_property.params_limit

class RePredictOBMDN(OneBranchMDN, LoadNet, LoadLoss, LoadChain, LoadHparams):
    def __init__(self, path='mdn', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)

    
#%%
class MultiBranchMDN(MultiBranchMLP):
    def __init__(self, train_set, param_names, vali_set=[None,None], obs_errors=None, cov_matrix=None, params_dict=None,
                 comp_type='Gaussian', comp_n=3, branch_hiddenLayer=2, trunk_hiddenLayer=2,
                 activation_func='softplus', noise_type='singleNormal', factor_sigma=1, multi_noise=5):
        #data
        self.spectra, self.params = train_set
        self.branch_n = len(self.spectra)
        self.spectra_base = [np.mean(self.spectra[i], axis=0) for i in range(self.branch_n)]
        self.params_base = np.mean(self.params, axis=0)
        self.param_names = param_names
        self.params_n = len(param_names)
        self.spectra_vali, self.params_vali = vali_set
        self.obs_errors = self._obs_errors(obs_errors)
        self.cholesky_factor = self._cholesky_factor(cov_matrix)
        self.params_dict = params_dict
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict)
        self.params_limit = p_property.params_limit
        #MDN model
        self.comp_type = comp_type
        self.comp_n = comp_n
        self.branch_hiddenLayer = branch_hiddenLayer
        self.trunk_hiddenLayer = trunk_hiddenLayer
        self.activation_func = activation_func
        self.lr = 1e-2
        self.lr_branch = 1e-2
        self.lr_min = 1e-8
        self.batch_size = 750
        self.auto_batchSize = True
        self.epoch = 2000
        self.epoch_branch = 2000
        self.base_epoch = 1000
        self.auto_epoch = True
        self.fix_initialize = False
        self.print_info = False
        #data preprocessing
        self.noise_type = noise_type
        self.factor_sigma = factor_sigma
        self.multi_noise = multi_noise
        self.scale_spectra = True
        self.scale_params = True
        self.norm_inputs = True
        self.norm_target = True
        self.independent_norm = False
        self.norm_type = 'z_score'
        #training
        self.spaceSigma_min = 5
        self.auto_repeat_n = False
        self.burn_in = False
        self.burnIn_step = None
        self.transfer_learning = False
        self.randn_num = round(abs(np.random.randn()/5.), 5)

    def _net(self):
        if self.fix_initialize:
            torch.manual_seed(1000) #Fixed parameter initialization
        self.nodes_in = []
        self.node_out = self.params.shape[1]
        self.fc_hidden = self.branch_hiddenLayer*2 + 1
        # self.fc_hidden = self.branch_hiddenLayer + self.trunk_hiddenLayer + 1 #also works, but not necessary
        if self.params_n==1:
            self.fc_out = self.comp_n + self.node_out*self.comp_n*2
        else:
            self.fc_out = self.comp_n + self.node_out*self.comp_n*2+self.comp_n*(self.node_out**2-self.node_out)//2
        for i in range(self.branch_n):
            self.nodes_in.append(self.spectra[i].shape[1])
        #to be modified to train branch network, change fcnet ??? --> add class Branch in fcnet.py
        # for i in range(self.branch_n):
        #     exec('self.branch_net%s=fcnet.FcNet(node_in=self.nodes_in[i], node_out=self.fc_out,\
        #           hidden_layer=self.fc_hidden, nodes=None, activation_func=self.activation_func)'%(i+1))
        if self.comp_type=='Gaussian':
            if self.params_n==1:
                self.net = fcnet.MultiBranchGaussianMDN(nodes_in=self.nodes_in, node_out=self.node_out, branch_hiddenLayer=self.branch_hiddenLayer, 
                                                        trunk_hiddenLayer=self.trunk_hiddenLayer, comp_n=self.comp_n, nodes_all=None, activation_func=self.activation_func)
            else:
                self.net = fcnet.MultiBranchMultivariateGaussianMDN(nodes_in=self.nodes_in, node_out=self.node_out, branch_hiddenLayer=self.branch_hiddenLayer, 
                                                                    trunk_hiddenLayer=self.trunk_hiddenLayer, comp_n=self.comp_n, nodes_all=None, activation_func=self.activation_func)
        elif self.comp_type=='Beta':
            if self.params_n==1:
                self.net = fcnet.MultiBranchBetaMDN(nodes_in=self.nodes_in, node_out=self.node_out, branch_hiddenLayer=self.branch_hiddenLayer, 
                                                    trunk_hiddenLayer=self.trunk_hiddenLayer, comp_n=self.comp_n, nodes_all=None, activation_func=self.activation_func)
            else:
                #MultiBranchMultivariateBeta?
                pass
        if self.print_info:
            print(self.net)

    @property
    def loss_func(self):
        if self.comp_type=='Gaussian':
            if self.params_n==1:
                return fcnet.gaussian_loss
            else:
                return fcnet.multivariateGaussian_loss
        elif self.comp_type=='Beta':
            if self.params_n==1:
                return fcnet.beta_loss
            else:
                pass
    
    def check_normType(self):
        if self.comp_type=='Gaussian':
            self.a = 0
            self.b = 1
        elif self.comp_type=='Beta':
            self.a = 1e-6
            self.b = 0.999999
            if not self.norm_target:
                self.norm_target = True
                print("'norm_target' is set to True because of using '%s' components."%self.comp_type)
            if self.norm_type!='minmax':
                self.norm_type = 'minmax'
                print("'norm_type' is set to 'minmax' because of using '%s' components."%self.comp_type)
    
    #change the branch net & trunck net (contain training) to use multiple GPUs ???
    def _train_branch(self, rank, repeat_n, showIter_n, device):
        
        optimizer = torch.optim.Adam(eval('self.branch_net%s.parameters()'%(rank+1)), lr=self.lr_branch)
        iteration = self.multi_noise*len(self.spectra[0])//self.batch_size * repeat_n
        
        self.inputs = self.spectra[rank]
        self.target = self.params
        self.error = self.obs_errors[rank]
        self.cholesky_f = self.cholesky_factor[rank]
#        self.transfer_subData(device=device)
        
        print('Training the branch network %s'%(rank+1))
        for subsample_num in range(1, self.epoch_branch+1):
            _inputs, _target = ds.AddGaussianNoise(self.inputs,params=self.target,obs_errors=self.error,cholesky_factor=self.cholesky_f,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            
            if self.scale_spectra:
                _inputs = _inputs / self.spectra_base_torch[rank] #to be tested !!!
            if self.norm_inputs:
                _inputs = dp.Normalize(_inputs, self.spectra_statistic[rank], norm_type=self.norm_type, a=self.a, b=self.b).norm()
            if self.norm_target:
                if self.independent_norm:
                    for i in range(self.params_n):
                        _target[:,i] = dp.Normalize(_target[:,i], self.params_statistic[i], norm_type=self.norm_type, a=self.a, b=self.b).norm()
                else:
                    _target = dp.Normalize(_target, self.params_statistic, norm_type=self.norm_type, a=self.a, b=self.b).norm()
            
            for iter_mid in range(1, iteration+1):
                batch_index = np.random.choice(len(_inputs), self.batch_size, replace=False)
                xx = _inputs[batch_index]
                yy = _target[batch_index]
                xx = Variable(xx)
                yy = Variable(yy, requires_grad=False)
                
                _omega, _mu_alpha, _sigma_cholesky_beta = eval('self.branch_net%s(xx)'%(rank+1))
                _loss = self.loss_func(_omega, _mu_alpha, _sigma_cholesky_beta, yy, logsumexp=True)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                
            if subsample_num%showIter_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch_branch, _loss.item(), optimizer.param_groups[0]['lr']))
            lrdc = optimize.LrDecay(subsample_num,iteration=self.epoch_branch,lr=self.lr_branch,lr_min=self.lr_min)
            optimizer.param_groups[0]['lr'] = lrdc.exp()
        
        #############################################################################
        # Note: hyperparameters must be transferred in the subprocess.
        #
        # Variables defined in the subprocess can not be called by the main process,
        # but, the hyperparameters of "self.branch_net%s"%i can be copied to "self.net",
        # the reason may be that hyperparameters of the network shared the memory.
        #############################################################################
        #print(eval("self.branch_net%s.fc[3].state_dict()['bias'][:5]"%(rank+1)))
        self._copyLayer_fromBranch(branch_index=rank+1)
    
    def _train_trunk(self, repeat_n=3, showIter_n=100, fix_lr=1e-4, reduce_fix_lr=False):
        branch_p = []
        for i in range(1, self.branch_n+1):
            branch_p.append(eval("{'params':self.net.branch%s.parameters(), 'lr':fix_lr}"%i)) #lr=fix_lr
        trunk_p = [{'params':self.net.trunk.parameters()}]
        optimizer = torch.optim.Adam(branch_p + trunk_p, lr=self.lr)

        print('Training the trunk network')
        for subsample_num in range(1, self.epoch_branch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.spectra,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            self.inputs = self.preprocessing_multiBranch_input(self.inputs, self.spectra_base_torch, a=self.a, b=self.b)
            self.target = self.preprocessing_multiBranch_target(self.target, a=self.a, b=self.b)
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.inputs[0]), self.batch_size, replace=False)
                xx = [self.inputs[i][batch_index] for i in range(self.branch_n)]
                yy = self.target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                
                _omega, _mu_alpha, _sigma_cholesky_beta = self.net(xx)
                _loss = self.loss_func(_omega, _mu_alpha, _sigma_cholesky_beta, yy, logsumexp=True)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                
            if subsample_num%showIter_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch_branch, _loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']))
            #test
            if reduce_fix_lr:
                lrdc_b = optimize.LrDecay(subsample_num,iteration=self.epoch_branch,lr=fix_lr,lr_min=self.lr_min)#change to lr=self.lr ?
                for i in range(self.branch_n):
                    optimizer.param_groups[i]['lr'] = lrdc_b.exp()
            lrdc_t = optimize.LrDecay(subsample_num,iteration=self.epoch_branch,lr=self.lr,lr_min=self.lr_min)
            for i in range(len(optimizer.param_groups)-self.branch_n):
                optimizer.param_groups[i+self.branch_n]['lr'] = lrdc_t.exp()
    
    def train(self, repeat_n=3, showIter_n=100, train_branch=True, parallel=True, train_trunk=False, fix_lr=1e-4, reduce_fix_lr=False):
        self._net()
        if self.transfer_learning==True and train_branch==False:
            self.copyLayers_fromTrainedNet()
        self.transfer_net(prints=self.print_info)
        
        #branch_p = [eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i) for i in range(1,self.branch_n+1)] #this will raise an error in python3.X
        #however, the following lines run well for both python2.X and python3.X, why?
        branch_p = []
        for i in range(1, self.branch_n+1):
            branch_p.append(eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i))
        trunk_p = [{'params':self.net.trunk.parameters()}]
        self.optimizer = torch.optim.Adam(branch_p + trunk_p, lr=self.lr)
        
        #added
        if self.auto_batchSize:
            self._auto_batchSize()
        self._check_batchSize()
        # print('batch size: %s'%self.batch_size)
        if self.auto_epoch:
            self._auto_epoch()
        if self.auto_repeat_n:
            repeat_n = self._auto_repeat_n(repeat_n)
        self.iteration = self.multi_noise*len(self.spectra[0])//self.batch_size * repeat_n
        # print('repeat_n:%s'%repeat_n)

        self.statistic()
        self.transfer_data()
        self.check_normType()
        
        # np.random.seed(1000)#
        print('randn_num: {}'.format(self.randn_num))
        if train_branch:
            if parallel:
                self._train_branchNet(repeat_n=repeat_n, showIter_n=showIter_n)
            else:
                self.transfer_branchNet()
                for rank in range(self.branch_n):
                    self._train_branch(rank, repeat_n, showIter_n, None)
        
        if train_trunk:
            self._train_trunk(repeat_n=repeat_n, showIter_n=showIter_n, fix_lr=fix_lr, reduce_fix_lr=reduce_fix_lr)
        
        self.train_loss = []
        self.vali_loss = []
        print('\nTraining the multibranch network')
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.spectra,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            self.inputs = self.preprocessing_multiBranch_input(self.inputs, self.spectra_base_torch, a=self.a, b=self.b)
            self.target = self.preprocessing_multiBranch_target(self.target, a=self.a, b=self.b)
            running_loss = []
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.inputs[0]), self.batch_size, replace=False)
                xx = [self.inputs[i][batch_index] for i in range(self.branch_n)]
                yy = self.target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                
                self.omega, self.mu_alpha, self.sigma_cholesky_beta = self.net(xx)
                _loss = self.loss_func(self.omega, self.mu_alpha, self.sigma_cholesky_beta, yy, logsumexp=True)
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                running_loss.append(_loss.item())
            loss_mean = np.mean(running_loss)
            self.train_loss.append(loss_mean)
            
            #vali_loss
            if self.spectra_vali is not None:
                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.spectra_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
                self.inputs_vali = self.preprocessing_multiBranch_input(self.inputs_vali, self.spectra_base_torch, a=self.a, b=self.b)
                self.target_vali = self.preprocessing_multiBranch_target(self.target_vali, a=self.a, b=self.b)
                self.net.eval()
                omega_vali, mu_alpha_vali, sigma_cholesky_beta_vali = self.net([Variable(self.inputs_vali[i]) for i in range(self.branch_n)])
                _vali_loss = self.loss_func(omega_vali, mu_alpha_vali, sigma_cholesky_beta_vali, Variable(self.target_vali, requires_grad=False), logsumexp=True)
                self.vali_loss.append(_vali_loss.item())
                self.net.train()

            if subsample_num%showIter_n==0:
                if self.lr==self.lr_branch:
                    if self.spectra_vali is None:
                        print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.optimizer.param_groups[0]['lr']))
                    else:
                        print('(epoch:%s/%s; train_loss/vali_loss:%.5f/%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.vali_loss[-1], self.optimizer.param_groups[0]['lr']))
                else:
                    if self.spectra_vali is None:
                        print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch, loss_mean, self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[-1]['lr']))
                    else:
                        print('(epoch:%s/%s; train_loss/vali_loss:%.5f/%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch, loss_mean, self.vali_loss[-1], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[-1]['lr']))
            lrdc_b = optimize.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr_branch,lr_min=self.lr_min)#change to lr=self.lr ?
            for i in range(self.branch_n):
                self.optimizer.param_groups[i]['lr'] = lrdc_b.exp()
            lrdc_t = optimize.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            for i in range(len(self.optimizer.param_groups)-self.branch_n):
                self.optimizer.param_groups[i+self.branch_n]['lr'] = lrdc_t.exp()    
        
        if self.use_multiGPU:
            self.net = self.net.module.cpu()
        else:
            self.net = self.net.cpu()
        self.train_loss = np.array(self.train_loss)
        self.vali_loss = np.array(self.vali_loss)
        return self.net, self.train_loss

    @property
    def sampler(self):
        if self.comp_type=='Gaussian':
            if self.params_n==1:
                return fcnet.gaussian_sampler
            else:
                return fcnet.multivariateGaussian_sampler
        elif self.comp_type=='Beta':
            if self.params_n==1:
                return fcnet.beta_sampler
            else:
                pass
    
    def _predict(self, spectra, use_GPU=False, in_type='numpy'):
        """Make predictions using a well-trained network.
        
        Parameters
        ----------
        spectra : numpy array or torch tensor
            The inputs of the network.
        use_GPU : bool
            If True, calculate using GPU, otherwise, calculate using CPU.
        in_type : str
            The data type of the spectra, 'numpy' or 'torch'.
        """
        if use_GPU:
            self.net = self.net.cuda()
            if in_type=='numpy':
                spectra = [dp.numpy2cuda(spectra[i]) for i in range(len(spectra))]
            elif in_type=='torch':
                spectra = [dp.torch2cuda(spectra[i]) for i in range(len(spectra))]
        else:
            if in_type=='numpy':
                spectra = [dp.numpy2torch(spectra[i]) for i in range(len(spectra))]
        self.net = self.net.eval() #this works for the batch normalization layers
        spectra = [Variable(spectra[i]) for i in range(len(spectra))]
        self.omega, self.mu_alpha, self.sigma_cholesky_beta = self.net(spectra)
        pred = self.sampler(self.omega, self.mu_alpha, self.sigma_cholesky_beta)
        if use_GPU:
            pred = dp.cuda2numpy(pred.data)
        else:
            pred = dp.torch2numpy(pred.data)
        #reshape chain
        if len(pred.shape)==1:
            pred = pred.reshape(-1, 1)
        return pred
    
    def predict_chain(self, obs_spectra, chain_leng=10000):
        # obs_spectra: observational spectrum in a list [spectra1, spectra2, ...], each element has shape (N, 3)
        # torch.manual_seed(1000)#
        obs_spectra = [dp.numpy2torch(obs_spectra[i]) for i in range(len(obs_spectra))]
        obs_best = [obs_spectra[i][:,1] for i in range(len(obs_spectra))]
        obs_best_multi = [torch.ones((chain_leng, len(obs_best[i]))) * obs_best[i] for i in range(len(obs_spectra))]
        obs_best_multi = self.preprocessing_multiBranch_input(obs_best_multi, [dp.numpy2torch(self.spectra_base[i]) for i in range(len(obs_spectra))], a=self.a, b=self.b)
        self.chain = self._predict(obs_best_multi, in_type='torch')
        self.chain = self.preprocessing_target_inv(self.chain, a=self.a, b=self.b)
        self.chain = self.cut_params(self.chain) #remove non-physical parameters
        return self.chain
    
    def save_net(self, path='mdn', sample='TT'):
        if sample is None:
            fileName = 'net_branch%s_train%s_batch%s_epoch%s_epochBranch%s_comp%s_%s.pt'%(self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.comp_n,self.randn_num)
        else:
            fileName = 'net-%s_branch%s_train%s_batch%s_epoch%s_epochBranch%s_comp%s_%s.pt'%(sample,self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.comp_n,self.randn_num)
        utils.saveTorchPt(path+'/net', fileName, self.net)
        
    def save_loss(self, path='mdn', sample='TT'):
        if sample is None:
            fileName = 'loss_branch%s_train%s_batch%s_epoch%s_epochBranch%s_comp%s_%s'%(self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.comp_n,self.randn_num)
        else:
            fileName = 'loss-%s_branch%s_train%s_batch%s_epoch%s_epochBranch%s_comp%s_%s'%(sample,self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.comp_n,self.randn_num)
        utils.savenpy(path+'/loss', fileName, [self.train_loss, self.vali_loss])
    
    def save_chain(self, path='mdn', sample='TT'):
        if sample is None:
            fileName = 'chain_branch%s_train%s_batch%s_epoch%s_epochBranch%s_comp%s_%s'%(self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.comp_n,self.randn_num)
        else:
            fileName = 'chain-%s_branch%s_train%s_batch%s_epoch%s_epochBranch%s_comp%s_%s'%(sample,self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.comp_n,self.randn_num)
        utils.savenpy(path+'/chains', fileName, self.chain)
    
    def save_hparams(self, path='mdn', sample='TT'):
        if sample is None:
            fileName = 'hparams_branch%s_train%s_batch%s_epoch%s_epochBranch%s_comp%s_%s'%(self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.comp_n,self.randn_num)
        else:
            fileName = 'hparams-%s_branch%s_train%s_batch%s_epoch%s_epochBranch%s_comp%s_%s'%(sample,self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.comp_n,self.randn_num)
        utils.savenpy(path+'/hparams', fileName, [self.spectra_statistic, self.params_statistic, self.spectra_base, self.params_base, self.param_names, self.params_dict, self.comp_type, 
                                                  self.scale_spectra, self.scale_params, self.norm_inputs, self.norm_target, self.independent_norm, self.norm_type, self.a, self.b, 
                                                  self.burnIn_step])

class LoadHparams_MB(object):
    def __init__(self, path='ann', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)

    def load_hparams(self):
        file_path = evaluate.FilePath(filedir=self.path+'/hparams', randn_num=self.randn_num, suffix='.npy').filePath()
        self.spectra_statistic, self.params_statistic, self.spectra_base, self.params_base, self.param_names, self.params_dict, self.comp_type, self.scale_spectra, self.scale_params, self.norm_inputs, self.norm_target, self.independent_norm, self.norm_type, self.a, self.b, self.burnIn_step = np.load(file_path, allow_pickle=True)
        self.params_n = len(self.param_names)
        p_property = cosmic_params.ParamsProperty(self.param_names, params_dict=self.params_dict)
        self.params_limit = p_property.params_limit
        self.branch_n = len(self.spectra_statistic)

class RePredictMBMDN(MultiBranchMDN, LoadNet, LoadLoss, LoadChain, LoadHparams_MB):
    def __init__(self, path='mdn', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)

