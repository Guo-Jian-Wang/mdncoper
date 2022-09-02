MDNcoper
========

**MDNcoper (Mixture Density Network Cosmological Parameter Estimator)**

MDNcoper is a method to estimate cosmological parameters based on mixture density network (MDN). It is an alternative to the traditional `Markov chain Monte Carlo (MCMC) <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ method, and can obtain almost the same results as MCMC.

MDNcoper can be applied to the research of cosmology and even other broader scientific fields.



Attribution
-----------

If you use this code in your research, please cite `Guo-Jian Wang, Cheng Cheng, Yin-Zhe Ma, Jun-Qing Xia, ApJS, 262, 24 (2022) <https://doi.org/10.3847/1538-4365/ac7da1>`_.



Dependencies
------------

The main dependencies of mdncoper are:

* `PyTorch <https://pytorch.org/>`_
* `CUDA <https://developer.nvidia.com/cuda-downloads>`_
* `ecopann <https://github.com/Guo-Jian-Wang/ecopann>`_
* `coplot-0.1.2 <https://github.com/Guo-Jian-Wang/coplot>`_
* os
* sys
* numpy



Installation
------------

You can install mdncoper by using::
	
	$ git clone https://github.com/Guo-Jian-Wang/mdncoper.git    
	$ cd mdncoper
	$ sudo python setup.py install



License
-------

Copyright 2022-2022 Guojian Wang

mdncoper is free software made available under the MIT License. For details see the LICENSE file.
