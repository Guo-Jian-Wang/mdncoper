MDNcoper
========

**MDNcoper (Mixture Density Network Cosmological Parameter Estimator)**

MDNcoper is a method to estimate cosmological parameters based on mixture density network (MDN). It is an alternative to the traditional `Markov chain Monte Carlo (MCMC) <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ method, and can obtain almost the same results as MCMC.

MDNcoper can be applied to the research of cosmology and even other broader scientific fields.



Dependencies
------------

The main dependencies of cmbnncs are:

* `PyTorch <https://pytorch.org/>`_
* `CUDA <https://developer.nvidia.com/cuda-downloads>`_
* `ecopann <https://github.com/Guo-Jian-Wang/ecopann>`_
* os
* sys
* numpy



Installation
------------

$ git clone https://github.com/Guo-Jian-Wang/mdncoper.git    
$ cd mdncoper
$ sudo python setup.py install



License
-------

Copyright 2022-2022 Guojian Wang

mdncoper is free software made available under the MIT License. For details see the LICENSE file.
