# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:17:03 2019

@author: Guojian Wang
"""

import os
import re
from setuptools import setup, find_packages


def read(filename):
    f = open(filename)
    r = f.read()
    f.close()
    return r

ver = re.compile("__version__ = \"(.*?)\"")
#m = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "refann", "__init__.py"))
#m = read(os.path.join(os.path.dirname(__file__), "refann", "__init__.py"))
m = read(os.path.join(os.getcwd(), "mdncoper", "__init__.py"))
version = ver.findall(m)[0]



setup(
    name = "mdncoper",
    version = version,
    keywords = ("pip", "MDN"),
    description = "Mixture Density Network Cosmological Parameter Estimator",
    long_description = "",
    license = "MIT",

    url = "",
    author = "Guojian Wang",
    author_email = "gjwang2018@gmail.com",

#    packages = find_packages(),
    packages = ["mdncoper", "examples"],
    include_package_data = True,
    data_files = ["examples/data/Pantheon_SNe_NoName.txt",
                  "examples/data/chain_fwCDM_3params.npy",
                  ],
    platforms = "any",
    install_requires = []
)

