# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:40:17 2020

@author: meizihang
"""

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

__version__ = '1.0.1'
__author__ = 'Zihang.Mei'

setup(
    name='autoML',
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'pip==20.2.4',
        'IPython==7.8.0'
        'numpy==1.17.4',
        'shap==0.32.1',
        'toad=0.0.64',
        'pandas==0.25.3',
        'scikit-learn==0.22',
        'lightgbm==2.2.3',
        'xgboost==0.90'
    ],
    url='',
    author=__author__,
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    include_package_data=True,
    package_data={'': ['*.py', '*.txt', '*.csv']},
    zip_safe=False,
    platform='any',

    descripition='Automated Machine Learning for VIVO Finance Data Mining Group',
    long_descripition=__doc__
)