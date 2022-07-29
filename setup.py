#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs 7 July 2022

@author : minh.ngo
"""
from setuptools import setup

setup(name='hlsf',
    description='Models for LSF',
    author="Minh NGO",
    author_email="ngoc-minh.ngo@insa-lyon.fr",
    version='1.0',
    packages=['hlsf'],
    package_dir={'hlsf': 'src/hlsf'},
    install_requires=['lmfit', 'numpyencoder', 'hpylib']
    )