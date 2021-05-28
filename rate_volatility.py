#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:11:14 2021

@author: jtobin
"""

import os
import git

# local location
local_dir = os.path.expanduser('~/Documents/projects/rate_volatility')
os.chdir(local_dir)

# location of git repository
repo_url = 'https://github.com/james-j-tobin/interest-rate-volatility.git'

