#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:03:46 2019

@author: joao
"""
#%% Imports
import numpy as np
import pandas as pd

fruits = pd.read_csv('database.csv');

#%% Tests
fruits

fruits.columns

fruits.index

fruits['weight']