#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:03:46 2019

@author: joao
"""
#%% Imports
import numpy as np
import pandas as pd

#%% Importar database e factorize
fruits = pd.read_csv('database.csv');

# Alguns atributos de database são alfa numéricos, portanto devemos deixá-los
# de forma muerica e para isso usa-se. O método *factorize* divide dois termos
# o dumbs separa para mais termos.

fruits['class'] = pd.factorize(fruits['class'])[0]
fruits['taste'] = pd.factorize(fruits['taste'])[0]
fruits['Color-F'] = pd.factorize(fruits['Color-F'])[0]
fruits = pd.get_dummies(fruits, columns=['Color-D'])
fruits['texture'] = pd.factorize(fruits['texture'])[0]

# Agora deve-se separar o vetor de rótulos

y = fruits['class'].values

fruits = fruits.drop(['class'],axis=1)
X = fruits.iloc[:,:].values

#%%