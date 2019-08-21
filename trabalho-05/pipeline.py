# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 22:23:51 2019

@author: joao_
"""

#%% Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#%% Importar database e factorize
fruits = pd.read_csv('database.csv');

# Alguns atributos de database são alfa numéricos, portanto devemos deixá-los
# de forma muerica e para isso usa-se. O método *factorize* divide dois termos
# o dummiess separa para mais termos.
fruits['class'] = pd.factorize(fruits['class'])[0]
fruits['taste'] = pd.factorize(fruits['taste'])[0]
fruits['Color-F'] = pd.factorize(fruits['Color-F'])[0]
fruits = pd.get_dummies(fruits, columns=['Color-D'])
fruits['texture'] = pd.factorize(fruits['texture'])[0]

# Agora deve-se separar o vetor de rótulos
y = fruits['class']

fruits = fruits.drop(['class'],axis=1)
X = fruits.iloc[:,:]

y = y.values
X = X.values

#%% Separar em teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=1, stratify=y)

#%% Modelo Perceptron
perc_model = Perceptron(eta0=1, tol=1e-3, random_state=0, max_iter=20)
perc_model.fit(X_train, y_train)

# Predict
y_train_pred = perc_model.predict(X_train)
y_test_pred = perc_model.predict(X_test)

# Accuracy
print("Perceptron accuracy score")
print(accuracy_score(y_train_pred, y_train))
print(accuracy_score(y_test_pred, y_test))

#%% Modelo Adaline
adal_model = SGDClassifier(eta0=0.01, tol=1e-3, max_iter=100,
                           loss='squared_loss')
adal_model.fit(X_train, y_train)

# Predict
y_train_pred = adal_model.predict(X_train)
y_test_pred = adal_model.predict(X_test)

# Accuracy
print("Stochastic Gradient Descent accuracy score")
print(accuracy_score(y_train_pred, y_train))
print(accuracy_score(y_test_pred, y_test))

#%% Modelo Regressao Logistica
logr_model = LogisticRegression(random_state=0, solver='liblinear', 
                                multi_class='auto', max_iter=100)
logr_model.fit(X_train, y_train)

# Predict
y_train_pred = logr_model.predict(X_train)
y_test_pred = logr_model.predict(X_test)

# Accuracy
print("Logistic Regression accuracy score")
print(accuracy_score(y_train_pred, y_train))
print(accuracy_score(y_test_pred, y_test))

#%% Normalizacao
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#%% Modelo Perceptron
perc_model = Perceptron(eta0=0.5, tol=1e-3, random_state=0, max_iter=10)
perc_model.fit(X_train_std, y_train)

# Predict
y_train_pred = perc_model.predict(X_train_std)
y_test_pred = perc_model.predict(X_test_std)

# Accuracy
print("STD - Perceptron accuracy score")
print(accuracy_score(y_train_pred, y_train))
print(accuracy_score(y_test_pred, y_test))

#%% Modelo Adaline
adal_model = SGDClassifier(eta0=0.01, tol=1e-3, max_iter=100,
                           loss='squared_loss')
adal_model.fit(X_train_std, y_train)

# Predict
y_train_pred = adal_model.predict(X_train_std)
y_test_pred = adal_model.predict(X_test_std)

# Accuracy
print("STD - Stochastic Gradient Descent accuracy score")
print(accuracy_score(y_train_pred, y_train))
print(accuracy_score(y_test_pred, y_test))

#%% Modelo Regressao Logistica
logr_model = LogisticRegression(random_state=0, solver='liblinear', 
                                multi_class='auto', max_iter=100)
logr_model.fit(X_train_std, y_train)

# Predict
y_train_pred = logr_model.predict(X_train_std)
y_test_pred = logr_model.predict(X_test_std)

# Accuracy
print("STD - Logistic Regression accuracy score")
print(accuracy_score(y_train_pred, y_train))
print(accuracy_score(y_test_pred, y_test))

#%% Regularizacao
adal_model = SGDClassifier(penalty='l2', alpha=0.8, loss='squared_loss', 
                           eta0=0.01, learning_rate='constant',
                           max_iter=100, tol=1e-3)

adal_model.fit(X_train_std, y_train)

# Predict
y_train_pred = adal_model.predict(X_train_std)
y_test_pred = adal_model.predict(X_test_std)

# Accuracy
print("REG - STD - Stochastic Gradient Descent accuracy score")
print(accuracy_score(y_train_pred, y_train))
print(accuracy_score(y_test_pred, y_test))

print('Coeficientes sem regularizacao')
print(adal_model.coef_)

#%% Tetnar um plot
eta = np.arange(0.01, 0.5, 0.01)
accuracy = []

for i in eta:
    adal_model = SGDClassifier(penalty='l2', alpha=0.8, loss='squared_loss', 
                               eta0=i, learning_rate='constant',
                               max_iter=100, tol=1e-3)
    adal_model.fit(X_train_std, y_train)
    y_test_pred = adal_model.predict(X_test_std)
    accuracy.append(accuracy_score(y_test_pred, y_test))

fig = plt.figure()
plt.plot(eta, accuracy)
plt.ylabel('accuracy score')
plt.xlabel('eta')
plt.show()

alf = np.arange(0.1, 1, 0.1)
accuracy = []

print('Coef em l1, com variacao de alpha')
for i in alf:
    adal_model = SGDClassifier(penalty='l1', alpha=i, loss='squared_loss', 
                               eta0=0.5, learning_rate='constant',
                               max_iter=100, tol=1e-3)
    adal_model.fit(X_train_std, y_train)
    y_test_pred = adal_model.predict(X_test_std)
    accuracy.append(accuracy_score(y_test_pred, y_test))
    print(adal_model.coef_)

fig = plt.figure()
plt.plot(alf, accuracy)
plt.ylabel('accuracy score')
plt.xlabel('alpha')
plt.title('l1')
plt.show()

alf = np.arange(0.1, 1, 0.1)
accuracy = []

print('Coef em l2, com variacao de alpha')
for i in alf:
    adal_model = SGDClassifier(penalty='l2', alpha=i, loss='squared_loss', 
                               eta0=0.5, learning_rate='constant',
                               max_iter=100, tol=1e-3)
    adal_model.fit(X_train_std, y_train)
    y_test_pred = adal_model.predict(X_test_std)
    accuracy.append(accuracy_score(y_test_pred, y_test))
    print(adal_model.coef_)


fig = plt.figure()
plt.plot(alf, accuracy)
plt.ylabel('accuracy score')
plt.xlabel('alpha')
plt.title('l2')
plt.show()

niter = np.arange(10, 200, 1)
accuracy = []

for i in niter:
    adal_model = SGDClassifier(penalty='l2', alpha=0.5, loss='squared_loss', 
                               eta0=0.5, learning_rate='constant',
                               max_iter=i, tol=1e-3)
    adal_model.fit(X_train_std, y_train)
    y_test_pred = adal_model.predict(X_test_std)
    accuracy.append(accuracy_score(y_test_pred, y_test))

fig = plt.figure()
plt.plot(niter, accuracy)
plt.ylabel('accuracy score')
plt.xlabel('max_iter')
plt.title('n_iter')
plt.show()


