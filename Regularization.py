# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:19:00 2021

@author: Grant
"""

import random
import torch
from torch import nn, optim
import math
from IPython import display

import numpy as np

from matplotlib import pyplot as plt

def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)
    
def plot_data(X, y, d=0, auto=False, zoom=1):
    X = X.cpu()
    y = y.cpu()
    plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    plt.axis('square')
    plt.axis(np.array((-1.1, 1.1, -1.1, 1.1)) * zoom)
    if auto is True: plt.axis('equal')
    plt.axis('off')

    _m, _c = 0, '.15'
    plt.axvline(0, ymin=_m, color=_c, lw=1, zorder=0)
    plt.axhline(0, xmin=_m, color=_c, lw=1, zorder=0)

def plot_model(X, y, model):
    model.cpu()
    mesh = np.arange(-1.1, 1.1, 0.01)
    xx, yy = np.meshgrid(mesh, mesh)
    with torch.no_grad():
        data = torch.from_numpy(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
        Z = model(data).detach()
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
    plot_data(X, y)
    
set_default()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 1
random.seed(seed)
N = 1000
D = 1
C = 1
H = 100

X = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1).to(device)
y = X.pow(3) + 0.3 * torch.rand(X.size()).to(device)

criterion = torch.nn.MSELoss()

n = 10

for m in range(n):
    model = nn.Sequential(
        nn.Linear(D, 500),
        nn.Tanh(),
        nn.Linear(500, 1000),
        nn.ReLU(),
        #nn.Dropout(p=0.01),
        nn.Linear(1000, 500),
        nn.Tanh(),
        nn.Linear(500, C)
        )
    model.to(device)
    
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    for t in range(50):
        #gets the result from the model
        y_pred = model(X)
    
        #compares the result to the expected output
        loss = criterion(y_pred, y)
        
        #fc1_params = torch.cat([x.view(-1) for x in model.parameters()])
        #loss += 0.001 * torch.norm(fc1_params, 1)
    
        #zero the gradient before running backward pass
        optimiser.zero_grad()
    
        #Backpropagation to calculate the gradient of loss vs learnable parameters
        loss.backward()
    
        #update the parameters
        optimiser.step()
    

print(model)

plt.scatter(X.data.cpu().numpy(), y.data.cpu().numpy())
plt.plot(X.data.cpu().numpy(), y_pred.data.cpu().numpy(), 'r-', lw=5)
plt.axis('equal')