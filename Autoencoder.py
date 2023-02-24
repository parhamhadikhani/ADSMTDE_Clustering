# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:30:49 2021

@author: User
"""




from torch import nn

class autoencoder(nn.Module):
    def __init__(self, numLayers, encoders=False):
        super().__init__()
        self.layers = nn.ModuleList()
        if encoders:
            for i in range(len(numLayers) - 2):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[-2], numLayers[-1]))
        else:
            for i in range(len(numLayers) - 2):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[-2], numLayers[-1]))
            for i in range(len(numLayers) - 1, 1, -1):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i-1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[1], numLayers[0]))

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        
        return y