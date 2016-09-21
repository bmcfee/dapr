#!/usr/bin/env python
''''DAPR types'''

from collections import namedtuple

# 1-dimensional convnets
Conv1Input = namedtuple('Conv1Input', ['name', 'x', 'y', 'n'])
'''1D Convolutional input'''

Conv1Filter = namedtuple('Conv1Filter', ['name', 'x', 'y', 'n'])
'''1D Convolutional filter'''

Pool1 = namedtuple('Pool1', ['name', 'x', 'y'])
'''1D pooling operation'''


# 2-dimensional convnets
Conv2Input = namedtuple('Conv2Input', ['name', 'x', 'y', 'n'])
'''2D Convolutional input'''

Conv2Filter = namedtuple('Conv2Filter', ['name', 'x', 'y', 'n'])
'''2D Convolutional filter'''

Pool2 = namedtuple('Pool2', ['name', 'x', 'y'])
'''2D pooling operation'''

# Dense networks
DenseLayer = namedtuple('Dense', ['name', 'd'])
'''Dense layer'''
