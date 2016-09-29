#!/usr/bin/env python
''''DAPR types'''

from collections import namedtuple

# 1-dimensional convnets
Conv1Input = namedtuple('Conv1Input', ['name', 'n', 'x'])
'''1D Convolutional input'''

Conv1 = namedtuple('Conv1', ['name', 'n', 'x'])
'''1D Convolutional filter'''

Pool1 = namedtuple('Pool1', ['name', 'x', 's_x'])
'''1D pooling operation'''


# 2-dimensional convnets
Conv2Input = namedtuple('Conv2Input', ['name', 'n', 'x', 'y'])
'''2D Convolutional input'''

Conv2 = namedtuple('Conv2', ['name', 'n', 'x', 'y'])
'''2D Convolutional filter'''

Pool2 = namedtuple('Pool2', ['name', 'x', 'y', 's_x', 's_y'])
'''2D pooling operation'''

# Dense networks
DenseLayer = namedtuple('Dense', ['name', 'n'])
'''Dense layer'''
