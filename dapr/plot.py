#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''DAPR plotting module'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.collections as mcoll

from .types import Conv2Input, Conv2, Pool2, Conv1Input, Conv1, Pool1


def convnet(shapes, image=None, cmap='magma', **kwargs):
    '''Plot a convolutional network'''
    # TODO:
    #    account for pooling in receptive field sizing

    ax = plt.gca()

    if 'facecolor' not in kwargs:
        kwargs.setdefault('facecolor',
                          next(ax._get_lines.prop_cycler)['color'])

    fc = kwargs['facecolor']

    posx, tickl = [], []

    offset, volume = 0, None

    spacing = shapes[0].x / 4

    center = None

    pool_scale = (1, 1)

    for i, s in enumerate(shapes):
        if i == 0:
            kwargs['facecolor'] = next(ax._get_lines.prop_cycler)['color']
        elif i == len(shapes) - 1:
            kwargs['facecolor'] = next(ax._get_lines.prop_cycler)['color']
        else:
            kwargs['facecolor'] = fc

        if isinstance(s, Conv2Input):
            volume = Conv2('', s.n, s.x, s.y)
            # Move all the verticals down by b
            b = volume.y * 0.5

            # Make sure to pad by our depth
            offset += volume.n / np.sqrt(2)

            draw_volume(ax, offset, -b, volume.x, volume.y, volume.n,
                        alpha=0.5, zorder=-1, **kwargs)
            posx.append(offset + volume.x * 0.5)

            tickl.append('{:s}\n{:d}'.format(s.name, volume.n))
            tickl[-1] += '\n$({:d}\\times{:d})$'.format(volume.x, volume.y)

            if image is not None:
                ax.imshow(image,
                          origin='lower',
                          interpolation='nearest',
                          cmap=cmap,
                          extent=[offset, offset + volume.x,
                                  -b, -b + volume.y],
                          clip_on=True)

            center = offset + volume.x * 0.5
            ind = volume.n

            offset += volume.x + spacing

        elif isinstance(s, Conv1Input):
            volume = Conv2('', 0, s.x, s.n)

            b = volume.y * 0.5

            draw_volume(ax, offset, -b, volume.x, volume.y, 0,
                        alpha=0.5, zorder=-1, **kwargs)
            posx.append(offset + volume.x * 0.5)

            tickl.append('{:s}\n{:d}'.format(s.name, volume.y))
            tickl[-1] += '\n$({:d})$'.format(volume.x)

            if image is not None:
                ax.imshow(image,
                          origin='lower',
                          interpolation='nearest',
                          cmap=cmap,
                          extent=[offset, offset + volume.x,
                                  -b, -b + volume.y],
                          clip_on=True)

            center = offset + volume.x * 0.5
            ind = volume.n

            offset += volume.x + spacing

        elif isinstance(s, Pool2):
            pool_scale = (s.x, s.y)

            # This math is probably wrong
            volume = Conv2('', volume.n,
                           (volume.x // s.s_x),
                           (volume.y // s.s_y))

            posx.append(offset + spacing * 0.5)
            tickl.append('{:s}\n${:d}\\times{:d}$'.format(s.name, s.x, s.y))
            offset += spacing

        elif isinstance(s, Pool1):
            pool_scale = (s.x, 1)

            volume = Conv2('', volume.n,
                           (volume.x // s.s_x),
                           volume.y)

            posx.append(offset + spacing * 0.5)

            tickl.append('{:s}\n${:d}$'.format(s.name, s.x))
            offset += spacing

        elif isinstance(s, Conv2):
            if s.x is None:
                inx = volume.x
                outx = 1
            else:
                inx = pool_scale[0] * s.x
                outx = volume.x - s.x + 1

            if s.y is None:
                iny = volume.y
                outy = 1
            else:
                iny = pool_scale[1] * s.y
                outy = volume.y - s.y + 1

            # Reset the pooling scale
            pool_scale = (1, 1)

            volume = Conv2('', s.n, outx, outy)

            # Move all the verticals down by b
            b = volume.y * 0.5

            # Make sure to pad by our depth
            offset += volume.n / np.sqrt(2)

            draw_volume(ax, offset, -b, volume.x, volume.y, volume.n,
                        alpha=0.5, zorder=-1, **kwargs)
            posx.append(offset + volume.x * 0.5 * (1 - volume.n / np.sqrt(2)))

            tickl.append('{:s}\n{:d}'.format(s.name, volume.n))
            tickl[-1] += '\n$({:d}\\times{:d})$'.format(volume.x, volume.y)

            # Draw the filter receptive field
            draw_volume(ax,
                        center - 0.5 * inx,
                        - 0.5 * iny,
                        inx, iny, ind,
                        alpha=0.5, zorder=1, facecolor='white')

            # Draw the target point: halfway in dimension, center in y
            cpt = (offset - volume.n / np.sqrt(2) / 2,
                   volume.n / np.sqrt(2) / 2)
            draw_zoom(ax, cpt,
                      center + 0.5 * inx,
                      - 0.5 * iny,
                      iny,
                      ind,
                      alpha=0.1, facecolor='none')

            center = offset + volume.x * 0.5
            ind = volume.n
            offset += volume.x + spacing

        elif isinstance(s, Conv1):
            if s.x is None:
                inx = volume.x
                outx = 1
            else:
                inx = pool_scale[0] * s.x
                outx = volume.x - s.x + 1

            iny = volume.y
            outy = s.n

            pool_scale = (1, 1)

            volume = Conv2('', 0, outx, outy)

            b = volume.y * 0.5

            draw_volume(ax, offset, -b, volume.x, volume.y, volume.n,
                        alpha=0.5, zorder=-1, **kwargs)
            posx.append(offset + volume.x * 0.5)

            tickl.append('{:s}\n{:d}'.format(s.name, volume.y))
            tickl[-1] += '\n${:d}$'.format(volume.x)

            draw_volume(ax,
                        center - 0.5 * inx,
                        - 0.5 * iny,
                        inx, iny, 0,
                        alpha=0.5, zorder=1, facecolor='white')

            cpt = (offset, 0)

            draw_zoom(ax, cpt,
                      center + 0.5 * inx,
                      -0.5 * iny,
                      iny,
                      0,
                      alpha=0.1, facecolor='none')

            center = offset + volume.x * 0.5
            ind = 0
            offset += volume.x + spacing

    # Tick the layers
    ax.set_xticks(posx)
    ax.set_xticklabels(tickl, ha='center')

    # Wipe out the spines and ticks
    clear_spines(ax)

    ax.axis('tight')
    return ax


def clear_spines(ax):
    for s in ax.spines:
        ax.spines[s].set_visible(False)

    ax.set_yticks([])

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    return ax


def draw_volume(ax, xpos, ypos, width, height, depth, **kwargs):
    '''Draw a volume'''
    patches = []

    off = depth / np.sqrt(2)

    if depth > 0:
        patches.append(mpatch.Polygon([(xpos, ypos + height),
                                       (xpos - off, ypos + height + off),
                                       (xpos - off + width,
                                        ypos + height + off),
                                       (xpos + width, ypos + height)]))

        patches.append(mpatch.Polygon([(xpos, ypos + height),
                                       (xpos - off, ypos + height + off),
                                       (xpos - off, ypos + off),
                                       (xpos, ypos)]))

    patches.append(mpatch.Rectangle((xpos, ypos), width, height))
    return ax.add_collection(mcoll.PatchCollection(patches, **kwargs))


def draw_zoom(ax, cpt, xpos, ypos, height, depth, **kwargs):
    '''Draw a 2d convolution zoom'''
    patches = []
    off = depth / np.sqrt(2)

    # Front
    patches.append(mpatch.Polygon([cpt,
                                   (xpos, ypos),
                                   (xpos, ypos + height)]))
    if depth > 0:
        # Top
        patches.append(mpatch.Polygon([cpt,
                                       (xpos, ypos + height),
                                       (xpos - off, ypos + height + off)]))
        # Bottom
        patches.append(mpatch.Polygon([cpt,
                                       (xpos, ypos),
                                       (xpos - off, ypos + off)]))
        # Back
        patches.append(mpatch.Polygon([cpt,
                                       (xpos - off, ypos + off),
                                       (xpos - off, ypos + height + off)]))

    return ax.add_collection(mcoll.PatchCollection(patches, **kwargs))
