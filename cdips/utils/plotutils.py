"""
plotutils.py - Luke Bouma (bouma.luke@gmail) - Nov 2020

Reusable plotting functions, including:

    compass: embed a compass to show North and East directions
    rainbow_text: write strings with matching colors.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import matplotlib.transforms as transforms

def compass(ax, x, y, size, invert_x=False, invert_y=False):
    """Add a compass to indicate the north and east directions.

    Assumes ax has a wcs projection.

    Parameters
    ----------
    x, y : float
        Position of compass vertex in axes coordinates.
    size : float
        Size of compass in axes coordinates.

    """
    xy = x, y
    scale = ax.wcs.pixel_scale_matrix
    scale /= np.sqrt(np.abs(np.linalg.det(scale)))

    for n, label, ha, va in zip(
        scale, 'EN', ['right', 'center'], ['center', 'bottom']
    ):
        if invert_x:
            n[0] *= -1
        if invert_y:
            n[1] *= -1

        ax.annotate(
            label, xy, xy + size * n, ax.transAxes, ax.transAxes,
            ha='center', va='center',
            arrowprops=dict(arrowstyle='<-', shrinkA=0.0, shrinkB=0.0)
        )


def rainbow_text(x, y, strings, colors, orientation='horizontal',
                 prefactor=0.75, ax=None, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    NOTE: `prefactor` is set for .pdf output. It renders differently for .png
    output, for insane reasons.

    Example call:
        ```
        words = ['Field', 'Corona', 'Core', 'TOI1937'][::-1]
        colors = ['gray', 'lightskyblue', 'k', 'lightskyblue'][::-1]
        rainbow_text(0.98, 0.02, words, colors, size='medium', ax=axs[0])
        ```

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transAxes
    canvas = ax.figure.canvas

    props = dict(boxstyle='square', facecolor='white', alpha=1, pad=0.08,
                 linewidth=0)

    kwargs.update(rotation=0, va='center', ha='center')

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, bbox=props,
                       **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(
            text.get_transform(), y=prefactor*ex.height, units='dots'
        )

