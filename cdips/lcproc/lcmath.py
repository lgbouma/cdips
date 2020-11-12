"""
lcmath.py - Luke Bouma (bouma.luke@gmail) - Nov 2020

Quick centering and scaling functions.

Contents
    scale_to_unitvariance
    center_and_unitvariance
    rescale_ipr
"""

import numpy as np

def rescale_ipr(vec, target_ipr):
    current_ipr = np.nanpercentile(vec, 95) - np.nanpercentile(vec, 5)
    factor = target_ipr / current_ipr
    return vec*factor

def scale_to_unitvariance(vec):
    return vec/np.nanstd(vec)

def center_and_unitvariance(vec):
    return (vec-np.nanmean(vec))/(np.nanstd(vec))


