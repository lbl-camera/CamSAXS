#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import camsaxs
import numpy as np
import pyFAI
import matplotlib.pylab as plt
import ipdb
import timeit

def func(args):
    camsaxs.remesh(*args)

filename = 'GIWAXS_sfloat_2m.gb'
center = [632., 69.5]
pixel = [172E-06, 172E-06]
poni = np.multiply(center, pixel)
dist = 300.
det = pyFAI.detectors.Pilatus2M
g = pyFAI.geometry.Geometry(dist=0.3, poni1=poni[1], poni2=poni[0], 
            pixel1=pixel[0], pixel2=pixel[1], wavelength=0.123984E-09)

data = np.fromfile(filename, dtype=np.float32).reshape(1679,1475)
data = np.flipud(data)
ai = np.deg2rad(0.14)
lims = [[0, 20],[0, 20]]
res = [512, 512]


# default qp-qz, same size as detector
img, x, y = camsaxs.remesh(data, g, ai)
box = [x.min(), x.max(), y.min(), y.max()]
plt.imshow(np.log(img+4), origin='lower', extent=box)
plt.savefig('qp_qz_full.png')

plt.figure()
img, x , y = camsaxs.remesh(data, g, ai, res=res, coord_sys='qp_qz', out_range=lims)
box = [x.min(), x.max(), y.min(), y.max()]
plt.imshow(np.log(img+4), origin='lower', extent=box)
plt.savefig('qp_qz_roi.png')

plt.figure()
img, x , y = camsaxs.remesh(data, g, ai, coord_sys='qy_qz')
box = [x.min(), x.max(), y.min(), y.max()]
plt.imshow(np.log(img+4), origin='lower', extent=box)
plt.savefig('qy_qz_full.png')

plt.figure()
img, x , y = camsaxs.remesh(data, g, ai, res=res, coord_sys='qy_qz')
box = [x.min(), x.max(), y.min(), y.max()]
plt.imshow(np.log(img+4), origin='lower', extent=box)
plt.savefig('qy_qz_roi.png')

plt.figure()
img, x , y = camsaxs.remesh(data, g, ai, coord_sys='theta_alpha')
box = [x.min(), x.max(), y.min(), y.max()]
plt.imshow(np.log(img+4), origin='lower', extent=box)
plt.savefig('th_al_full.png')

plt.figure()
lims = [[-0.1, 0.1],[0, 0.2]]
res = [2000, 2000]
img, x , y = camsaxs.remesh(data, g, ai, res=res, coord_sys='theta_alpha', out_range=lims)
box = [x.min(), x.max(), y.min(), y.max()]
plt.imshow(np.log(img+4), origin='lower', extent=box)
plt.savefig('th_al_roi.png')
