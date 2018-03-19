#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import camsaxs
import numpy as np
import pyFAI
import ipdb

filename = 'GIWAXS_sfloat_2m.gb'
center = [632., 69.5]
pixel = [172E-06, 172E-06]
poni = np.multiply(center, pixel)
dist = 300.
det = pyFAI.detectors.Pilatus2M
g = pyFAI.geometry.Geometry(dist=0.3, poni1=poni[1], poni2=poni[0], 
            pixel1=pixel[0], pixel2=pixel[1], wavelength=0.123984E-09)

data = np.fromfile(filename, dtype=np.float32).reshape(1679,1475)

ipdb.set_trace()
img, qp, qz = camsaxs.remesh(data, g, np.deg2rad(0.14))
