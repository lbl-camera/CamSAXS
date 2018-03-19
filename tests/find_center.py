#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import camsaxs
import matplotlib.pyplot as plt
import numpy as np
import ipdb

filename = 'AgB_sfloat_2m.gb'

data = np.fromfile(filename, dtype=np.float32).reshape(1679, 1475)
center, radius = camsaxs.cwt2d(data, domain=[35, 45], log=True)

print(center, radius)
th = np.linspace(0, 2 * np.pi, 100)
x = center[1] + radius * np.cos(th)
y = center[0] + radius * np.sin(th)
plt.imshow(np.log(data+3))
plt.scatter(x, y)
plt.show()
