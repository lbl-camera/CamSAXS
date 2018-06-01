#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import camsaxs
import matplotlib.pyplot as plt
import numpy as np


# create a SASModel
category = 'Cylinder Functions'
name = 'Cylinder'
parameters = camsaxs.models[category][name]['params']
model = camsaxs.XicamSASModel(name, parameters)

q = np.linspace(0, 1, 512)
I0 = model(q)
r0 = model.radius.value
l0 = model.length.value

# add noise
data = I0 + np.random.normal(I0, 0.1 * np.sqrt(I0))

# fit it
model.radius = 5.
model.length = 100
model.fixed['radius'] = False
model.fixed['length'] = False


g, vals = camsaxs.fit_sasmodel(model, q, data, acc=0.001)
print('radius = %f, length = %f' % (r0, l0))
print('model init: r = %f, l = %f' % (model.radius.value, model.length.value))
print('Fit = %f, %f' % (g[3], g[2]))
plt.semilogy(q, I0)
plt.semilogy(q, data, 'ko')
plt.semilogy(q, vals)
plt.show()
