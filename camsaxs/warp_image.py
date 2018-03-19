#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np

def qp_qz(image, qp, qz, pixel, center, alphai, k0, dist, method):
    # find reverse map for every pair or qp,qz
    sin_al = qz / k0 - np.sin(alphai)
    cos_al = np.sqrt(1 - sin_al**2)
    with np.errstate(divide='ignore'):
        tan_al = sin_al / cos_al

    cos_ai = np.cos(alphai)
    with np.errstate(invalid='ignore'):
        cos_th = (cos_al**2 + cos_ai**2 - (qp / k0)**2) / (2 * cos_al * cos_ai)
    sin_th = np.sign(qp) * np.sqrt(1 - cos_th**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        tan_th = sin_th / cos_th

    map_x = ((tan_th * dist + center[0]) / pixel[0])
    with np.errstate(divide='ignore'):
        map_y = ((tan_al * dist / cos_th + center[1]) / pixel[1])

    mask = np.isnan(map_x) | np.isnan(map_y)
    map_x = map_x.astype(int).tolist()
    map_y = map_y.astype(int).tolist()
    q_image = np.fromiter([
        image[i,j] if mask[i,j] else 0. for i, j in zip(map_x,map_y)
    ], np.float)
    return q_image.reshape(mask.shape)


def qy_qz(image, qy, qz, pixel, center, alphai, k0, dist, method):
    sin_al = qz / k0 - np.sin(alphai)
    cos_al = np.sqrt(1 - sin_al**2)
    sin_th = qy / (k0 * cos_al)
    cos_th = np.sqrt(1 - sin_th**2)

    with np.errstate(divide='ignore'):
        tan_al = sin_al / cos_al

    with np.errstate(divide='ignore'):
        tan_th = sin_th / cos_th

    map_x = ((tan_th * dist + center[0]) / pixel[0])
    with np.errstate(divide='ignore'):
        map_y = ((tan_al * dist / cos_th + center[1]) / pixel[1])

    mask = np.isfinite(map_x) & np.isfinite(map_y)
    map_x = map_x.astype(int).tolist()
    map_y = map_y.astype(int).tolist()
    q_image = np.fromiter([
        image[i, j] if mask[i, j] else 0. for i, j in zip([map_y, map_x])
    ], np.float)
    return q_image.reshape(mask.shape)


def theta_alpha(image, theta, alpha, pixel, center, alphai, k0, dist, method):
    tan_th = np.tan(theta)
    cos_th = np.cos(theta)
    tan_al = np.tan(alpha)
    map_x = (tan_th * dist + center[0]) / pixel[0]

    with np.errstate(divide='ignore'):
        map_y = (tan_al * dist / cos_th + center[1]) / pixel[1]

    mask = np.isfinite(map_x) & np.isfinite(map_y)
    map_x = map_x.astype(int).tolist()
    map_y = map_y.astype(int).tolist()
    q_image = np.fromiter([
        image[i, j] if mask[i, j] else 0. for i, j in zip([map_y, map_x])
    ], np.float)
    return q_image.reshape(mask.shape)


def x2angles(x, y, d):
    s1 = np.sqrt(d**2 + x**2)
    theta = np.arctan2(d, x)
    alpha = np.arctan2(s1, y)
    return theta, alpha


def x2q(x, y, d, alphai, k0):
    s1 = np.sqrt(x**2 + d**2)
    s2 = np.sqrt(s1**2 + y**2)
    sin_th = x / s1
    cos_th = d / s1
    sin_al = y / s2
    cos_al = s1 / s2
    qx = k0 * (cos_al * cos_th - np.cos(alphai))
    qy = k0 * cos_al * sin_th
    qz = k0 * (sin_al + np.sin(alphai))
    return qx, qy, qz
