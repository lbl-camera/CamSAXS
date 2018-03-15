#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
try:
    import cWarpImage
    C_WARP = False
    remesh = {
        'qp_qz': cWarpImage.qp_qz,
        'qy_qz': cWarpImage.qy_qz,
        'theta_alpha': cWarpImage.theta_alpha
    }
except ImportError:
    sys.stderr.write('Could not import C extension')
    C_WARP = True
    remesh = {'qp_qz': qp_qz, 'qy_qz': qy_qz, 'theta_alpha': theta_alpha}


def remesh(image,
           geometry,
           alphai,
           out_range=None,
           res=None,
           coord_sys='qp_qz'):
    """
    Redraw the GI Image in (qp, qz) coordinates.

    Parameters:
    -----------
    image: ndarray, 
        detector image 
    geometry: pyFAI Geometry
        PONI, Pixel Size, Sample Det dist etc
    alphai: scalar, 
        angle of incedence
    out_range: list, optional
        [[lower left], [upper right]] for the output image
    res: list, optional
        resolution of the output image
    coord_sys: str
        'qp_qz', 'qy_qz' or 'theta_alpha' 
    Returns
    -------
    qimage: ndarray
        remeshed/warped image
    xcrd: ndarray
        x-coords of warped image
    ycrd: ndarray
        y-coords of warped image
    """
    rows, cols = image.shape
    center = [geometry.get_poni2(), geometry.get_poni1()]
    pixel = [geometry.get_pixel1(), geometry.get_pixel2()]
    dist = geometry.get_dist()
    # convert wavelen to nanomters
    wavelen = geometry.get_wavelength() * 1.0E+09
    k0 = 2 * np.pi / wavelen

    if res is None:
        res = [rows, cols]
    else:
        if not len(res) == 2:
            sys.stderr.write('resolution should a sequence of two integers')
            exit(1)

    if coord_sys == 'qp_qz':
        if out_range is None:
            # coordinates for detector corners
            coord = np.multiply([[0, ncol], [0, nrow]], pixel) - center
            qx, qy, qz = x2q(coord[0, :], coord[1, :], dist, alphai, k0)
            qp = np.sign(qy) * np.sqrt(qx**2 + qy**2)
        else:
            qp = [out_range[0][0], out_range[1][0]]
            qz = [out_range[0][1], out_range[1][1]]

        xcrd = np.linspace(qp[0], qp[1], res[0])
        ycrd = np.linspace(qz[0], qz[1], res[1])

    elif coord_sys == 'qy_qz':
        if out_range is None:
            coord = np.multiply([[0, ncol], [0, nrow]], pixel) - center
            _, qy, qz = x2q(coord[0, :], coord[1, :], dist, alphai, k0)
        else:
            qy = [out_range[0][0], out_range[1][0]]
            qz = [out_range[0][1], out_range[1][1]]

        xcrd = np.linspace(qy[0], qy[1], res[0])
        ycrd = np.linspace(qz[0], qz[1], res[1])

    elif coord_sys == 'theta_alpha':
        if out_range is None:
            coord = np.multiply([[0, ncol], [0, nrow]], pixel) - center
            _, qy, qz = x2q(coord[0, :], coord[1, :], dist, alphai, k0)
        else:
            theta = [out_range[0][0], out_range[1][0]]
            alpha = [out_range[0][1], out_range[1][1]]

        xcrd = np.linspace(theta[0], theta[1], res[0])
        ycrd = np.linspace(alpha[0], alpha[1], res[1])

    xcrd, ycrd = np.meshgid(xcrd, ycrd)
    qimage = remeshed[coord_sys](image, xcrd, ycrd, pixel, center, alphai, k0,
                                 dist, 0)
    return qimg, xcrd, ycrd


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

    mask = np.isfinite(map_x) & np.isfinite(map_y)
    map_x = map_x.astype(int)
    map_y = map_y.astype(int)
    q_image = np.fromiter([
        image[i, j] if mask[i, j] else 0. for i, j in np.nditer([map_y, map_x])
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
    map_x = map_x.astype(int)
    map_y = map_y.astype(int)
    q_image = np.fromiter([
        image[i, j] if mask[i, j] else 0. for i, j in np.nditer([map_y, map_x])
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
    map_x = map_x.astype(int)
    map_y = map_y.astype(int)
    q_image = np.fromiter([
        image[i, j] if mask[i, j] else 0. for i, j in np.nditer([map_y, map_x])
    ], np.float)
    return q_image.reshape(mask.shape)


def x2ang(x, y, d):
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
