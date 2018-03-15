#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys

try:
    import cWarpImage
    C_WARP = False
except ImportError:
    sys.stderr.write('Could not import C extension')
    C_WARP = True

import numpy as np

def remesh(image, geometry, alphai, q_range=None, res=None):
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
    q_range: list, optional
        [qp_min, qz_min, qp_max, qz_max] for the output image
    res: list, optional
        resolution of the output image
    
    Returns
    -------
    qimage: ndarray
        redrawn image
    q_range: list, optional
    """
    rows, cols = image.shape
    center = [geometry.get_poni2(), geometry.get_poni1()]
    pixel = [geometry.get_pixel1(), geometry.get_pixel2()]
    dist = geometry.get_dist()
    # convert wavelen to nanomters
    wavelen = geometry.get_wavelength() * 1.0E+09 
    k0 = 2*np.pi/wavelen

    if res is None:
        res = [rows, cols]
    else:
        if not len(res) == 2:
            sys.stderr.write('resolution should a sequence of two integers')
            exit(1)

    if q_range is None:
        # coordinates for detector corners
        coord = np.multiply([[ 0, ncol], [0, nrow]], pixel) - center
        qx, qy, qz = x2q(coord[0,:], coord[1,:], dist, alphai, k0)

    qp = np.sign(qy) * np.sqrt(qx**2 + qy**2)
    qp = np.linspace(qp[0], qp[1], res[1])
    qz = np.linspace(qz[0], qz[1], res[2])

    if C_WARP:
        cWarpImage.qp_qz(image, qp, qz, pixel, center, alphai, k0, dist, 0)
    else:
        # find reverse map for every pair or qp,qz
        sin_al = qz/k0 - np.sin(alphai)
        cos_al = np.sqrt(1 - sin_al**2)
        with np.errstate(divide='ignore'):
            tan_al = sin_al/cos_al
        
        cos_ai = np.cos(alphai)
        with np.errstate(invalid='ignore'):
            cos_th = (cos_al**2 + cos_ai**2 - (qp/k0)**2)/(2*cos_al*cos_ai)
        sin_th = np.sign(qp) * np.sqrt(1 - cos_th**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            tan_th = sin_th / cos_th
 
        map_x = ((tan_th * dist + center[0])/pixel[0]).astype(np.int)
        with np.errstate(divide='ignore'):
            map_y = ((tan_al * dist / cos_th + center[1])/pixel[1]).astype(np.int)

        mask = np.isfinite(map_x) & np.isfinite(map_y)
        q_image = np.fromiter(image[i,j] for i,j in np.nditer([map_y,map_x]), np.float)
        return q_image;


def x2ang(x, y, d):
    s1 = np.sqrt(d**2 + x**2)
    theta = np.arctan2(d, x)
    alpha = np.arctan2(s1, y) 
    return theta,alpha
    

def x2q(x, y, d, alphai, k0):
    s1 = np.sqrt(x**2 + d**2)
    s2 = np.sqrt(s1**2 + y**2)
    sin_th = x/s1
    cos_th = d/s1
    sin_al = y/s2
    cos_al = s1/s2
    qx = k0 * (cos_al * cos_th - np.cos(alphai))
    qy = k0 * cos_al * sin_th
    qz = k0 * (sin_al + np.sin(alphai))
    return qx,qy,qz    
