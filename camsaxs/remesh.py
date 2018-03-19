#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
try:
    import cWarpImage
    remesh_fcn = {
        'qp_qz': cWarpImage.qp_qz,
        'qy_qz': cWarpImage.qy_qz,
        'theta_alpha': cWarpImage.theta_alpha
    }
except ImportError:
    from .warp_image import qp_qz, qy_qz, theta_alpha, x2q, x2angles 
    remesh_fcn = {'qp_qz': qp_qz, 'qy_qz': qy_qz, 'theta_alpha': theta_alpha}


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
    center = np.array([geometry.get_poni2(), geometry.get_poni1()])
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
            coord = np.multiply([[0, cols], [0, rows]], pixel) - center[:,np.newaxis]
            qx, qy, qz = x2q(coord[0, :], coord[1, :], dist, alphai, k0)
            qp = np.sign(qy) * np.sqrt(qx**2 + qy**2)
        else:
            qp = [out_range[0][0], out_range[1][0]]
            qz = [out_range[0][1], out_range[1][1]]

        xcrd = np.linspace(qp[0], qp[1], res[1])
        ycrd = np.linspace(qz[0], qz[1], res[0])

    elif coord_sys == 'qy_qz':
        if out_range is None:
            coord = np.multiply([[0, cols], [0, rows]], pixel) - center[:,np.newaxis]
            _, qy, qz = x2q(coord[0, :], coord[1, :], dist, alphai, k0)
        else:
            qy = [out_range[0][0], out_range[1][0]]
            qz = [out_range[0][1], out_range[1][1]]

        xcrd = np.linspace(qy[0], qy[1], res[1])
        ycrd = np.linspace(qz[0], qz[1], res[0])

    elif coord_sys == 'theta_alpha':
        if out_range is None:
            coord = np.multiply([[0, cols], [0, rows]], pixel) - center[:,np.newaxis]
            _, qy, qz = x2angles(coord[0, :], coord[1, :], dist, alphai, k0)
        else:
            theta = [out_range[0][0], out_range[1][0]]
            alpha = [out_range[0][1], out_range[1][1]]

        xcrd = np.linspace(theta[0], theta[1], res[1])
        ycrd = np.linspace(alpha[0], alpha[1], res[0])

    xcrd, ycrd = np.meshgrid(xcrd, ycrd)
    qimage = remesh_fcn[coord_sys](image, xcrd, ycrd, pixel, center, alphai, k0,
                                 dist, 0)
    return qimg, xcrd, ycrd
