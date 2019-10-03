#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from pyFAI.geometry import Geometry
from pyFAI.ext import splitBBox
from typing import Tuple
from pyFAI import units

# Limitations of this module:
# This module incompletely supports tilt/rotation. The alpha/phi angles are calculated correctly from the geometry,
# but its unclear how these should map to the reflection geometry -> tilt/rotation are unsupported in reflection.
#
# The transmission -> reflection mapping is done by subtracting alpha_i from the transmission alphas, effectively making
# the horizon at alpha=0. This mapping does not correct the solid angle normalization.
#
# The pyFAI integrators also have strange windowing behavior; if part of the image is cutoff, you need to transform it
# into a reduced coordinate system (and back afterwards).


def q_from_angles(phi, alpha, wavelength):
    r = 4 * np.pi / wavelength
    qx = r * np.sin(phi/2) * np.cos(alpha/2)
    qy = r * np.sin(alpha/2)
    qz = r * np.cos(alpha/2) * np.cos(alpha/2) - 1

    return np.array([qx, qy, qz])


def alpha(x, y, z):
    return np.arctan2(y, np.sqrt(x ** 2 + z ** 2))


def phi(x, y, z):
    return np.arctan2(x, np.sqrt(z ** 2))


def test_show(data, log=True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if log:
        image = np.log(data + 3)
    else:
        image = data
    im = plt.imshow(image)
    fig.colorbar(im, ax=fig.axes)
    plt.show()


def remesh(image: np.ndarray,
           geometry: Geometry,
           reflection: bool,
           alphai: float,
           bins: Tuple[int, int] = None,
           q_h_range: Tuple[float, float] = None,
           q_v_range: Tuple[float, float] = None,
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
    alphai: scalar, deg
        angle of incedence
    out_range: list, optional
        [[left, right],[lower, upper]] for the output image
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
    assert image.shape == geometry.detector.shape
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    px_x, px_y = np.meshgrid(x, y)
    r_z, r_y, r_x = geometry.calc_pos_zyx(d1=px_y, d2=px_x)

    if reflection:
        alphas = alpha(r_x, r_y, r_z) - alphai
        phis = phi(r_x, r_y, r_z)

        k_i = k_f = 2 * np.pi / geometry.wavelength * 1e-10

        q_x = k_f * np.cos(alphas) * np.cos(phis) - k_i * np.cos(alphai)
        q_y = k_f * np.cos(alphas) * np.sin(phis)
        q_z = -(k_f * np.sin(alphas) + k_i * np.sin(alphai))
        q_r = np.sqrt(q_x ** 2 + q_y ** 2) * np.sign(q_y)
        q_h = q_r
        q_v = -q_z
    else:
        alphas = alpha(r_x, r_y, r_z)
        phis = phi(r_x, r_y, r_z)

        q_x, q_y, q_z = q_from_angles(phis, alphas, geometry.wavelength) * 1e-10
        q_v = -q_y  # -q_y
        q_h = q_x

    if bins is None: bins = tuple(image.shape)
    if q_h_range is None: q_h_range = (q_h.min(), q_h.max())
    if q_v_range is None: q_v_range = (q_v.min(), q_v.max())

    I, q_h, q_v, _, _ = splitBBox.histoBBox2d(weights=image,
                                              pos0=q_h,
                                              delta_pos0=np.ones_like(image) * (q_h_range[1] - q_h_range[0]) / bins[0],
                                              pos1=q_v,
                                              delta_pos1=np.ones_like(image) * (q_v_range[1] - q_v_range[0]) / bins[1],
                                              bins=bins,
                                              pos0Range=q_h_range,
                                              pos1Range=q_v_range,
                                              dummy=None,
                                              delta_dummy=None,
                                              allow_pos0_neg=True,
                                              # mask=mask,
                                              # dark=dark,
                                              # flat=flat,
                                              # solidangle=solidangle,
                                              # polarization=polarization,
                                              # normalization_factor=normalization_factor,
                                              # chiDiscAtPi=self.chiDiscAtPi,
                                              # empty=dummy if dummy is not None else self._empty
                                              )
    q_h, q_v = np.meshgrid(q_h, q_v)
    return I, q_h, q_v


if __name__ == '__main__':
    from pyFAI import detectors
    import fabio

    filename = '/home/rp/data/YL1031/AGB_5S_USE_2_2m.edf'

    fit2d = {'centerX': 237.5,
             'centerY': 1678 - 30.500000000000004,
             'directDist': 283.26979219190343,
             'pixelX': 171.99999999999997,
             'pixelY': 171.99999999999997,
             'splineFile': None,
             'tilt': 0.0,
             'tiltPlanRotation': 0.0}

    det = detectors.Pilatus2M
    g = Geometry(wavelength=0.123984E-09)
    g.setFit2D(**fit2d)
    g.detector = det()

    data = fabio.open(filename).data
    ai = np.deg2rad(0.14)

    test_show(data)
    from pyFAI import calibrant

    test_show(calibrant.ALL_CALIBRANTS['AgBh'].fake_calibration_image(g))

    # default qp-qz, same size as detector
    I, q_h, q_v = remesh(data, g, False, 0)  # img, x, y =
    test_show(I)
