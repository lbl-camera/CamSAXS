#! /usr/bin/env python
# -*- coding: utf-8 -*-


def tophat2d(radius,  width=10):
    """
    convolution kernel is a Mexican Hat revolved in x-plane

    radius : peak position along the radius
    width: width of peak in pixels
    """
    N = np.int(np.round(radius) + 3 * np.round(width) + 1)
    x = np.arange(-N,N)
    x, y = np.meshgrid(x, x)
    t = np.sqrt(x**2 + y**2) - radius
    a = 1./np.sqrt(2 * np.pi) / width**3
    w = a * (1 - (t/width)**2) * np.exp(-t**2 / width**2 / 2.)
    return w

def cwt2d(img, domain=None, width=None, log=False):
    """
    
   
    """
    nrow,ncol = img.shape
    if domain is None:
        rmin = 0
        rmax = min(nrow, ncol)
    else:
        rmin = domain[0]
        rmax = domain[1]
    maxval = 0
    center = np.array([0,0], dtype=np.int)

    # if log is true
    if log: sig = np.log(img)
    else: sig = img

    for r in range(rmin, rmax):
        w = tophat2(r, width)
        im2 = signal.fftconvolve(sig, w, 'same')
        if im2.max() > maxval:
            maxval = im2.max()
            center = np.unravel_index(im2.argmax(), img.shape)
    return center
