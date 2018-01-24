#! /usr/bin/env python


def tophat2d(radius, scale=1, width=10):
    """
    convolution kernel is a Mexican Hat revolved in x-plane
    
    radius : peak position along the radius
    scale  : magnification factor
    """
    N = np.int(np.round(radius) + 3 * np.round(width) + 1)
    x = np.arange(-N,N)
    x, y = np.meshgrid(x, x)
    t = np.sqrt(x**2 + y**2) - radius
    a = scale/np.sqrt(2 * np.pi) / width**3
    w = a * (1 - (t/width)**2) * np.exp(-t**2 / width**2 / 2.)
    return w

def cwt2d(img, rmin=None, rmax=None, scale=None, width=10):
    """
    
   
    """
    nrow,ncol = img.shape
    if rmin is None: rmin = 0
    if rmax is None: rmax = min(nrow, ncol)
    if scale is None: scale = img.mean()
    maxval = 0
    center = np.array([0,0], dtype=np.int)
    for r in range(rmin, rmax):
        w = tophat2(r, scale, width)
        im2 = signal.fftconvolve(img, w, 'same')
        if im2.max() > maxval:
            maxval = im2.max()
            center = np.unravel_index(im2.argmax(), img.shape)
    return center
