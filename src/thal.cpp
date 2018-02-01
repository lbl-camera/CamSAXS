#include <Python.h>
#include "npy.h"
#include <iostream>

#include "util.h"
#include "cWarpImage.h"

PyObject *theta_alpha(PyObject *self, PyObject *args) {
    PyObject *pyImg, *pyTheta, *pyAlpha, *PX, *CTR;
    double alphai, k0, sdd;
    int method;
    /* Parse tuples */
    if (!PyArg_ParseTuple(args, "OOOOOdddI", &pyImg, &pyTheta, &pyAlpha, &PX, &CTR,
                          &alphai, &k0, &sdd, &method))
        return NULL;

    /* Interpret Python objects as C-arrays. */
    container_t Img(pyImg);
    container_t Theta(pyTheta);
    container_t Alpha(pyAlpha);

    /* put calibration parameters into struct */
    parameter_t p;

    /* center */
    get_pair(CTR, p.center);

    /* pixel size */
    get_pair(PX, p.pixel);

    /* rest of the bunch */
    p.sdd = sdd;
    p.k0 = k0;
    p.alphai = alphai;

    /* contruct output array */
    int outdims[2];
    if ((Theta.dim() == 2) && (Alpha.dim() == 2)){
        outdims[0] = Theta.row();
        outdims[1] = Theta.col();
    } else if ((Theta.dim() == 1) && (Alpha.dim() == 1)) {
        outdims[0] = Theta.row();
        outdims[1] = Alpha.row();
    } else {
        std::cerr << "error: output coordinates are ill-formed." << std::endl;
        return NULL;
    }

    const int DIMS = 2;
    PyArrayObject *PyOut = (PyArrayObject *)PyArray_FromDims(DIMS, outdims, NPY_DOUBLE);
    double *warped = (double *)PyArray_DATA(PyOut);

    for (int i = 0; i < outdims[0]; i++)
        for (int j = 0; j < outdims[1]; j++) {
            int ix = -1;
            int iy = -1;
            double al = Alpha[i];
            double th = Theta[j];
            inverse(cos(th), cos(al), ix, iy, p);

            /* use sane values only */
            ix = std::max(0, ix);
            ix = std::min(Img.col() - 1, ix);
            iy = std::max(0, iy);
            iy = std::min(Img.row() - 1, iy);
            if ((ix < 0) || (iy < 0)) {
                fprintf(stderr,
                        "error: either \"fwd\" or \"inv\" model failed.\n");
                return NULL;
            }
            /* TODO: do the fancy interpolations here */
            warped[i * outdims[1] + j] = Img(iy, ix);
        }
    return PyArray_Return(PyOut);
}
