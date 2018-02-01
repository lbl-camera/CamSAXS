#include <Python.h>
#include "npy.h"
#include <iostream>

#include "util.h"
#include "cWarpImage.h"

PyObject *qyqz(PyObject *self, PyObject *args) {
    PyObject *pyImg, *pyQy, *pyQz, *PX, *CTR;
    double alphai, k0, sdd;
    int method;
    /* Parse tuples */
    if (!PyArg_ParseTuple(args, "OOOOOdddI", &pyImg, &pyQy, &pyQz, &PX, &CTR,
                          &alphai, &k0, &sdd, &method))
        return NULL;

    /* Interpret Python objects as C-arrays. */
    container_t Img(pyImg);
    container_t Qy(pyQy);
    container_t Qz(pyQz);

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
    if ((Qy.dim() == 2) && (Qz.dim() == 2)){
        outdims[0] = Qy.row();
        outdims[1] = Qy.col();
    } else if ((Qy.dim() == 1) && (Qz.dim() == 1)) {
        outdims[0] = Qy.row();
        outdims[1] = Qz.row();
    } else {
        std::cerr << "error: output coordinates are ill-formed." << std::endl;
        return NULL;
    }

    const int DIMS = 2;
    PyArrayObject *PyOut = (PyArrayObject *)PyArray_FromDims(DIMS, outdims, NPY_DOUBLE);
    double *warped = (double *)PyArray_DATA(PyOut);

    double sin_ai = sin(p.alphai);
    for (int i = 0; i < outdims[0]; i++)
        for (int j = 0; j < outdims[1]; j++) {
            int ix = -1;
            int iy = -1;
            double qz = Qz[i];
            double qy = Qy[j];
            
            double sin_al = (qz/p.k0) - sin_ai;
            double cos_al = std::sqrt(1. - sin_al * sin_al);
            double sin_th = (qy/p.k0/cos_al);
            if (sin_th > 1) {
                warped[i * outdims[1] + j] = 0;
                continue;
            }
            // otherwise, calculate cos(theta)
            double cos_th = std::sqrt(1 - sin_th * sin_th);
            
            // map theta and alpha  to nearest indecies on the image
            inverse(cos_th, cos_al, ix, iy, p);

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
