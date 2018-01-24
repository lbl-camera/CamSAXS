#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "warpimage.h"
const int DIMS = 2;

static PyObject *qpqv(PyObject *self, PyObject *args) {
  PyObject *pyImg, *pyQp, *pyQz, *PX, *CTR;
  double alphai, k0, sdd;
  int method;

  /* Parse tuples */
  if (!PyArg_ParseTuple(args, "OOOOOdddI", &pyImg, &pyQp, &pyQz, &PX, &CTR,
                        &alphai, &k0, &sdd, &method))
    return NULL;

  /* Interpret Python objects as C-arrays. */
  container_t Img, Qp, Qz;
  unpack_array(arr1, Img);
  unpack_array(arr2, Qp);
  unpack_array(arr3, Qz);

  /* put calibration parameters into struct */
  parameter_t p;

  /* center */
  double center[DIMS];
  get_center(CTR, p.center);

  /* pixel size */
  double pixel[DIMS];
  get_pixel(PX, p.pixel);

  /* rest of the bunch */
  p.sdd = sdd;
  p.k0 = k0;
  p.alphai = alphai;

  /* contruct output array */
  int nrow, ncol;
  if (Qp.dim == 2) {
    outdims[0] = Qp.row;
    outdims[1] = Qp.col;
  } else if ((Qp.dim == 1) && (Qz.dim == 1)) {
    outdims[0] = Qp.row;
    outdims[1] = Qz.row;
  } else {
    fprintf(stderr, "error: output coordinates ill-formed.\n");
    return NULL;
  }
  PyOut = (PyArrayObject *)PyArray_FromDims(DIMS, outdims, NPY_DOUBLE);
  double *warped = (double *)PyArray_DATA(PyOut);

  for (int i = 0; i < outdims[0]; i++)
    for (int j = 0; j < outdims[1]; j++) {
      int ix = -1;
      int iy = -1;
      double theta, alpha;
      forward(j, i, theta, alpha, p);
      inverse(theta, alpha, ix, iy, p);

      /* use sane values only */
      ix = max(0, ix);
      ix = min(Img.col-1, ix);
      iy = max(0, iy);
      iy = min(Img.row-1, iy);
      int k = iy * Img.col + ix;

      if ((ix < 0) || (iy < 0)){
        fprintf(stderr,"error: either \"fwd\" or \"inv\" model failed.\n");
        return NULL;
      }
      switch (Img.typ){
        case NPY_INT:
        case NPY_INT32:
          warped[i * outdims[1] + j] = (double)(((int *) Img.buf)[k]);
          break;
        case NPY_FLOAT:
        case NPY_FLOAT32:
          warped[i * outdims[1] + j] = (double)(((float *) Img.buf)[k]);
          break;
        case NPY_DOUBLE:
        case NPY_FLOAT64: 
          warped[i * outdims[1] + j] = ((double *) Img.buf)[k];
          break;
        default:
          fprintf(stderr,"error: unknown/unsupported dtype.\n");
          return NULL;
    }
  return PyArray_Return(PyOut);
}
