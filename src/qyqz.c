#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "warpimage.h"
const int DIMS = 2;

static PyObject *qyqz(PyObject *self, PyObject *args) {
  PyObject *pyImg, *pyQp, *pyQz, *PX, *CTR;
  double alphai, k0, sdd;
  int method;

  /* Parse tuples */
  if (!PyArg_ParseTuple(args, "OOOOOdddI", &pyImg, &pyQp, &pyQz, &PX, &CTR,
                        &alphai, &k0, &sdd, &method))
    return NULL;

  /* Interpret Python objects as C-arrays. */
  Container_t Img, Qp, Qz;
  unpack_array(arr1, Img);
  unpack_array(arr2, Qp);
  unpack_array(arr3, Qz);

  /* center */
  double center[DIMS];
  get_center(CTR, center);

  /* pixel size */
  double pixel[DIMS];
  get_pixel(CTR, center);

  /* calculate theta, alpha  at every pixels */
  double *theta, *alpha;
  reciprocal(k0, sdd, alphai, center, pixel, nrow, ncol, theta, alpha);

  /* contruct output array */
  if (Qp.dim == 2) {
    outdims[0] = Qp.row;
    outdims[1] = Qp.col;
  } else if ((Qp.dim == 1) && (Qz.dim == 1))
    outdims[0] = Qp.row;
  outdims[1] = Qz.row;
}
else {
  fprintf(stderr, "error: output coordinates ill-formed.\n");
  return NULL;
}
PyOut = (PyArrayObject *)PyArray_FromDims(DIMS, outdims, NPY_DOUBLE);
double *warped = (double *)PyArray_DATA(PyOut);

for (int i = 0; i < outdims[0]; i++)
  for (int j = 0; j < outdims[1]; j++) {
    // invert (qp, qz)
  }
return PyArray_Return(PyOut);
}
