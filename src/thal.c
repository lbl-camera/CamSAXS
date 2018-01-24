#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "warpimage.h"
const int DIMS = 2;

static PyObject *theta_alpha(PyObject *self, PyObject *args) {
  PyObject *pyImg, *pyTheta, *pyAlpha, *PX, *CTR;
  double alphai, k0, sdd;
  int method;

  /* Parse tuples */
  if (!PyArg_ParseTuple(args, "OOOOOdddI", &pyImg, &pyTheta, &pyAlpha, &PX,
                        &CTR, &alphai, &k0, &sdd, &method))
    return NULL;

  /* Interpret Python objects as C-arrays. */
  Container_t Img, Th, Al;
  unpack_array(pyImg, Img);
  unpack_array(pyTheta, Th);
  unpack_array(pyAlpha, Al);

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
  if (Th.dim == 2) {
    outdims[0] = Th.row;
    outdims[1] = Th.col;
  } else if ((Th.dim == 1) && (Al.dim == 1))
    outdims[0] = Th.row;
  outdims[1] = Al.row;
}
else {
  fprintf(stderr, "error: output coordinates ill-formed.\n");
  return NULL;
}
PyOut = (PyArrayObject *)PyArray_FromDims(DIMS, outdims, NPY_DOUBLE);
double *warped = (double *)PyArray_DATA(PyOut);

for (int i = 0; i < outdims[0]; i++)
  for (int j = 0; j < outdims[1]; j++) {
    // invert
  }
return PyArray_Return(PyOut);
}
