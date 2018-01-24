#ifndef WARPIMAGE__H
#define WARPIMAGE__H

#include <math.h>

inline double sgn(x) {
  if (x < 0.)
    return -1.;
  else
    return 1.;
}

struct Parameter {
  double center[2];
  double pixel[2];
  double sdd;
  double alphai;
  double k0;
} parameter_t;

struct Container {
  int dim;
  int row;
  int col;
  NPY_TYPES typ;
  void *buf;
} container_t;

static PyObject *qpqz(PyObject *, PyObject *);
static PyObject *qyqz(PyObject *, PyObject *);
static PyObject *theta_alpha(PyObject *, PyObject *);

// utiliity functions
_Bool unpack_array(PyObject *, container_t &);
_Bool get_center(PyObject *, double *);
_Bool get_pixel(PyObject *, double *);

void forward(int, int, double &, double &, parameter_t &);
void inverse(double, double, int &, int &, parameter_t &);

#endif  // WARPIMAGE__H
