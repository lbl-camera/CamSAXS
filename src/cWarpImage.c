#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "warpimage.h"

/* setup methdods table */
static PyMethodDef cWarpImageMehods[] = {
    {"qp_qz", qp_qz, METH_VARARGS, NULL},
    {"qy_qz", qy_qz, METH_VARARGS, NULL},
    {"theta_alpha", theta_alpha, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

/* module struct */
static struct PyModuleDef cWarpImageModule = {
    PyModuleDef_HEAD_INIT, "cWarpImage", NULL, -1, cWarpImageMehods};

/* Initialize the module */
PyMODINIT_FUNC PyInit_cWarpImage(void) {
  import_array();
  return PyModule_Create(&cWarpImageModule);
}
