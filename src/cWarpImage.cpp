#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL CAMSAXS_ARRAY_API
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "cWarpImage.h"

static PyObject * f1(PyObject *self, PyObject *args){
    return qpqv(self, args);
}

static PyObject * f2(PyObject *self, PyObject *args){
    return qyqz(self, args);
}

static PyObject * f3(PyObject *self, PyObject *args){
    return theta_alpha(self, args);
}


/* setup methdods table */
static PyMethodDef cWarpImageMehods[] = {
    {"qp_qz", f1, METH_VARARGS, NULL},
    {"qy_qz", f2, METH_VARARGS, NULL},
    {"theta_alpha", f3, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

/* module struct */
static struct PyModuleDef cWarpImageModule = {
    PyModuleDef_HEAD_INIT, "cWarpImage", NULL, -1, cWarpImageMehods};

/* Initialize the module */
PyMODINIT_FUNC PyInit_cWarpImage(void) {
    import_array();
    return PyModule_Create(&cWarpImageModule);
}
