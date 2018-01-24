#include <Python.h>
#include <numpy/arrayobject.h>

#include "wrapimage.h"

/* Error codes */
const unsigned BAD_DIMS  = 111;
const unsigned BAD_ARRAY = 112;
const unsigned BAD_TYPE  = 113;

/*
 * The function unpacks array to a Container_t stuct.
 * It does not check for the object-type, that should
 * be done on the caller side.
 */
_Bool unpack_array(PyObject *obj, Container_t &ctn) {
    PyArrayObject *arr = (PyArrayObject *)obj;
    ctn.dim            = PyArray_NDIM(arr);
    ctn.row            = (int)PyArray_DIM(obj, 0);
    if (ctn.dim == 2)
        ctn.col = (int)PyArray_DIM(obj, 1);
    else
        ctn.col = 1;
    ctn.buf = PyArray_DATA(obj);
    ctn.typ = PyArray_Type(obj);
    return true;
}

unsigned _two_item_obj(PyObject *obj, double *data) {
    /* pixel size can be list, tuple or array */
    const Py_ssize_t DIMS = 2;
    if (PyArray_Check(PX)) {
        if (PyArray_NDIM(obj) != DIMS) return BAD_DIMS;
        Container_t ctr;
        if (!unpack_array(PX, ctr)) return BAD_ARRAY;
        switch (cpx.typ) {
            case NPY_FLOAT:
            case NPY_FLOAT32: {  // scoping
                float tmp = (float *)ctr.buf;
                for (int i = 0; i < DIMS; i++) data[i] = (double)ctr.buf[i];
            } break;
            case NPY_INT:
            case NPY_INT32:
            case NPY_UINT32:
            case NPY_UINT: {
                int tmp = (int *)ctr.buf;
                for (int i = 0; i < DIMS; i++) data[i] = (double)ctr.buf[i];
            } break;
            case NPY_DOUBLE:
            case NPY_FLOAT64:
                for (int i = 0; i < DIMS; i++) data[i] = (double)ctr.buf[i];
                break;
            default:
                fprintf(stderr, "error: un-supported dtype\n");
                return BAD_TYPE;
        }
    } else if (PyList_Check(obj)) {
        if (PyList_Size(obj) != DIMS) return BAD_DIMS;
        data[0] = PyFloat_AsDouble(PyList_GetItem(obj, 0));
        data[1] = PyFloat_AsDouble(PyList_GetItem(obj, 1));
    } else if (PyTuple_Check(obj)) {
        if (PyTuple_Size(obj) != DIMS) return BAD_DIMS;
        data[0] = PyFloat_AsDouble(PyList_GetItem(obj, 0));
        data[1] = PyFloat_AsDouble(PyList_GetItem(obj, 1));
    } else {
        return BAD_TYPE;
    }
    return 0;
}

_Bool get_center(PyObject *obj, double *center) {
    unsigned res = _two_item_obj(obj, center);
    if (res != 0) {
        fprintf(stderr, "_two_item_obj returned %u error code. investigate.\n",
                res);
        return false;
    }
    return true;
}

_Bool get_pixel(PyObject *obj, double *pixel) {
    unsigned res = _two_item_obj(obj, pixel);
    if (res != 0) {
        fprintf(stderr, "_two_item_obj returned %u error code. investigate.\n",
                res);
        return false;
    }
    return true;
}

void forward(int ix, int iy, double &theta, double &alpha, parameter_t &p) {
    double x = (ix - p.center[0]) * p.pixel[0];
    double y = (iy - p.center[1]) * p.pixel[1];
    double d = p.sdd;
    double l = sqrt(x * x + d * d);
    theta = atan(x / d);
    alpha = atan(y / l);
}

void inverse(double theta, double alpha, int &ix, int &iy, parameter_t &p) {
    double  d  = p.sdd;
    double  x  = d * tan(theta);
    double  l  = sqrt(d * d + x * x);
    double  y  = l * tan(alpha);
    ix = (int)(x/p.pixel[0] + p.center[0]);
    iy = (int)(y/p.pixel[1] + p.center[1]);
}
