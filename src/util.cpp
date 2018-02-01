#include <Python.h>
#include "npy.h"

#include "cWarpImage.h"
#include "util.h"

/*
 * Center and Pixel size can be either numpy.ndarray, list or tuple.
 * This utility function checks for each and returns an array of
 * two double-precision numbers.
 */ 
bool get_pair(PyObject * obj, double *val){
    if PyArray_Check(obj){
        container_t C(obj);
        val[0] = C[0];
        val[1] = C[1];
    } else if (PyList_Check(obj)){
        for (Py_ssize_t i = 0; i < 2; i++){
            PyObject * x = PyList_GetItem(obj, i);
            if (PyFloat_Check(x)) val[i] = PyFloat_AsDouble(x);
            else if (PyLong_Check(x)) val[i] = (double) PyLong_AsLong(x);
            else return false;
        }
    } else if (PyTuple_Check(obj)){
        for (Py_ssize_t i = 0; i < 2; i++){
            PyObject * x = PyTuple_GetItem(obj, i);
            if (PyFloat_Check(x)) val[i] = PyFloat_AsDouble(x);
            else if (PyLong_Check(x)) val[i] = (double) PyLong_AsLong(x);
            else return false;
        }
    } else {
        std::cerr << "error: unsupported data-type" << std::endl;
        return false;
    }
    return true;
}
        
void forward(int ix, int iy, double &theta, double &alpha, parameter_t &p) {
    double x = (ix - p.center[0]) * p.pixel[0];
    double y = (iy - p.center[1]) * p.pixel[1];
    double d = p.sdd;
    double l = std::sqrt(x * x + d * d);
    theta = std::atan2(d, x);
    alpha = std::atan2(l, y);
}

void inverse(double cos_theta, double cos_alpha, int &ix, int &iy, parameter_t &p) {
    double tan_theta = (1 - cos_theta*cos_theta)/cos_theta;
    double tan_alpha = (1 - cos_alpha*cos_alpha)/cos_alpha;
    double d = p.sdd;
    double x = d * tan_theta;
    double l = std::sqrt(d * d + x * x);
    double y = l * tan_alpha;
    ix = (int)(x / p.pixel[0] + p.center[0]);
    iy = (int)(y / p.pixel[1] + p.center[1]);
}

void qval(double theta, double alpha, parameter_t & p, double *q) {
    double ai = p.alphai;
    double k0 = p.k0;
    q[0] = k0 * (std::cos(alpha) * std::cos(theta) - std::cos(ai));
    q[1] = k0 * (std::cos(alpha) * std::sin(theta));
    q[2] = k0 * (std::sin(alpha) + std::sin(ai));
}
