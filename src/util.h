#ifndef WARPIMAGE__H
#define WARPIMAGE__H
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>

constexpr double sign(auto x) {
    if (x < 0) return -1.;
    if (x > 0) return 1;
    return 0;
}

struct Parameter {
    double pixel[2];
    double center[2];
    double sdd;
    double alphai;
    double k0;
};
typedef Parameter parameter_t;

class container_t {
    int dim_;
    int row_;
    int col_;
    int typ_;
    void *buf;

   public:
    // constructor
    container_t(PyObject * obj){
        PyArrayObject *arr = (PyArrayObject *)obj;
        dim_ = PyArray_NDIM(arr);
        row_ = static_cast<int>(PyArray_DIM(arr, 0));
        if (dim_ == 2) col_ = static_cast<int>(PyArray_DIM(arr, 1));
        else col_ = 1;
        typ_ = PyArray_TYPE(arr);
        buf = PyArray_DATA(arr);
    }

    // getters
    int dim() const { return dim_; }
    int row() const { return row_; }
    int col() const { return col_; }
    int dtype() const { return typ_; }
    double operator[](int i) const {
        size_t w = 0;
        switch(typ_){
            case NPY_INT:
                w = i  * sizeof(int);
                return static_cast<double>(*(static_cast<int *>(buf) + w));
            case NPY_FLOAT:
                w = i  * sizeof(float);
                return static_cast<double>(*(static_cast<float *>(buf) + w));
            case NPY_DOUBLE:
                w = i  * sizeof(double);
                return *(static_cast<int *>(buf) + w);
            default:
                std::cerr << "error: unsupported data-type." << std::endl;
                throw 1;
        }
    }
    double operator()(int i, int j) const { return (*this)[i*col_ + j]; }
};

// utiliity functions
bool get_pair(PyObject *, double *);
void forward(int, int, double &, double &, parameter_t &);
void inverse(double, double, int &, int &, parameter_t &);
void qval(double, double, parameter_t &, double *);

#endif  // WARPIMAGE__H
