
from setuptools import setup, Extension
import numpy


c_src = [ 'src/cWarpImage.cpp', 'src/util.cpp', 'src/qpqv.cpp', 'src/qyqz.cpp', 'src/thal.cpp' ]
c_inc = [ 'src/cWarpImage.h', 'src/util.h', 'src/npy.h' ]
ext = Extension(name='camsaxs.cWarpImage',
                sources = c_src,
                depends = c_inc, 
                language = "c++",
                define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                include_dirs = [ numpy.get_include()],
                libraries = [ 'm' ],
                extra_compile_args = [ '-g', '-O2' ]
        )


if __name__ == '__main__':
    setup(name='camsaxs',
        version='1.0.0',
        description='Xi-cam.SAXS companion functions',
        author='Dinesh Kumar',
        author_email='dkumar@lbl.gov',
        ext_modules = [ext],
        packages = ['camsaxs']

    )
