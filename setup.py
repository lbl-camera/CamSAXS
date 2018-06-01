
from setuptools import setup

if __name__ == '__main__':
    setup(name='camsaxs',
        version='1.0.0',
        description='Xi-cam.SAXS companion functions',
        author='Dinesh Kumar',
        author_email='dkumar@lbl.gov',
        ext_modules = [ext],
        install_requires = ['numpy', 'scipy', 'astropy'],
        packages = ['camsaxs']

    )
