
from setuptools import setup

if __name__ == '__main__':
    setup(name='camsaxs',
        version='1.0.0',
        description='Xi-cam.SAXS companion functions',
        author='Dinesh Kumar',
        author_email='dkumar@lbl.gov',
        url = "git@github.com:lbl-camera/CamSAXS.git",
        install_requires = ['numpy', 'scipy', 'astropy', 'pyFAI', 'sasmodels'],
        packages = ['camsaxs'],
        data_files=[('camsaxs', ['camsaxs/config.yml'])]
    )
