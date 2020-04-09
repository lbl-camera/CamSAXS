
from setuptools import setup

if __name__ == '__main__':
    setup(name='camsaxs',
          version='1.0.1',
          description='Xi-cam.SAXS companion functions',
          author='Dinesh Kumar',
          author_email='dkumar@lbl.gov',
          url="http://github.com/lbl-camera/CamSAXS",
          install_requires = ['numpy', 'scipy', 'astropy', 'pyFAI', 'sasmodels', 'pyyaml'],
          packages = ['camsaxs'],
          include_package_data = True,
          package_data={'camsaxs': ['camsaxs/config.yml']}
          )
