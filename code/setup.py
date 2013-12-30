# python setup.py build_ext --inplace --rpath=...
import os

from distutils.core import setup, Extension
from Cython.Distutils import build_ext

ext = [Extension('interpolation', ['./crushinator/interpolation.pyx'],
                 libraries=['gsl', 'gslcblas'],
                 library_dirs=['/home/rfadely/local/lib/'],
                 include_dirs=['/home/rfadely/local/include/', '.']),
       Extension('flux_calculation', ['./crushinator/flux_calculation.pyx'])]

setup(cmdclass={'build_ext':build_ext}, ext_modules=ext)

os.system('mv interpolation.so ./crushinator/')
os.system('mv flux_calculation.so ./crushinator/')
