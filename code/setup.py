from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'flux_calculation',
  ext_modules = cythonize("./crushinator/flux_calculation.pyx"),
)
