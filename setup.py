import os
import sys
from setuptools import setup, find_namespace_packages
from pybind11.setup_helpers import Pybind11Extension
import numpy

__version__ = '1.0.0'

extension_args = {'extra_compile_args': ['-fopenmp', '-std=c++17'],
                  'extra_link_args': ['-lgomp'],
                  'library_dirs': ['/usr/local/lib',
                                   os.path.join(sys.prefix, 'lib')],
                  'include_dirs': [numpy.get_include(),
                                   os.path.join(sys.prefix, 'include'),
                                   os.path.join(os.path.dirname(__file__), 'cbclib/include')]}

extensions = [Pybind11Extension("streak_finder.src.fft_functions",
                                sources=["streak_finder/src/fft_functions.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                libraries = ['fftw3', 'fftw3f', 'fftw3l', 'fftw3_omp', 'fftw3f_omp', 'fftw3l_omp'],
                                **extension_args),
              Pybind11Extension("streak_finder.src.image_proc",
                                sources=["streak_finder/src/image_proc.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args),
              Pybind11Extension("streak_finder.src.median",
                                sources=["streak_finder/src/median.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args)]

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(name='streak_finder',
      version=__version__,
      author='Nikolay Ivanov',
      author_email="nikolay.ivanov@desy.de",
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_namespace_packages(),
      install_requires=['numpy', 'scipy'],
      ext_modules=extensions,
      python_requires='>=3.7')
