import os
import sys
from setuptools import setup, find_namespace_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy

__version__ = '1.3.0'

extension_args = {'extra_compile_args': ['-fopenmp', '-std=c++20'],
                  'extra_link_args': ['-lgomp'],
                  'library_dirs': ['/usr/local/lib',
                                   os.path.join(sys.prefix, 'lib')],
                  'include_dirs': [numpy.get_include(),
                                   os.path.join(sys.prefix, 'include')]}

extensions = [Pybind11Extension("streak_finder._src.src.bresenham",
                                sources=["streak_finder/_src/src/bresenham.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args),
              Pybind11Extension("streak_finder._src.src.label",
                                sources=["streak_finder/_src/src/label.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args),
              Pybind11Extension("streak_finder._src.src.signal_proc",
                                sources=["streak_finder/_src/src/signal_proc.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args),
              Pybind11Extension("streak_finder._src.src.streak_finder",
                                sources=["streak_finder/_src/src/streak_finder.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args)]

setup(version=__version__,
      packages=find_namespace_packages(),
      install_package_data=True,
      install_requires=['numpy', 'scipy'],
      ext_modules=extensions,)
