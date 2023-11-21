# Convergent beam streak finder
Connection-based streak finding algorithm for convergent beam diffraction patterns.

## Dependencies

- [Pybind11](https://github.com/pybind/pybind11) 2.11 or later.
- [Python](https://www.python.org/) 3.7 or later (Python 2.x is **not** supported).
- [h5py](https://www.h5py.org) 2.10.0 or later.
- [NumPy](https://numpy.org) 1.19.0 or later.
- [SciPy](https://scipy.org) 1.5.2 or later.
- [tqdm](https://github.com/tqdm/tqdm) 4.66 or later.

## Installation from source
In order to build the package from source simply execute the following command:

    python setup.py install

or:

    pip install -r requirements.txt -e . -v

That builds the C++ extensions into ``/streak_finder/src``.