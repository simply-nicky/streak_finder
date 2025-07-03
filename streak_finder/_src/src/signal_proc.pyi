from typing import List, Optional, Sequence, Tuple, overload
from ..annotations import IntArray, NDRealArray, RealArray, IntSequence

def binterpolate(inp: RealArray, grid: Sequence[RealArray | IntArray],
                 coords: RealArray | IntArray, num_threads: int=1) -> NDRealArray:
    """Perform bilinear multidimensional interpolation on regular grids. The integer grid starting
    from ``(0, 0, ...)`` to ``(inp.shape[0] - 1, inp.shape[1] - 1, ...)`` is implied.

    Args:
        inp : The data on the regular grid in n dimensions.
        grid : A tuple of grid coordinates along each dimension (`x`, `y`, `z`, ...).
        coords : A list of N coordinates [(`x_hat`, `y_hat`, `z_hat`), ...] to sample the gridded
            data at.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If ``inp`` and ``coords`` have incompatible shapes.

    Returns:
        Interpolated values at input coordinates.
    """
    ...

def kr_predict(y: RealArray, x: RealArray, x_hat: RealArray, sigma: float,
               kernel: str='gaussian', w: Optional[RealArray]=None,
               num_threads: int=1) -> NDRealArray:
    """Perform the multi-dimensional Nadaraya-Watson kernel regression [KerReg]_.

    Args:
        y : The data to fit.
        x : Coordinates array.
        x_hat : Set of coordinates where the fit is to be calculated.
        sigma : Kernel bandwidth.
        kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
            are available:

            * 'biweigth' : Quartic (biweight) kernel.
            * 'gaussian' : Gaussian kernel.
            * 'parabolic' : Epanechnikov (parabolic) kernel.
            * 'rectangular' : Uniform (rectangular) kernel.
            * 'triangular' : Triangular kernel.

        w : A set of weights, unitary weights are assumed if it's not provided.
        num_threads : Number of threads used in the calculations.

    Returns:
        The regression result.

    Raises:
        ValueError : If ``x`` and ``x_hat`` have incompatible shapes.
        ValueError : If ``x`` and ``y`` have incompatible shapes.

    References:
        .. [KerReg] E. A. Nadaraya, “On estimating regression,” Theory Probab. & Its Appl. 9,
            141-142 (1964).
        .. [Krn]    Kernel (statictics), https://en.wikipedia.org/wiki/Kernel_(statistics).
    """
    ...

def kr_grid(y: RealArray, x: RealArray, grid: Sequence[RealArray], sigma: float,
            kernel: str='gaussian', w: Optional[RealArray]=None, num_threads: int=1
            ) -> Tuple[RealArray, List[float]]:
    """Perform the multi-dimensional Nadaraya-Watson kernel regression [KerReg]_ over a grid of
    points.

    Args:
        y : The data to fit.
        x : Coordinates array.
        grid : A tuple of grid coordinates along each dimension (`x`, `y`, `z`, ...).
        sigma : Kernel bandwidth.
        w : A set of weights, unitary weights are assumed if it's not provided.
        num_threads : Number of threads used in the calculations.

    Returns:
        The regression result and the region of interest.
    """
    ...

@overload
def local_maxima(inp: RealArray, axis: IntSequence, num_threads: int=1) -> NDRealArray:
    ...

@overload
def local_maxima(inp: IntArray, axis: IntSequence,
                 num_threads: int=1) -> IntArray:
    ...

def local_maxima(inp: RealArray | IntArray, axis: IntSequence, num_threads: int=1
                 ) -> NDRealArray | IntArray:
    """
    Find local maxima in a multidimensional array along a set of axes. This function returns
    the indices of the maxima.

    Args:
        x : The array to search for local maxima.
        axis : Choose an axis along which the maxima are sought for.

    Returns:


    Notes:
        - Compared to `scipy.signal.argrelmax` this function is significantly faster and can
          detect maxima that are more than one sample wide.
        - A maxima is defined as one or more samples of equal value that are
          surrounded on both sides by at least one smaller sample.
    """
    ...
