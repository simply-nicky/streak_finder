from typing import overload
from ..annotations import NDIntArray, NDRealArray, IntSequence

@overload
def local_maxima(inp: NDRealArray, axis: IntSequence, num_threads: int=1) -> NDRealArray:
    ...

@overload
def local_maxima(inp: NDIntArray, axis: IntSequence,
                 num_threads: int=1) -> NDIntArray:
    ...

def local_maxima(inp: NDRealArray | NDIntArray, axis: IntSequence, num_threads: int=1
                 ) -> NDRealArray | NDIntArray:
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
