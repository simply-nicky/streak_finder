from typing import Any, Dict, List, Literal, NamedTuple, Sequence, Set, Tuple, Union
import numpy as np
import numpy.typing as npt

Scalar = Union[int, float, np.number]
Shape = Tuple[int, ...]

IntTuple = Tuple[int, ...]
FloatTuple = Tuple[float, ...]

NDArray = npt.NDArray[Any]
NDBoolArray = npt.NDArray[np.bool_]
NDIntArray = npt.NDArray[np.integer[Any]]
NDRealArray = npt.NDArray[np.floating[Any]]
NDComplexArray = npt.NDArray[Union[np.floating[Any], np.complexfloating[Any, Any]]]

Indices = Union[int, slice, NDIntArray, Sequence[int]]

IntSequence = Union[int, np.integer[Any], Sequence[int], NDIntArray]
ROIType = Union[List[int], Tuple[int, int, int, int], NDIntArray]
RealSequence = Union[float, np.floating[Any], Sequence[float], NDRealArray]
CPPIntSequence = Union[Sequence[int], NDIntArray]

NDRealArrayLike = Union[NDRealArray, List[float], Tuple[float, ...]]

Table = Dict[Tuple[int, int], float]

Norm = Literal['backward', 'forward', 'ortho']
Mode = Literal['constant', 'nearest', 'mirror', 'reflect', 'wrap']

Line = List[float]
Streak = Tuple[Set[Tuple[int, int, float]], Dict[float, List[float]],
               Dict[float, List[int]], Line]

class Pattern(NamedTuple):
    index   : NDIntArray
    frames  : NDIntArray
    y       : NDIntArray
    x       : NDIntArray
    rp      : NDRealArray

class PatternWithHKL(NamedTuple):
    index   : NDIntArray
    frames  : NDIntArray
    y       : NDIntArray
    x       : NDIntArray
    rp      : NDRealArray
    h       : NDIntArray
    k       : NDIntArray
    l       : NDIntArray

class PatternWithHKLID(NamedTuple):
    index   : NDIntArray
    frames  : NDIntArray
    y       : NDIntArray
    x       : NDIntArray
    rp      : NDRealArray
    h       : NDIntArray
    k       : NDIntArray
    l       : NDIntArray
    hkl_id  : NDIntArray
