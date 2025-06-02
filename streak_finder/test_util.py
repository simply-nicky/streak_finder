from typing import Dict, Optional
import numpy as np
from ._src.annotations import ComplexArray, RealArray

_atol = {np.dtype(np.float32): 1e-4, np.dtype(np.float64): 1e-5,
         np.dtype(np.complex64): 1e-4, np.dtype(np.complex128): 1e-5}
_rtol = {np.dtype(np.float32): 1e-3, np.dtype(np.float64): 1e-4,
         np.dtype(np.complex64): 1e-3, np.dtype(np.complex128): 1e-4}

def default_tolerance() -> Dict[np.dtype, float]:
    return _atol

def tolerance(dtype: np.dtype, tol: Optional[float]=None) -> float:
    if tol is None:
        return default_tolerance()[dtype]
    return default_tolerance().get(dtype, tol)

def default_gradient_tolerance() -> Dict[np.dtype, float]:
    return _rtol

def gradient_tolerance(dtype: np.dtype, tol: Optional[float]=None) -> float:
    if tol is None:
        return default_gradient_tolerance()[dtype]
    return default_gradient_tolerance().get(dtype, tol)

def check_close(a: RealArray | ComplexArray, b: RealArray | ComplexArray,
                rtol: Optional[float]=None, atol: Optional[float]=None):
    if rtol is None:
        rtol = max(gradient_tolerance(a.dtype, rtol),
                   gradient_tolerance(b.dtype, rtol))
    if atol is None:
        atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
