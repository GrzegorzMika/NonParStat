# License: Apache 2.0
# Carl Kadie
# https://fastlmm.github.io/

import os
import numpy as np
import inspect
import logging
from types import ModuleType

_warn_array_module_once = False


def array_module(xp=None):
    """
    Find the array module to use, for example **numpy** or **cupy**.
    Args:
        xp (optional, string or Python module): The array module to use, for example, 'numpy'
            (normal CPU-based module) or 'cupy' (GPU-based module).
            If not given, will try to read  from the ARRAY_MODULE environment variable.
            If not given and ARRAY_MODULE is not set,
            will use numpy. If 'cupy' is requested, will
            try to 'import cupy'. If that import fails, willrevert to numpy.

    Returns:
        Python module

    Example:
        >>> xp = array_module() # will look at environment variable
        >>> print(xp.zeros(3))
        [0. 0. 0.]
        >>> xp = array_module('cupy') # will try to import 'cupy'
        >>> print(xp.zeros(3))
        [0. 0. 0.]
    """
    xp = xp or os.environ.get("ARRAY_MODULE", "numpy")

    if isinstance(xp, ModuleType):
        return xp

    if xp == "numpy":
        return np

    if xp == "cupy":
        try:
            import cupy as cp

            return cp
        except ModuleNotFoundError as e:
            global _warn_array_module_once
            if not _warn_array_module_once:
                logging.warning(f"Using numpy. ({e})")
                _warn_array_module_once = True
            return np

    raise ValueError(f"Don't know ARRAY_MODULE '{xp}'")


def asnumpy(a):
    """
    Given an array created with any array module, return the equivalent
    numpy array. (Returns a numpy array unchanged.)

    Args:
        a (numpy.ndarray or cupy.ndarray): array created with any array module

    Returns:
        numpy.ndarray: numpy equivalent of array a

    Example:
        >>> xp = array_module('cupy')
        >>> zeros_xp = xp.zeros(3) # will be cupy if available
        >>> zeros_np = asnumpy(zeros_xp) # will be numpy
        >>> zeros_np
        array([0., 0., 0.])
    """
    if isinstance(a, np.ndarray):
        return a
    return a.get()


def get_array_module(a):
    """
    Given an array, returns the array's
    module, for example, **numpy** or **cupy**.
    Works for numpy even when cupy is not available.

    Args:
        a (numpy.ndarray or cupy.ndarray): array created with any array module

    Returns:
        Python module used to create array a

    Example:
        >>> import numpy as np
        >>> zeros_np = np.zeros(3)
        >>> xp = get_array_module(zeros_np)
        >>> xp.ones(3)
        array([1., 1., 1.])
    """
    submodule = inspect.getmodule(type(a))
    module_name = submodule.__name__.split(".")[0]
    xp = array_module(module_name)
    return xp