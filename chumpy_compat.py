"""
Compatibility shim for chumpy with Python 3.12 and modern numpy.

This module patches the Python environment to make chumpy (designed for Python 2.7/3.6)
work with Python 3.12 and numpy 1.x/2.x.

Import this BEFORE importing chumpy or any module that imports chumpy.
"""
import sys
import inspect
import numpy as np

def patch_environment():
    """Apply all necessary patches for chumpy compatibility."""
    
    # Patch 1: inspect.getargspec → inspect.getfullargspec (removed in Python 3.11+)
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
        print(" Patched inspect.getargspec")
    
    # Patch 2: numpy type aliases (removed in numpy 1.20+)
    # Chumpy tries: from numpy import bool, int, float, complex, object, unicode, str
    if not hasattr(np, 'bool'):
        np.bool = np.bool_
    if not hasattr(np, 'int'):
        np.int = np.int_
    if not hasattr(np, 'float'):
        np.float = np.float_
    if not hasattr(np, 'complex'):
        np.complex = np.complex_
    if not hasattr(np, 'object'):
        np.object = np.object_
    if not hasattr(np, 'unicode'):
        np.unicode = np.str_
    if not hasattr(np, 'str'):
        np.str = np.str_
    
    print(" Patched numpy type aliases")

# Apply patches immediately when this module is imported
patch_environment()
