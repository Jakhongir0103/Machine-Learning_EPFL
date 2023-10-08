# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    f = np.ones(len(x))
    for j in range(degree):
        f = np.vstack((f,x**(j+1)))
    return f.T
