# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    w_star=np.linalg.solve(tx.T.dot(tx),tx.T.dot(y))
    err_star=tx.dot(w_star)
    N=len(y)
    mse=0.5*np.sum((y-err_star)**2)/N
    return w_star, mse

