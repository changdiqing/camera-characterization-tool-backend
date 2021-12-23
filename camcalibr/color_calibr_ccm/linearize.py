import numpy as np
from math import log, exp
from .utils import *


class Linear:
    """linearization base"""

    def __init__(self, *args):
        """get args"""
        pass

    def calc(self):
        """calculate parameters"""
        pass

    def linearize(self, inp):
        """inference"""
        return inp

    def value(self):
        """evaluate linearization model"""
        pass


class Linear_gamma(Linear):
    """
        gamma correction;
    see Linearization.py for details;
    """

    def __init__(self, gamma, *_):
        self.gamma = gamma

    def linearize(self, inp):
        return gamma_correction(inp, self.gamma)
