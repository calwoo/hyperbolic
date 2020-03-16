import jax
import jax.numpy as np
import numpy as onp


class Euclidean:
    """
    Class containing objects relevant to computing the Riemannian
    structure of Euclidean space.

    Points in the Euclidean space are modeled normally as n-vectors.
    """

    def __init__(self, dim=2):
        assert dim > 0
        self.dim = dim
