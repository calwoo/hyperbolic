import jax
import jax.numpy as np
import numpy as onp


class Hyperbolic:
    """
    Class containing objects relevant to computing the Riemannian
    structure of hyperbolic space. We will model hyperbolic geometry
    using the (2-dimensional) Poincare disk model.

    Points in the Poincare disk are modeled in polar coordinates
        x = (r, theta) where r \in [0, 1)
                             theta \in [0, 2 * \pi)
    """

    def __init__(self, dim=2, emb_radius=1):
        assert dim > 1

        self.dim = dim
        self.emb_radius = emb_radius

    def distance(self, x, y):
        if self.dim == 2:
            x_euc = self._to_euclidean(x)
            y_euc = self._to_euclidean(y)

            diff_norm = np.sum((x_euc - y_euc) ** 2, axis=-1)
            return np.arccosh(
                1
                + 2
                * (
                    diff_norm
                    / (
                        (1 - self._euclidean_norm(x) ** 2)
                        * (1 - self._euclidean_norm(y) ** 2)
                    )
                )
            )
        else:
            diff_norm = np.sum(x - y, axis=-1)
            return np.arccosh(
                1
                + 2
                * (
                    diff_norm
                    / ((1 - np.sum(x ** 2, axis=-1)) * (1 - np.sum(y ** 2, axis=-1)))
                )
            )

    def norm(self, x):
        if self.dim == 2:
            return 2 * np.arctanh(x[:, 0])
        else:
            euc_norms = np.sqrt(np.sum(x ** 2, axis=-1))
            return 2 * np.arctanh(euc_norms)

    def dot(self, x, y):
        if self.dim == 2:
            rxs, thxs = x[:, 0], x[:, 1]
            rys, thys = y[:, 0], y[:, 1]
            return 4 * np.arctanh(rxs) * np.arctanh(rys) * np.cos(thxs - thys)
        else:
            pass

    def _euclidean_norm(self, x):
        # convert to euclidean coordinates first
        x_euc = self._to_euclidean(x)
        return np.sqrt(np.sum(x_euc ** 2, axis=-1))

    def _to_euclidean(self, x):
        # x is sent as a 2-dim polar coordinate vector
        rs, thetas = x[:, 0], x[:, 1]
        eucs = np.stack([rs * np.cos(thetas), rs * np.sin(thetas)], axis=1)
        return eucs


if __name__ == "__main__":
    hyp = Hyperbolic(dim=2)
    x = np.array([[0.5, np.pi / 2], [0.2, np.pi / 14]])
    y = np.array([[0.2, np.pi / 14], [0.7, 1.0]])

    print(hyp.distance(x, y))
    print(hyp.dot(x, y))
