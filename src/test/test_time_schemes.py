import unittest
import numpy as np
import scipy.sparse as sp

from time_stepping.finite_differences import step_theta

class TestThetaStep(unittest.TestCase):
    def test_step_theta_manual(self) -> None:
        rng = np.random.default_rng(123)
        n = 7

        M = sp.random(n, n, density=0.4, format="csr", random_state=1)
        M = (M + M.T) * 0.5 + sp.eye(n)  # make it nonsingular-ish
        A = sp.random(n, n, density=0.4, format="csr", random_state=2)
        w = rng.normal(size=n)

        dt = 0.123
        theta = 0.37

        lhs, rhs = step_theta(M, dt, A, w, theta)

        # manual
        t_m = (1.0 / dt) * M.tocsr()
        A_csr = A.tocsr()
        lhs_ref = t_m + theta * A_csr
        rhs_ref = (t_m - (1.0 - theta) * A_csr) @ w

        np.testing.assert_allclose((lhs - lhs_ref).data, 0.0, atol=0, rtol=0)
        np.testing.assert_allclose(rhs, rhs_ref, rtol=0, atol=0)

    def test_step_theta_identity(self) -> None:
        n = 5
        M = sp.diags([1, 2, 3, 4, 5], format="csr")
        A = sp.csr_matrix((n, n))  # A=0
        w0 = np.array([1.0, -2.0, 3.5, 0.0, 4.2])

        dt = 0.5
        for theta in [0.0, 0.5, 1.0]:
            lhs, rhs = step_theta(M, dt, A, w0, theta)
            w1 = sp.linalg.spsolve(lhs.tocsr(), rhs)
            np.testing.assert_allclose(w1, w0, atol=1e-14)

    def test_step_theta_scalar(self) -> None:
        m = 2.0
        a = 3.0
        M = sp.csr_matrix([[m]])
        A = sp.csr_matrix([[a]])
        w0 = np.array([4.0])

        dt = 0.1
        theta = 0.5

        lhs, rhs = step_theta(M, dt, A, w0, theta)
        w1 = sp.linalg.spsolve(lhs.tocsr(), rhs)

        factor = (m/dt - (1.0-theta)*a) / (m/dt + theta*a)
        w1_ref = factor * w0
        np.testing.assert_allclose(w1, w1_ref, atol=1e-14)
