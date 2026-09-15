"""Independent NumPy checks of the derivations; does NOT execute the C++ code.
Run the separate CMake tests to test the actual Eigen implementation.
"""
import unittest
import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def exp(v):
    a = np.linalg.norm(v)
    k = skew(v)
    if a < 1e-12:
        return np.eye(3) + k + .5 * k @ k
    return np.eye(3) + np.sin(a) / a * k + (1 - np.cos(a)) / a**2 * k @ k


def log(r):
    w = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]]) / 2
    s = np.linalg.norm(w)
    a = np.arctan2(s, (np.trace(r)-1)/2)
    return w if s < 1e-12 else a/s*w


def jr(v):
    a = np.linalg.norm(v)
    k = skew(v)
    if a < 1e-4:
        return np.eye(3) - (.5-a*a/24)*k + (1/6-a*a/120)*k@k
    return np.eye(3) - (1-np.cos(a))/a**2*k + (a-np.sin(a))/a**3*k@k


def fd(fun, dim):
    eps = 1e-6
    return np.column_stack([(fun(np.eye(dim)[i]*eps)-fun(-np.eye(dim)[i]*eps))/(2*eps)
                            for i in range(dim)])


class Derivations(unittest.TestCase):
    def test_right_reset_and_prior_transport(self):
        for d in [np.array([.4, -.2, .3]), np.array([1e-7, -2e-7, 0.])]:
            numeric = fd(lambda e: log(exp(-d)@exp(d+e)), 3)
            np.testing.assert_allclose(jr(d), numeric, atol=2e-8)
            prior_log = fd(lambda e: log(exp(d)@exp(e)), 3)
            np.testing.assert_allclose(jr(d)@prior_log, np.eye(3), atol=2e-8)

    def test_world_point_with_extrinsic_and_cross_covariance(self):
        rng = np.random.default_rng(123)
        r = exp(np.array([.4, -.2, .3]))
        ri = exp(np.array([-.3, .1, .5]))
        point = ri@np.array([1., -.2, 2.]) + np.array([.1, -.1, .4])
        j = np.column_stack([-r@skew(point), np.eye(3)])
        numeric = fd(lambda e: r@exp(e[:3])@point+e[3:], 6)
        np.testing.assert_allclose(j, numeric, atol=2e-8)
        a = rng.normal(size=(6, 6)); p = a@a.T + np.eye(6)
        beam = np.diag([.01, .04, .09])
        c = r@ri@beam@ri.T@r.T + j@p@j.T
        q = exp(np.array([.2, .6, -.3])); t = np.eye(6); t[3:, 3:] = q
        jq = np.column_stack([-q@r@skew(point), np.eye(3)])
        cq = q@r@ri@beam@ri.T@r.T@q.T + jq@t@p@t.T@jq.T
        np.testing.assert_allclose(cq, q@c@q.T, atol=1e-10)

    def test_reference_residual_coordinates(self):
        r = exp(np.array([.4, -.2, .3])); ri = exp(np.array([-.3, .1, .5]))
        pi = np.array([1., -.2, 2.]); pos = np.array([.1, -.1, .4]); point = r@pi+pos
        photo = np.array([2., -.3, 1.]); rw = ri@r.T
        j = np.concatenate([-photo@rw@r@skew(pi), photo@rw, -photo@rw])
        numeric = fd(lambda e: np.atleast_1d(-photo@ri@(r@exp(e[:3])).T@(point+e[6:]-pos-e[3:6])), 9)
        np.testing.assert_allclose(j, numeric[0], atol=2e-8)

    def test_reference_pose_time_shift(self):
        r = exp(np.array([.4, -.2, .3])); rate = np.array([.3, .4, -.2])
        velocity = np.array([.5, -.2, 1.]); dt = .04; frame = r@exp(rate*dt)
        def error(e):
            return np.r_[log(frame.T@r@exp(e[:3])@exp(rate*(dt+e[9]))),
                         e[3:6]+(velocity+e[6:9])*(dt+e[9])-velocity*dt]
        j = np.zeros((6, 10)); j[:3, :3] = exp(rate*dt).T
        j[3:, 3:6] = np.eye(3); j[3:, 6:9] = np.eye(3)*dt
        j[:3, 9] = rate; j[3:, 9] = velocity
        np.testing.assert_allclose(j, fd(error, 10), atol=2e-8)

    def test_full_and_schmidt_information_solver(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            n, m = 21, 11
            a = rng.normal(size=(n, n)); p = a@a.T + np.eye(n)
            h = rng.normal(size=(m, n)); r = np.diag(rng.uniform(.1, 1., m))
            y = rng.normal(size=m); delta = rng.normal(size=n)
            info = h.T@np.linalg.solve(r, h); vec = h.T@np.linalg.solve(r, y)
            b = np.linalg.inv(np.linalg.inv(p)+info)
            k = np.linalg.solve(h@p@h.T+r, h@p).T
            for freeze in [False, True]:
                mask = np.eye(n)
                if freeze: mask[1::3, 1::3] = 0
                selected_b = mask@b; selected_k = mask@k
                g = selected_b@info; j = np.eye(n)-g
                actual = j@p@j.T + selected_b@info@selected_b.T
                j2 = np.eye(n)-selected_k@h
                expected = j2@p@j2.T+selected_k@r@selected_k.T
                np.testing.assert_allclose(actual, expected, atol=2e-10)
                np.testing.assert_allclose(selected_b@vec+mask@delta-g@delta,
                                           mask@delta+selected_k@(y-h@delta), atol=2e-10)
                self.assertGreater(np.linalg.eigvalsh(actual).min(), 0)

    def test_indefinite_old_cross_block_counterexample(self):
        old = np.array([[1/101, .9], [.9, 1.]])
        corrected = np.array([[1/101, .9/101], [.9/101, 1.]])
        self.assertLess(np.linalg.eigvalsh(old).min(), 0)
        self.assertGreater(np.linalg.eigvalsh(corrected).min(), 0)

    def test_low_rank_covariance_and_shared_information(self):
        rng = np.random.default_rng(12)
        d = rng.uniform(.01, .1, 64); u = rng.normal(size=(64, 12)); b = rng.normal(size=(64, 18))
        di_u = u/d[:, None]; di_b = b/d[:, None]
        woodbury = di_b-di_u@np.linalg.solve(np.eye(12)+u.T@di_u, u.T@di_b)
        np.testing.assert_allclose(woodbury, np.linalg.solve(np.diag(d)+u@u.T, b), atol=2e-10)
        n = 100; c = np.eye(n)*.03**2 + np.ones((n, n))*.01**2
        variance = 1/(np.ones(n)@np.linalg.solve(c, np.ones(n)))
        self.assertAlmostEqual(variance, .01**2+.03**2/n, places=12)

    def test_unknown_correlation_bound(self):
        rng = np.random.default_rng(51)
        for _ in range(50):
            factors = [rng.normal(size=(6, 8)) for _ in range(4)]
            terms = [a@a.T for a in factors]
            weights = np.sqrt([np.trace(c) for c in terms]); weights /= weights.sum()
            bound = sum(c/w for c, w in zip(terms, weights))
            actual = sum(factors)@sum(factors).T
            self.assertGreater(np.linalg.eigvalsh(bound-actual).min(), -1e-9)

    def test_imu_noise_units(self):
        psd, dt, duration = .02, .005, 1.
        n = round(duration/dt)
        self.assertAlmostEqual(n*psd*dt, psd*duration)
        discrete_variance = psd/dt
        self.assertAlmostEqual(n*discrete_variance*dt**2, psd*duration)


if __name__ == '__main__':
    unittest.main(verbosity=2)
