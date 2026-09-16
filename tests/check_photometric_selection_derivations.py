"""Independent numerical checks, not execution of the C++ implementation.

The separate photometric_selection_math_test CMake target exercises the actual
Eigen selector against a dense joint-covariance oracle.
"""
import unittest
import numpy as np


def posterior(p, h, r):
    return np.linalg.inv(np.linalg.inv(p) + h.T @ np.linalg.solve(r, h))


class SelectionDerivations(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(54)

    def assertClose(self, a, b, tol=1e-9):
        np.testing.assert_allclose(a, b, atol=tol, rtol=tol)

    def test_conditional_increment_matches_joint_model(self):
        for _ in range(12):
            d, m, n = 10, 8, 5
            x = self.rng.normal(size=(d, d))
            p = x @ x.T + np.eye(d)
            h = self.rng.normal(size=(m + n, d))
            b = self.rng.normal(size=(m + n, 4))
            r = np.eye(m + n) + b @ b.T
            a = np.linalg.solve(r[:m, :m], r[:m, m:])
            hc = h[m:] - a.T @ h[:m]
            rc = r[m:, m:] - r[m:, :m] @ a
            after = posterior(posterior(p, h[:m], r[:m, :m]), hc, rc)
            self.assertClose(after, posterior(p, h, r))
            self.assertGreater(np.linalg.eigvalsh(rc).min(), 0.)

    def test_full_state_pose_marginal_keeps_exposure_uncertainty(self):
        p = np.diag([1.] * 6 + [100.])
        h = np.zeros((1, 7)); h[0, [0, 6]] = 10.
        full = posterior(p, h, np.eye(1))[0, 0]
        wrongly_conditioned = posterior(p[:6, :6], h[:, :6], np.eye(1))[0, 0]
        self.assertGreater(full, .99)
        self.assertLess(wrongly_conditioned, .01)

    def test_incremental_candidate_cache_matches_dense_conditioning(self):
        sizes = [3, 4, 2, 5]
        rows = sum(sizes)
        h = self.rng.normal(size=(rows, 9))
        b = self.rng.normal(size=(rows, 7))
        r = np.eye(rows) + b @ b.T
        target = slice(rows - sizes[-1], rows)
        hc, rc = h[target].copy(), r[target, target].copy()
        lower_cross = np.empty((0, sizes[-1]))
        selected_rows = 0
        for size in sizes[:-1]:
            added = slice(selected_rows, selected_rows + size)
            selected_rows += size
            l = np.linalg.cholesky(r[:selected_rows, :selected_rows])
            whitened = np.linalg.solve(l, h[:selected_rows])
            cross = r[added, target] - l[added, :added.start] @ lower_cross
            new_cross = np.linalg.solve(l[added, added], cross)
            hc -= new_cross.T @ whitened[added]
            rc -= new_cross.T @ new_cross
            lower_cross = np.vstack((lower_cross, new_cross))
            direct = np.linalg.solve(r[:selected_rows, :selected_rows], r[:selected_rows, target])
            self.assertClose(hc, h[target] - direct.T @ h[:selected_rows])
            self.assertClose(rc, r[target, target] - r[target, :selected_rows] @ direct)

    def test_conditional_gain_can_increase(self):
        p = np.eye(6)
        h = np.zeros((2, 6)); h[:, 0] = 2.
        b = np.array([[5.], [-5.]])
        r = np.eye(2) + b @ b.T
        first = posterior(p, h[:1], r[:1, :1])
        both = posterior(p, h, r)
        logdet = lambda x: np.linalg.slogdet(x)[1]
        self.assertGreater(logdet(first) - logdet(both), logdet(p) - logdet(first))

    def test_plane_center_projection_preserves_center_cross_terms(self):
        # The frontend actually inserts plane centers: dp/d[n,c] = [0 I].
        j = np.hstack((np.zeros((3, 3)), np.eye(3)))
        root = self.rng.normal(size=(6, 6)); cov = root @ root.T
        self.assertClose(j @ cov @ j.T, cov[3:, 3:])
        self.assertGreater(np.linalg.norm(j @ cov @ j.T - np.diag(np.diag(cov[3:, 3:]))), 1e-4)
        # Two observations of the same plane-center source share its error;
        # separate independent copies would incorrectly double its information.
        h = np.eye(3); p = np.eye(3)
        r = np.block([[np.eye(3) + cov[3:, 3:], cov[3:, 3:]],
                      [cov[3:, 3:], np.eye(3) + cov[3:, 3:]]])
        joint = posterior(p, np.vstack((h, h)), r)
        independent = posterior(p, np.vstack((h, h)), np.kron(np.eye(2), np.eye(3) + cov[3:, 3:]))
        self.assertGreater(np.trace(joint), np.trace(independent))

    def test_two_stage_pixel_ancestry_and_normalization(self):
        # S represents raw->virtual resampling, W the affine support read.
        s = self.rng.uniform(size=(12, 20)); s /= s.sum(axis=1, keepdims=True)
        w = self.rng.uniform(size=(8, 12)); w /= w.sum(axis=1, keepdims=True)
        raw = self.rng.uniform(10., 240., size=20)
        a = w @ s
        self.assertClose(w @ (s @ raw), a @ raw)
        values = a @ raw; centered = values - values.mean(); sigma = values.std()
        jnorm = (np.eye(8) - np.ones((8, 8)) / 8 - np.outer(centered, centered) / (8 * sigma**2)) / sigma
        def normalized(x):
            v = a @ x
            return (v - v.mean()) / v.std()
        numeric = np.column_stack([(normalized(raw + np.eye(20)[k] * 1e-4) -
                                    normalized(raw - np.eye(20)[k] * 1e-4)) / 2e-4 for k in range(20)])
        self.assertClose(jnorm @ a, numeric, 1e-7)
        # Shared original pixels induce cross-patch covariance after resampling.
        self.assertGreater(abs((a @ a.T)[0, 1]), 0.)

    def test_projection_independent_information_and_covariance(self):
        h = self.rng.normal(size=(24, 4)) @ self.rng.normal(size=(4, 9))
        u, singular, _ = np.linalg.svd(h, full_matrices=False)
        q = u[:, singular > 1e-10 * singular[0]]
        projected = q.T @ h
        self.assertClose(h.T @ h, projected.T @ projected)
        self.assertClose(q.T @ q, np.eye(4))

    def test_score_cannot_increase_pose_covariance(self):
        for _ in range(30):
            d = 12; x = self.rng.normal(size=(d, d)); p = x @ x.T + np.eye(d)
            h = self.rng.normal(size=(6, d)); b = self.rng.normal(size=(6, 3))
            updated = posterior(p, h, np.eye(6) + b @ b.T)
            self.assertGreaterEqual(np.linalg.eigvalsh(p[:6, :6] - updated[:6, :6]).min(), -1e-10)

    def test_active_marginal_recovers_full_posterior(self):
        for _ in range(15):
            d = 32
            x = self.rng.normal(size=(d, d)); p = x @ x.T + np.eye(d)
            active = list(range(6)) + [9, 17]
            h = np.zeros((13, d)); h[:, active] = self.rng.normal(size=(13, len(active)))
            u = self.rng.normal(size=(13, 5)); r = np.eye(13) + u @ u.T
            pa = p[np.ix_(active, active)]
            reduced = posterior(pa, h[:, active], r)
            g = np.linalg.solve(pa, p[:, active].T).T
            recovered = p + g @ (reduced - pa) @ g.T
            self.assertClose(recovered, posterior(p, h, r))

    def test_early_pixel_projection_matches_original_noise_model(self):
        for normalized in [False, True]:
            m = 64; rows = 47; rank = 7
            indices = self.rng.choice(m, size=rows, replace=False)
            q, _ = np.linalg.qr(self.rng.normal(size=(rows, rank)))
            weights = self.rng.uniform(.1, 1., size=rows)
            projection = q.T * weights / np.sqrt(1000.)
            a = self.rng.uniform(size=(m, 91)); a /= a.sum(axis=1, keepdims=True)
            values = self.rng.uniform(20., 230., size=m)
            centered = values - values.mean(); sigma = values.std()
            if normalized:
                norm_j = (np.eye(m) - np.ones((m, m)) / m -
                          np.outer(centered, centered) / (m * sigma**2)) / sigma
            else:
                norm_j = 1.3 * np.eye(m)
            original = -projection @ (norm_j @ a)[indices]
            early = np.zeros((rank, m)); early[:, indices] = -projection
            if normalized:
                early = (early - early.mean(axis=1, keepdims=True)) / sigma - \
                    np.outer(early @ centered, centered) / (m * sigma**3)
            else:
                early *= 1.3
            self.assertClose(early @ a, original)

    def test_sparse_source_cache_matches_dense_joint_covariance(self):
        sizes = [3, 5, 2, 4, 6]
        sources = [{i // 2: self.rng.normal(size=(m, 2)),
                    10 + i % 2: self.rng.normal(size=(m, 1))}
                   for i, m in enumerate(sizes)]
        blocks = {}; owners = {}
        for i, source in enumerate(sources):
            for key, b in source.items():
                for j, other in owners.get(key, []):
                    blocks[i, j] = blocks.get((i, j), np.zeros((sizes[i], sizes[j]))) + b @ other.T
                owners.setdefault(key, []).append((i, b))
        for i in range(len(sizes)):
            for j in range(i):
                dense = sum((sources[i][k] @ sources[j][k].T
                             for k in sources[i].keys() & sources[j].keys()),
                            np.zeros((sizes[i], sizes[j])))
                self.assertClose(blocks.get((i, j), np.zeros_like(dense)), dense)


if __name__ == '__main__':
    unittest.main(verbosity=2)
