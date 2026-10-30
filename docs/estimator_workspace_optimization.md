# Covariance and visual workspace optimization

This change preserves the measurement model, active-state selection, mean update,
patch admission, reduction order, and SO(3) covariance reset. It does not change
the OpenMP backend or any YAML configuration.

## Covariance identity

Let `B = (P^-1 + information)^-1`, let `S` select the active state coordinates,
and let `Q = I-S`. The original implementation uses `selected_B = S B` and
`gain_times_jacobian = S B information`. In exact arithmetic:

```
A = I - S B information = Q + S B P^-1
P_new = A P A' + S B information B S
      = Q P Q + Q B S + S B Q + S B S
      = B + Q (P-B) Q
```

The implementation symmetrizes `B` and restores only its frozen/frozen submatrix
from `P`. Frozen/active cross-covariances remain from `B`. Restoring entire frozen
rows or columns would be incorrect. The same helper is used by both LIO and VIO.

The prior and posterior information still require successful LLT factorizations.
The candidate covariance still requires finite entries and a successful LLT.
If the simplified covariance fails, the original Joseph expression is evaluated
as a numerical fallback. No jitter or noise adjustment is introduced. A failure
of both candidates rejects the update without overwriting the previous output.
Callers retain their original rollback and post-injection covariance reset.

## Storage ownership

- `UpdateWorkspace` owns one active layout: fixed Eigen storage for dimensions
  19, 20 and 21, or reusable dynamic storage for other layouts. It belongs to an
  estimator instance, not a global variable or OpenMP thread identifier.
- `updateInto` reuses caller-owned outputs. The value-returning `update` wrapper
  remains available; both production estimator call sites use `updateInto`.
- Historical-reference pose-affine nuisance deltas belong to the existing
  per-camera, per-worker patch workspace.
- Current/current pair Jacobians, their difference vector, Hessian, gradient,
  and normalized-patch outputs belong to a reusable pair workspace. This path
  remains serial. Incomplete-patch rejection and ordered commits are unchanged.
- VIO Hessian, gradient, full-state remapping buffers, solution and active-index
  storage persist across iterations. The redundant solve Hessian/gradient
  copies are removed.
- Accepted, proposed and rollback state remain separate. The saved prior
  covariance refers only to the immutable prior state snapshot, never to the
  state being updated.

Layout changes may allocate. Same-layout warmed update kernels and the tested
patch/pair operations are checked with Eigen heap allocation disabled. This is
not a claim that the entire visual update, optional ZNCC processing, or covariance
tangent transport is allocation-free.

## Regression checks

Configure and run the independent tests without ROS:

```sh
cmake -S tests -B /tmp/cake-optimization-tests -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cake-optimization-tests -j2
cd /tmp/cake-optimization-tests
ctest --output-on-failure
./estimator_update_workspace_math_test --benchmark
```

The new test exercises production update/workspace code. Its Joseph reference
retains the previous implementation and fixed-size dispatch. It also checks an
independent measurement-space Kalman implementation with `R=I`.

The 1,296 numerical cases cover dimensions 2, 13, 18, 19, 20, 21, 22, 39 and 61;
prior condition numbers up to 1e8; prior scales from 1e-5 to 1e5; zero, low-rank
and full-rank information; full, partial and empty active sets; duplicate and
reversed indices; and covariance tangent reset. Separate checks cover rejected
inputs, recovery after rejection, preserving previous outputs, layout changes,
and stale statistics from an incomplete previous pair.

Floating-point results are not required to be bitwise identical. Large prior
scale with low-rank information can make the posterior information ill-conditioned
even when the prior condition number is one. Stress-test gain/correction tolerances
therefore use the known synthetic-input condition bound, capped at 2e-7; ordinary
cases retain the 2e-10 comparison. Both implementations are independently checked
against measurement-space Kalman, rather than treating the old result as exact.

## Validation record: 2026-10-30

- Windows-hosted Ubuntu 22.04 / GCC 11: all 8 independent tests passed.
- Target ROS machine / GCC 9 and Eigen 3.3, native CPU instructions: all 8 tests
  passed. Maximum normalized covariance difference from Joseph: 3.36541e-9;
  maximum gain difference: 1.34458e-9. Both old and new maximum gain error against
  the measurement-space calculation: 1.68366e-9.
- Both covariance tests passed ASan/UBSan after the final source revision.
- The complete ROS1 `cake_slam_mapping_multi_cam` target compiled on the remote
  machine. The final isolated build used the gold linker; the original workspace
  was not rebuilt. This is not an identical-build baseline comparison.
- With compressed-to-raw image adapters, 20-second bag excerpts produced 193
  visual frames for room01 (three cameras) and 166 for mocap8 (one camera, as in
  the remote configuration read for this test). No covariance rejection was
  reported during these excerpts. Both nodes nevertheless exited with SIGSEGV
  after SIGINT. Therefore the integration check is NOT a clean pass.

### Outstanding shutdown failure

The debugger stops in `free`, called from
`boost::detail::sp_counted_impl_p<pcl::PointCloud<pcl::PointXYZINormal>>::dispose`,
inside `LIVMapper::~LIVMapper`. The point-vector release reads the allocation
pointer preceding the buffer. The same release instruction pattern exists in
the untouched original library; that alone does not prove the old executable
would also crash on these inputs.

A local two-translation-unit Eigen-only probe (no estimator changes or PCL)
reproduced SIGSEGV when an SSE-built library allocated the vector and a native
AVX-built executable released it. Matching both sides to SSE exited normally.
This supports an alignment/allocator ABI hypothesis, not a definitive attribution
of the remote failure. Setting only `EIGEN_MAX_ALIGN_BYTES=16` did not resolve
the local mixed-build probe, so no unverified global alignment workaround has
been added to the project.

The original remote configuration/executable remain untouched by this work.
Following the user's local-only editing preference, further code changes belong
to the local checkout and Git synchronization belongs to the user. The patch is
not marked deployment-ready: the shutdown failure and complete-sequence
trajectory comparison remain unresolved.

The isolated target validation root is
`/home/cak/cake_opt_check_20261030_nnjq8e`. Original target sources, configuration,
executable and historical outputs are not deployment targets of this change.
Runtime tests use a separate ROS master and redirect memory statistics and all
compile-time `ROOT_DIR` outputs into this validation root.

Microbenchmarks measure only `estimator_covariance` update, with the original
value-returning implementation versus the reused output/workspace implementation.
They do not measure the full multi-level, iterative Joint EKF Update or virtual
patch construction. Seven alternating trials of 2,000 updates report medians;
performance depends on compiler, CPU and system load.
