#ifndef CAKE_SPLIT_STATE_MATH_H_
#define CAKE_SPLIT_STATE_MATH_H_

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace split_state_math
{
constexpr int kBodyDim = 18;
constexpr int kMotionDim = 9;  // rotation, position, velocity; bias/gravity Jacobians are zero.
using BodyMatrix = Eigen::Matrix<double, kBodyDim, kBodyDim>;
using BodyVector = Eigen::Matrix<double, kBodyDim, 1>;
using MotionMatrix = Eigen::Matrix<double, kMotionDim, kMotionDim>;
using MotionVector = Eigen::Matrix<double, kMotionDim, 1>;

struct PixelJacobian
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MotionVector motion;
  Eigen::VectorXd camera;
  void clear() { motion.setZero(); camera.setZero(); }
  bool allFinite() const { return motion.allFinite() && camera.allFinite(); }
};

// Solve order is [body(18), camera parameters]. Keep the off-diagonal block:
// splitting storage must not turn the joint estimator into independent filters.
struct Information
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BodyMatrix body;
  BodyVector body_gradient;
  Eigen::Matrix<double, kBodyDim, Eigen::Dynamic> cross;
  Eigen::MatrixXd camera;
  Eigen::VectorXd camera_gradient;

  void reset(int camera_dim)
  {
    body.setZero();
    body_gradient.setZero();
    cross.setZero(kBodyDim, camera_dim);
    camera.setZero(camera_dim, camera_dim);
    camera_gradient.setZero(camera_dim);
  }

  void dense(Eigen::MatrixXd &h, Eigen::VectorXd &g) const
  {
    const int n = camera.rows();
    h.resize(kBodyDim + n, kBodyDim + n);
    g.resize(kBodyDim + n);
    h.topLeftCorner<kBodyDim, kBodyDim>() = body;
    h.topRightCorner(kBodyDim, n) = cross;
    h.bottomLeftCorner(n, kBodyDim) = cross.transpose();
    h.bottomRightCorner(n, n) = camera;
    g.head<kBodyDim>() = body_gradient;
    g.tail(n) = camera_gradient;
  }
};

// One reusable workspace per camera. A historical-reference residual involves
// only this camera's parameters, not every camera in the rig.
struct PatchWorkspace
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using MotionRows = Eigen::Matrix<double, Eigen::Dynamic, kMotionDim>;
  MotionRows motion, weighted_motion;
  Eigen::MatrixXd camera, weighted_camera;
  PixelJacobian pixel_split;
  Eigen::VectorXd residual, weight, weighted_residual, pixel;
  Eigen::MatrixXd nuisance, reference_noise, compact;
  MotionMatrix h_motion;
  MotionVector g_motion;
  Eigen::Matrix<double, kMotionDim, Eigen::Dynamic> h_cross;
  Eigen::MatrixXd h_camera;
  Eigen::VectorXd g_camera;
  std::vector<int> camera_indices, solve_to_patch, patch_to_solve;

  void configure(int rows, int solve_dim, const std::vector<int> &indices, bool need_nis)
  {
    camera_indices = indices;
    const int n = indices.size();
    motion.resize(rows, kMotionDim);
    weighted_motion.resize(rows, kMotionDim);
    camera.resize(rows, n);
    weighted_camera.resize(rows, n);
    residual.resize(rows);
    weight.resize(rows);
    weighted_residual.resize(rows);
    pixel.resize(kMotionDim + n);
    pixel_split.camera.resize(n);
    compact.resize(rows, kMotionDim + n);
    h_cross.resize(kMotionDim, n);
    h_camera.resize(n, n);
    g_camera.resize(n);
    if (need_nis) nuisance.resize(rows, 9);
    solve_to_patch.assign(solve_dim, -1);
    patch_to_solve.resize(kMotionDim + n);
    for (int i = 0; i < kMotionDim; ++i)
      solve_to_patch[i] = patch_to_solve[i] = i;
    for (int i = 0; i < n; ++i)
    {
      solve_to_patch[indices[i]] = kMotionDim + i;
      patch_to_solve[kMotionDim + i] = indices[i];
    }
  }

  void compute(int rows)
  {
    weighted_motion.topRows(rows).noalias() = weight.head(rows).asDiagonal() * motion.topRows(rows);
    weighted_residual.head(rows) = weight.head(rows).asDiagonal() * residual.head(rows);
    h_motion.noalias() = weighted_motion.topRows(rows).transpose() * weighted_motion.topRows(rows);
    g_motion.noalias() = weighted_motion.topRows(rows).transpose() * weighted_residual.head(rows);
    if (camera.cols() != 0)
    {
      weighted_camera.topRows(rows).noalias() = weight.head(rows).asDiagonal() * camera.topRows(rows);
      h_cross.noalias() = weighted_motion.topRows(rows).transpose() * weighted_camera.topRows(rows);
      h_camera.noalias() = weighted_camera.topRows(rows).transpose() * weighted_camera.topRows(rows);
      g_camera.noalias() = weighted_camera.topRows(rows).transpose() * weighted_residual.head(rows);
    }
  }

  void addTo(Information &information) const
  {
    information.body.topLeftCorner<kMotionDim, kMotionDim>() += h_motion;
    information.body_gradient.head<kMotionDim>() += g_motion;
    for (int i = 0; i < static_cast<int>(camera_indices.size()); ++i)
    {
      const int ci = camera_indices[i] - kBodyDim;
      information.cross.col(ci).head<kMotionDim>() += h_cross.col(i);
      information.camera_gradient[ci] += g_camera[i];
      for (int j = 0; j < static_cast<int>(camera_indices.size()); ++j)
        information.camera(ci, camera_indices[j] - kBodyDim) += h_camera(i, j);
    }
  }

  void addHessianTo(Eigen::MatrixXd &h) const
  {
    h.topLeftCorner<kMotionDim, kMotionDim>() += h_motion;
    for (int i = 0; i < static_cast<int>(camera_indices.size()); ++i)
    {
      const int ci = camera_indices[i];
      h.block<kMotionDim, 1>(0, ci) += h_cross.col(i);
      h.block<1, kMotionDim>(ci, 0) += h_cross.col(i).transpose();
      for (int j = 0; j < static_cast<int>(camera_indices.size()); ++j)
        h(ci, camera_indices[j]) += h_camera(i, j);
    }
  }

  void pack(int rows)
  {
    compact.topLeftCorner(rows, kMotionDim) = motion.topRows(rows);
    compact.topRightCorner(rows, camera.cols()) = camera.topRows(rows);
  }
};

// IMU propagation has F = diag(F_body, I_camera) and diagonal camera process
// noise. Propagate the fixed 18-state block and its camera cross-covariance
// directly, without multiplying a full matrix containing identity/zero blocks.
struct PropagationWorkspace
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BodyMatrix transition, noise, body, propagated_body;
  Eigen::Matrix<double, kBodyDim, Eigen::Dynamic> cross, propagated_cross;
  Eigen::VectorXd camera_noise;
  std::array<int, kBodyDim> body_indices;
  std::vector<int> camera_indices;

  void configure(int full_dim, int velocity_index)
  {
    for (int i = 0; i < 6; ++i) body_indices[i] = i;
    for (int i = 6; i < kBodyDim; ++i) body_indices[i] = velocity_index + i - 6;
    camera_indices.clear();
    for (int i = 6; i < velocity_index; ++i) camera_indices.push_back(i);
    for (int i = velocity_index + 12; i < full_dim; ++i) camera_indices.push_back(i);
    cross.resize(kBodyDim, camera_indices.size());
    propagated_cross.resize(kBodyDim, camera_indices.size());
    camera_noise.setZero(camera_indices.size());
    transition.setIdentity();
    noise.setZero();
  }

  void propagate(Eigen::MatrixXd &covariance)
  {
    for (int i = 0; i < kBodyDim; ++i)
    {
      for (int j = 0; j < kBodyDim; ++j)
        body(i, j) = covariance(body_indices[i], body_indices[j]);
      for (int j = 0; j < static_cast<int>(camera_indices.size()); ++j)
        cross(i, j) = 0.5 * (covariance(body_indices[i], camera_indices[j]) +
                              covariance(camera_indices[j], body_indices[i]));
    }
    propagated_body.noalias() = transition * body * transition.transpose();
    propagated_body += noise;
    propagated_cross.noalias() = transition * cross;
    for (int i = 0; i < kBodyDim; ++i)
    {
      for (int j = 0; j < kBodyDim; ++j)
        covariance(body_indices[i], body_indices[j]) =
            0.5 * (propagated_body(i, j) + propagated_body(j, i));
      for (int j = 0; j < static_cast<int>(camera_indices.size()); ++j)
        covariance(body_indices[i], camera_indices[j]) =
            covariance(camera_indices[j], body_indices[i]) = propagated_cross(i, j);
    }
    for (int i = 0; i < static_cast<int>(camera_indices.size()); ++i)
    {
      covariance(camera_indices[i], camera_indices[i]) += camera_noise[i];
      for (int j = i + 1; j < static_cast<int>(camera_indices.size()); ++j)
      {
        const double value = 0.5 * (covariance(camera_indices[i], camera_indices[j]) +
                                    covariance(camera_indices[j], camera_indices[i]));
        covariance(camera_indices[i], camera_indices[j]) =
            covariance(camera_indices[j], camera_indices[i]) = value;
      }
    }
  }
};
} // namespace split_state_math
#endif
