# 多相机估计器协方差修正

本次修改针对 `cake_slam_mapping_multi_cam` 的实际链路，对应 2026-09-15 审计中的 F1–F6。单相机历史文件 `vio.cpp`、`vio_fisheye.cpp`、`voxel_map.cpp` 未做同步移植。

## 已修改的实现

| 审计项 | 修正 | 主要位置 |
| --- | --- | --- |
| F1 子块更新后遗留旧交叉协方差 | LIO/VIO 使用完整先验求增益，冻结状态采用零增益行，完整协方差使用 Joseph 公式；删除恢复旧行列的函数。在线标定暂时冻结时仍保留其测量 Jacobian。 | `include/estimator_covariance.h`，两个估计器源文件 |
| F2 点协方差坐标不一致 | 建图、关联门控、地图更新统一使用 `J=[-R_WI[p_I]×, I]` 和完整位姿 6×6 块；点测量噪声通过 `R_WI R_IL` 旋转。 | `RefreshWorldPoints`、`StateEstimation` |
| F3 参考 NIS 坐标混用 | 从参考光度残差构造参考位姿及地图点的混合坐标 Jacobian；在残差空间构造共享误差，并同时用于光度权重和创新门控。删除直接套 SE(3) adjoint 的做法。 | `referenceUncertaintyJacobian`、主视觉更新 |
| F4 流形坐标转换缺失 | 每次迭代把固定先验协方差转到当前线性化点，最终按实际注入量重置姿态误差坐标；覆盖 IMU 与相机外参旋转的交叉块。另修正重力对齐时世界系位置、速度、重力协方差。 | `StatesGroup::covarianceAt/resetCovariance`、`gravityAlignment` |
| F5 初始化自匹配及地图共享误差 | 首帧只建图，不用自己刚建的地图进行 Kalman 更新，也不重复插入。仍查询法向供视觉使用。平面拟合聚合同一扫描的位姿误差，同一平面对应的多个残差保留非对角共享协方差。 | `BuildVoxelMap`、`handleLIO`、平面拟合与点面权重 |
| F6 IMU 参数未接通及单位不明确 | 接通 `imu.b_gyr_cov`、`imu.b_acc_cov`，新增明确的噪声模型选择，启动时用加粗彩色 `printf` 打印实际参数。 | `LIVMapper`、`ImuProcess`、多相机 YAML |

另外，平面拟合改为中心化的对称特征分解，拒绝没有唯一法向的点集，避免特征值差为零时计算无效的平面协方差。旧的 S2 入口转接主联合求解器，避免保留第二套不一致的协方差更新。

## 更新公式与边界

设完整先验为 P，测量信息为 Λ=HᵀR⁻¹H，b=HᵀR⁻¹(z−h)，B=(P⁻¹+Λ)⁻¹。D 是活动状态的对角选择矩阵，A=DB，G=AΛ。对于当前状态指向先验均值的增量 d，使用：

```
correction = A b + D d - G d
P_local = (I-G) P (I-G)^T + A Λ A^T
```

这是完整增益 K 的非活动行置零后得到的 Joseph 更新。冻结均值及其边缘协方差不意味着保留旧的活动—冻结交叉块。

旋转采用右扰动。固定先验到当前线性化点的协方差变换使用 `Jr(current ⊖ prior)`；后验注入后的重置使用 `Jr(correction)`。每轮迭代都从同一个先验构造协方差，最终只提交一次后验，不把各次迭代作为独立测量累积。右扰动重置可参照 [Solà 的 ESKF 推导](https://arxiv.org/abs/1711.02508)。

对于同一扫描中的点，先累加它们对共享位姿误差的 Jacobian，再传播一次共享协方差。对于未保存交叉项的不同扫描、不同平面，采用 Cauchy–Schwarz 上界：

```
Cov(sum e_i) <= sum C_i / w_i,
w_i = sqrt(trace(C_i)) / sum_j sqrt(trace(C_j)).
```

这会改变地图协方差、关联权重及有效测量信息；并非只调整日志。该上界可能保守，不能仅以 NEES 下降评价它的收益，需要同时看误差与不确定性大小。

视觉部分在每个 patch 内保留参考位姿和地图点引起的共享误差。参考 Jacobian 对应前端当前使用的局部、固定仿射 warp 近似；参考与地图点的未知相关性也用上述上界处理。原始与归一化光度残差都使用相应的梯度/归一化 Jacobian。参考位姿协方差在关闭地图管理时也保存；它通过帧时偏修正的姿态、位置、速度和时偏 Jacobian 传播，当前帧姿态 Jacobian 同步转换到估计器的扰动坐标。

**本次修改没有把地图扩展为完整联合随机状态。** 当前状态与历史地图的全部交叉协方差、不同视觉 patch 之间的共享误差，以及可选当前帧跨相机残差与其他视觉观测的相关性，仍未显式保存。IMU 传播也仍采用一阶协方差离散化。因此，本次修正不构成整个 SLAM 系统的严格统计一致性证明；内部视觉 NIS 仍是近似门控量，尤其不能把通过鲁棒筛选后的数值当作无选择偏差的标准 NIS 实验。

## IMU 噪声配置

现有多相机配置显式加入：

```yaml
imu:
  noise_model: discrete
```

`discrete` 保留原来的 `Q ∝ dt²` 约定：`gyr_cov`、`acc_cov` 分别表示送入传播的有效中点角速度、加速度的离散方差；两个 `b_*_cov` 表示离散偏置变化率的方差。加速度参数对应代码完成重力尺度归一化后的物理加速度。原始 IMU 采样方差并不自动等于中点平均后的有效方差，相邻中点共享样本的时间相关性仍属近似。

`continuous` 使用 `Q ∝ dt` 的一阶白噪声模型，此时上述四项应填写连续功率谱密度（噪声密度或偏置随机游走强度的平方）。它们不是标准差。已有的曝光噪声继续保持原约定，时偏过程噪声继续使用原有 `dt` 约定。

若从 [Kalibr IMU 噪声模型](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model) 转换参数，需要先平方噪声密度，并使用对应的单位。不要只切换模型字符串而保留另一种含义的参数数值；本次没有凭 NEES 结果调整噪声大小。

## 评估输出

仍输出原轨迹文件 `Log/result/<seq_name>.txt` 和独立评估文件 `Log/result/<seq_name>_nees_eval.txt`。原轨迹文件的写出逻辑、评估数据行的 6×6 协方差排列和 stage 列保持原格式。评估文件额外包含注释：

```
# covariance_model=full_joseph_tangent_shared_map_v2; imu_noise_model=discrete
# full_cov timestamp stage dimension min_eigenvalue max_eigenvalue max_asymmetry
```

第二种注释每条位姿记录一行，保存完整状态协方差的健康信息。按 `#` 跳过注释的现有解析方式仍可使用。完整状态异常会用加粗红色 `printf` 提示；导出的位姿协方差不做裁剪或人为缩放。健康指标反映矩阵结构，不代替 NEES。

重新实验请使用新的 `seq_name`，保留旧结果；保持噪声参数、外参、时间对齐及场景一致，运行 only-LIO off/on 和 LIVO off/on。先检查完整矩阵健康记录，再按同一 stage 比较 ATE/RPE、实际误差、协方差及 NEES。此前用 mocap4 本身拟合的外参/时延带来的评估不确定性，不会由这些代码修正自动消除。

## 验证状态（2026-09-15）

- 已运行 `python tests/check_covariance_derivations.py`：9 组独立 NumPy 推导检查通过，包括有限差分、与稠密 Kalman/Joseph 更新对照、共享误差上界和低秩残差求解。这些检查不执行 C++ 源码。
- 已添加直接调用实际 Eigen 辅助函数的 C++ 测试 `tests/estimator_covariance_math_test.cpp`，独立 CMake 入口同时包含原有方向性更新测试。
- `git diff --check`、新增/修改 C++ 文件的词法括号检查通过；原 TUM 写出代码块与修改前一致。这些静态检查不等于编译验证。
- 尚未编译 C++、运行 C++ 测试或重放 ROS 数据。运行时间、轨迹误差和 NEES 变化尚未验证。

在有 Eigen3 的构建环境中，可以单独运行数学测试，无需 ROS：

```bash
cmake -S tests -B /tmp/cake-covariance-tests
cmake --build /tmp/cake-covariance-tests
ctest --test-dir /tmp/cake-covariance-tests --output-on-failure
```
