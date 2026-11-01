# 多相机视觉候选预算：实现与远端验证

总开关为 `vio.patch_budget_en`，默认 `false`。关闭时走加入模块之前（提交 `23e710b`）的原始处理路径，沿用原参数；设为 `true` 时才启用新模块。未修改后端协方差更新逻辑。

## 开关与启动

| 参数 | 可选值 | 含义 |
|---|---|---|
| `vio.patch_budget_en` | `false/true` | 总开关；false 为原始路径，true 为新模块，启动时读取 |
| `vio.max_total_points` | 沿用各数据集原值 | 关闭时使用的原预算参数（自测/HILTI22 为 500，M2DGR 为 150） |
| `vio.patch_budget_points` | 默认 `400` | 仅开启时使用的新模块预算，不覆盖原参数 |
| `vio.patch_budget_mode` | `legacy` | 在新模块路径中使用按有效点数扣减的旧规则，候选尝试数可能超过配置值 |
| | `sequential` | 原相机顺序，失败也扣预算，严格限制候选尝试数 |
| | `global` | 所有相机先收集和评分，再用保底/上限约束分配全局名额 |
| `vio.patch_score_mode` | `shi_tomasi` | 原角点评分 |
| | `isotropic` | 平移光度敏感度，方向矩阵为单位阵 |
| | `lidar_directional` | 使用 LiDAR 平面法向量形成方向权重 |
| `vio.patch_budget_log_en` | `false/true` | 输出逐帧预算、几何有效性和耗时 |

参数分别保存在各数据集原有 YAML 的 `vio` 段：自测使用 `config/mid360_mei_multi_cam.yaml`，HILTI22 使用 `config/HILTI22_multi_cam.yaml`，M2DGR 使用 `config/M2DGR_multi_cam.yaml`。各自原有 launch 直接加载本数据集 YAML 和相机 YAML。

直接修改所用数据集 YAML 中已存在的参数，修改后重启节点：

```yaml
vio:
  patch_budget_en: false  # false：原始路径；true：启用新模块
  patch_budget_points: 400
  patch_budget_mode: global
  patch_score_mode: lidar_directional
  patch_budget_log_en: false
```

三套 YAML 各自配置为 `false + global + lidar_directional + 400`，可以分别调整。关闭后，新模式、400 配额和新增日志选项均不生效，继续使用数据集原来的 `max_total_points` 及其预算语义。地图管理、跨相机残差、相机标定和原预算值继续使用各自原配置。

原始路径与 `patch_budget_mode: legacy` 有明确区别：总开关关闭时调用保留的原始 VIO 函数，跳过候选准备/处理拆分、全局调度和新增日志；LiDAR 调用原始残差函数，不收集方向统计，地图点 ID 也保留原行为。`legacy` 则是开启模块时可选的旧预算规则，用于对照实验。原始函数保存在 `src/vio_patch_budget_baseline.inc`，新增 LiDAR 统计放在 `src/voxel_patch_budget_geometry.inc`；两者由对应 cpp 包含，提交代码时需一并包含。

在所用 YAML 中将 `patch_budget_en` 设为 `true` 后，继续使用对应的原启动文件。以下是三个入口，选择当前数据集的一条：

```bash
roslaunch cake_slam mapping_mid360_mei_multi_cam.launch
roslaunch cake_slam mapping_hilti22_multi_cam.launch
roslaunch cake_slam mapping_m2dgr_multi_cam.launch
```

需要同时启用实际地图管理及跨相机残差时，在当前数据集 YAML 的 `vio` 段修改对应已有项：`visual_map_manage_en: true`、`visual_map_manage_shadow_en: false`、`visual_map_retirement_apply_en: true`、`cross_camera_current_residual_en: true`。HILTI22 原配置为 shadow 模式且不实际应用退休，开启管理时需留意这两项。启动日志会打印最终生效的管理、shadow、退休和当前跨相机残差状态。

新增数据集仍按原有方式配置自己的 YAML 和 launch。同一前端与残差模型使用相同的 C++ 兼容性修复，不依赖数据集名称，也不要求改变参数文件的组织方式。

恢复原始路径只需在 YAML 中设置 `patch_budget_en: false`。`patch_budget_points` 不修改原始预算。自测 launch 原有的五项预算/评分命令行覆盖参数继续保留，省略时不覆盖 YAML；HILTI22 和 M2DGR 直接在各自 YAML 中配置。最低/最高相机配额继续使用数据集的 `points_per_camera_min/max`。

新预算/评分模式限定为 `frontend_mode=0`、dense virtual patch、`virtual_raw_score_select_en=true`，与这三套配置的当前前端一致。支持同时开启 `visual_map_manage_en` 和 `cross_camera_current_residual_en`，保留多参考选择、回退、地图生命周期及当前相机对残差的原始门限、归一化权重和 Jacobian。当前仍不支持 adaptive sampling、sparse patch 或 ZNCC 残差组合；这属于功能前提，不按数据集名称判断。基础图像噪声必须为有限正数，总预算须满足相机最低配额。

与地图管理一起使用时，在实验配置中设置 `patch_budget_en: true` 和 `visual_map_manage_en: true`；原有 `visual_ref_current_select_en`、`visual_ref_fallback_en` 等管理选项继续生效。

这些新增检查只在总开关为 true 时执行。报错逐项给出参数名、实际值和要求，包括相机数量变化导致的最低配额不足；模式名拼写错误也显示实际输入。主程序统一报告异常并返回失败状态，不再用一条混合提示让用户猜测冲突来自地图管理、相机对或其他功能。新增数据集只要满足同一前端及参数要求，就经过完全相同的共享路径；更换前端或观测模型仍须满足对应数学和实现前提。

## 数学对象与开销

每个候选复用原 Shi–Tomasi 的 8×8 中心差分扫描，得到 `G = mean(g g^T)`；在 patch 中心使用相机解析投影 Jacobian。分数为：

```text
s = inverse_exposure² / fixed_image_noise_variance
    * trace(G * J_projection * R_cw * W * R_cw^T * J_projection^T)
```

它是局部平移光度敏感度，未建模成功概率、旋转消元、观测相关性或真实后验信息增益。所有相机沿用当前曝光模型及统一基础图像噪声尺度，没有读取状态协方差 P，也没有按各相机平均分重新归一化。

LiDAR 方向统计只在第一轮几何关联中均匀抽取至多 1024 个点，在原主体素遍历期间、协方差门控之前收集最近的有效平面法向量。固定点到面距离阈值为 0.3 m；不收集依赖协方差门控结果触发的相邻体素搜索。法向量不少于 30 个时计算：

```text
H = mean(n n^T)
W = 0.1 * inverse(H + 0.1 I)
```

统计不足、无效或时间差不在 [-0.05, 0.5] 秒时，当帧回退到单位方向权重，日志 `geometry_fresh=0`。这不意味着整个估计器不使用 P，只意味着给定姿态与地图时，选择器不读取 P。

每台相机以地图点分组，同一点只计算一次评分，主参考和备用参考随组保留，不分别参与排序。全局模式每台相机最多缓存 `points_per_camera_max` 个点组的元数据及其参考索引，不预先生成 support。先给每台相机保留配额，再根据各相机的下一点组评分分配剩余配额；相机 ID 与地图点 ID 用于确定同分顺序。

在 global/sequential 模式下，配额限制的是主参考与备用参考合计的实际处理次数。每批按参考链的最大可能次数预留配额，不同点组仍可并行；同一点组内按顺序处理参考，主参考只有发生 NCC/光度拒绝时才触发原有回退，任何一个参考成功后立即停止该组。未使用的预留配额可供本相机下一批点组使用，实际尝试失败的次数不会退回。尾部参考链若超过最终剩余配额，则只处理预算允许的前缀；没有处理的参考不记为匹配失败，也不产生拒绝证据。每台相机中同一点最多加入一次被接受的观测；同一点在不同相机中的观测仍可分别参与更新。legacy 模式仍按有效点数扣除跨相机预算，不提供严格尝试次数上限。

对相机配额 q，每批预留量不超过剩余配额，该批实际尝试数又不超过预留量，因此累计尝试数不超过 q；各相机配额之和不超过总预算。400 表示最多 400 次参考处理尝试，不保证 400 个不同点或有效点。保底配额也是处理机会，而非有效点数量；候选不足时可以少用预算，全局模式不在相机之间二次转移未用配额。

分配器对“每个主参考单位成本”的静态评分目标仍是最优分配；启用回退后，备用参考会占用原本可用于后续点的名额，所以该目标只是分配依据，不代表实际最终观测集合最优。一个名额不等于固定毫秒数；新地图点生成、联合更新及其他后台工作不包含在次数上限中。此次兼容修改没有新增图像试读或读取 P；分组、配额检查与批次调度会有额外开销，尚未测量。

当前跨相机残差复用已经选中的有效观测，不新增一轮用于选择器评分的图像试读，不截断现有的相机对列表。设进入配对的观测数为 N、相机数为 C，同一点 p 被 m_p 台相机观测，则可能的相机对数为 `sum_p m_p*(m_p-1)/2 <= N*(C-1)/2`。严格预算模式下 `N <= B`，所以相机对尝试数至多 `floor(B*(C-1)/2)`；三相机、B=400 时至多另有 400 次相机对尝试。400 是历史参考尝试预算，不是历史参考与相机对计算的合计上限，也不是整帧耗时上限。拒绝相机对也会计入尝试次数；其真实开销包含在 VIO/Joint EKF 阶段中。

每帧在应用地图退休前清空上一帧的相机对引用；零观测帧即使跳过联合更新，也不会保留上一帧相机对的地图指针。

## 建议对照

保持 bag、播放速率、日志和其他配置一致，仅按表改变策略及预算，每次启动独立进程。若评估地图管理开启的系统，各组均应保持 `visual_map_manage_en: true` 和相同的管理子选项：

| 对照 | patch_budget_en | budget mode | score mode | 预算来源 |
|---|---|---|---|---|
| 原配置基线 | false | 忽略 | 忽略 | 数据集原 max_total_points |
| 严格预算基线 | true | sequential | shi_tomasi | patch_budget_points=400 |
| 只改全局分配 | true | global | shi_tomasi | patch_budget_points=400 |
| 新评分，顺序分配 | true | sequential | lidar_directional | patch_budget_points=400 |
| 新评分，全局分配 | true | global | lidar_directional | patch_budget_points=400 |
| 去掉方向权重 | true | global | isotropic | patch_budget_points=400 |

总开关为 false 时不输出 `[PATCH_BUDGET]`，使用原有系统阶段耗时做基线对照。开启模块且开启日志时，`[PATCH_BUDGET]` 中：

- `mode=0/1/2` 对应 legacy/sequential/global；`score=0/1/2` 对应 shi_tomasi/isotropic/lidar_directional。
- `scored` 按每台相机的不同地图点计数，多个历史参考只评分一次。
- `attempted` 是实际进入处理的主参考和备用参考次数，严格模式应始终不超过 `limit`；未执行的备用参考不计数。`accepted` 是实际有效观测数，同一相机内不会重复接收同一点。
- `cameras_quota_attempted_accepted` 展示每台相机的配额、尝试数、有效数。
- `prepare_ms` 包含候选收集、评分及排序；`sort_ms` 是其中的排序子阶段，不能再次相加。
- `allocate_ms` 是全局标量分配与截取；`vio_ms` 是整个 VIO 阶段，已经包含前述工作。
- `geometry_aggregate_us` 只包含法向量汇总和 3×3 分解，不包含关联时的收集开销。几何收集的全部开销在既有 LIO 阶段计时内。
- `pair_observations` 是进入当前相机配对的观测数；`pair_bound` 是由该数量及相机数计算出的理论数量上界，不是另一个配置预算。
- `pair_attempted`/`pair_accepted` 分别统计实际开始处理及通过门限的当前相机对；它们单独统计，不混入历史参考的 `attempted`。原相机对计算与权重规则保留。

先核对实际配额和上限，再比较公共时间段的 LIO+VIO 耗时、有效点数和共同真值时间戳上的 ATE。ROS 回放相位或线程调度可能造成波动，建议至少重复两次。新增调度开销平均 0.2 ms 是目标，不是已验证结果。

## 本地数学测试

不依赖 ROS，使用 Eigen 的独立 CMake 测试：

```bash
cmake -S tests -B /tmp/cake-patch-budget-tests -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cake-patch-budget-tests \
  --target visual_patch_budget_test visual_patch_score_math_test -j2
ctest --test-dir /tmp/cake-patch-budget-tests -R visual_patch --output-on-failure
```

测试包括 2000 组小规模穷举最优值对照、相机置换和同分不变性、缺失候选、配额边界、显式 64 像素 Jacobian 能量对照、亮度/曝光尺度不变性、世界坐标旋转一致性与几何失效回退。纯分配器计时不包含图像扫描、候选排序或 ROS 运行时，不能当成完整系统开销。

总开关测试检查 false 时忽略新模式、配额与日志设置，包括无效实验配置；原函数源码检查使用原提交的固定指纹，验证关闭路径的两个 VIO 与两个 LiDAR 函数保持原始函数体，以及 launch 配额参数隔离。源码一致性不代表编译后的二进制、每次回放轨迹或耗时逐位一致。

最初版本完成过 4 项独立检查及 YAML/launch 解析。之后的地图管理、跨相机残差兼容修改，以及恢复各数据集独立参数配置，均按用户要求只做简单静态检查：阅读差异、XML 结构与加载路径检查、git diff --check；没有编译、运行测试、离线实验或连接远端。既往检查结果不覆盖这些后续改动，完整编译和运行效果尚未验证。
