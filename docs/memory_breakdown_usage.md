# 自动分项内存统计：NUC 开启与查看

适用：多相机 `cake_slam_mapping_multi_cam`。新增功能独立于 `vio.usage_stats_en`。
本地未编译、未运行 ROS；以下编译命令只在 NUC 执行。

## 1. 同步并在 NUC 编译

把本轮源码同步到原工作空间 `/home/cak/cake_ws/src/cake_slam`，保留原来的
`build/devel` 和已配置的 OpenCV 路径，然后在 **NUC** 执行：

```bash
cd /home/cak/cake_ws
catkin_make -j2
source devel/setup.bash
```

原来的 roslaunch 和 rosbag 命令继续使用。

## 2. YAML 开关

在实际启动时使用的 YAML 中，找到已有的 `debug:` 段并加入/修改：

```yaml
debug:
  memory_stats_en: true
  memory_stats_interval_sec: 1.0
  memory_stats_print_interval_sec: 10.0
  memory_stats_output_dir: "/home/cak/tangent_memory"
```

三个多相机主配置（M2DGR、HILTI22、mid360）已加入以上参数，默认关闭。
不要再添加第二个同名 `debug:` 段。参数在节点启动时读取，修改后重新启动节点。

- `memory_stats_en: false`：不创建 reporter、文件或对象遍历，不读取 `/proc`。
- `memory_stats_interval_sec`：单调墙钟采样间隔；0 表示每个安全检查点都采样，
  数据较大时不建议这样设置。
- `memory_stats_print_interval_sec`：控制台完整分项汇报间隔。
- 输出目录为空：使用编译源码根目录下的 `Log/memory/`。
- 每轮自动创建 `<seq_name>_<时间>_<PID>` 子目录，避免覆盖此前实验。

启用后会打印 `Memory statistics output: ...` 的完整路径。内存统计使用独立
`ros.cake_slam.memory_stats` INFO logger，不必恢复所有 IMU/图像到达 INFO。

正式测量时可同时设置 `vio.usage_stats_en: false`，关闭几何近似误差/NCC/使用统计。
这不关闭基础 VIO 计时，也不关闭新的内存统计。

启动后确认实际参数：

```bash
rosparam get /laserMapping/debug/memory_stats_en
rosparam get /laserMapping/debug/memory_stats_interval_sec
```

## 3. 自动输出

每轮目录内：

| 文件 | 内容 |
|---|---|
| `memory_samples.csv` | 每个采样时刻、每个分项的有效数据字节、保留容量、节点开销估计、数量、共享引用数 |
| `memory_summary.csv` | 每个分项及已覆盖合计的当前、采样峰值、算术均值、时间加权均值、最终占用；合计估算峰值的位置 |
| `process_samples.csv` | Linux RSS/HWM、匿名/文件/共享驻留、swap、采集耗时 |
| `process_summary.csv` | 整进程字段的当前、峰值、均值、最终汇总及峰值观察时刻 |
| `coverage.txt` | 口径、去重规则、估计项、采样边界和外部堆剖析范围 |

文件每次采样都会刷新。汇总不必等结束才生成；正常 Ctrl+C 后增加 `final` 汇总。
最后业务状态与上一采样完全相同时，不重复增加业务统计样本或改变均值。
整进程最终 RSS 会另外刷新，额外最终观察不参与均值，CSV 有
`is_statistical_sample=0` 标记。

采样位于主线程安全检查点（初始化、处理完 LIO/VIO、等待输入、正常结束），
不会在另一个线程无锁遍历地图。`frame` 对 VIO 使用当前处理的多相机帧号，
对 LIO 使用当前处理计数；`data_time_sec` 和 `stage` 同时保留。

## 4. 覆盖分项

- `vio.map.*`：地图体素索引、地标、参考观测、core patch、参考图像、动态数组。
- `vio.retirement.*`：待删除地标/参考及影子维护集合；未释放的对象仍计入。
- `vio.camera_N.*`：相机对象、当前帧图像、网格/射线 LUT、待建参考、子图、
  变换 patch、有效掩膜、梯度缓存、跨相机及自适应支持。
- `vio.ekf.*`：信息矩阵、每线程线性化/NIS/LLT 工作区、逐观测结果及雅可比。
- `vio.cross_camera.*`、`vio.calibration.*`、`vio.usage.*`、`vio.optical_flow.*`、
  `vio.color.*`、`vio.dump.*`：相应功能的实际保留缓存。
- `lidar.map.*`：实际 LiDAR 地图索引、八叉树节点、平面、保留点和层级数组。
- `lidar.workspace.*`：协方差、点面对应、残差工作区；共享扫描协方差只计一次。
- `preprocess.*`：完整/角点/面点云、128 条扫描线点云与类型缓存。
- `imu.*`：去畸变点云、姿态缓存、传播工作区和消息缓冲。
- `mapper.state*`、`mapper.propagated_state*`、`lidar.state*`、`imu.*state*`：
  各自独立的状态动态数组及协方差。
- `input.*`：待同步图像、完整图像组、LiDAR/IMU 队列及当前同步批次。
- `output.*`：轨迹、待发布点云及存盘缓存。
- `mapper.config.*`：标定表、去畸变映射及配置动态数据。
- `optional.color_gray.*`：颜色初始化对象、完成结果、worker 输入快照已知容量。

图像按底层 `UMatData` 去重，点云/消息/地图对象按独立对象身份去重。
同一存储由多个分项引用时，归属给首先访问的分项；参考地图先于当前帧/输入缓冲。
`shared_references` 表示复用次数，不再增加一份字节。

### 字节口径

`payload_bytes` 是有效数据；`capacity_bytes` 已包含 payload 和预留容量。
`overhead_estimate_bytes` 是额外的容器节点/bucket 开销估计。
`accounted_estimate_bytes = capacity_bytes + overhead_estimate_bytes`。
不要把 payload 再加到 capacity 上。

动态 Eigen 按当前系数存储计量；vector 计 capacity；list/map 的节点开销是估计；
deque 的公开接口只能给出活元素字节下界，不能完整取得其闲置块和内部指针表。
这些是业务对象容量/已覆盖估算，不是各模块 RSS 的精确分摊。

算术均值按统计样本等权；时间加权均值采用相邻样本间的左保持值，时间轴是
steady-clock 墙钟。观察区间内的启动等待、暂停和结束等待会计入；实验比较应保持
播放和停止流程一致。单样本时加权均值留空。各分项自身峰值不应相加；需要内存构成
时读取同一个 sample_id 下的各项。

`VmHWM` 是内核 RSS 高水位；分项 `sampled_peak` 只是在采样点观察到的峰值。
`collector_ms` 记录业务遍历开销，未包含 reporter 聚合/打印/写盘开销。

## 5. 查看方法

在 NUC：

```bash
source /home/cak/cake_ws/devel/setup.bash
PKG_PATH=$(rospack find cake_slam)
python3 "$PKG_PATH/scripts/view_memory_stats.py" /home/cak/tangent_memory
```

传父目录自动选择最近更新的运行。也可传入一轮输出的完整子目录。

细看视觉地图：

```bash
python3 "$PKG_PATH/scripts/view_memory_stats.py" /home/cak/tangent_memory \
  --prefix vio.map --group-level 0
```

逐相机/模块的二级分类：

```bash
python3 "$PKG_PATH/scripts/view_memory_stats.py" /home/cak/tangent_memory --group-level 2
```

工具的分组峰值是先逐快照加总分组，再取峰值，不是把各子项的独立峰值相加。
实时刷新可用 `watch -n 10` 包裹上面的命令；直接读 CSV 时文件正更新可能短暂不完整，
正常结束后的文件适合归档和论文统计。

## 6. 私有库分配与瞬态峰值：可选 Heaptrack

容量快照不能遍历 ROS 内部传输/序列化队列、标准库/分配器控制块、PCL 私有叶布局、
第三方 PinholeCamera 私有映射图、颜色 worker 的不透明函数捕获/相机副本，
或读取已经销毁的函数内临时矩阵。匿名 threadprivate virtual patch workspace 也
不是跨翻译单元可遍历的成员。这些列为未单独归属，不能写成测得 0 字节。
完整 RSS 已包含它们；若要追踪其申请来源和短时峰值，用 Heaptrack：

```bash
# 只在 NUC 执行；保留现有优化，增加调用栈所需符号。
sudo apt install heaptrack
cd /home/cak/cake_ws
catkin_make -j2 -DCAKE_MEMORY_PROFILE_SYMBOLS=ON
source devel/setup.bash
roslaunch cake_slam mapping_m2dgr_multi_cam.launch \
  use_rviz:=false memory_profile_prefix:=heaptrack
```

HILTI/mid360 的三个多相机 launch 都支持同样的可选 `memory_profile_prefix` 参数。
默认空字符串，原启动命令不变。Heaptrack 从节点启动时追踪，ROS 参数加载方式不变。
结果文件路径由 Heaptrack 自己打印。结束后：

```bash
heaptrack_print /替换为实际输出路径/heaptrack.cake_slam_mapping_multi_cam.PID.gz
```

有 GUI 时也可使用 `heaptrack_gui` 打开对应文件。其结果是分配调用栈意义的 heap，
不是当前数据归属自动分类；一张由回调分配的图像可能随后由地图参考长期保留。
需结合业务对象明细解读。工具剖析和正常性能测试分开运行。

## 7. 地图管理消融

同一条数据分别运行 w/o MM 和 Ours，每次重新启动进程。两组内存统计选项一致，
各轮独立目录。MM 主开关和几何筛选开关的配置沿用原联合实验教程。

论文可汇总进程峰值 RSS、平均 VIO 时间，并用分项快照解释内存节省来自哪里。
内存采集本身有开销；正式时延基准可另外关闭 memory_stats_en 运行。
不要把 RSS 减去容量估算后命名为“库内存”，二者没有可直接相减的统一口径。
