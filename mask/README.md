# Mocap / MID360-MEI 相机 mask

只在 `config/mid360_mei.yaml` 和 `config/mid360_mei_multi_cam.yaml` 中配置。
其他数据集不设置 `cameras.cameraN.mask_path`，保持原有行为；代码不通过 topic 子串自动匹配。

| 镜头 | 当前 topic | 文件 | 状态 |
| --- | --- | --- | --- |
| left | `/fisheye/left/image_raw` | `mask/left.png` | 已启用 |
| right | `/fisheye/right/image_raw` | `mask/right.png` | 已启用 |
| bright | `/fisheye/bright/image_raw` | 暂无 | `mask_path: ""`，尚未过滤 |

两张 PNG 均为 **1088×1280、8 位单通道**，白色有效、黑色无效。
圆内设备平台、线缆以及真实黑色物体保留。圆边界根据图片估计；当前版本圆心保持 left `(535,649)`、right `(550,631)`，半径从 541 向内缩 10 像素至 531。

- 初始化时读取一次，之后只查询内存中的 mask。相对路径以构建时项目根目录为基准，也支持绝对路径。
- 文件必须对应实际用于 VIO 的图像坐标和尺寸。当前 MEI 配置 `scale: 1.0` 且不开启输入去畸变，因此直接使用这两张原图坐标的 mask。
- 指定路径但文件读取失败或尺寸不匹配时，初始化报错；不静默忽略、不自动拉伸。若改变缩放或启用去畸变，应同时提供相应坐标下的 mask。
- 视觉侧检查选点评分、插值、梯度、patch 和滤波的实际像素范围。参考帧使用对应参考相机的 mask，ROI 采样换回完整图像坐标。
- 着色逐台相机遍历点云，仅接受变换到该相机坐标系后 `z > 0` 的点（前方 180° 半球，`z = 0` 也排除），并在读取颜色之前检查该镜头 mask 中双线性插值的四个像素。每个有效相机观测都输出一份；同一个三维点在多个镜头中有效时，会保留坐标相同、颜色分别来自对应镜头的多份点。所有相机均无有效颜色时，不加入彩色点云，LiDAR 几何仍保留。前半球限制只用于着色，不改变视觉跟踪的视场。
- 日志 `colored` 是至少在一台相机中成功着色的输入点数，`output_points` 是实际输出的 RGB 条目数（包含重叠视野中的重复坐标）。原始 RGB 发布和保存保留这些条目；下采样导出仍按原来的体素过滤设置处理。
- 着色距离开关为 `publish.color_range_en`，范围由 `publish.color_min_distance` / `publish.color_max_distance` 指定（米，包含上下限）。距离是点到当前相机的三维直线距离，每台相机独立判断；仍需满足 `z > 0`、mask 和原有 `blind_rgb_points` 下限。范围过滤只影响 RGB 输出，不改变定位和 LiDAR 几何。日志 `range_rejects` 记录通过投影、mask 和盲区检查后被距离范围排除的相机观测数。
- 两份 `mid360_mei` 配置启用着色距离范围 `[0, 5]` 米；其他配置不填写时默认关闭。设 `color_range_en: false` 关闭新增距离范围限制；前半球、mask 和原有盲区过滤仍生效。启用时要求有限的 `0 <= color_min_distance <= color_max_distance`，不合法则初始化报错。
- 设 `mask_path: ""` 可单独关闭某一路。第三路 bright 尚未提供 mask，仍可能从该路黑边取色；不能用 left/right 的 mask 代替。

当前 ROS1 分支的单相机 `mapping_mid360_mei.launch` 也使用多相机节点，并将 left 配置为 camera0。

验证：`tests/camera_mask_test.cpp` 编译生产代码中的 mask、采样和取色函数，覆盖边界邻居、跨相机取色、参考相机、ROI、滤波及无 mask 行为。该测试使用简化相机/帧夹具，不代表完整 ROS 节点或 rosbag 已验证。
