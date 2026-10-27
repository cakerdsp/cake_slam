# Tangent-LIVO 内存与运行时间联合实验教程

2026-10-26。适用：NUC 上 Ubuntu / ROS Noetic，已有工作空间 /home/cak/cake_ws。依据当前本地源码静态核对，未在 NUC 执行、未编译。

本轮序列：Street03、Gate03、Exp03、Exp21，以及作者选定的自采序列。每条序列比较 w/o MM 与 Ours。内存与时间在同一轮运行中采集。

## 1. 旧实验如何做，这次能否复现

旧稿记录的是：同一序列分别开启和关闭地图管理，pidstat 以 1 Hz 记录整个 SLAM 进程的 RSS，取采样最大值，并报告 ATE。当前仓库没有找到旧实验的 pidstat 原始日志或专用采集脚本，因此能复现测量方法，不能据此保证复现旧的六组数值。

内存测量无需在 SLAM 中打开专门的内存开关，由外部工具读取进程即可。当前代码默认输出 LIO Mapping Time 和 VIO Time，不需要另开计时开关。

这次建议保留逐秒 RSS 曲线，同时记录 /proc/PID/status 中的 VmHWM。前者便于与旧实验口径对照，后者是内核记录的 RSS 高水位。二者分别命名 Sampled peak RSS 和 Peak RSS (VmHWM)，不要混用。Linux 对 RSS 的记账有自身近似，VmHWM 也不是逐字节的实际分配峰值。

## 2. 正式运行前必须明确的配置问题

完整地图管理涉及两个独立入口：vio.visual_map_manage_en 和 vio.visual_geom_filter_en。前者管理路标与参考；后者控制几何候选筛选。若论文的 w/o MM 指去掉整个第二贡献，二者应一起关闭。

| 参数 | w/o MM | Ours |
|---|---:|---:|
| vio.visual_map_manage_en | false | true |
| vio.visual_geom_filter_en | false | true |
| vio.visual_map_manage_shadow_en | false | false |
| vio.visual_map_retirement_apply_en | true（管理关闭时不执行） | true |
| common.directional_update_en | false | false |
| vio.virtual_adaptive_sampling_en | false | false |
| vio.color_gray_init_en | false | false |

参考选择、回退、生命周期、视角覆盖、NIS、种子验证、足迹去重、信息筛选及替换等子开关均保持 true；主开关关闭后走对应旧路径。几何筛选和管理阈值沿用各数据集当前配置，不为实验临时放宽。

当前本地配置特别需要纠正：
M2DGR：管理与几何筛选均关闭。
HILTI22：管理与几何筛选均关闭，而且 shadow=true、retirement=false；只打开管理主开关不会得到完整的实际删除行为。
Mid360 自采：管理与几何筛选均关闭，directional_update_en=true，需要为本轮实验关闭。

下方参数副本生成步骤已处理这些开关，不修改源 YAML。

### 论文与代码的一处实际差异

当前 selectReferenceCandidates、shouldAddObservation 和 manageReferenceBank 中，距离排序与距离覆盖仍受 virtual_adaptive_sampling_en 控制。关闭 SCPS 后，这些距离准则也关闭。

因此，按当前代码可以完成“未使用距离准则的地图管理”测试。如果最终论文继续包含距离准则，应先在代码中将距离管理与 SCPS 解耦，再运行正式结果。本教程没有擅自修改该依赖，也没有通过重新打开 SCPS 绕过它。

## 3. 表格如何合并

建议用一张表，逐个序列列出两个版本：

| Sequence | Variant | Peak RSS (MiB) | Mean LIO (ms) | Mean VIO (ms) | P95 VIO (ms) |
|---|---|---:|---:|---:|---:|
| Street03 | w/o MM | 待测 | 待测 | 待测 | 待测 |
| Street03 | Ours | 待测 | 待测 | 待测 | 待测 |
| Gate03 | w/o MM / Ours，分两行 | 待测 | 待测 | 待测 | 待测 |
| Exp03 | w/o MM / Ours，分两行 | 待测 | 待测 | 待测 | 待测 |
| Exp21 | w/o MM / Ours，分两行 | 待测 | 待测 | 待测 | 待测 |
| 自采序列 | w/o MM / Ours，分两行 | 待测 | 待测 | 待测 | 待测 |

ATE 保存在原有精度实验中；测试时仍保存轨迹。若希望表格独立说明精度是否变化，可以增加 ATE 一列。

这张表回答的是完整系统相对关闭地图管理版本的内存和时间差异。若还要做 Omni-LIVO 效率比较，应另补同一硬件、输入和计时边界的基线运行；现有原生计时边界不能直接视为完全一致。

## 4. 现有计时究竟测到哪里

LIO Current Total Time：从下采样前到体素地图更新后。包含其中的状态估计和部分输出，但不包含前面的 IMU 传播/去畸变，也不包含后面的地图滑动与部分发布。

VIO Current Total Time：从多相机图像准备开始，到参考维护与部分调试绘制结束。包括参考维护、实际删除与主要视觉处理；不包含之后遍历地图统计 Active Visual Points、Mean Reuse Count 及打印耗时表的开销，也不是完整 handleVIO 调用耗时。

所以表头写 Mean LIO / Mean VIO，表注说明使用代码现有计时边界。不要简单相加后称为完整一帧延迟，也不要只凭这些均值证明端到端实时性。

Current Total Time 的单位是秒，下方提取步骤转换为毫秒。Average Total Time 是累计均值，不能再把所有行累计均值平均一次。

## 5. NUC 实际执行：终端一准备参数并启动 SLAM

沿用当前已有的编译产物。不需要为了采集 RSS 和日志重新编译。

先每个版本跑一遍确认流程，再按同样流程重复正式测试。每轮自动使用单独目录，避免日志覆盖。

下方只需要改 BENCH_FAMILY、BENCH_SEQ 和 BENCH_MODE：
BENCH_FAMILY：m2dgr / hilti22 / mid360。
BENCH_SEQ：street03 / gate03 / exp03 / exp21 / 你的自采序列名。
BENCH_MODE：on 表示 Ours；off 表示 w/o MM。

~~~bash
source /home/cak/cake_ws/devel/setup.bash
export BENCH_FAMILY=m2dgr
export BENCH_SEQ=street03
export BENCH_MODE=on
export BENCH_ROOT="$HOME/tangent_bench"
export BENCH_RUN="$BENCH_ROOT/$BENCH_SEQ"_mm_"$BENCH_MODE"_$(date +%Y%m%d_%H%M%S)
export BENCH_PKG="$(rospack find cake_slam)"
mkdir -p "$BENCH_RUN"

case "$BENCH_FAMILY" in
m2dgr)
BENCH_BASE="$BENCH_PKG/config/M2DGR_multi_cam.yaml"
BENCH_CAMERA="$BENCH_PKG/config/camera_M2DGR_multi_cam.yaml"
BENCH_LAUNCH=mapping_m2dgr_multi_cam.launch
BENCH_ARG=m2dgr_params_file
;;
hilti22)
BENCH_BASE="$BENCH_PKG/config/HILTI22_multi_cam.yaml"
BENCH_CAMERA="$BENCH_PKG/config/camera_fisheye_HILTI22_multi_cam.yaml"
BENCH_LAUNCH=mapping_hilti22_multi_cam.launch
BENCH_ARG=hilti22_params_file
;;
mid360)
BENCH_BASE="$BENCH_PKG/config/mid360_mei_multi_cam.yaml"
BENCH_CAMERA="$BENCH_PKG/config/camera_mei_multi_cam.yaml"
BENCH_LAUNCH=mapping_mid360_mei_multi_cam.launch
BENCH_ARG=mid360_params_file
;;
*)
printf 'BENCH_FAMILY must be m2dgr, hilti22, or mid360\n'
exit 1
;;
esac

cp "$BENCH_CAMERA" "$BENCH_RUN/camera.yaml"
cp "$BENCH_BASE" "$BENCH_RUN/base_params.yaml"

python3 - "$BENCH_BASE" "$BENCH_RUN/params.yaml" "$BENCH_MODE" "$(basename "$BENCH_RUN")" <<'PY'
import sys
import yaml
source_path, output_path, mode, tag = sys.argv[1:]
assert mode in ("on", "off"), "mode must be on/off"
cfg = yaml.safe_load(open(source_path, encoding="utf-8"))
enabled = mode == "on"
cfg.setdefault("common", {}).update(directional_update_en=False)
vio = cfg.setdefault("vio", {})
vio.update(visual_map_manage_en=enabled, visual_geom_filter_en=enabled, visual_map_manage_shadow_en=False, visual_map_retirement_apply_en=True, virtual_adaptive_sampling_en=False, virtual_pose_affine_en=False, color_gray_init_en=False, visual_map_manage_log_en=False, visual_geom_filter_log_en=False, usage_stats_en=False)
vio.update(dict.fromkeys(["visual_ref_current_select_en", "visual_ref_fallback_en", "visual_ref_lifecycle_en", "visual_ref_view_coverage_en", "visual_ref_nis_en", "visual_point_seed_validation_en", "visual_point_footprint_redundancy_en", "visual_point_information_prune_en", "visual_point_replacement_en", "visual_ref_post_ekf_build_en"], True))
cfg.setdefault("pcd_save", {}).update(pcd_save_en=False, colmap_output_en=False)
cfg.setdefault("image_save", {}).update(img_save_en=False)
cfg.setdefault("debug", {}).update(plot_time=-10.0, ref_patch_dump_en=False, runtime_support_dump_en=False)
cfg.setdefault("evo", {}).update(seq_name=tag, pose_output_en=True)
with open(output_path, "w", encoding="utf-8") as stream: yaml.safe_dump(cfg, stream, sort_keys=False)
print("MM:", mode, "cameras:", cfg["common"]["num_cameras"], "output:", output_path)
PY
~~~

若 Python 报缺少 yaml 模块，安装 python3-yaml 后重新执行上面的生成步骤。若生成失败，不要继续启动。

~~~bash
sudo apt install python3-yaml
~~~

上述副本关闭 PCD/图片导出和额外诊断，保留轨迹输出。原因是 PCD 导出会在进程内累计点云，可能使内存曲线主要反映导出缓存。两组采用完全相同的导出与日志设置。相机数量、图像分辨率、线程配置、曝光估计、几何阈值、LiDAR 滑动地图等均沿用各数据集配置。

继续在终端一执行：

~~~bash
printf 'log4j.logger.ros=WARN\n' > "$BENCH_RUN/rosconsole.conf"
export ROSCONSOLE_CONFIG_FILE="$BENCH_RUN/rosconsole.conf"
printf '%s\n' "$BENCH_RUN" > "$BENCH_ROOT/current_run.txt"
git -C "$BENCH_PKG" rev-parse HEAD > "$BENCH_RUN/source_commit.txt" 2>/dev/null
git -C "$BENCH_PKG" diff > "$BENCH_RUN/source_changes.diff"
printf 'Run directory: %s\n' "$BENCH_RUN"
roslaunch cake_slam "$BENCH_LAUNCH" use_rviz:=false "$BENCH_ARG:=$BENCH_RUN/params.yaml" "camera_params_file:=$BENCH_RUN/camera.yaml" > "$BENCH_RUN/run.log" 2>&1
~~~

终端一保持运行。ROS_INFO 级别的 IMU/图像到达日志被抑制，printf 输出的计时表保留。关闭 RViz 不会自动取消程序内部的图像绘制；两组保持一致即可。

如果你平时使用定制过的相机参数文件，BENCH_CAMERA 应指向那份文件，不要替换成另一套标定。

## 6. 终端二：确认实际参数并采集内存

先启动终端一，再执行本节。一次只运行一个 SLAM 实验，current_run.txt 指向本轮目录。

~~~bash
source /home/cak/cake_ws/devel/setup.bash
BENCH_RUN="$(cat "$HOME/tangent_bench/current_run.txt")"
SLAM_PID="$(rosnode info /laserMapping | awk '$1 == "Pid:" {print $2; exit}')"
test -n "$SLAM_PID" && test -r "/proc/$SLAM_PID/status" || { printf 'SLAM node is not ready; inspect run.log first.\n'; exit 1; }
printf '%s\n' "$SLAM_PID" > "$BENCH_RUN/slam.pid"
rosparam dump "$BENCH_RUN/actual_rosparams.yaml" /laserMapping
readlink "/proc/$SLAM_PID/exe" > "$BENCH_RUN/executable_path.txt"
sha256sum "/proc/$SLAM_PID/exe" > "$BENCH_RUN/executable_sha256.txt"
grep -E 'VIO Map Config|VIO Ref Build' "$BENCH_RUN/run.log"

printf 'wall_time_s,rss_kib,hwm_kib,swap_kib\n' > "$BENCH_RUN/memory.csv"
while test -r "/proc/$SLAM_PID/status"
do
awk -v t="$(date +%s.%N)" '
/^VmRSS:/ {rss=$2; seen=1}
/^VmHWM:/ {hwm=$2}
/^VmSwap:/ {swap=$2}
END {if (seen) printf "%s,%d,%d,%d\n", t, rss, hwm, swap}
' "/proc/$SLAM_PID/status" >> "$BENCH_RUN/memory.csv" 2>/dev/null
sleep 1
done
~~~

Ours 启动日志应包含 manage=1 shadow=0，以及 retirement=1。地图管理每帧统计日志关闭，不影响启动时的 VIO Map Config。

采集对象来自 /laserMapping 的实际 PID。不要采 roslaunch、rosbag 或 RViz 的进程，也不要用整机 free/htop 的已用内存代替 SLAM 内存。

如果只是严格照搬旧的采样方法，等价的 pidstat 采集命令是以下这一条；本教程的 memory.csv 已提供 RSS 曲线，无需再同时运行它。

~~~bash
LC_ALL=C pidstat -r -h -p "$SLAM_PID" 1 > "$BENCH_RUN/pidstat_memory.log"
~~~

## 7. 终端三：播放数据并正常结束

沿用平时已经正确使用的话题转换、图像解压和数据播放流程。下面仅给单个 bag 的示例，替换 BENCH_BAG 为真实文件路径；不增加新的图像预处理。

~~~bash
source /home/cak/cake_ws/devel/setup.bash
BENCH_RUN="$(cat "$HOME/tangent_bench/current_run.txt")"
BENCH_BAG="/你的实际路径/street03.bag"
rosbag info "$BENCH_BAG" > "$BENCH_RUN/bag_info.txt"
date +%s.%N > "$BENCH_RUN/play_command_walltime.txt"
rosbag play --clock -r 1.0 --pause "$BENCH_BAG"
~~~

确认内存监控已经启动，再按空格开始播放。两组使用相同 bag、播放范围、速率和输入预处理。

bag 播放结束后，等 SLAM 的处理帧号与输出轨迹时间停止推进，再停止 SLAM。不要在 bag 刚播完时立即强行结束尚未处理完的数据。

可在另一个终端保存最终进程状态：

~~~bash
BENCH_RUN="$(cat "$HOME/tangent_bench/current_run.txt")"
SLAM_PID="$(cat "$BENCH_RUN/slam.pid")"
cp "/proc/$SLAM_PID/status" "$BENCH_RUN/proc_final_status.txt"
~~~

随后在终端一按 Ctrl+C 正常结束。终端二会在 SLAM 进程退出后自行结束。

轨迹默认写在编译源码目录的 Log/result/<本轮目录名>.txt，另有同名前缀的评估文件。代码按编译时 ROOT_DIR 输出，不一定等于另一个后来复制的源码目录。启动日志中的 EVAL INFO 会打印实际路径。归档这些文件，之后使用原有数据集的 ATE 评估流程，不修改已有坐标、时间对齐约定。

## 8. 汇总内存、平均时间和 P95

节点结束、run.log 完整写入后执行。该步骤只解析日志，不执行 SLAM。

~~~bash
BENCH_RUN="$(cat "$HOME/tangent_bench/current_run.txt")"

awk -F, '
FNR==NR {if(FNR>1){n++; if($2>rss)rss=$2; if($3>hwm)hwm=$3; if($4>swap)swap=$4} next}
/^VmHWM:/ {split($0,fields,/[[:space:]]+/); if(fields[2]>hwm)hwm=fields[2]}
/^VmSwap:/ {split($0,fields,/[[:space:]]+/); if(fields[2]>swap)swap=fields[2]}
END {
if(n==0){print "No memory samples collected."; exit 1}
printf "metric,value\nsampled_peak_rss_mib,%.3f\nkernel_hwm_mib,%.3f\nmax_swap_mib,%.3f\n",rss/1024,hwm/1024,swap/1024
}
' "$BENCH_RUN/memory.csv" "$BENCH_RUN/proc_final_status.txt" > "$BENCH_RUN/memory_summary.csv"

LC_ALL=C sed -r 's/\x1B\[[0-9;]*[mK]//g' "$BENCH_RUN/run.log" |
awk -F'|' '
/LIO Mapping Time/ {stage="LIO"}
/\|[[:space:]]*VIO Time[[:space:]]*\|/ {stage="VIO"}
$2 ~ /^[[:space:]]*Current Total Time[[:space:]]*$/ && stage!="" {printf "%s,%.6f\n",stage,$3*1000}
' > "$BENCH_RUN/stage_times_ms.csv"

LC_ALL=C sort -t, -k1,1 -k2,2n "$BENCH_RUN/stage_times_ms.csv" |
awk -F, '
{stage=$1; n[stage]++; sum[stage]+=$2; value[stage,n[stage]]=$2}
END {
print "stage,count,mean_ms,p95_ms"
for(stage in n){
rank=int(0.95*n[stage]); if(rank<0.95*n[stage])rank++;
printf "%s,%d,%.6f,%.6f\n",stage,n[stage],sum[stage]/n[stage],value[stage,rank]
}
}
' > "$BENCH_RUN/runtime_summary.csv"

cat "$BENCH_RUN/memory_summary.csv"
cat "$BENCH_RUN/runtime_summary.csv"
~~~

P95 使用 nearest-rank 定义。计时统计覆盖本轮所有输出正常计时记录的处理调用，不另行裁掉启动阶段。空点等提前返回的调用没有计时记录；count 是输出计时记录的调用数量，不等于原始相机消息总数。

当前直接多相机前端使用同一 VIO 表头，该解析适用于当前 frontend_mode=0。若 runtime_summary.csv 只有表头，说明没有匹配的计时输出，需要检查 run.log 中的实际日志格式或启动失败信息，不能填成零。

表格的 Peak RSS 采用 kernel_hwm_mib；与旧稿比较时采用 sampled_peak_rss_mib。MiB=KiB/1024。

kernel_hwm_mib 同时读取停止前保存的 proc_final_status.txt，避免仅依赖最后一次逐秒采样。如果节点异常退出而未保存该文件，保留已采集的日志并标记本轮异常结束，不把它当作完整成功运行。

## 9. 结果如何组织，哪些做法更有价值

每条序列先做 on/off 各一轮。正式耗时建议各做 3 轮，串行运行，可交替 on/off 减少固定顺序与温度变化影响。论文可用 3 轮的每轮峰值 RSS、每轮平均耗时及每轮 P95 的中位数，表注注明；同时保留各轮结果，不挑最好的一轮。不要把三轮内存峰值的平均数称为单次运行峰值。

主要检查同一对实验的处理帧数、轨迹时间覆盖和跟踪是否成功。明显少处理了数据的一组不能直接当作更省内存或更快。发生失败时如实记录。

如果任一版本在 1 倍速下明显积压，先对两组使用相同较低播放速率，完成数据量一致的算法开销对照；记录该速率。这类运行不用于证明原速实时性。程序缓存增长也可能来自积压，不能全归因于视觉地图。

保持两组的相机数量、分辨率、输入降频、点数预算、线程数、LiDAR 滑动地图、发布订阅者和输出设置一致。不同数据集保留各自正常配置，不把它们混成一个统一阈值。

最有价值的附图是 Street03 或 Gate03 的 RSS 随时间曲线，同时查看已有 Active Visual Points 日志。这样能观察内存增长是否与地图增长同步。Active Visual Points 不等于全部内存归属，不足以计算视觉地图的实际字节数。

如果只需要内存节省与时间代价，这套外部采样加现有计时日志已够用。若要严谨声称端到端实时性，再增加覆盖同步、IMU、LIO、VIO及必要输出的统一计时和积压观测；这是新的最小插桩工作，不应把现有阶段计时直接改名充当。

## 10. 代码与工具依据

当前读取参数：src/LIVMapper_multi_cam.cpp，visual_map_manage_en、visual_map_manage_shadow_en、visual_map_retirement_apply_en、visual_geom_filter_en 分别独立读取。
实际删除：src/vio_multi_cam.cpp::queueReferenceRetirement、queuePointRetirement、flushVisualMapRetirements。
距离依赖：src/vio_multi_cam.cpp::selectReferenceCandidates、shouldAddObservation、manageReferenceBank。
VIO 计时：src/vio_multi_cam.cpp::processMultiCameraFrame，frame_start 至 reference_end。
LIO 计时：src/LIVMapper_multi_cam.cpp::handleLIO，t0 至 t4。
PCD 累计：src/LIVMapper_multi_cam.cpp::publish_frame_world。
ROS1 私有参数：include/utils/ros1_param.h 将点号名称转换为斜杠路径；各 launch 在 laserMapping 节点内加载 YAML。
旧实验协议：此前稿件的 Memory and Accuracy Comparison，1 Hz pidstat + position ATE RMSE。

Linux /proc 文档：https://docs.kernel.org/filesystems/proc.html
pidstat 手册：https://man7.org/linux/man-pages/man1/pidstat.1.html
ROS Noetic 日志配置实现：https://github.com/ros/rosconsole/blob/noetic-devel/src/rosconsole/impl/rosconsole_log4cxx.cpp

## 11. 新增自动分项统计

`debug.memory_stats_en` 为独立开关，三个多相机主配置默认关闭。
完整分项、自动峰值/均值/最终汇总及 NUC 查看方法见
[自动分项内存统计](memory_breakdown_usage.md)。不需要为统计重新开启 usage_stats_en。
如需保持本教程原先只测外部 RSS 的时间口径，请保持 memory_stats_en=false。

