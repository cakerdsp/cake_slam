# 自适应切平面采样：文献与论文论证记录

记录日期：2026-10-17。用途：按用户要求，保留本对话已核查的近邻工作、引用用途及贡献边界，供后续 cake_slam 论文写作调用。本次为调研记录归档；不修改论文、Overleaf 或算法实现。

## 研究问题与当前判断

- 用户指出的现象：前后运动时，参考与当前观测的视线方向可能几乎不变，但同一物体在图像中的尺度显著变化；几何 warp 后，两块 patch 仍可能具有不同的有效细节尺度。
- 当前方案：把参考和当前原始像素的局部采样足迹映射到共同切平面，联合选择采样间隔与预滤波尺度，再构造光度残差。保留人工基础虚拟焦距、固定 patch 样本数和现有迭代框架。
- 判断：这是针对明确现象、有数学依据的解决方案，值得作为切平面视觉前端中的小贡献继续验证；已有相似工作不等于该方向没有价值。
- 尚未证实：相对合理的几何选层与预滤波基线，是否有独立且稳定的精度、参数敏感性或效率收益。不要把“有贡献空间”写成“创新性和有效性已经完成验证”。

## 文献清单

### 1. SVO: Fast Semi-Direct Monocular Visual Odometry

- ICRA 2014。
- 论文：https://rpg.ifi.uzh.ch/docs/ICRA14_Forster.pdf
- 官方代码：https://github.com/uzh-rpg/rpg_svo/blob/master/svo/src/matcher.cpp
- 已核查位置：Sec. IV-A/B、VI；`getBestSearchLevel`、reference/current pyramid 取样路径。
- 可引用内容：局部 photometric patch、affine warp、图像金字塔及逐 patch 的离散尺度选择。官方代码按 `det(A)>3` 等条件选层。
- 与本方案的区别：当前采样还使用两侧原图到共同切平面的局部投影密度与主轴尺度。
- 避免误写：SVO 并非只有全局统一的粗到细，也不是仅在原图上扩大步长；其相关路径实际读取图像金字塔。

### 2. SVO: Semidirect Visual Odometry for Monocular and Multicamera Systems

- IEEE Transactions on Robotics, 2017。
- 论文：https://rpg.ifi.uzh.ch/docs/TRO17_Forster-SVO.pdf
- 后续公开 SVO Pro 实现：https://github.com/uzh-rpg/rpg_svo_pro_open/blob/ca371f304637e7fb355cf4624d0a02da4e3da220/svo_direct/src/patch_warp.cpp
- 已核查位置：Sec. VII、X-B；公开实现的 affine / pixelwise warp 路径。
- 可引用内容：鱼眼与大视场投影、相应 Jacobian，以及多相机直接法的已有基础。
- 边界：当前公开实现中的每个分支不能直接追溯为 2017 年论文已具备的功能。原生鱼眼处理本身不是我们的新增贡献。

### 3. FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry

- 论文版本：https://arxiv.org/html/2408.14035v2
- 官方代码：https://github.com/hku-mars/FAST-LIVO2/blob/main/src/vio.cpp
- 已核查位置：Sec. V-C/D/E，特别是 Eq. 12–13；`getBestSearchLevel`、`warpAffine`、`getImagePatch`。
- 可引用内容：多层 photometric patch、几何引导 warp、参考选择及 LIVO 中的光度更新。
- 与本方案的区别：所查公开采样路径采用离散选层和原图步长采样，未见当前这种两侧共同域采样足迹与预滤波设计。
- 边界：论文明确描述通过相机投影/反投影支持鱼眼；论文声明与公开代码各路径的实际模型支持应分别核查。不能把原始鱼眼输入本身当作区别。

### 4. Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR Odometry via Photometric Migration and ESIKF Fusion

- 所查论文版本：https://arxiv.org/html/2509.15673v5
- 固定版本官方代码：https://github.com/elon876/Omni-LIVO/blob/842311d6929b65caea6ab059d3af6a1894e2b5dd/Omni-LIVO/src/vio.cpp
- 已核查位置：Sec. III-C/E/F，Eq. 9、11、20；几何 warp、选层与 patch 读取路径。
- 可引用内容：多相机 LIVO 中跨时间、跨相机的光度约束与参考处理。
- 与本方案的区别：所查路径未发现当前 `D_ref * inverse(A)` 与 `D_cur` 共同决定连续采样间隔和双侧预滤波的机制。
- 边界：adaptive covariance 是观测权重处理，不能等同于 adaptive sampling；不能凭名称或示意图判断实现。

### 5. Frequency-Aware Self-Supervised Monocular Depth Estimation

- WACV 2023；其中的 Auto-Blur 为重点近邻。
- 论文：https://arxiv.org/pdf/2210.05479
- 官方 Auto-Blur：https://github.com/xingyuuchen/freq-aware-depth/blob/master/networks/auto_blur.py
- 两侧调用：https://github.com/xingyuuchen/freq-aware-depth/blob/master/trainer.py
- 已核查位置：Sec. 4.3–4.4，Eq. 10–21；官方 `AutoBlur` 与各 frame 的调用。
- 可引用内容：重采样和高频纹理可能使光度误差惩罚正确几何；在比较之前进行适当滤波具有先例。
- 方法细节：目标图和源图各自处理，局部高梯度密度控制原图与模糊图的混合权重。所查代码高斯核为 11×11、sigma=5.0，而不是根据几何自适应 sigma。
- 与本方案的区别：我们的采样间隔和滤波参数受成对投影与几何约束；Auto-Blur 未联合选择两侧共同几何网格。
- 避免误写：它不是仅对一侧模糊，也不能说它从未关注重采样造成的伪光度误差。

### 6. EWA Splatting 与 EWA 重采样技术

- EWA Splatting，2002；作者托管论文：https://cgl.ethz.ch/Downloads/Publications/Papers/2002/p_Zwi02b.pdf
- PBRT 技术章节：https://www.pbr-book.org/3ed-2018/Texture/Image_Texture#EllipticallyWeightedAverage
- 已核查位置：EWA Splatting Sec. 6.1/6.3，Eq. 21–22、35–36；PBRT Sec. 10.4.4–10.4.5。
- 可引用内容：投影 Jacobian 传递采样足迹；几何驱动的各向异性预滤波、连续 LOD 与抗混叠基础。
- 与本方案的区别：标准 EWA 对既定输出网格计算输入滤波足迹；当前方案还根据参考和当前两份观测决定比较网格的间隔。
- 边界：上述“成对选择网格”仍需结合下一条双侧选层先例讨论，不能仅凭与单输入 EWA 的区别证明新颖性。
- 避免误写：几何驱动滤波、连续层级、协方差经 Jacobian 变换均不是我们首次提出；当前实现也不是完整各向异性 EWA。

### 7. A Visual Odometry Framework Robust to Motion Blur

- Pretto 等，ICRA 2009。
- 作者论文：https://albertopretto.altervista.org/papers/prettoICRA2009.pdf
- 已核查位置：Sec. III，Eq. 3–8。
- 可引用内容：估计局部运动模糊，再求额外各向异性滤波，使结果接近目标等方高斯，随后进行尺度空间匹配。
- 与本方案的区别：该工作建模运动 PSF；当前重点是两侧投影与跨帧尺度变化造成的采样不一致。
- 避免误写：“补偿到可比较的模糊后再匹配”已有先例。

### 8. Direct scale level selection for multilevel feature tracking under motion blur

- 技术先例：专利申请公开 US20220377238A1，2022；不是同行评议论文。
- 全文：https://patents.google.com/patent/US20220377238A1/en
- 已核查位置：Fig. 10–11、Claims 4–6。
- 关键重合：明确包含无运动模糊、仅靠近或远离导致尺度变化的情况；把两端尺度映射后联合决定匹配分辨率。
- 可引用用途：双侧几何尺度选择的直接先例，以及界定我们新增处理的范围。
- 与本方案的区别：所读方法未给出当前两侧原始像素足迹进入共同切平面的 `D_ref * inverse(A)`、`D_cur` 设计；该具体差异的价值必须用对照实验验证。
- 避免误写：不能声称它只处理运动模糊，不能把“前后运动时联合选择两侧分辨率”单独包装成首次提出。

## 后续写作如何使用

- 问题动机：优先使用 Frequency-Aware / Auto-Blur，说明几何对齐并不保证重采样后的光度比较可靠。
- 直接法背景与近邻：SVO、FAST-LIVO2、Omni-LIVO，区分已有 warp、选层和本方案新增的两侧投影密度约束。
- 方法依据：EWA 及采样理论，用于解释预滤波与像素足迹传播；这些基础应明确承认。
- 更接近的边界：Pretto 与双侧选层专利，避免把共同模糊或双侧分辨率选择本身写成新概念。
- 不要求把所有记录都塞进正文；按具体论点选择引用。正式生成 BibTeX 时核对作者、出版版本、卷期、页码与 DOI，优先正式出版记录，并保留已有 manuscript citation keys。

## 可发展的贡献表述与待验证项

候选论证：针对大视场图像中跨帧尺度变化与非均匀投影共同造成的 patch 采样不一致，将两侧原始像素足迹映射到共同切平面，联合决定采样间隔与预滤波尺度，在固定 patch 样本数下构造受两侧实际采样能力约束的光度观测。

这里的新增重点是更完整的成对采样依据；连续调节只是实现特点。当前为标量滤波与金字塔层间混合近似，不宣称严格的双侧 PSF 匹配、参数无关或新尺度空间理论。

最关键的验证：

1. 与包含适当预滤波的几何选层基线比较，排除“收益仅来自补上抗混叠”。
2. 分开检验两侧投影密度与连续步长的贡献，并固定参考维护策略；当前总开关可能同时影响距离参考维护。
3. 测试前后运动、鱼眼不同位置与基础虚拟焦距敏感性，同时报告光度一致性、轨迹误差和耗时。

代码对应：`include/virtual_sampling.h` 的 `makePlan`；实现近似与衔接边界见 `docs/adaptive_virtual_sampling.md`。
