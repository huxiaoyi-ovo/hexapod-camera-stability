# Machines 相关论文方法级调查：20 篇（2026-10-20）

**研究问题**：Compensation-Aware Gait Adaptation for Camera-Stable Hexapod Locomotion on Uneven Terrain，即给定运动进度与足端可行性，利用机身有限姿态补偿能力预测并在线选择能降低**补偿后图像空间残余扰动**的步态。

**本刊检索边界**：仅列 *Machines*（ISSN 2075-1702）的研究论文，不把 MDPI 其他刊物误纳入。按方法重合度排序，**不是**按相同关键词排序。网页可访问的部分已核对方法、控制变量及实验；部分出版商正文访问触发 429，对这些条目只记录已经检索到的具体信息，标注“有限核查”，不冒充通读。论文可能出现在多个关键类别，但仅计数一次。

**重合等级**：A=对课题构成实质先例（但不一定覆盖独特机制）；B=具体控制机制或成像评价先例；C=场景/实验参照；D=仅宽泛背景，不构成直接竞争。核查深度：M=核到正文方法与实验段落，P=核到摘要和部分正文（不能声称精读完整全文）。

## 一、最接近的核心论文（优先用于 Related Work）

### 01. Yu et al., 2025 — Collaborative Optimization Control of Gravity Center and Pose of Hexapod Robot in Complex Terrains
- DOI: https://doi.org/10.3390/machines13090871
- 研究目标：复杂地形下稳定、方向保持、稳定裕度和能耗。
- 机制：按三腿支撑/六腿支撑阶段优化机身姿态与质心；支撑多边形、IK、三角步态；在摆腿前预移质心，步后重调姿态。
- 证据：ADAMS–Simulink 和物理样机崎岖地形 25 个运动周期；机身欧拉角、轨迹、稳定裕度和能耗。
- 与本文关系：**机身姿态-步态时序协调早已成立**。未以图像运动、相机曝光和当前可补偿能力下的残余扰动为调度依据。
- 定位：**A，M**。不能写“前人未考虑步态与机身协调”。

### 02. Liu et al., 2026 — Trajectory Tracking of Underwater Hexapod Robot Based on Model Predictive Control
- DOI: https://doi.org/10.3390/machines14020171
- 目标：水下六足沿指定轨迹平顺运动。
- 机制：Hopf CPG 为下层；上层 MPC 根据动态/运动学模型在线算轨迹控制量，经实验拟合映射到 CPG 节律参数；平滑步态切换。
- 证据：水下六足圆形、直线轨迹跟踪的数值仿真；未见完整实机性能验证。
- 与本文关系：**在线上层优化 CPG 变量的分层框架不是创新**。不同点应是预测可补偿/不可补偿的相机扰动并选择步态，既不是改航向，也不是单纯换 MPC 目标权重。
- 定位：**A，M**。

### 03. 2022 — Mechanical Design and a Novel Structural Optimization Approach for Hexapod Walking Robots
- DOI: https://doi.org/10.3390/machines10060466
- 目标：在速度和运动学可达条件下，减少六足的能耗、**无用机身运动、振动**与关节力矩差异；正文明确指出相机、LiDAR、IMU 会受机身晃动影响。
- 机制：经验证的六足动力学模型 + 多目标适应度 + PSO；优化机器人几何、弹簧阻尼、行走算法参数，包含机身转动/平移振动惩罚；平地及 ±30° 坡度场景。
- 证据：模型驱动的结构参数优化与仿真评价，目标是机电结构设计和参数选型，不是接收 IMU 状态的在线自适应步态补偿控制。
- 与本文关系：**这是本刊最容易被遗漏的实质先例。** 不能声称“首次用步态优化减少传感器相关机身振动”。明确区别应是控制时变补偿限制下的残余成像扰动，而非机械静态设计时优化振动。
- 定位：**A，M**。

### 04. Gao et al., 2022 — Synergistic Motion Stability of a Scorpion-like Composite Robot
- DOI: https://doi.org/10.3390/machines10100834
- 目标：减小足端触地反弹、姿态变化和行走不稳定。
- 机制：支撑裕度决定步态，复合摆线平滑摆腿轨迹，动态修正接触力/足端轨迹以减小冲击；接触柔顺控制。
- 证据：ADAMS 动力学仿真 + 户外运动样机；报告姿态角变化及落地时接触点振荡。
- 与本文关系：**通过足端接触优化从源头抑制扰动已经存在**。研究目标不是实际图像质量，也没有将机身补偿可达集与候选步态评分耦合。
- 定位：**A，M**。

### 05. 2023 — Unknown Slope-Oriented Research on Model Predictive Control for Quadruped Robot
- DOI: https://doi.org/10.3390/machines11020133
- 目标：未知坡地稳定性，避免足滑。
- 机制：IMU + 关节估计坡度、SAMPC 依据坡度调整身体姿态和摆动腿触地点，控制摩擦锥。
- 证据：不同坡度仿真（包括 42.4° 爬坡案例）。
- 与本文关系：**地形中机身姿态与足端接触协同是现有技术**，但非视频稳定任务，也没有补偿后图像残差预测。
- 定位：**B，M**。

### 06. You et al., 2022 — A New Trajectory Tracking Control Method for Fully Electrically Driven Quadruped Robot
- DOI: https://doi.org/10.3390/machines10050292
- 目标：质心轨迹和摆动腿轨迹准确跟踪、稳定行走。
- 机制：机身 MPC 求支撑足接触力，Bezier 摆腿轨迹，虚拟弹簧阻尼与 RBFNN 监督跟踪；IMU+运动学状态估计。
- 证据：直线/圆形运动轨迹跟踪与实验。
- 与本文关系：支撑足主动姿态补偿、摆腿轨迹柔顺已存在；与相机补偿能力无关。
- 定位：**B，M**。

### 07. Yang et al., 2026 — Design of Multi-Legged Locomotion Control System for Reconfigurable Robots Integrating Decoupled Virtual Model Control with BP Neural Network
- DOI: https://doi.org/10.3390/machines14020184
- 目标：可重构二/四/六足模块下统一实现步态稳定。
- 机制：BP 网络自适应调整 VMC 参数，分离姿态/轨迹控制；多腿组合步态。
- 证据：Simscape Multibody + 物理样机单腿摆动及 Roll/Pitch 相关测试。
- 与本文关系：运动状态到机身稳定参数自适应已有，非相机成像目标、非预测机身补偿后残余扰动。
- 定位：**B，M**。

### 08. Liang et al., 2024 — Control of Parallel Quadruped Robots Based on Adaptive Dynamic Programming Control
- DOI: https://doi.org/10.3390/machines12120875
- 目标：并联四足未知场景下的足端位移跟踪。
- 机制：运动学和逆运动学 + Generalized Policy Iteration (GPI) 在线自适应跟踪；AMESim/Simulink 联仿。
- 证据：仿真和样机速度/位移跟踪，深度相机/激光雷达用于测量。
- 与本文关系：自适应腿部控制，而相机在此是测量器，不是需稳定的受控对象。
- 定位：**B，M**。

### 09. Van Hauwermeiren et al., 2025 — Tactile Force Sensing for Admittance Control on a Quadruped Robot
- DOI: https://doi.org/10.3390/machines13050426
- 目标：触地瞬间和外界冲击后尽快重新平衡。
- 机制：足端球形触觉传感器三维重建力，导纳控制修正腿部运动以重新分配触地反力。
- 证据：横梁扰动、落地冲击实机；报告峰值 GRF 降低 55%，五次实验重复。
- 与本文关系：**抑制触地冲击的强方法参照**，需要在 Discussion 中承认高频冲击可以采用力反馈柔顺控制，但本研究主要选择低频/步态相关且机身未补偿的剩余像素运动。
- 定位：**B，M**。

## 二、控制方法和实验标准参照

### 10. Yu et al., 2026 — Quadruped Robot Motion Control Based on an Improved PPO Algorithm
- DOI: https://doi.org/10.3390/machines14060621
- 机制：LSTM + 多头注意力增强 PPO，输入机身/关节/足接触/地形历史；奖励包括速度跟踪和稳定性。
- 实验：Isaac Gym 多混合地形、多种子、Reward 消融与少量真实地面行走，强调 sim-to-real 初步能力。
- 与本文关系：**多地形＋控制消融＋少量实机**的刊物写作先例，非视频成像稳定控制。
- 定位：**C，M**。

### 11. Dai et al., 2026 — Robust Locomotion Control of Quadrupedal Wheel-Legged Robots via Contrastive History-Aware Reinforcement Learning in Complex Environments
- DOI: https://doi.org/10.3390/machines14050568
- 机制：利用本体感觉历史的对比学习隐式估计地形与外力，基于 RL 自适应轮腿运动。
- 实验：Isaac Gym 多地形训练、跨仿真器和实机部署，重点鲁棒速度跟踪。
- 与本文关系：实验规模和统计证据标准参照；不是视觉稳定。
- 定位：**C，M**。

### 12. Li et al., 2026 — Reinforcement Learning-Based Landing Impact Mitigation and Stabilization Control for Lunar Quadruped Robots Under Complex Operating Conditions
- DOI: https://doi.org/10.3390/machines14040417
- 机制：MuJoCo 50Hz 策略下对着陆冲击/稳定恢复进行 RL，考虑复杂坡地、重力及载荷变化。
- 实验：物理仿真训练和定量着陆/姿态指标；其“接触冲击抑制”目标接近高频干扰来源，但不是六足持续行走视觉稳定。
- 与本文关系：触地冲击下可控范围与评估方法先例，不构成残余图像步态选择直接竞争。
- 定位：**C，P**。

### 13. Zhu & Rosendo, 2022 — PSTO: Learning Energy-Efficient Locomotion for Quadruped Robots
- DOI: https://doi.org/10.3390/machines10030185
- 机制：TD3 仿真学步态，转移到物理机再优化开环轨迹参数，提高每距离能效。
- 实验：四足 Ant 的实际不同步速和腿长实验；期刊公开同行评议里有审稿人质疑“RL + 轨迹优化只是简单拼装”的创新性。
- 与本文关系：直接警示“先做优化再控制”的组合若无新机制会被批评；不是图像防抖。
- 定位：**C，M**。

### 14. Saputra et al., 2023 — Neuro-Cognitive Locomotion with Dynamic Attention on Topological Structure
- DOI: https://doi.org/10.3390/machines11060619
- 机制：拓扑注意力 / 可供性认知与四足 CPG 运动生成结合；外感信息指导运动行为。
- 实验：点云注意力、CPG 等模块对照，ODE + 物理四足。
- 与本文关系：视觉/外感调节 CPG 不是新框架，但目标是导航感知/动作生成，不是对相机本身进行防抖。
- 定位：**B，M**。

### 15. Khan et al., 2024 — Multi-Agent Reinforcement Learning Tracking Control of a Bionic Wheel-Legged Quadruped
- DOI: https://doi.org/10.3390/machines12120902
- 机制：多智能体策略分别控制各腿，运动专家指导轨迹/步态奖励，四种运动模式。
- 证据：速度、转向和外扰恢复的实机对照。
- 与本文关系：机身/腿耦合运动策略，核心差异是训练范式与任务目标；不涉及成像稳定。
- 定位：**C，M**。

## 三、传感器、机电与边界文献

### 16. Peng et al., 2022 — Constrained Image-Based Visual Servoing of Robot Manipulator with Third-Order Sliding-Mode Observer
- DOI: https://doi.org/10.3390/machines10060465
- 机制：基于视觉特征误差的 MPC + 三阶滑模观测器，约束目标可见性和执行器，求机械臂控制转矩。
- 证据：2/6-DOF 机械臂仿真和理论稳定性分析。
- 与本文关系：**图像代价+机器人控制约束**在本刊已存在；它处理目标视觉伺服，不是持续移动中的步态周期扰动/机身可补偿能力。
- 定位：**B，M**。

### 17. 2023 — Advancing Simultaneous Localization and Mapping with Multi-Sensor Fusion and Point Cloud De-Distortion
- DOI: https://doi.org/10.3390/machines11060588
- 机制：四足上的 D435 + 激光 + 里程计，通过点云/栅格融合与去畸变改善 SLAM。
- 实验：四足实机在室内障碍场景下比较去畸变前后的地图。
- 与本文关系：视觉/点云运动失真是真实工程问题；这里是感知后处理而非主动改变足端步态，**只能作为动机，不是核心竞争**。
- 定位：**C，M**。

### 18. Liu et al., 2025 — Ground Type Classification for Hexapod Robots Using Foot-Mounted Force Sensors
- DOI: https://doi.org/10.3390/machines13100900
- 机制：单足三轴力时间序列→1D CNN/LSTM/CBAM；自动超参调节用于地形分类。
- 实验：单腿试验台、五类地面、四种坡度、三档力载荷；96.94% 分类准确率。
- 与本文关系：可选的地形上下文输入参考，但不是步态控制，更不评价视频抖动。
- 定位：**D，M**。

### 19. Liu et al., 2022 — Design and Control of a Hydraulic Hexapod Robot with a Two-Stage Supply Pressure Hydraulic System
- DOI: https://doi.org/10.3390/machines10050305
- 机制：浙大 ZJUHEX01 液压六足，两级供压系统 + 滑模重复控制 + 关节速度/加速度估计。
- 实验：ADAMS/Simulink 联仿，重点关节跟踪与液压节能。
- 与本文关系：机器人执行器与运动控制工程，非步态成像稳定；对我们的无额外液压硬件平台构成极弱关联。
- 定位：**D，P**。

### 20. Hu et al., 2025 — Design and Dynamic Performance Evaluation of a Novel 6W4L Wheel-Legged Robot
- DOI: https://doi.org/10.3390/machines13080662
- 机制：6 轮 4 腿并联结构设计、工作空间 / 速度灵巧度 / 承载能力的几何尺寸优化。
- 证据：机构分析和数值动态性能评价。
- 与本文关系：机器人补偿可行域可能借鉴运动学思路，**不是**步态-图像质量的竞争论文。
- 定位：**D，P**。

## 四、从这 20 篇可以得出的确切结论

1. **不能声称首先想到“传感器怕机身晃动，所以优化六足步态”**。Machines 2022 的结构优化论文已经明确写了机身不必要的平移/旋转/振动对相机、LiDAR、IMU 的影响，并在目标函数里减少这些量。
2. **不能声称首个“上层 MPC 实时调节 CPG”**。Machines 2026 的水下六足已明确使用上层 MPC 与下层 Hopf CPG 的参数映射。
3. **不能声称首先做“复杂地形机身姿态与支撑步态协调”**。Machines 2025 的六足论文有三腿/六腿阶段质心与机身补偿及实机。
4. **不能声称“通过平滑摆腿和触地减少冲击”是新机制**。Machines 2022 的蝎式复合机器人已有复合摆线、接触柔顺/反力补偿。
5. **在这些 Machines 文章的已检索方法部分，没有发现“以**当前机身补偿能力**为条件，预测每套步态经机身控制之后的图像空间残余运动，再选步态”的完全等价控制器**。这是有限检索结论，而非全球首创证明。
6. 需保留最强的替代方案：
    - M1：固定步态 + IMU 机身补偿；
    - M2：离线调优的单套固定步态 + 相同的机身补偿；
    - M3：只最小化**补偿前**机身振动的在线步态选择 + 相同的机身补偿；
    - M4：最小化**补偿后残余图像运动**的在线步态选择 + 相同补偿器。
    M3 对应最容易被认为“2022 结构振动优化 + 在线实施”的方法，**它是我们的关键基线**。
7. **论文贡献应写在可实现的候选预测机制**：
   - 给定步态候选 θ、运动速度 v 和接触/地形状态 s，预测相机原始周期扰动 d(θ,s)。
   - 在相同执行器与足端约束下，求机身补偿器能实现的校正 c*(θ,s)。
   - 使用深度与相机几何把残余相机 twist d-c* 变换到图像运动，计算 J_residual(θ,s)，在运动进度不掉线的条件下选 θ。
   - 不允许模型直接使用未观测的未来地形真值作为在线输入。
8. **实验要用真实成像证明图像质量差异**：不要把 IMU roll/pitch 下降直接解释为图像稳定效果。至少展示实际 RGB 视频特征抖动 RMS 和目标 AprilTag 像素偏差，并对比补偿前后残余扰动的预测误差。

## 五、Machines 审稿与论文工作量提示

- 2026 年机器狗 LA-PPO：平/坡/随机地形 + 多种子 + 消融 + 初步实机。该类型论文报告较完整的统计，不宜只放几条姿态曲线。
- 2024 年并联四足 GPI：联仿 + 真实速度/位移跟踪；适合展示数值控制也能发表，但本文必须测量真实图像抖动而非仅轨迹误差。
- 2025 年触觉导纳：真实接触扰动实验显示 55% 峰值足力降低；足端物理扰动并非只有步态周期，论文应明确无法控制的高频冲击。
- 2022 PSTO 的公开审稿意见对“RL + 优化简单组合”提出直接质疑：我们应优先展示**残余预测 / 补偿可行域**的独立机制和必要性消融，避免仅增加一项成本函数。

**当前建议**：固定 Machines 为目标刊，收敛成一项核心方法贡献“补偿可行域下的步态条件化相机残余扰动预测与选择”，把“视觉伺服、目标居中、复杂地形姿态调平”写成现有技术及评测功能，不当作三项独立创新。
