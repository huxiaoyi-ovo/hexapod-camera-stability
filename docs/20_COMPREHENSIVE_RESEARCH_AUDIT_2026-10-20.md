# Machines 稿件与代码审计：2026-10-20

> **Historical audit snapshot (2026-10-20).** This audit predates the standalone simulator architecture. Preserve its review of prototype limitations and scientific risks; PCR/HexGround runtime recommendations and status statements are historical, not current instructions. See [the active architecture and experiment contract](25_ARCHITECTURE_AND_EXPERIMENT_CONTRACT.md) and [verification record](27_REFACTOR_VERIFICATION.md).

> 审计范围：本仓库源码、33 项离线测试、Methods/Experiments/Results、PCR-Net 原仓库公开接口。未运行 Isaac Gym，不作任何动力学性能结论。此文件用于团队内部执行，不是投稿正文。

## 结论

方向成立，但本仓库处于 **offline-prototype / manuscript-draft**，距离“可投稿的机器人控制论文”仍缺**真实动力学闭环、模型证据、对照实验、实机短序列**。现在不应该继续大规模扩写 Introduction/Related Work；研发优先级应以真实 Results 为中心。

## P0：影响论文成立与投稿的关键问题

1. **真实低层闭环尚未实现。** `tripod_gait.foot_targets` 只是机体系下的足端轨迹，尚未完成 PCR-Net 足端 IK/FK、支撑足世界锚定、连续相位切换、接触滑移/力矩饱和处理。`pcr_action_interface` 只是 18D action 转换与静态约束检查，不能视为已实现行走控制。
2. **补偿器没有真实响应模型。** `controller.py::select_gait` 属于历史理想化逐轴速度限幅求解，不应作为 V3 的闭环控制器；主方法实际依赖 `closed_loop_surrogate.py` 和未来补偿开/关动力学轨迹。此历史实现已显式标记为 LEGACY。
3. **可行域未落实到实际接触/驱动。** 目前 `pre_screened_feasible` 是外部布尔输入，`planned_stance_count`、`predicted_joint_headroom` 依赖尚不存在的在线计算器。没有可靠的支撑可行性，就没有严谨的候选筛选。
4. **关键预测器尚未训练。** Ridge 模型只有合成接口测试。其预决策特征不能包含仿真未来接触或真实地形真值；训练/验证/测试必须按 terrain_instance + seed 分组，不能按相邻帧随机划分。
5. **图像运动模型存在可辨识性风险。** `nominal camera twist` 当前由命令速度推导；机器人真实速度偏差会被并入残余图像运动。必须在“给定相同实际速度”的实验设计下分离未完成速度跟踪与非期望抖动；几何代理必须使用校验过的光学系外参和明确场景深度，RGB 提供独立证据。
6. **核心机制没有实测。** 不知道补偿能力是否随步态显著变化，也不知道原始与补偿后抖动排序是否反转。不能以测试中的人为构造反转代替论文机制证据。

## P1：已定位并采取的修正

- **Grasp 平台残留：** 旧 `integration/INTEGRATION_CONTRACT.md` 错指 Grasp_hexapod，已改为 PCR-Net HexGround API 和待开发清单。
- **测试依赖环境变量：** 裸 `pytest` 无法导入 `src/`。新增 `pytest.ini`，现在裸 `pytest -q` 与 CI 一致。
- **配对样本混淆：** 原 `FixedCompensatorResponseModel.fit` 只按 terrain/pair_id 判断 on/off 覆盖，不能排除 seed、gait 或预决策条件不一致。现加入 seed/gait 键、同一条件重复拒绝和特征相同检查，两项新单元测试。
- **Result 基线编号：** 文中应为 B0–B6 加 Ours 共 8 法，已修订 Results 章节的错误范围。
- **把示意测试当实验：** `test_mechanism.py` 的排序反转仅是合成数值例子，不构成现场效应存在的证据。

## P1：可发表性实验（按优先级）

**E0/P0：第一批真实动力学日志。** 同一仿真场景、相同种子、相同步态，补偿器 off/on 两次独立 rollout；保存 root pose、q/dq/torque、contact、计划支撑、成像外参、命令和真实速度。所有失败轨迹保留。输出校验后 CSV，跑 `scripts/build_pcr_sim_windows.py`。

**E1/P0：补偿器机理。** 至少 3 个步态 × 2 地形 × 多种子记录 raw/residual jitter、支持接触和执行器响应。图示回归与排序反转出现频率（可以为零），而不是仅选一个有利例子。

**E2/P0：主效应。** 先做四法：B1 固定步态+相同补偿器、B2 训练地形最优固定步态+补偿器、B3 最小原始扰动调度+补偿器、Ours 预测残余调度+补偿器。必要时再纳入 B5 gait-independent response 与 B6 history-only feedback，二者对创新性非常关键。相同候选集合、信息、补偿器、计算预算与速度门槛。

**E3/P1：模型真实性。** 在严格未见过的地形实例/种子上报告：预测误差、候选 top-1 排序正确率、regret、决策时延。对比 candidate-independent、history-only 简化方法。

**E4/P1：多地形与鲁棒性。** 平地、坡地、台阶/障碍条、随机粗糙，随机接触摩擦与执行器迟滞；对照达到速度、通过率和图像扰动。

**E5/P1：轻量实机。** 固定步态、普通减振调度和 Ours；两类安全工况、短 RGB/IMU 轨迹；不用于训练。标注帧率、曝光、同步和实际速度。

## 审稿风险

- **Novelty:** Shaw 2019 已有六足 CPG + SE(3) 姿态 + 视觉注视；Machines 2025 已有质心/姿态协同；因此不能声称这些基础组合新。须由“候选步态—实际补偿后残余扰动”预测与公平对照证明区别。
- **Mechanism:** 若最优 raw gait 和最优 residual gait 几乎总相同，论文的新方法带来的增益会小。此时应改成“对真实可执行补偿响应的统一估计与评估”而非夸大排序反转。
- **Evidence:** 只有 IMU 曲线或理想相机模型不足以证明图像稳定；至少需要真实/渲染 RGB 的静态特征运动检查。
- **Time:** 目标为尽早形成合格稿件并投稿；编辑、审稿和最终接收日期不可保证。

## 冻结约束

1. **只使用 Pcrnet_hexapod** 的 URDF、Isaac Gym 和实机通信；不混入 Grasp。
2. 仿真负责训练和绝大多数实验；实机只做小规模冻结模型验证。
3. 预测器输入仅限决策时可得量；future contacts、ground-truth terrain seeds 只能用于离线诊断。
4. 扰动指标不把有意平移导致的正常光流全部算作抖动。
5. 论证须控制实际速度，不能通过减速“赢”得相机稳定。

## 远端工程卫生

仓库初始化时的空目录占位 README（`src/README.md` 等）和原 `docs/STATUS.md` 包含“尚未同步论文/算法”的旧状态，完整工程推送后必须清理，避免与本审计及根目录 README 矛盾。
