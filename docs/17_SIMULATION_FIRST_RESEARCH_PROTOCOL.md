# Machines 论文 V3：仿真为主、实机为辅的固定研究协议

日期：2026-10-20；代码基础：`huxiaoyi-ovo/Pcrnet_hexapod`（**不是** Grasp_hexapod）。

## 研究主张的边界

**论文研究对象**：在固定物理机身补偿器 π_c 和固定行进目标下，候选步态 θ 既改变扰动产生机制，也改变实际闭环补偿后的残余图像运动。我们采用仿真条件下的补偿关/开配对样本训练轻量预测器，并用其在线选择步态。

**控制目标**：真实实现的补偿器执行后，静态场景的非期望图像位移较小，同时具备相近的**实际**前进速度、足端接触有效性和通过能力。不能通过自动减速获得虚假的相机稳定性提升。

**证据安排**：开发工作 85–90% 投入 Isaac Gym 控制与定量实验；实机 10–15% 用于冻结模型后少量短序列核验。此比例只是资源策略，**不是**期刊官方标准，也不是证据可统计权重。

## 对比协议

必须保持相同机器人模型、相同虚拟相机外参、相同控制器 π_c、相同步态候选集合、相同目标速度与同类地形实例：

| ID | 控制系统 | 证实的问题 |
|---|---|---|
| F | 训练地形优选固定步态 + π_c | 在线调度是否有必要 |
| R | 基于预测原始扰动排名选择步态 + π_c | 只做减振是否足够 |
| H | 由历史残余抖动反馈选步态 + π_c | 提前预测比事后调整是否更有效 |
| C | 使用候选无关的补偿效果模型选步态 + π_c | 步态条件化补偿效果的贡献 |
| O | 预测**候选补偿后残余**并调度 + π_c | 完整方法 |

另提供无 π_c 的固定步态作为诊断参考，不充当唯一“弱基线”。

## 核心实验按信息增益组织

1. **仿真标定**：`hex_ground` 的机器人质量、动作/电机/摩擦配置，特别是执行器跟踪速率、饱和、延迟和接触随机化。真实机器人只需用少量安全动作估计合理变化范围；不需要实机采集训练大数据集。记录实际配置 SHA、随机种子与物理参数。
2. **扰动数据池**：平地、坡地、台阶/障碍条、随机起伏四类；在多目标速度、多步频/占空比/触地曲线下分别执行相同 π_c 的开/关工况。两次试验的**初始条件相同**，但后续轨迹不会逐帧完全相同。按不同 `terrain_instance_id / seed / pair_id` 区分，失败记录保留。
3. **监督目标**：高频日志记录 Isaac Gym **真实发生的**机身位姿、关节状态和接触；转成相机光学系运动，再减去运动任务引起的预期平滑运动，计算静态场景非期望图像位移 `px/frame`。该值是**几何代理指标**，不是渲染 RGB 模糊。
4. **轻量响应预测**：按候选计划步态及当前可观测状态训练补偿关/开两支模型；只按地形实例/随机种子划分数据，严禁把相邻时间窗随机分给训练和测试。比较 MAE、候选排序准确率、top-1 regret 与计算耗时。
5. **排序机制实验**：在所有合法且速度匹配的候选对上统计原始排名与补偿后排名不一致的比例、发生工况以及预测模型的收益；不挑选对自己有利的单个案例代替总体统计。
6. **完整闭环**：四类地形与相同指令、相同可用候选集，报告像面位移、实际速度、通过率、关节限制、足滑及控制器计算开销。比较 F/R/H/C/O；按**独立地形/独立 rollout** 做配对统计，而不是按 30 fps 视频帧伪重复。
7. **物理随机化**：分别变化电机位置响应延迟、关节跟踪误差、足地摩擦、接触噪声和相机外参误差。论文只展示与最终仿真配置一致的实际执行参数，不宣称做了未实现的随机化。

## 独立实机证据（冻结模型、无再训练）

用原六足+D435i+IMU，选择平地轻扰动和一类安全非平整地形，初始目标 **3 方法 × 2 场景 × 3 次 = 18 条序列**；三方法是 `F/R/O`。仅在速度、曝光、光照和角度基本可比的条件下比较。记录全部重复和失败情况；这只是初始方案，差异较小应如实报告不确定性。RGB 中需要用静态特征/AprilTag 测量非期望图像运动，不能用仅 IMU 结果替代。记录曝光与 RGB rolling-shutter 相关限制。

**不要求**完整实机多地形泛化、不要求实机为预测器训练大量样本、不要求新硬件云台，也不需要修改已有 PCR 高层 Follow/Avoid 模型。

## 尚未完成的工程衔接（必须实事求是）

V3 已给出 PCR 接口的无侵入日志 `src/pcr_sim_bridge.py`、动作映射与限制检查 `src/pcr_action_interface.py`、残余运动标签生成 `src/pcr_sim_windows.py`、组划分 `src/sim_split.py`，这些已通过**合成状态的单元测试**。

**当前仍需在用户 Isaac Gym 环境执行**：新的参数化步态足端规划 → PCR 机器人运动学 IK → 18D action → PhysX 电机力矩；真实 Roll/Pitch/Z 补偿开关及伺服响应；相机外参实际核对；完整地形批处理。当前没有任何合格的物理仿真性能结果。下面不能被称为“已集成 Isaac Gym 运行”。

### 在 PCR-Net Isaac Gym 的嵌入点（伪代码，不能直接运行）

```python
# env: actual HexGround task from PCR-Net; not a synthetic mock
# controller.make_targets MUST provide safe absolute joint references via IK
# and the real body stabilizer (on or off).
# q_target is (num_envs,18); planned_support is (num_envs,6).
from pcr_action_interface import targets_to_hexground_actions
from pcr_sim_bridge import PcrSimTap, TrialIdentity
import torch

identity = TrialIdentity(trial_id, pair_id, terrain_id, seed, controller_on, gait_id)
with PcrSimTap(log_path, identity) as tap:
    for step_id in range(n_steps):
        q_target, planned_support = controller.make_targets(env, controller_on)
        action_np = targets_to_hexground_actions(env, q_target)
        actions = torch.as_tensor(action_np, device=env.device, dtype=torch.float32)
        env.step(actions)
        tap.append(env, env_id=0, t_s=(step_id+1)*env.dt,
                   nominal_cmd_body=[vx_body,vy_body,wz_body],
                   planned_support=planned_support[0])
```

注意：环境 `HexGround.step` 使用**相对默认关节角的 18D action**，其电机模型把 position error 和 velocity error 转成力矩。不能把角度目标直接喂到 action，也不能在自己的代码里假设刚体运动瞬时完美执行。这里仅针对**独立 rollouts** 采样，若仿真自己重置环境，必须停止或剔除该段窗口。

### 从真实采集日志生成监督样本

1. 先复制 `templates/pcr_sim_calibration.UNVERIFIED.json`，填入实际的相机光学系外参、K、静态参考场景深度与曝光参数；`extrinsic_verified=true` 应在完成坐标标定后才设置。
2. 收集所有开/关仿真日志，编写 `trial_manifest.json`（见 `templates/pcr_sim_trial_manifest.example.json`），必须在执行各条候选前存储 predecision 特征。
3. `PYTHONPATH=src python scripts/build_pcr_sim_windows.py --manifest outputs/trial_manifest.json --calibration config/camera.json --output outputs/response_windows.csv`
4. `PYTHONPATH=src python scripts/evaluate_response_surrogate.py --input outputs/response_windows.csv --test-terrains <若干独立的地形ID>`

主张及新颖性没有因为合成单元测试成功而成立：只有真实动力学多地形和少量实际相机结果才支持 Methods 和 Results。
