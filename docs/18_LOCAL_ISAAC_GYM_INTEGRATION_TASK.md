# 给本地 Codex / 开发 Agent 的精确开发任务（PCR-Net，V3）

仓库：`https://github.com/huxiaoyi-ovo/Pcrnet_hexapod`，项目本地 `RL_hexapod_gym`。不要修改已有 PCR-Net 论文评测、`src_real/interface/scripts/physical_robot/run_agent2.py` 和现有网络 checkpoint。新增独立任务，不覆盖原行为。

## 目标：真正跑通首条仿真开/关补偿配对 rollout

创建独立实验脚本 `legged_gym/scripts/collect_camera_stable_gaits.py` 和必要的独立 gait/IK/stabilizer 模块。尽量继承现有 `HexGround`，仅在新任务配置覆盖必要参数。先固定机器人数量为 1，后续支持并行加速。使用 PCR 坐标：`+x` 右、`+y` 前、`+z` 上；leg order `LB,LF,LM,RB,RF,RM`。检查 joint direction mapping，禁止照搬 Grasp 控制器。

已核对的实际 PCR API（GitHub `main`, 2026-10-20）：
- `legged_gym/envs/hex_v4/hex_ground.py::step(actions)` 接收 `(N,18)` Torch actions。
- `hex_ground.py::_compute_torques`: `action_scaled = actions * cfg.control.action_scale; pos_err = (action_scaled+default_dof_pos)-dof_pos`；实际 torques 经 actuator model 和 torque_limits 限幅。
- `env.root_states[:,0:13]` = `[p_world(3),quat_xyzw(4),v_world(3),omega_world(3)]`。
- `env.dof_pos / dof_vel` = `(N,18)`；`env.contact_forces` = `(N,num_bodies,3)`；`env.feet_indices` = `(6,)`。
- `env.dt` = `cfg.control.decimation * sim_params.dt`（用于高层单步时间，实际以运行时为准）。

已提供工具位于这份 V3 包的 `src/`：
`tripod_gait.py`, `pcr_action_interface.py`, `pcr_sim_bridge.py`, `pcr_sim_windows.py`, `closed_loop_surrogate.py`, `sim_split.py`。把它们作为独立 `paper_camera_stability` 包复制到新实验目录，明确 Python 导入路径。

## 必须补做的执行器/控制代码

1. `tripod_gait.foot_targets(...)` 只返回 `(6,3)` 足端目标和 `(6,)` 支撑计划；用 PCR 现有运动学实现，按其各腿坐标变换、关节顺序和限位求实际 `(1,18)` `q_target`。如果原 `CurvAdapt_multi` 绑定了吸附/攀爬状态机，只抽取运动学，不直接照搬吸附动作。
2. 固定机身补偿器 `pi_c`: 输入实时 IMU/机身姿态、脚接触估计和轨迹参考，输出**实际可执行**的 body roll/pitch/height 修正；使用 IK 分配到支撑脚，限制位置、关节速度、力矩饱和（物理上由 actuator model 处理）、最大补偿速率。用 `controller_on=0/1` 控制是否加入该修正，其余各环节一致。
3. 用 `targets_to_hexground_actions(env, q_target)` 转 `(N,18)` action，不能直接把绝对角度送进 `env.step()`；若物理候选不合规则拒绝而不是静默裁剪。参数化步频与步幅保持速度约束，切换时相位连续。
4. 相机几何：真实固定的 RGB 光学系外参必须从模型/标定求得，独立测试轴向；`pcr_sim_calibration.UNVERIFIED.json` 中标注的初始零位是占位符，不能直接用于实际论文数据。若没有相机渲染，需要在明确固定静态特征深度假设下计算几何图像运动代理。
5. 同种地形/目标速度/候选步态/初始种子下 run 2 次 `controller_on=0/1`，真实读取每个 `env.step` 结束后的状态，调用 `PcrSimTap.append`；每次输出独立日志，不能跨 episode 自动重置并混合连续窗口。
6. 显式记录 `predecision`：候选 gait、相位、当前 roll_rate、pitch_rate、局部坡度/粗糙度的**在线可估计量**。不得用模拟器未来接触/奖励/随机种子等预测目标泄露未来信息。

## 本地验收条件

- 对固定平地 +Y 正向速度执行有效连续运动，真实前进位移/速度可记录，关节、接触日志维度正确；若机器人摔倒，保留失败标签。
- 对同一场景与步态完成补偿关/开两条独立配对日志，并通过 `build_pcr_sim_windows.py` 生成 CSV 响应标签；不能使用人工合成轨迹冒充 Isaac 数据。
- 能读取多地形且可用 seed 精确复现相同初始布局；所有 trials 保留完整元数据与配置 SHA。
- 单元测试通过，并提供真实 Isaac Gym 运行命令与输出路径；统计指标保持未定直到实验完成。

## 后续最先安排的实验

完成 3×3×2 小型候选步态集合覆盖，平地/斜坡先出原始 versus 补偿后残余的候选排序矩阵，再扩展四地形与 held-out seeds。这里是正常研发步骤，**不是重新审议立项或前置否决实验**。
