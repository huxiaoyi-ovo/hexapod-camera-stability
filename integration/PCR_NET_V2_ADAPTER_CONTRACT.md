# PCR-Net 六足实验适配接口（历史草案；当前不活动）

> **历史接口方案，当前不活动。** 以下内容记录既往 PCR/HexGround 适配审查，保留其研究与公平对照注意事项，不再是实现指令。本项目运行时仅使用本地 robot asset，不依赖 PCR-Net、`HexGround` 或 `legged_gym`。当前边界见[架构与实验契约](../docs/25_ARCHITECTURE_AND_EXPERIMENT_CONTRACT.md)。

仓库：<https://github.com/huxiaoyi-ovo/Pcrnet_hexapod>，`main`。已核查文件：
`legged_gym/envs/hex_v4/hex_ground.py`, `hex_ground_config.py`, `src_real/interface/scripts/physical_robot/run_agent2.py`, `src_real/locomotion/curv_adapt/scripts/curv_adapt_multi.py`。

## 实验代码应落在哪里

- 新增独立任务 `legged_gym/envs/hex_v4/hex_camera_stability.py`，建议继承 `HexGround`，将训练原有 PCR-Net 任务保持不变。由 host code 执行 `reset() / step()`，传入不同候选步态生成的 18 关节目标；不更改 Follow/Avoid 网络权重。
- 新建 `legged_gym/scripts/collect_gait_response.py`，只采**原始动力学状态、真实命令、接触信息、相机外参**。离线用 `src/trace_metrics.py` 算相对 nominal trajectory 的 px/frame；一个短窗只能输出一条监督样本，不能把相关 50-Hz 帧当独立重复。
- 开闭环配对：以 `terrain_instance_id`、`pair_id`、`controller_on=0/1`、`start_seed`、候选 gait θ、导航速度标识同一条件；重置/采集原始动作不保证后续轨迹相同。
- 两个候选在同一段 current state 下的*预测比较*，必须由当前状态和**候选计划足端轨迹**推断 support/headroom；不能在做决定时调用另一条真实 roll-out 的未来受力和 camera motion。
- 本研究中外环更新频率应根据实际实验设置；脚下状态切换只允许相位安全更新。`50 Hz` 是 PCR 现有低层关节推理/控制路径的目标频率，并不证明机身物理补偿也有 50-Hz 带宽。

## 初始字段对应

| 逻辑量 | PCR 仿真来源/推导 | 关键注意 |
|---|---|---|
| q, dq, torque | HexGround / LeggedRobot 的 DOF 状态 | 六腿 × 三自由度顺序必须明确 |
| body pose, angular velocity | root state/IMU simulation | 必须转为相机光学系；不要直接把 world twist 代入 L |
| contact/support | PhysX feet contact forces + gait planner phases | 接触力是观测标签；未来真实接触状态不允许作为在线 oracle |
| T_BC, K | `hex_ground_config.py` camera config + 标定 | depth/RGB 相机配置不同；现有欧拉角轴向不可直接假定就是 camera optical |
| commanded motion | 统一指定 vx, vy, wz | 相机 nominal flow 必须由任务运动定义，不能使用低通后的实际晃动代替 |
| terrain roughness/slope | 与 D435i/状态估计能力对应的在线估计 | ground truth 地形高度图只能训练分析，不可直接放实机部署特征 |
| future support count / joint headroom | 候选步态 generator + 当前 FK/IK 推算 | 候选预估，不是未来真实接触反馈 |
| speed outcome | root velocities 计算实际行进速度 | 这是离线监督目标，不是在线候选先验 |
| unwanted image displacement | `trace_metrics` 或真实 RGB static feature tracks | 后处理窗口对齐；真实视频指标不直接等同模型代理指标 |

## 数据契约与运行脚本

`templates/response_trial_schema.csv` 是**空表头**，不含实验数据。由每段 0.5–1.0 秒窗口生成一个 row：

- `controller_on=0`：相同 gait 条件下未补偿相机抖动；
- `controller_on=1`：相同 gait 条件下固定 Roll/Pitch/Z 补偿控制器执行后抖动。

`src/closed_loop_surrogate.py` 读取这个表训练两个响应模型+速度模型。`scripts/evaluate_response_surrogate.py` 要求明确提供独立测试地形 ID；不支持训练/验证用同一个 terrain_instance_id 偷看测试数据。

命令（仅在你已有数据后执行）：

```bash
cd Machines_Hexapod_v2_2026-10-20
PYTHONPATH=src python scripts/evaluate_response_surrogate.py \
  --input outputs/response_windows.csv --test-terrains rough_heldout_1 slope_heldout_2
```

## 安全边界

新增模块只作为独立仿真任务和旁路实机节点开发，**先不要**直接改动 `run_agent2.py` 的稳定电机通信逻辑。物理机器人实验需要单独进行零速度、关节方向/工作空间校验、驱动限速和急停联锁。仿真代码通过不等于可以直接接实机输出。
