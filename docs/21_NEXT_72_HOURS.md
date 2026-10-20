# 下一个 72 小时：产出真实 Results 的最短路径

**P0 / Day 1：PCR-Net 仿真最小闭环**

在 `Pcrnet_hexapod` 添加独立实验任务；使用其 HexGround / URDF / actuator，验证 18 关节顺序及 FK/IK。首先让同一参数化步态在平地行走，输出 `PcrSimTap` 轨迹（含有效脚接触和无人工重置的时间序列）。不需处理 PCR-Net Follow/Avoid 网络。

**P0 / Day 2：固定补偿器和 off/on 配对轨迹**

启用固定 Roll/Pitch/Z 机身补偿，将姿态修正通过足端 IK/支撑约束送到 HexGround action，明确延迟/限幅与安全约束；在两种地形、若干步态上输出配对 `off/on` 日志及几何图像扰动，记录速度与失败率。这里的图才算真实物理仿真结果。

**P0 / Day 3：首版预测与四组公平对照**

锁定训练/验证/测试地形实例，训练 `closed_loop_surrogate`，比较 raw、residual 预测和候选排序；至少跑固定/最优固定/raw optimizer/ours 四组，按实际速度约束分析。先获得表和图，再增补论文方法。

## 最小启动门槛

- 动力学仿真真实走动、未摔倒或如实统计失败；原始 `root_states` 和 `contact_forces` 来自运行中的 HexGround。
- 保存配置、相机光学系外参、seed、terrain_id、controller_on、gait_id、checkpoint/代码修订 SHA。
- 下游 `build_pcr_sim_windows` 能读取采样且输出非构造数据；预测模型不接触测试集。

## 论文交付检查

当前 `paper/` 是 methods-first 工作稿；Results 占位、作者/资助/数据声明缺失。任何提交前必须清除这些占位符，迁移模板并以真实物理与视频证据填图表。
