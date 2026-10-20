# Machines 专业审稿人逐项验稿清单

## 投稿前硬条件

- [ ] 本文仍为未公开、未被其他期刊同时审理的原始研究稿。
- [ ] 作者顺序、贡献、单位、基金确认；全体共同作者同意。
- [ ] Abstract 用真实量化结果替换“设计期望”；不得继续保留开发状态的文案。
- [ ] 所有图表有真实来源；每张图的 n、单位和误差解释可追溯至 raw logs。
- [ ] 参考文献核实作者、刊名、卷期、DOI 和引述是否准确；Related Work 不用关键词拼凑。
- [ ] 所有算法对照的迭代、优化候选数量、相机深度输入和控制频率公平。
- [ ] 速度、越障成功率、相机抖动同时报告，不能只报告“看起来稳”。
- [ ] 补偿可行域加入真实 support-foot anchor、IK 行程与 qdot；模型响应有测量或明确限幅。
- [ ] 采集真实 RGB：至少静态特征跟踪残差或 AprilTag 指标；标注 RGB shutter / exposure 处理。
- [ ] 所有失败案例、接触违反、速度下降都在结果中。
- [ ] 数据可用性说明与实际可分享的源码/日志一致。
- [ ] 将英文稿迁移到 Machines 官方 LaTeX 模板；提供可重新编译完整 ZIP。

## Reviewer 1：Novelty

1. Shaw 2019 早就视觉反馈 + CPG + body gaze；本文有何独立增量？→ 应回答「根据 candidate-conditioned attainable correction 计算最终残余图像运动」，不是简单加视觉损失项。
2. Saggar 2007 早已按视觉稳定性学习步态；本文和它的真正差别是什么？→ 物理约束导致的候选补偿能力变化，而非视觉评价本身。
3. Burkus Machines 2022 已优化 sensor-relevant vibration；该工作与本文是不是同一个优化目标？→ 本文时变、在线、候选专属补偿可行域、残余像素运动；需要 B3/B5 独立证据。

## Reviewer 2：Control soundness

1. 脚固定在地面而机身变姿态，是否导致 stance slip？→ 明确脚的世界锚点和 IK 计算，记录 slip。
2. position-controlled servos 的高频响应不足怎么办？→ 只在控制带宽范围内建模；触地冲击通过改变摆腿轨迹减少源头，不能声称全频段主动消振。
3. 候选预测是否利用未来地形 ground truth？→ 训练可用真值标签，在线输入只能用已有观测。
4. 步频变化的同时步长怎样保持实际速度？→ 基于 gait cycle 明确步长-步频映射与可达工作空间；报告 achieved speed。

## Reviewer 3：Experiments

1. 与 best fixed gait、raw disturbance、image cost w/o compensation awareness 的强对照是否齐全？
2. 按相同 terrain seeds 和相同相机参数完成 paired test 吗？
3. 模型预测准确度是否在未见 terrain seed 上测试？
4. “改步态防抖”会不会只是减速或摔倒次数增加？
5. Results 中每项效果是否能由原始可复现日志重新计算？

## 英语与排版（科研 Skills 流程）

- literature-review：所有 gap 都依赖原文方法细节，非全文标题印象。
- paper-narrative：Fig.1 问题/机制；Fig.2 核心结果；Fig.3 机制原因；Fig.4 消融；Fig.5 trade-off；Fig.6 实机。无独立证据的图删除。
- figure-style：有 n，轴有单位，baseline 命名具体，不能用示意图冒充实验。
- peer-review：用以上 novelty / correctness / attribution 三类 reviewer 挑缺陷并修复。
