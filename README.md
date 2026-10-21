# Hexapod Camera Stability Research

**研究目标：** 研究在受限机身补偿能力下，按补偿后相机残余运动选择步态是否优于按原始机身扰动选择。论文贡献仍是待验证假设。

## 当前可运行状态

默认 `configs/flat_stand.json` 是本地 URDF 的平地站立诊断；已完成并通过其配置门限的 6 s GPU PhysX 运行。低速 Python 三脚步态配置也完成过平地多周期前进运行，但这只证明该配置的一次局部仿真，不代表 OpenSHC、复杂地形或相机稳定性已通过验收。

OpenSHC 已接入独立运行器，但截至当前尚未通过连续物理验收；失败运行和原始轨迹均保留。步态配置在 50 Hz 更新上层足端/关节参考，400 Hz PhysX 以显式建模的 MIT 式位置目标伺服推进；伺服力矩是模型输出，不是传感器实测。默认站立诊断保留原有 400 Hz PD 路径。物理运行使用 GPU PhysX 和 CPU tensor pipeline。所有仿真结论限于对应配置与原始 trace，不代表真实硬件性能。

运行和记录由 `scripts/run_hexapod_pilot.py` 驱动，机器人模型来自本地 URDF 与所需网格；PCR-Net/`legged_gym` 不作为运行依赖。OpenSHC 源码及许可记录位于 `third_party/openshc/`，构建脚本为 `scripts/build_openshc.sh`。

## 运行与验证

先按[站立模拟器指南](docs/24_STANDALONE_SIMULATOR.md)配置已有 Isaac Gym 环境，不安装或升级依赖。典型站立运行：

```bash
ENV=/home/artrc/miniconda3/envs/grasp_hexapod
PATH="$ENV/bin:$PATH" TORCH_EXTENSIONS_DIR=/tmp/physx_pilot_torch_extensions \
  "$ENV/bin/python" scripts/run_hexapod_pilot.py --headless --sim-device cuda:0 --config configs/flat_stand.json
```

低速 Python 步态使用 `--config configs/flat_walk.json`。OpenSHC 配置 `configs/flat_walk_openshc.json` 需先按本地 ROS Noetic 环境运行 `scripts/build_openshc.sh`，再传给相同入口；它仍是未通过连续物理验收的实验后端。每次运行创建独立 `data/raw/` 记录，失败记录应保留。

离线检查：

```bash
PYTHONPATH=src python -m pytest -q tests/
```

离线检查不运行 PhysX，也不能证明步态稳定或相机改善。

## 项目指南

- [架构与实验契约](docs/25_ARCHITECTURE_AND_EXPERIMENT_CONTRACT.md)：当前实现边界、复现实验和失败记录要求。
- [研究方法](docs/16_REVISED_CORE_METHOD_AND_EXECUTION.md) · [仿真研究协议](docs/17_SIMULATION_FIRST_RESEARCH_PROTOCOL.md)：研究假设、配对与分析设计。
- [写作指南](docs/26_WRITING_GUIDE.md) · [验证记录](docs/27_REFACTOR_VERIFICATION.md)：论文主张边界与已记录的验证证据。
- [竞品重评](docs/02B_COMPETITOR_REASSESSMENT_2026-10-20.md) · [文献综述](docs/10_MACHINES_20_LITERATURE_REVIEW.md) · [审稿清单](docs/14_MANUSCRIPT_REVIEW_CHECKLIST.md)。

机器人资产来源、哈希与资产许可边界见 `assets/hexapod/PROVENANCE.md`。仓库未声明通用软件发布许可。保留失败试验；不得将其标为成功或训练数据，不要提交原始/私人视频、凭证、检查点或大规模原始数据。
