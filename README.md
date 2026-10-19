# Mini-WA-JEPA

> **从 MetaDrive 到 3D 赛车游戏**：基于 JEPA 潜空间预测思想的轻量级 BEV 时空联合世界模型与闭环轨迹规划系统。

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ROCm%20%7C%20CUDA%20%7C%20CPU-red.svg)](https://pytorch.org/)
[![MetaDrive](https://img.shields.io/badge/MetaDrive-0.4.3-green.svg)](https://github.com/metadriverse/metadrive)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[项目目标与规范 (GOAL.md)](GOAL.md) · [开发进度追踪 (TASKS.md)](TASKS.md) · [可视化规范 (VISUALIZATION.md)](docs/VISUALIZATION.md)

---

## 🎯 项目目标

本项目旨在构建一个基于 BEV（鸟瞰图）的轻量端到端自主驾驶与轨迹规划系统。

**最终目标**：
1. 先在 **MetaDrive** 虚拟仿真环境中训练并验证能够自主闭环驾驶的 BEV 世界模型；
2. 再将学到的驾驶能力迁移到实际运行的 **3D 赛车游戏** 中（通过 BEV 感知与控制适配器）。

核心探索与待回答的科学问题：
- **潜空间预测的有效性**：在潜空间学习未来表征（JEPA），能否真正改善驾驶决策，并在感知退化（遮挡、丢帧）时提供超越传统 CNN/端到端回归的鲁棒性？
- **跨环境迁移能力**：在仿真环境中学习到的空间表征与驾驶策略，能否通过轻量适配层有效迁移到全新的 3D 赛车游戏？

> [!NOTE]
> 本项目借鉴 JEPA (Joint-Embedding Predictive Architecture) 思想，在潜空间预测未来而非高维像素级重建。全套模型结构、数据管线与训练逻辑均由本项目独立设计与实现，不依赖外部大型预训练权重或私有依赖。

---

## 🧠 系统架构

Mini-WA-JEPA 采用“时空编码 → 潜空间演化 → 显式轨迹规划 → 控制器执行”的分层架构：

```
[4 帧历史 BEV 观测] (过去 0.3s, 4×128×128)
       │
       ▼ Conv-Stem + Transformer 编码器
[时空潜空间 Token 序列]
       │
       ├────────────────────────────────────────┐
       ▼ JEPA 世界预测器 (Transformer)          ▼ 显式轨迹规划头 (MLP / Flow Matching)
[未来 4 步潜空间表征]                           [未来 10 步轨迹路点] (0.2s~2.0s)
(受 VICReg 防塌缩正则约束)                      (x, y, v, heading)
                                                │
                                                ▼ Adaptive Pure Pursuit + PID 控制器
                                          MetaDrive 闭环仿真 / 游戏环境 (dt=0.02s)
```

1. **BEV 感知输入**：128×128 四通道语义 BEV（覆盖自车周边 64m×64m），自车居中、朝向固定朝上，提供清洁且几何对齐的空间拓扑。
2. **JEPA 潜空间世界模型**：仅在特征潜空间预测未来演化，避免昂贵且容易过度拟合无关细节的像素级重建，采用 VICReg 正则（不变性 + 方差 + 协方差）防止表征坍缩。
3. **显式轨迹规划**：模型输出物理意义明确的未来 10 个路点 `(x, y, v, heading)`，而非黑盒底层控制量。
4. **闭环控制**：通过自适应 Pure Pursuit 与纵向速度控制器将轨迹转化为平滑动力学输入，兼顾规划表现与控制稳定性。

---

## 🗺️ 项目路线与进展

项目严格遵循 [GOAL.md](GOAL.md) 制定的 9 阶段里程碑推进：

| 阶段 | 阶段目标 | 核心任务 | 当前状态 |
| :--- | :--- | :--- | :---: |
| **阶段 1** | 环境与 BEV 验证 | MetaDrive EGL 无头渲染、自车坐标系与 4 通道 BEV 标定 | ✅ 已完成 |
| **阶段 2** | 专家数据与驾驶基线 | 120k 帧专家数据集构建，ResNet-18 行为克隆基线建立 | ✅ 已完成 |
| **阶段 3** | 独立未来潜空间预测 | 编码器 + JEPA 预测器，VICReg 正则，验证潜空间预测优于简单基线 | ✅ 已完成 |
| **阶段 4** | 联合世界预测与轨迹预测 | 联合训练时空潜变量与 10 步显式轨迹，严格动作消融对照 (Action-only) | ✅ 已完成 |
| **阶段 5** | 遮挡与丢帧压力测试 | BEV 遮挡、边界缺失、丢帧下测试 Mini-WA-JEPA 相比 CNN 基线的鲁棒性 | ✅ 实测完成 |
| **阶段 6** | 动作条件世界预测 | 引入候选轨迹作为条件，预测不同驾驶动作对应的未来演化 | ✅ 实测通过 |
| **阶段 7** | Flow Matching 轨迹生成与候选规划 | 多样化候选轨迹生成，结合风险评估头与评分机制择优规划 | 📋 计划中 |
| **阶段 8** | MetaDrive 完整闭环验收 | 多种子与极端场景无接管自主闭环驾驶验收，为游戏迁移把关 | 📋 计划中 |
| **阶段 9** | 迁移到 3D 赛车游戏 | 适配目标游戏画面提取 BEV，控制量映射，完成跨平台闭环迁移 | 🎯 最终目标 |

阶段6已完成两组种子的条件/无条件对照：测试潜变量误差降低45.05%/45.54%，后果差分误差降低66.43%/66.62%；全部科学与交付门槛通过，71项测试通过。
结果限于32个测试起点、8个独立种子下的动作后果预测，没有实现候选选路器，也不代表新的自主驾驶成功率。详见 [阶段6报告](artifacts/restart_stage6/REPORT.md)。
按用户排期，接下来先做现有chapter的marimo入门交互，不添加框架；阶段7尚未启动。

---

## 🔬 核心对照实验设计

本项目坚持以实验与数据为依据，所有关键模块均设有基准对照：

- **世界模型有效性**：联合训练模型 vs 动作对照消融 (Action-only，同架构仅关停世界损失)。
- **网络表征能力**：Mini-WA-JEPA vs 经典 CNN / ResNet 行为克隆基线。
- **环境鲁棒性**：完整 BEV vs 动态遮挡、边界缺失与突发连续丢帧。
- **规划机制对比**：确定性 MSE 回归规划 vs Flow Matching 多样化生成 + 风险评分规划。
- **跨域迁移效果**：赛车游戏零样本直接迁移 vs 少量目标域数据适配。

---

## ⚡ 快速开始

面向入门开发者的六章交互课程：运行 `./start.sh`，或用 `./start.sh --edit` 查看 Marimo 单元格。
每章包含联动滑块、逐步讲解、中文注释代码、练习与正式源码；教学示例和实测结果分别标明来源。
详见 [课程说明](chapters/README.md) 与 [交互验证记录](docs/LEARNING_WORKBENCH.md)。

### 1. 环境准备

本项目支持 Linux 平台，具备硬件自适应能力（支持 AMD ROCm / NVIDIA CUDA / CPU）：

```bash
# 克隆仓库
git clone https://github.com/Ant1mage/mini-wa-jepa.git
cd mini-wa-jepa

# 安装依赖
bash scripts/bootstrap.sh
.venv/bin/pip install -r requirements-training.txt -r requirements-visualization.txt

# 硬件自适应与环境诊断
.venv/bin/python -m scripts.check_rocm
```

### 2. 运行单元测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### 3. 数据采集与模型训练

```bash
# 采集 BEV 专家数据
.venv/bin/python -m scripts.collect_stage2_parallel

# 训练行为克隆基线 (CNN/ResNet)
.venv/bin/python -m scripts.train_baseline

# 阶段 3：训练 VICReg 潜空间自监督世界模型
.venv/bin/python -m scripts.train_latent

# 阶段 4：训练联合模型与动作消融对照
.venv/bin/python -m scripts.train_joint --output artifacts/stage4_run/joint
.venv/bin/python -m scripts.train_joint --action-only --output artifacts/stage4_run/action_only
```

### 4. 闭环仿真与评估

阶段 5 已完成冻结模型压力评测与交付审计，结果见 [阶段 5 报告](artifacts/restart_stage5/REPORT.md)。协议见 [STAGE5_PROTOCOL.md](docs/STAGE5_PROTOCOL.md)，
运行记录和进度位于 `artifacts/restart_stage5/`；`open_loop/report.json` 逐条件更新，
`suite_progress.json` 记录闭环队列，最终报告为 `REPORT.md`。正式结果不使用章节演示曲线。

```bash
# 运行闭环驾驶评测
.venv/bin/python -m scripts.evaluate_joint_closed_loop --checkpoint artifacts/stage4_run/joint/best.pt --record

# 启动 Marimo 响应式指标看板 (ADE/FDE、完成率、消融对比)
.venv/bin/marimo run scripts/dashboard_marimo.py --host 127.0.0.1 --port 2718 --no-token
```

---

## 📁 目录导览

- [`configs/`](configs/): 各阶段超参数配置 (YAML / dataclass)
- [`env/`](env/): MetaDrive 环境封装、四通道 EGL 语义 BEV 渲染、专家控制器
- [`models/`](models/): 模型实现（ResNet 基线、JEPA 编码器、时空预测器与联合轨迹头）
- [`training/`](training/): 损失函数（VICReg、轨迹 MSE、联合损失）与训练循环
- [`controller/`](controller/): Pure Pursuit 轨迹跟踪与速度控制器
- [`mini_wa_jepa/`](mini_wa_jepa/): BEV 几何变换、硬件检测与 Rerun 多模态可视化
- [`scripts/`](scripts/): 数据采集、训练、开闭环评估与看板脚本
- [`docs/`](docs/): 阶段设计协议与技术规范深入文档

---

## 📜 许可

本项目遵循 [MIT License](LICENSE)。
