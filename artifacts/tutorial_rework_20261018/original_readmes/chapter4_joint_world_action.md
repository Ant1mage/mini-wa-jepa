# Chapter 4: 联合世界与动作模型 (Joint World & Action Model)

本章节对应项目的**阶段 4 (Stage 4)**，实现了基于端到端隐变量世界模型（Latent World Model）与动作规划头（Action Trajectory Head）联合训练的自主驾驶决策智能体。

---

## 核心设计理念

### 1. 为什么“仅动作模仿学习（Action-Only BC）”会惨败？
在行为克隆中，单纯监督轨迹容易受到**因果混淆（Causal Confusion）**和**开环/闭环协变量偏移（Covariate Shift）**的影响：
- 当自车进入训练集罕见的偏差位姿时，Action-Only 模型由于缺乏环境动态理解，会持续输出错误的微调动作，陷入不可逆的失控震荡。
- 实测表明：在 MetaDrive 测试路线上，Action-Only 模型路线完成率仅为 **29.71%**，且容易在弯道冲出道路。

### 2. 联合模型（Joint Model）的救赎
Joint 模型引入了“双任务驱动”：
- **世界预测任务（World Prediction）**：利用 4 步历史 BEV，预测未来 4 步（0.4 秒）环境隐空间 Token 的演变，由 VICReg 损失约束，防止表征塌缩。
- **动作规划任务（Action Planning）**：将当前隐空间 Token 与预测的未来 4 步隐空间 Token 联合送入动作头，规划未来 2 秒内的 10 个运动学路标 $(x, y, v, \theta)$。
- **实测成果**：联合训练使路线完成率飙升至 **96.98%**！在闭环驾驶中表现出极强的避障和车道线贴合能力。

### 3. 空间 Token 绝对保真（Spatial Token Preservation）
绝大多数通用视觉模型喜欢在末端使用全局平均池化（`AdaptiveAvgPool2d`），但这会彻底摧毁障碍物和道路边缘的精确几何位置！
- 本架构将 64 个空间 Token 降维为 128 维后直接 Flatten 为 8192 维高维向量。
- 每一个 Token 的相对空间拓扑关系被完全保留并直接输入给多层感知机（MLP）规划头。

### 4. 无参数运动学常曲率先验（Constant Turn Trajectory Prior）
利用自车当前的物理车速 $v$ 与横摆角速度 $\omega$，基于刚体几何计算无参数圆弧推演：
$$x(t) = v \cdot t \cdot \text{sinc}\left(\frac{\omega t}{\pi}\right)$$
$$y(t) = v \cdot t \cdot \frac{\omega t}{2} \cdot \text{sinc}^2\left(\frac{\omega t}{2\pi}\right)$$
神经网络仅需预测相对于常曲率物理先验的微小残差 $\Delta \tau$。初始化时将输出层权重置零，模型在训练起点便具备合乎牛顿力学的平滑转弯能力！

---

## 模块结构
- `config.py`：阶段 4 独立配置类 `Stage4Config`。
- `model.py`：包含 `BEVEncoder`、`OneStepPredictor`、`constant_turn_trajectory` 与 `JointModel`。
- `loss.py`：包含 `joint_loss`、`masked_trajectory_loss` 与 `regularization`。
- `dataset.py`：多帧时序数据切片 `JointSequencesDataset`。
- `train.py`：支持联合训练与 Action-only 消融对照的独立训练入口。
- `app.py`：Marimo 交互式教学笔记（含先验模拟沙盒、Token 维度推演、Joint vs Action-only 效果对比图表）。
