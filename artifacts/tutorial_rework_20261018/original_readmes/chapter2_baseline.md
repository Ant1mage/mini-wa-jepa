# Chapter 2: 行为克隆基线 (Behavior Cloning Baseline) 与因果困境

## 📚 章节导读

**行为克隆 (Behavior Cloning, BC)** 是自动驾驶最直接的端到端学习范式：
> 将传感器感知的鸟瞰图输入（BEV）作为状态 $s_t$，将专家人类驾驶员在未来 2 秒内的轨迹作为动作监督信号 $a_t$，通过监督学习回归最优控制路径。

在 Chapter 2 中，我们构建并训练一个标准的 **ResNet-18 + MLP 行为克隆基线**（20,116,648 参量）：
1. **端到端网络架构**：适配 4 通道 BEV 输入的 ResNet-18 骨干，结合自车当前速度与角速度，通过全连接层输出未来 10 个航向点 $(x, y, v, \text{heading})$；
2. **掩码轨迹损失函数 (Masked Trajectory Loss)**：处理终点提前停止与截断掩码，防止网络拟合无效填充点；
3. **开环预测 vs 闭环仿真中的因果困境**：开环评测表现良好的网络，为什么在闭环实际跑车时会遇到**复合误差 (Compounding Error)** 与**虚假因果 (Causal Confusion)**？这也是为什么我们后续必须引入世界模型（JEPA）的核心动机！

---

## 🚀 交互式学习

```bash
# 启动 Chapter 2 Marimo 教学走读应用
marimo edit chapters/chapter2_baseline/app.py
```

在教学模式下，你可以：
- 单独分块运行 ResNet-18 网络的每一层，观察张量形状从 $[B, 4, 128, 128]$ 压缩为 $[B, 10, 4]$ 的全过程；
- 在 BEV 画布上对比模型预测航向点与专家真实驾驶轨迹；
- 深入剖析纯模仿学习在闭环仿真中遇到的“由于状态分布偏移而脱轨”的数学本质。

---

## 📂 文件结构

- [`app.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter2_baseline/app.py): Marimo 交互教学应用（网络分块前向、轨迹可视化、损失函数推导）。
- [`model.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter2_baseline/model.py): 独立的 ResNet-18 轨迹回归网络与损失函数实现。
- [`config.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter2_baseline/config.py): 基线超参数配置类。
- [`dataset.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter2_baseline/dataset.py): 阶段 2 专家数据集读取器。
- [`train.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter2_baseline/train.py): 独立的基线训练脚本。
