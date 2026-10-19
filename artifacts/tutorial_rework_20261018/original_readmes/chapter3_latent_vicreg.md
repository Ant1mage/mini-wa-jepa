# Chapter 3: 自监督隐空间世界模型与 VICReg (Latent World Model & VICReg)

## 📚 章节导读

在 Chapter 2 中，我们发现了纯行为克隆的根本瓶颈：缺少对未来环境动态演化的因果推演能力。

**世界模型 (World Model)** 旨在让自动驾驶系统拥有“想象力”：
> “如果我现在保持 8 m/s 并向左打方向盘，0.2 秒后周围的道路、车辆和自车相对位置会变成什么样？”

然而，传统的基于像素重构的世界模型（如基于 VAE 或 Diffusion 的模型）面临计算极其沉重、过分拟合路面沥青噪点等无关细节的问题。**Yann LeCun 提出的 JEPA (Joint Embedding Predictive Architecture, 联合嵌入预测架构)** 指明了一条全新的路径：
> **不在像素空间重构图像，而在高度抽象的潜变量特征空间 (Latent Space) 预测未来！**

在 Chapter 3 中，我们实现并训练一个原生的 **自监督隐空间预测模型 + VICReg 防塌缩正则化系统**（6~8M 编码器 + 6~8M 单步预测器）：
1. **ConvStem + Transformer 空间编码器**：将 $4 \times 128 \times 128$ BEV 无损压缩为 $8 \times 8 = 64$ 个空间特征 Token；
2. **动作条件残差预测器**：以自车当前线速度与角速度为条件，从当前隐变量出发预测下一帧的隐空间特征；
3. **VICReg 正则化防塌缩**：通过方差正则化强制特征空间展开，通过协方差正则化消除维度冗余，彻底解决无重构自监督学习中的表征塌缩问题。

---

## 🚀 交互式学习

```bash
# 启动 Chapter 3 Marimo 教学走读应用
marimo edit chapters/chapter3_latent_vicreg/app.py
```

在教学模式下，你可以：
- 观察 BEV 图像是如何被 ConvStem 与 Transformer 切片成 64 个高阶特征 Token 的；
- 亲眼看方差与协方差公式里的每一个真实数字，理解 VICReg 损失函数如何抑制塌缩；
- 使用 PCA 算法将抽象的 256 维隐空间特征降维为肉眼可见的伪彩色特征图！

---

## 📂 文件结构

- [`app.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter3_latent_vicreg/app.py): Marimo 交互教学应用（逐层结构可视化、VICReg 矩阵运算走读、PCA 特征图展示）。
- [`model.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter3_latent_vicreg/model.py): 独立的 BEVEncoder、OneStepPredictor 与 LatentModel。
- [`loss.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter3_latent_vicreg/loss.py): 独立的 VICReg 损失函数（方差、协方差、欧氏距离）。
- [`config.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter3_latent_vicreg/config.py): 阶段 3 自监督超参数配置。
- [`dataset.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter3_latent_vicreg/dataset.py): 相邻帧对 (t, t+1) 数据集加载器。
- [`train.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter3_latent_vicreg/train.py): 阶段 3 自监督训练脚本。
