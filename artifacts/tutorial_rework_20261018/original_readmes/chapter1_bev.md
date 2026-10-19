# Chapter 1: 正交鸟瞰图 (BEV) 空间感知与语义渲染

## 📚 章节导读

在自动驾驶中，感知系统的输入通常有两种选择：
1. **前视摄像头 RGB 图像**：包含丰富的纹理和颜色，但存在透视投影畸变（近大远小）、物理尺度非线性、被前方大车严重遮挡等问题，使得下游的轨迹规划非常困难；
2. **正交鸟瞰图 (Bird's-Eye View, BEV)**：从车辆正上方垂直俯视，**具备天然的度量空间性质（Metric Space）**。每个像素对应真实世界中恒定的物理尺寸（如 $0.5\text{ m/pixel}$），车辆旋转平移遵循严格的欧几里得几何变换。

在 Chapter 1 中，我们将深入剖析 **Mini-WA-JEPA** 的感知地基——$4 \times 128 \times 128$ 正交语义鸟瞰图：

1. **4 通道语义栅格定义**：路面标线、移动障碍物、自车车体、物理路沿边界；
2. **车辆自车坐标系与像素坐标转换**：以自车中心为原点，前向为 $+X$、左向为 $+Y$，通过几何仿射变换精确映射到 $128 \times 128$ 图像；
3. **无头 EGL 高性能渲染流水线**：利用 Panda3D EGL 离屏渲染上下文，在服务器或无显示器环境下实现 $\ge 120\text{ FPS}$ 的高效渲染与显存读回。

---

## 🚀 交互式学习

```bash
# 启动 Chapter 1 Marimo 教学走读应用
marimo edit chapters/chapter1_bev/app.py
```

在教学模式下，你可以：
- 单独分块运行每个 Cell，直观查看 4 个语义通道的独立热力图与合成 RGB 图；
- 拖拽自车航向与坐标滑块，亲手验证世界坐标系到自车坐标系、自车坐标系到像素坐标系的数学转换；
- 测试离屏 EGL 渲染吞吐量与显存占用曲线。

---

## 📂 文件结构

- [`app.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter1_bev/app.py): Marimo 交互教学应用（包含完整可分块独立执行的 BEV 生成、通道分解与坐标系可视化）。
- [`semantic_bev.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter1_bev/semantic_bev.py): 独立的语义 BEV 提取器与坐标标定算法。
- [`verify.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter1_bev/verify.py): 阶段 1 验收验证脚本（100 轮无头渲染、FPS 与显存泄漏测试）。
