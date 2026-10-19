# Chapter 0: 硬件自动检测、算子优化与训练自适应校准

## 📚 章节导读

在开始自动驾驶世界模型的训练之前，首要任务是**让算法适配底层算力**。不同开发者可能拥有不同的计算设备（AMD ROCm、NVIDIA CUDA、Apple Silicon MPS 或 CPU），显存容量也从 8GB 到 24GB 不等。

传统深度学习项目往往将 Batch Size、显存上限、Attention 算子硬编码在配置中（例如原项目遗留的 14GB 人工上限与强制 `SDPBackend.MATH`）。在 Chapter 0 中，我们将构建一套**通用、自适应、高容错的底层训练基础设施**：

1. **硬件无感自动检测**：只读探测 GPU 厂商 (AMD/NVIDIA)、型号、显存大小、CPU 拓扑、内存与 PyTorch 计算后端；
2. **厂商优化注意力算子 (SDPA)**：支持 `AUTO`、`FLASH_ATTENTION`、`EFFICIENT_ATTENTION` 与 `MATH` 原生基准，在优化核不可用时自动无损回退；
3. **参数自适应推荐与批大小校准**：依据物理显存动态预留 15% 安全余量（无硬编码上限），综合考虑 VICReg 批内样本协方差的统计无偏性要求，推荐最优批大小与梯度累积步数；
4. **有限重试 OOM 恢复机制**：在发生显存溢出时自动清空缓存、持久化紧急检查点、按比例缩减批次并加倍梯度累积，记录可追溯审计日志，杜绝静默失败。

---

## 🚀 快速开始

### 1. 启动 Marimo 交互走读应用

```bash
# 在浏览器中以交互方式探索 Chapter 0
marimo edit chapters/chapter0_hardware/app.py
```

### 2. 命令行静态检测与推荐

```bash
# 探测当前系统硬件并输出推荐配置至 JSON
python3 chapters/chapter0_hardware/calibration.py
```

### 3. 运行独立算子与硬件单元测试

```bash
python3 -m unittest chapters/chapter0_hardware/test_chapter0.py
```

---

## 📂 文件结构

- [`app.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter0_hardware/app.py): Marimo 交互式教学应用（硬件仪表盘、算子速度比对、批大小滑块计算器、OOM 模拟器）。
- [`hardware.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter0_hardware/hardware.py): 独立硬件探测模块（GPU, CPU, RAM, 精度格式）。
- [`operators.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter0_hardware/operators.py): SDPA 算子调度器、自动回退机制与数值精度检验。
- [`calibration.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter0_hardware/calibration.py): 考虑 VICReg 统计特性的参数推荐器与候选批大小试运行器。
- [`oom_recovery.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter0_hardware/oom_recovery.py): 显存溢出有限重试与紧急状态保存管理器。
- [`test_chapter0.py`](file:///home/auronox/Desktop/mini-wa-jepa/chapters/chapter0_hardware/test_chapter0.py): Chapter 0 自动化单测。
