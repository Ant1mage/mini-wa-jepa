# Chapter 2 · 从专家轨迹学会计算误差

面向入门开发者，先修知识：自车坐标、均值和欧氏距离。

## 学习目标

分清输入/实际未来标签/预测；计算掩码ADE；固定2秒FDE缺失时不能补零。

本章分成三个小实验，每次围绕一个问题操作控件、观察图示与解释；中文注释代码、深入原理和正式源码可以按需展开。
可以调节：手工预测偏差、噪声、有效点数量和随机种子。拖动滑块即更新旁边图示与解释；在总入口切换章节会保留各章参数。

## 打开与阅读代码

```bash
# 从仓库根目录运行
.venv/bin/marimo edit chapters/chapter2_baseline/app.py
# 也可 ./start.sh 打开七章统一入口
```

`app.py` 将控件创建与消费分成独立 cell；`ui.py` 内的 `compute(values)`
是页面展示的实际计算代码，包含中文注释。
页面还提供完整本章 Python 源码和最小调用示例；`src/` 的原有模型/训练实现保留，
但教学版本不保证兼容正式训练 checkpoint。正式实验不由滑块启动。

## 先做这道练习

噪声设0、横向偏差设1，检查ADE=FDE=1；有效点改7后FDE必须缺失。

## 结果边界

手工构造参考/预测轨迹，用于说明指标；不加载训练权重，不声称是ResNet推理。正式基线使用加权L1。

进一步阅读：models/baseline.py、training/baseline.py、dataset/baseline.py。共享展示工具位于 `chapters/learning.py`，
不参与正式训练。完整课程说明见上级 `chapters/README.md`。

