# Chapter 0 · 先认识硬件与张量内存

面向入门开发者，先修知识：Python 变量、乘法和数组形状。

## 学习目标

解释显存为什么不只包含输入；手算单层注意力矩阵占用；区分物理 Batch 和有效 Batch。

本章分成三个小实验，每次围绕一个问题操作控件、观察图示与解释；中文注释代码、深入原理和正式源码可以按需展开。
可以调节：Batch、Token 数、数值类型与梯度累积。拖动滑块即更新旁边图示与解释；在总入口切换章节会保留各章参数。

## 打开与阅读代码

```bash
# 从仓库根目录运行
.venv/bin/marimo edit chapters/chapter0_hardware/app.py
# 也可 ./start.sh 打开七章统一入口
```

`app.py` 将控件创建与消费分成独立 cell；`ui.py` 内的 `compute(values)`
是页面展示的实际计算代码，包含中文注释。
页面还提供完整本章 Python 源码和最小调用示例；`src/` 的原有模型/训练实现保留，
但教学版本不保证兼容正式训练 checkpoint。正式实验不由滑块启动。

## 先做这道练习

固定 N=64，把物理 Batch 翻倍，再把 N 变为四倍，观察存储量分别如何变化。

## 结果边界

单层显式注意力矩阵的理论存储，不是训练峰值实测。硬件卡片为会话首次只读检测。

进一步阅读：mini_wa_jepa/hardware.py、mini_wa_jepa/operators.py。共享展示工具位于 `chapters/learning.py`，
不参与正式训练。完整课程说明见上级 `chapters/README.md`。

