# Chapter 5 · 设计可复现的感知退化实验

面向入门开发者，先修知识：正式BEV通道、历史帧、开环/闭环指标。

## 学习目标

区别空间不可见与时间过期；写出匹配测试设计；避免将退化像素比例当模型鲁棒性。

本章分成三个小实验，每次围绕一个问题操作控件、观察图示与解释；中文注释代码、深入原理和正式源码可以按需展开。
可以调节：遮挡宽深、边界缺失比例、最近连续丢帧、查看帧和种子。拖动滑块即更新旁边图示与解释；在总入口切换章节会保留各章参数。

## 打开与阅读代码

```bash
# 从仓库根目录运行
.venv/bin/marimo edit chapters/chapter5_stress_testing/app.py
# 也可 ./start.sh 打开七章统一入口
```

`app.py` 将控件创建与消费分成独立 cell；`ui.py` 内的 `compute(values)`
是页面展示的实际计算代码，包含中文注释。
页面还提供完整本章 Python 源码和最小调用示例；`src/` 的原有模型/训练实现保留，
但教学版本不保证兼容正式训练 checkpoint。正式实验不由滑块启动。

## 先做这道练习

关闭所有退化应完全一致；最近3帧丢失时收到的4帧都重复最早帧。

## 结果边界

当前三个小节用于解释输入退化；折叠区读取阶段5正式鲁棒性报告并保留来源。旧benchmark仍待解决正式checkpoint兼容问题，不用于正式验收。

进一步阅读：chapters/chapter5_stress_testing/src/degradation.py、docs/VISUALIZATION.md。共享展示工具位于 `chapters/learning.py`，
不参与正式训练。完整课程说明见上级 `chapters/README.md`。

