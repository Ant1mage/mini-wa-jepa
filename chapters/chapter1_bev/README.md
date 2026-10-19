# Chapter 1 · 把驾驶场景变成 BEV 数组

面向入门开发者，先修知识：数组切片、Chapter 0 的 CHW/dtype。

## 学习目标

按正式 schema 读四通道；用中心63.5、比例2像素/米做坐标正反变换。

本章分成三个小实验，每次围绕一个问题操作控件、观察图示与解释；中文注释代码、深入原理和正式源码可以按需展开。
可以调节：目标 x/y/朝向与道路宽度。拖动滑块即更新旁边图示与解释；在总入口切换章节会保留各章参数。

## 打开与阅读代码

```bash
# 从仓库根目录运行
.venv/bin/marimo edit chapters/chapter1_bev/app.py
# 也可 ./start.sh 打开七章统一入口
```

`app.py` 将控件创建与消费分成独立 cell；`ui.py` 内的 `compute(values)`
是页面展示的实际计算代码，包含中文注释。
页面还提供完整本章 Python 源码和最小调用示例；`src/` 的原有模型/训练实现保留，
但教学版本不保证兼容正式训练 checkpoint。正式实验不由滑块启动。

## 先做这道练习

令 x=10,y=5，检查 u=53.5,v=43.5；只改朝向时位置应不动。

## 结果边界

合成地图示意；通道0道路/标线、1静态边界要素、2其他参与者（排除自车）、3可见性。

进一步阅读：env/semantic_bev.py、chapters/chapter1_bev/src/semantic_bev.py。共享展示工具位于 `chapters/learning.py`，
不参与正式训练。完整课程说明见上级 `chapters/README.md`。

