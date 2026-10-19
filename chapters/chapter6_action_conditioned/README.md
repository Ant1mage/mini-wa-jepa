# Chapter6：不同驾驶方案会带来什么后果？

在marimo里按三个小节学习：同场景执行四种方案、把方案告诉模型、故意给错方案。
第一节支持0–2秒时间拖动和方向键逐步查看；后两节使用冻结模型真实输出与真实未来特征做匹配检查。

```bash
.venv/bin/marimo run chapters/chapter6_action_conditioned/app.py
```

`src/lesson_data.py`只读加载`artifacts/restart_stage6/`的正式分支、特征目标与输出。
本页不训练、不调用随机模型、不改写原始结果；数据缺失时明确显示缺失。
图中的未来地图都来自实际执行记录，不能解释为模型生成的未来图像；匹配框不是规划器选择。
