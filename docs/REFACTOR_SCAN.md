# 重构后结构扫描（2026-10-18）

本次为结构、接口与已有验收证据检查，未启动正式训练或模拟器闭环。

来源：HEAD `14373760ab74275f36fc9f4a4e13cdb14d07b7f4`，工作区包含尚未提交的用户重构。
GOAL.md SHA-256：`151368fe3f69b63ceb80fab351be6ed676fa56a822e9f0466d3738dd7d035511`。

## 当前结构和阶段

- `app.py`、`start.sh`、`Mini-WA-JEPA.desktop`：统一教学工作台入口。
- `chapters/chapter0_hardware` 至 `chapter5_stress_testing`：分章节 `app.py`、`ui.py` 和独立 `src/` 实现。
- `configs/`、`models/`、`dataset/`、`training/`、`evaluation/`、`controller/`、`scripts/`：正式实验流程。
- `env/`、`mini_wa_jepa/`：模拟器、BEV、几何、硬件和 Rerun 基础设施。
- `artifacts/restart_stage{1,2,3,4}/`：既有阶段证据；`artifacts/hardware_upgrade/`：硬件升级记录。

阶段 4 的 `acceptance.json` 为 completed=true、passed=true，全部 7 项 gate 通过。`delivery_checks.json` 记录 9,600 个匹配测试锚点、验收视频、4 个闭环 RRD 与既有测试记录。Joint 和 Action-only 的 `best.pt`、v2 数据目录均存在；ResNet 权重实际位于 `artifacts/restart_stage2/baseline/best.pt`。

依据 AGENTS.md，阶段 5 已获授权。目前只有实现雏形：未在 artifacts 中发现文件名或路径包含 stage5/chapter5 的实测产物，缺少正式匹配开环、闭环及验收交付。阶段 5 结果与鲁棒性增益应保持未测状态。

## 已确认的问题

1. **统一入口导航值不匹配**：`app.py` 的 radio 使用 `{chapter_key: label}`，Marimo `.value` 返回 label，后续分支却比较 `ch0` 等 key。最小探针 `mo.ui.radio(options={'ch0': 'Chapter 0'}, value='ch0').value` 返回 `'Chapter 0'`。当前分支无法选中相应章节。
   - 后续修复（2026-10-18）：选项改为 `{label: chapter_key}`，默认选择使用 label。`marimo check app.py` 通过；通过真实 Marimo radio 与 `app.run(defs=...)` 逐一选择六个章节，均返回对应 selected 值和章节内容。此项已解决；未进行浏览器点击验证。
2. **Chapter 5 默认权重不兼容**：benchmark 实例化章节版 Chapter 4 模型，却加载正式阶段 4 权重。CPU 比对确认 115 个缺失 key、155 个额外 key，且共有 key 存在形状不一致。例如 encoder 首层正式权重为 `(64,4,5,5)`、章节版为 `(128,4,7,7)`；predictor position 分别为 `(1,64,384)` 和 `(1,64,256)`。这不能仅通过重命名 key 修复。
3. **演示数据被表达成实测结论**：Chapter 5 的 `ui.py` 与 `app.py` 将 ADE 曲线写死为 `[0.31,0.35,0.42,0.58,0.81]`、`[0.85,1.15,1.62,2.45,3.80]`，随后声称证明鲁棒性优势。轨迹演示仅新建未训练模型，无 checkpoint、真实目标或 ResNet 对照；clean/degraded 预测之间的漂移不能等同于对真值的 ADE。
4. **BEV 语义分叉**：正式 `env/semantic_bev.py` 定义 0=道路/标线，1=人行道/围栏/标志，2=其他交通参与者且排除自车，3=visibility。Chapter 1 声称 1=车辆、2=自车、3=边界；Chapter 5 合成示例又定义 1=车道线、3=自车。Chapter 1 几何使用中心 64，正式标定中心为 63.5。应对齐已验收 schema，再定义边界退化的具体含义。
5. **评测尚不满足阶段 5 契约**：benchmark 默认取 val 子集，仅评 Joint，没有 Action-only/ResNet、匹配的 9,600 个 test 锚点、闭环注入、连续 3 帧丢失条件、原始配对误差与置信区间。缺 checkpoint 时会继续运行随机权重；无有效 FDE 时写 0.0。输出不包含完整 config/seed/source/GOAL hash，也未接入正式指标格式、TensorBoard、Rerun、预测对真值图或视频。
6. **文档路径过时**：章节 README 与 TASKS 的重构描述仍使用 `chapterN/train.py`、`chapter5/benchmark.py`；实际代码在 `src/`。从仓库根目录运行的模块入口 `python -m chapters.chapter5_stress_testing.src.benchmark --help` 能退出成功，但存在包初始化提前导入 benchmark 的 runpy warning。

## 训练基础设施静态检查待办

`training/joint.py` OOM 分支只更新 batch-size 变量并 `continue`，没有重建 DataLoader 或切分当前 batch，因此不能据此认定真实训练会降低物理 batch 并重试同一批数据。启用梯度累积时，第一次 optimizer step 前 `step` 仍为 0，日志分支可能引用尚未赋值的 `norm`。这些是静态代码发现，本次未制造 GPU OOM 或运行训练复现。

## 本次验证

- `.venv/bin/python -m unittest discover -s tests -v`：48 项测试，14.512 秒，全部通过；日志见 `artifacts/refactor_scan_20261018/tests.log`。未单独运行 chapters 内部测试。
- Chapter 5 模块 `--help`：成功，带上述 runpy warning。
- 章节数据集 val 单样本 mmap 初始化、读取 normalizer：成功。
- 章节 Joint 模型 CPU 合成输入 forward：成功；这不验证训练权重兼容性或实际鲁棒性。
- 统一入口 radio 返回值、正式 checkpoint 与章节模型 state_dict 形状：已实际核对。

本次仅同步扫描记录与 TASKS 状态；模型、训练代码、历史权重和验收证据未修改。
