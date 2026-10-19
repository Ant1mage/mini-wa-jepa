# 阶段 2：四通道专家数据与行为克隆 Baseline

本报告只覆盖阶段 2；阶段 3 未启动。所有统计来自原始数据及固定留出地图评测。

训练帧 100,000；验证/测试各 10,000 / 10,000。
ResNet-18 + MLP 20,116,648 参数，从零训练 12 epochs，按验证 ADE 选择第 10 epoch。

| 指标 | 专家 | Baseline |
|---|---:|---:|
| 到达成功率 | 1.0000 | 1.0000 |
| 路线完成率 | 0.9850 | 0.9850 |
| 碰撞率 | 0.0000 | 0.0000 |
| 越界率 | 0.0000 | 0.0000 |
| 平均横向误差 (m) | 0.0327 | 0.1409 |

开环测试 ADE/FDE：0.180 / 0.273 m（完整 2s 轨迹）。
决策延迟 P50/P95/P99：3.05 / 3.22 / 3.43 ms。
完整管线 P50/P95/P99：8.34 / 11.27 / 12.90 ms；控制周期 20ms（50Hz）。
训练峰值分配显存 0.891 GiB；另外保留各 epoch RSS/DRM 与 allocator reserved 统计。

## 门槛与结论

- PASS: training_frames
- PASS: parameters
- PASS: training_vram
- PASS: decision_latency_p95
- PASS: full_pipeline_p95_40hz
- PASS: heldout_episodes
- PASS: desktop_keyboard_integration
- PASS: rerun_desktop_rendering

阶段 2 的路线分数是基线锚点，没有预设成功率下限；不能把得到基线分数写成世界模型已验证。
若存在失败门槛或闭环失败，保留原始结果并在进入下一阶段前明确说明。

## 数据监督统计

| split | 帧数 | 扰动帧 | 恢复帧 | 参与动作监督的扰动/恢复帧 | 终止失败帧 |
|---|---:|---:|---:|---:|---:|
| train | 100000 | 26021 | 33999 | 26021 / 33999 | 0 |
| val | 10000 | 2647 | 3577 | 2647 / 3577 | 0 |
| test | 10000 | 2667 | 3511 | 2667 / 3511 | 0 |

本批数据的终止碰撞/全车离道失败帧为 0；扰动和恢复监督已保留，但这批数据没有提供终止失败的正样本，不能据此验证后续安全过滤头或盲区避障。
路线完成率按 MetaDrive 原始 info 统计；到达终点条件可在 route_completion 略小于 1 时满足，因此到达成功率另列。

## 桌面与交互实测

- 独立桌面键盘与 EGL 集成：通过。使用 XSendEvent 将按键送到键盘子进程回传的实际 SDL 客户端窗口，核验暂停、单步、重置、手动模式、WASD 与退出；这不是人工按键测试。
- 初轮自动探测误选同标题的窗口装饰层，事件未传到 SDL；失败日志与按键 MRE 保留。最终按子进程回传的窗口 ID 定向发送，不向桌面其他窗口注入全局按键。
- Rerun 原生窗口显示：已记录截图；截图自动退出：失败，保留原始日志/MRE。
- 旧同进程 Panda3D/Pygame 路径仍未修复；本次使用进程隔离，不重启旧路径。
- Rerun 0.22.1 `--screenshot-to` 在生成图像后触发 `Failed to take store hub from the Viewer` panic；已记录可见内容与失败退出，不能据此宣称查看器完全稳定。

## 数据与评测口径

- 4×128×128 CHW uint8、64m 方形原生 EGL BEV；visibility=255，未实现阶段 5 退化器。
- 10×4 `(x,y,v,heading)`，间隔 0.2s；0.1s 数据决策，闭环控制 0.02s，物理积分仍为 0.02s。
- 扰动事件概率为每个 eligible 决策 10%，持续 0.4–0.8s，之后至少 0.8s 专家满权重恢复。实际 noisy-frame 比例另见 data_stats.json。
- 动作损失保留扰动/恢复与大偏差片段，仅终止失败观察/结构上无控制或无未来的元素不参与相应监督；世界和安全标签保留。
- 原生默认中心越界已改为中心和四角均离开车道持续 1s。修改前采集的完整片段没有越界事件，因此轨迹不受终止规则修正影响；各来源 hash 与续采链记录在 manifest。
- 全地图几何 hash 与 seed 集合互斥；测试闭环 seed 为每张测试地图保留的 +9000 区间，没有用于训练或模型选择。
- 两个独立 EGL 采集进程按地图固定配额，合并时重新编号 episode，所有分片/计数与标签经校验。
- Rerun 录制、视频、PNG、TensorBoard 分别保留；延迟不含磁盘/可视化编码和 reset，reset 耗时单列。
- 逐步计时去掉每 episode 前 20 步；决策时间也包含状态读取和仅用于评分的导航横向误差计算，模型输入没有导航信息。
- 本次完整重跑 20 episodes；此前主动中断的 15/20 原报告、逐步计时及录制保存在 baseline_eval_interrupted_001，不混入最终统计。
- 只训练一个预先固定 seed 的基线；20 个闭环地图/seed episode 的碰撞率不是无限运行安全保证。

## 证据

- [原始数据统计](data_stats.json) / [训练记录](baseline/report.json) / [TensorBoard](baseline/tensorboard)
- [专家闭环](expert_test/report.json) / [基线闭环与开环](baseline_eval/report.json) / [逐步延迟](baseline_eval/latency_samples_ms.npz)
- [Marimo 指标源](metrics.json) / [Marimo 导出](dashboard.html) / [结果图](stage2_results.png) / [阶段 1 验收样本仍保留](../restart_stage1/SUMMARY.md)
- [桌面键盘与 EGL 集成](keyboard_desktop/report.json) / [手动驾驶录制](keyboard_desktop/manual.rrd)
- [Rerun 桌面截图](rerun_desktop_completion.png) / [截图退出失败日志](rerun_desktop_completion.log) / [桌面探测与 MRE 命令](rerun_desktop_probe.json) / [86KB 最小录制](rerun_screenshot_mre/minimal.rrd)
- [本次 26 项测试](tests_completion.log) / [中断现场](baseline_eval_interrupted_001/report.json)
- [旧数据清理清单](old_data_cleanup.json) / [迁移前源代码](pre_migration_source.tar.gz)

- expert / SCS / seed 309000: [video](../../outputs/videos/stage2_v2_expert/map-SCS_seed-309000_20261016_031528_848297.mp4)
- expert / CSC / seed 319000: [video](../../outputs/videos/stage2_v2_expert/map-CSC_seed-319000_20261016_031809_954838.mp4)
- baseline / SCS / seed 309000: [video](../../outputs/videos/stage2_v2_baseline/map-SCS_seed-309000_20261016_221012_014919.mp4)
- baseline / CSC / seed 319000: [video](../../outputs/videos/stage2_v2_baseline/map-CSC_seed-319000_20261016_221338_966021.mp4)
- 桌面按键驱动 EGL 手动驾驶：[video](../../outputs/videos/stage2_keyboard_desktop/map-S_seed-500000_20261016_222043_835500.mp4)
