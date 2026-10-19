# 可视化标准（阶段 1 用户验收后）

统一接口是 `from mini_wa_jepa.visualization import RerunLogger`，交互指标入口是 `scripts/dashboard_marimo.py`。阶段 2 已切换到正式四通道数据与含速度轨迹；旧 schema-1 训练数据已按用户要求清理。训练/评测进度见 TASKS.md。

2026-10-19 用户澄清：教学主界面应让初学者直接操作、逐步理解，不继续制作视频；播放可以是暂停、单步、拖动或自动推进。
各章讲清各自原理，见 [交互教材方案](INTERACTIVE_LEARNING_DESIGN.md)。未来规划章节在同一交互画面展示全部候选、选择、执行和下一轮规划，
见 [规划过程可视化契约](PLANNING_VISUALIZATION.md)。以下实验记录与历史验收要求不代表新的教学视频任务。

## 安装与运行

```bash
.venv/bin/python -m pip install -r requirements-visualization.txt
.venv/bin/marimo run scripts/dashboard_marimo.py --headless --port 2718

# 现有数据的独立 Rerun 回放；output 和 rrd 都须使用新路径
.venv/bin/python -m scripts.replay_data --data outputs/data/stage2_v2_100k \
  --rerun --rerun-save artifacts/visualization/my_replay.rrd \
  --output artifacts/visualization/my_replay --video

# 无窗口保存；不启动 MetaDrive，不需要 DISPLAY
.venv/bin/python -m scripts.replay_data --data outputs/data/stage2_v2_100k \
  --headless --rerun-save artifacts/visualization/offline.rrd \
  --output artifacts/visualization/offline

# 新规格小规模采集，先验证三个 split；键盘在隔离进程中运行
.venv/bin/python -m scripts.collect_data --frames 320 --val-frames 80 --test-frames 80 --camera --keyboard \
  --output outputs/data/visualization_smoke \
  --rerun --rerun-save artifacts/visualization/collection.rrd

.venv/bin/python -m rerun artifacts/visualization/my_replay.rrd
```

可视化依赖固定为 rerun-sdk 0.22.1、pyarrow 18.1.0、marimo 0.25.1，与已验收的 NumPy 1.26.4 兼容。新版 Rerun SDK 声明 NumPy >=2，不能直接装入已验收的模拟器环境；桥接层也适配已有的 0.39 API。不同版本生成的旧 `.rrd` 原样保留，应使用对应 SDK/viewer 版本审阅，不能假定跨版本格式兼容。

`--rerun` 启动当前 SDK 自带的独立 Rerun 进程，并分配独立端口，避免系统 PATH 上的另一版本查看器/默认端口冲突；0.22 使用 TCP，0.39 使用 gRPC。`--rerun-save` 单独指定时不弹窗，两者可以同时使用；未开启时不导入或初始化 Rerun SDK。采集/回放的 `--live` 是 `--rerun` 的兼容别名。采集使用原生 EGL 四通道 BEV，普通回放直接读分片；同进程 Pygame/Panda3D 组合窗口不再由这两个脚本启动。回放原生第三人称重建诊断保留为显式 `--reconstruct-camera`，不会自动运行。

采集增加 `--keyboard` 后由独立进程接收 Space/N/R/Q/M/WASD；无头事件测试和与 EGL 同时运行的手动油门测试已通过，真实桌面交互不等同于这些自动化测试。Rerun 的播放、暂停、时间轴拖动属于数据审阅，不等同于暂停模拟器或手动驾驶；旧窗口原生失败也没有被宣称修复。

阶段 2 收尾另运行了 `scripts.probe_keyboard_desktop --output <全新目录>`：通过键盘子进程回传的原生 SDL 客户端窗口 ID 定向发送 XSendEvent，验证全部控制键及 40 步 EGL 手动驾驶（视频/rrd 在 `artifacts/restart_stage2/keyboard_desktop`）。窗口装饰层可能与客户端同标题，不能靠标题选择注入目标。此项验证实际桌面事件路径，不代表人工体验测试。Rerun 0.22.1 已生成原生窗口截图，但截图自动退出 panic；最小录制和 MRE 命令见 `artifacts/restart_stage2/rerun_desktop_probe.json`，显示成功与退出失败分别记录。

## 流式记录契约

```python
from mini_wa_jepa.visualization import RerunLogger

with RerunLogger(app_id="mini-wa-jepa", spawn=args.rerun,
                 save_path=args.rerun_save) as logger:
    for step in range(num_steps):
        logger.set_step(step, sim_time_s=step * dt_s)
        logger.log_bev(bev)  # v2: uint8 (4,128,128), extent 64m
        logger.log_camera(camera_rgb)  # 真正第三人称 RGB；缺失传 None
        logger.log_telemetry(speed_m_s=v, steering=steer, lateral_error_m=lat_err)
        logger.log_trajectories(planned=pred_traj, ground_truth=gt_traj)
```

- 时间：按真实决策 `dt_s`，不能把未来路点间隔 0.2s 当成仿真决策间隔。多 episode 采集使用累计模拟步数，另记 episode_id、episode_step、seed，避免 reset 后覆盖之前的时间线。
- BEV：标准四通道 CHW uint8；旧三通道仅为历史回放兼容，不填造 visibility。颜色融合、逐通道查看统一由 logger 处理。
- 状态：`log_state` 记录世界 x/y、heading、速度、世界 vx/vy（SI 单位）；遥测速度只记录实际 m/s，FPS 放到 `log_metrics`，不占用速度通道。
- 轨迹：自车 x 前、y 左（m）；新版 `(K,4)` 中后两维为速度（m/s）、相对 heading（rad），像素投影使用前两维。实际 BEV 分辨率/extent 必须传入，默认 128/64。空数组清除旧轨迹；None 表示该调用不更新这一条流，便于只回填真值。
- 采集：当前控制前的 BEV/状态/执行动作与专家导航预览即时记录。未来实际执行的扰动/恢复轨迹在 episode 标签就绪后回填到原观测时间，不将专家预览误记为预测模型输出或真实未来。
- 相机：仅记录实际可用的 RGB；旧数据不含相机，不默认启动模拟器重建。阶段 1 的 `native_rgb` 是顶视语义相机，不是第三人称画面。
- 生命周期：显式 close/context manager 刷新并关闭流，不覆盖已有 `.rrd`。同时保存和显示使用各自的 recording stream，避免 SDK 的后续 sink 覆盖先前 sink。

## 阶段 3～5 的训练与评测要求

未来各阶段开发时，配置 `visualization_interval_steps`、固定 `visualization_sample_ids` 和采样 seed；以 dataclass/YAML 和运行 manifest 保存，不能只留在看板会话里。每次周期抽样输出一个新的 `.rrd`，通过 `set_step` 与 TensorBoard 的训练 global_step 对齐，另记样本 ID、预测 horizon 与真实时间。轨迹用当前自车坐标、只记录有效真值路点；潜变量用 `log_latents(predicted=..., ground_truth=...)` 保存真实 `[horizon, token, channel]` 数组（训练侧 detach 到 CPU）。不将潜变量冒充重建图像。

阶段 3 尚无动作头时，先保存预测/真实潜变量和表示健康度；ADE/FDE 保持未产出。阶段 4～5 加入开环预测/真实轨迹及开闭环指标、遮挡/丢帧条件和消融模型标识。TensorBoard、周期性预测 vs 真值图、每阶段验收视频继续保留，`.rrd` 不能替代这些交付。此处定义标准，不提前实现或启动后续阶段训练。

阶段 3 已获用户确认并开始执行，具体口径见 [STAGE3_PROTOCOL.md](STAGE3_PROTOCOL.md)。每 400 个 optimizer step 及 epoch 末保存四个固定验证样本，各自使用一个独立 `.rrd`，避免相同训练 step 覆盖不同样本；训练 step 与 TensorBoard 对齐，实际仿真时间及 episode/sample/horizon 单独记录。每个目录也保存实际 current/predicted/target 数组和输入/目标 BEV。周期图中的全验证 rank/PCA 参考来自最近完成的验证；最终 PCA 重新使用选中 checkpoint 的对应验证统计。热图与 PCA 均明确标为潜变量而非像素重建。Marimo 增加正式运行选择、健康曲线、原始标量表和周期图选择；缺失轨迹/闭环指标保持空值。

## Marimo 实测报告格式

看板扫描 `artifacts/stage{2,3,4,5}/**/metrics.json` 和 `artifacts/restart_stage{2,3,4,5}/**/metrics.json`。点击“刷新阶段报告”重新读取；表格按模型、消融、split、seed 展示实测值，比较图不跨 seed 擅自求均值。格式如下（示意字段，**不要把示例作为实验报告保存**）：

```json
{
  "visualization_schema_version": 1,
  "spec_version": 2,
  "stage": 4,
  "config": {"path": "configs/run.yaml"},
  "seeds": [7],
  "git_commit": "SOURCE_COMMIT",
  "goal_sha256": "GOAL_SHA256",
  "runs": [{
    "run_id": "RUN_ID",
    "model": "MODEL_NAME",
    "ablation": "ABLATION_NAME",
    "split": "test",
    "seed": 7,
    "metrics": {
      "ade_m": null,
      "fde_m": null,
      "route_completion": null,
      "collision_rate": null,
      "out_of_road_rate": null,
      "lateral_error_m": null,
      "inference_ms": null,
      "effective_rank_ratio": null,
      "latent_mse": null,
      "copy_last_mse": null
    }
  }]
}
```

`visualization_schema_version` 是指标报告格式，`spec_version=2` 指当前 GOAL 规格，不是旧数据的 schema_version=1。指标可以缺省或 null，不能伪造零值。比例用 0–1，距离用米，延迟用毫秒；同一报告须附配置、seed、源码 commit、GOAL hash，并保留原始评测数据、TensorBoard、视频和 `.rrd`。格式/来源缺失、非有限值或 legacy 报告会显示错误而不会混入新版对比。
