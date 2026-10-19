# mini-wa-jepa

MetaDrive + 自车中心 BEV 的小型 JEPA 世界与动作模型作品集。**目前只实现阶段 1（环境与渲染验证）**；阶段 2–7 尚未开始，必须由用户逐阶段确认。

阶段 1 已验收：100 个 episode、34,494 步、0 次程序崩溃；GPU EGL 下含每步 BEV/RGB 提取 96.74 步/秒，端到端 89.45 步/秒。ROCm 矩阵乘、SDPA MATH、bf16 小算子前反向和 4 个单元测试通过；已保存各 20 张图。内置 IDM 到达率 36%，不作为后续专家成绩。现已停止，等待阶段 2（先 5k 帧）的确认。

用户补充的 [GOAL.md](GOAL.md) 已阅读：公共 Python 包名为 `mini_wa_jepa`；阶段 2 将定义可替换 `BEVProvider`（CHW uint8 + 元数据）和 `SimulatorGroundTruthBEV`，数据管线加入可配置 BEV 漏检/边界抖动/帧间闪烁；模型不依赖真值来源特有字段。后续对照使用相同调参预算，并增加 BEV 噪声下的退化对比。当前诊断图仍是原生 HWC RGB 预览，不能冒充该数据接口。对话中要求的未来/随机掩码、EMA/高斯正则消融仍保留。

## 来源与边界

后续仅借鉴需求中描述的架构思想：在线/EMA 编码器、未来掩码、潜空间 x-prediction flow matching、双流联合注意力与动作路径 stop-gradient。没有使用、复制或 clone WA-JEPA 代码、权重、数据，没有使用 V-JEPA 权重。模型代码将自行实现；当前没有模型或训练结果。

模拟器使用官方 MetaDrive，地图、交通、碰撞和车辆动力学都来自模拟器。阶段 1 的 RGB 仅用于安装验收，**不作为训练输入**。未加入 F1、分割/深度模型或其他感知系统。训练计划限制为 PyTorch ROCm + numpy + dataclass + CSV，注意力只通过 `torch.nn.functional.scaled_dot_product_attention` 调用。

## 安装与复现

已在 Ubuntu 24.04.5、Python 3.12.3、Ryzen 7 7800X3D、RX 7900 XT 20GB 上验证。系统已有 ROCm 7.2.1、内核 amdgpu 与 Mesa 25.2.8；项目脚本不修改系统驱动。

```bash
bash scripts/bootstrap.sh
.venv/bin/python -m scripts.check_rocm
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.diagnose_egl --device 1
.venv/bin/python -m scripts.verify_stage1 \
  --backend egl --egl-device 1 --msaa-limit 8 \
  --episodes 100 --horizon 500 --capture-count 20 \
  --output artifacts/stage1/reproduction
```

MetaDrive 固定为官方 commit `fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1`（0.4.3），资产版本 0.4.3。第一次下载资产较慢。`requirements-runtime.txt` 锁定模拟器及基础 Python 依赖；其中绘图和 IPython 等是 MetaDrive 的依赖链，不是额外训练框架。`.deps/` 和 `.venv/` 不纳入项目 Git。

安装 PyTorch 使用官方 `torch==2.9.1+rocm6.4` wheel，`--no-deps` **有意排除其声明的 `pytorch-triton-rocm` 依赖**，基础依赖另行固定安装。`pip check` 因此会报告这一个缺失依赖，不能声称它全部通过；所需 eager、SDPA MATH、bf16 前反向均以实测为准。不调用 `torch.compile`。环境中未安装 CUDA、flash-attn、xformers、Triton 或 CuPy 包。`torch.cuda` 和设备字符串 `cuda` 是 PyTorch ROCm 共用的 Python API 拼写，`torch.version.cuda is None`，并不引入 NVIDIA CUDA 运行时。系统 ROCm 7.2.1 与 wheel 内置 HIP 6.4 的组合通过了本阶段小测试，尚未验证后续训练。

EGL 设备索引与机器有关：本机 0 是集成显卡、1 是 RX 7900 XT、2 是 Mesa llvmpipe；不要假设其他机器也相同，使用诊断脚本查看实际 renderer。正式 headless 测试清除 `DISPLAY`/`WAYLAND_DISPLAY`，使用 `p3headlessgl`，不启动窗口或 Xvfb。

## 阶段 1 实验与验收

完整数字见 [验收报告](artifacts/stage1/SUMMARY.md)，逐 episode 数据及计时见 [完整报告 JSON](artifacts/stage1/egl_100/report.json)，GPU 数值检查见 [ROCm JSON](artifacts/stage1/rocm.json)。配置、随机种子和运行代码 commit 记录在各报告中。

正式运行：随机地图 3 个生成块，seed 0–99，交通密度 0.1，最多 500 个决策步/episode。每个决策步均提取 BEV 和 RGB。这里使用 **MetaDrive 内置 IDMPolicy** 作为安装测试驱动，不是阶段 2 待实现的纯追踪 + PID 专家。行驶成功、碰撞和超时与程序是否崩溃分别统计；不能将“100 个 episode 不崩”称作“100 次路线完成”。

20 张 [语义 BEV](artifacts/stage1/egl_100/bev/) 和 20 张 [RGB 相机图](artifacts/stage1/egl_100/rgb/) 按 20 个分散种子采样，通常在第 100 步保存，提前结束时保存终止帧。均为原始图像，没有轨迹叠加或感知加工。

![20 张语义 BEV](artifacts/stage1/egl_100/bev_contact_sheet.png)
![20 张 RGB](artifacts/stage1/egl_100/rgb_contact_sheet.png)

### 渲染问题与适配

1. 默认 `use_render=False, image_observation=True` 能输出，但 `glxGraphicsPipe` 仍连接本机 X `:0.0`。清除环境变量不足以阻止驱动默认连接 X；这次结果不算真正 headless。
2. 显式 `p3headlessgl` 原生配置失败：MetaDrive 的引擎 PBR 与 RGBCamera 均硬编码请求 16× MSAA，本机 EGL 浮点帧缓冲仅支持至 8×，导致 tonemapping quad 为 `None`。独立 Panda3D 探针确认 0/4/8 成功，16 失败；源文件没有被修改。
3. 按顺序尝试软件渲染：直接 `LIBGL_ALWAYS_SOFTWARE=1` 与 EGL 自动选硬件冲突，进程以 SIGSEGV 退出；指定软件设备后，Panda3D 的 `framebuffer-software=true` 又错误识别 llvmpipe，初始化失败。最终使用软件设备 2、`framebuffer-hardware=false`、`framebuffer-software=false` 和 4× MSAA，实际 renderer 由报告核实。所有尝试均留记录。
4. 项目提供显式 `--msaa-limit` 图形兼容适配：仅在 Panda3D `FilterManager.createBuffer` 入口复制并限制抗锯齿请求，退出时恢复原方法，报告记录每次调整。不更改 MetaDrive 源码、物理、地图或传感器语义。正式验收使用 GPU EGL + 8×，软件回退另测。
5. 曾误判“未注册 main_camera 图像”会导致 RGB 未挂载，并中断了一次运行。补充源码与位姿实验确认：引擎始终内部创建主相机，原生 reset hook 已正确挂载 RGB；早期同 seed 同步数的 RGB 与显式挂载后的图像逐像素一致（平均差 0）。这不是 MetaDrive 的挂载缺陷。最终仍采用重新从 seed 0 开始的完整 100 episode，保留显式挂载与逐 episode 断言，纠正记录而不隐藏误判。

默认、EGL 失败、软件失败/回退、GPU 正式测试的差异及退出码见 [尝试记录](artifacts/stage1/attempts.json)。软件回退复现命令：

```bash
.venv/bin/python -m scripts.verify_stage1 \
  --backend software --egl-device 2 --msaa-limit 4 \
  --episodes 1 --horizon 120 --capture-count 1 \
  --output artifacts/stage1/software_reproduction
```

### 经源码与运行验证的 MetaDrive 接口

下述文件路径均相对于 `.deps/metadrive/`，可在 [固定版本官方源码](https://github.com/metadriverse/metadrive/tree/fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1/metadrive) 中检查。实际数值快照见正式报告的 `initial_state` / `final_state`。

| 需求 | 实际接口、单位与限制 | 源码 |
|---|---|---|
| 位姿 | `env.agent.position`：世界坐标二维米；`heading_theta`：弧度；heading 向量为 `[cos(theta), sin(theta)]` | `metadrive/base_class/base_object.py` |
| 速度 | `speed`：非负 m/s；`velocity`：世界坐标 m/s 向量；`speed_km_h` 是另一单位，勿混用 | 同上 |
| 当前/下一段参考线 | `agent.navigation.current_ref_lanes` / `next_ref_lanes`（末段可为 `None`），每段可能有多条车道；根据当前 lane/导航选取，不是现成的一整条连续折线 | `component/navigation_module/node_network_navigation.py` |
| 参考点与方向 | `lane.local_coordinates(world_xy) -> (s,l)`；`lane.position(s,l)`；`lane.heading_theta_at(s)`；`lane.length`，直道、弯道实现相同接口。预瞄跨段时需切到下一段，不能仅把 s 一直裁剪到当前段尾 | `component/lane/abs_lane.py` 及具体 lane 类 |
| 导航终点/完成比例 | `navigation.checkpoints` 是路网节点 ID；`final_lane.position(final_lane.length, 0)` 给终点参考位置；`navigation.route_completion` / step info 的 `route_completion` 给完成比例。成功以 `info['arrive_dest']` 为准 | 导航模块及 `envs/metadrive_env.py` |
| 碰撞 | step info：`crash_vehicle`、`crash_object`、`crash_building`、`crash_human`、`crash_sidewalk`，`crash` 是这些事件的逻辑或；agent 有对应属性 | `MetaDriveEnv.done_function` |
| 越界、成功、超时 | `info['out_of_road']`、`info['arrive_dest']`、`info['max_step']`；Gymnasium `step -> obs,reward,terminated,truncated,info`。越界由 MetaDrive 规则综合判定，不能替换成简单车道外距离；检查 `terminated or truncated` 后 reset | `envs/metadrive_env.py`、`envs/base_env.py` |
| 语义俯视图 | `env.render(mode='topdown', window=False, semantic_map=True, target_agent_heading_up=True, center_on_map=False, screen_size=(256,256), film_size=(2048,2048), scaling=4, num_stack=1)`，返回 uint8 HWC **RGB 类别着色图**，自车居中、车头向上。Pygame 在 CPU 绘制，不需 3D GPU 渲染 | `engine/top_down_renderer.py`、`obs/top_down_obs_impl.py` |
| BEV 范围 | `scaling` 单位 pixel/m，实际视野 `screen_size/scaling`，本次目标 64m×64m。原生 renderer 会按整张地图大小下调 scaling；应读取 `env.engine.top_down_renderer.scaling`，不能把请求值当作实际值。旋转含插值，颜色不保证只有离散 palette | 同上 |
| BEV 原生多通道选项 | `TopDownMetaDrive` 的多通道 observation 是道路/导航/历史位置/交通历史等，不等同于要求的三个通道。`resolution_size` / `distance` 配置与 `TopDownRenderer.screen_size` / `scaling` 是不同接口，不应混写 | `envs/top_down_env.py`、`obs/top_down_obs_multi_channel.py` |
| RGB | 注册 `sensors={'rgb_camera': (RGBCamera,256,144)}`，`vehicle_config.image_source='rgb_camera'`，`image_observation=True, image_on_cuda=False`；本项目通过 `get_sensor('rgb_camera').get_image(agent, mode='rgb')` 明确取 RGB。原生 observation 为 BGR、形状 HWC×stack，`norm_pixel` 决定范围 | `component/sensors/base_camera.py`、`obs/image_obs.py` |
| 控制 | 默认连续 action `[steering, throttle_brake]` 各在 [-1,1]；正油门、负刹车，转向乘 `max_steering`（度）。默认 `enable_reverse=False`。未来轨迹跟踪器需要输出这两个量 | `component/vehicle/base_vehicle.py` |
| 动力学配置 | `vehicle_config` 可指定 `vehicle_model`、`max_engine_force`、`max_brake_force`、`max_steering`、`wheel_friction`、`mass`、`enable_reverse`；`agent.get_dynamics_parameters()` 返回实际值，默认车型会对力参数随机采样。发动机/刹车力可动态读取配置，转角/摩擦应在初始化时设定 | `component/vehicle/base_vehicle.py`、`component/pg_space.py` |
| 几何尺寸与轴距 | 标准车型的 `LENGTH/WIDTH/HEIGHT` 和前后轮轴偏移由车型类定义；不能假设任意 `length/width` 配置都会改变标准车型物理尺寸。`top_down_length/width` **仅改变画图**。默认车型轴距 = `FRONT_WHEELBASE + REAR_WHEELBASE` = 2.46894m，运行车型需再核实 | `component/vehicle/vehicle_type.py` |
| 时间步长 | 默认 `physics_world_step_size=0.02` s，`decision_repeat=5`，所以一个 `env.step` 对应 0.1 s。后续轨迹秒数必须按此计算 | `envs/base_env.py` |

**阶段 2 的 BEV 建议：**保留自车中心和车头向上约定，用 `current_map.get_map_features(sample_interval)` 的 lane polygon、道路边界 polyline，以及动态对象的位置/尺寸/方向在自车坐标系栅格化，输出 64×64×3 uint8（可行驶区域、边界/障碍、其他车辆）。原生彩色语义图适合预览，但类别色/旋转插值及自车着色不适合作为精确三通道标签。实现前仍需针对固定源码与小实验验证该转换；本阶段没有提前实现数据集。

## 待用户确认的后续阶段

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| 2 | 纯追踪 + PID 专家、噪声恢复、按地图/种子划分、先采 5k 帧 | 未开始；5k 汇报后再确认扩到 100k+ |
| 3 | 小 ViT + EMA + 确定性潜变量预测、方差/有效秩/复制末帧基线 | 未开始 |
| 4 | x-prediction flow、adaLN、混合未来掩码、4/8 步采样 | 未开始 |
| 5 | 双流世界/轨迹、动作 stop-gradient、风险头、候选重排序 | 未开始 |
| 6 | 独立轨迹跟踪、测试地图闭环指标、3 段录屏 | 未开始 |
| 7 | 全部消融、如实整理 README | 未开始 |

总参数预算后续按 20–40M 设计，模型 bf16 训练稳定性尚未验收。每个实验将记录配置、随机种子、代码 commit 和原始 CSV/JSON；每阶段结束停止。

| 必做基线/消融 | 结果 |
|---|---|
| 匀速轨迹外推 | 未实验 |
| 同输入 CNN + 轨迹回归 BC | 未实验 |
| 世界 + 动作 vs 仅动作损失 | 未实验 |
| flow matching vs 确定性回归 | 未实验 |
| 未来掩码 vs 随机掩码 | 未实验 |
| EMA vs 各向同性高斯正则 | 未实验 |
| 有/无碰撞头重排序 | 未实验 |

## 官方参考资料

- [MetaDrive 安装](https://metadrive-simulator.readthedocs.io/en/latest/install.html)：明确建议源码安装；本机 PyPI 旧包依赖 Gym 0.19，在 Python 3.12 构建失败，因此改用官方源码固定 commit。
- [MetaDrive 观测](https://metadrive-simulator.readthedocs.io/en/latest/obs.html)、[Top-down Renderer](https://metadrive-simulator.readthedocs.io/en/latest/top_down_render.html)：用于定位 API，最终以固定版本源码与实测为准。
- [PyTorch ROCm wheel](https://pytorch.org/get-started/previous-versions/) 与 [HIP API 语义](https://docs.pytorch.org/docs/stable/notes/hip.html)。
- [Panda3D FrameBufferProperties](https://docs.panda3d.org/1.10/python/reference/panda3d.core.FrameBufferProperties)、[EGL 设备选择源码](https://github.com/panda3d/panda3d/blob/v1.10.16/panda/src/egldisplay/eglGraphicsPipe.cxx)。
