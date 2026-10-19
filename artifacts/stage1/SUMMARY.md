# 阶段 1 验收报告

阶段状态：环境与渲染验收通过，停止等待用户确认阶段 2。

## 正式 GPU headless 测试

- 完整连续运行 **100/100 episode**、**34,494 决策步**，程序异常/崩溃 **0**。
- seed 0–99；每地图 3 个生成块；交通密度 0.1；每 episode 最多 500 步；MetaDrive 内置 IDM 驱动。
- 路线到达 36/100、碰撞事件 30/100（车辆 30、物体 0、建筑 0）、越界 14/100、到时限 21/100。事件可重叠，不能假设这些计数互斥。
- 行驶表现不是阶段 2 专家验收；这些随机地图也不是已建立的独立测试划分。
- renderer：`Radeon RX 7900 XT (radeonsi, navi31, LLVM 20.1.2, DRM 3.57, 6.8.0-147-generic)`；`4.6 (Compatibility Profile) Mesa 25.2.8-0ubuntu0.24.04.4`；`eglGraphicsPipe`；DISPLAY/WAYLAND_DISPLAY 均清除；EGL device 1；8× MSAA 适配生效。

| 测量口径 | 步/秒 |
|---|---:|
| env.step（包括内建 RGB observation 渲染/回读） | 155.55 |
| env.step + 每步 BEV 与 RGB 显式提取 | 96.74 |
| 端到端（含首次引擎初始化、地图 reset、图像检查/保存与关闭） | 89.45 |

总耗时 385.63 秒。接触表生成不计入计时。训练和后续规模采集速度尚未测量。

## 图像验收

- BEV 20 张，256×256×3，uint8；RGB 20 张，144×256×3，uint8。
- 两类各 20 张像素哈希均不同，形状/类型/非空检查通过；全部保存帧的 RGB camera parent 均为自车。
- BEV 保存帧实际 scaling 范围 4.0000–4.0000 pixel/m；4 pixel/m 对应 64m×64m。
- BEV 是 MetaDrive 原生类别着色预览，不是阶段 2 三通道二值栅格。
- [BEV 总览](egl_100/bev_contact_sheet.png)、[RGB 总览](egl_100/rgb_contact_sheet.png)、[原始逐 episode JSON](egl_100/report.json)。

## ROCm 与数值验收

| 项目 | 实测 |
|---|---|
| PyTorch | 2.9.1+rocm6.4 |
| torch.version.hip | 6.4.43484-123eb5128 |
| torch.version.cuda | None |
| torch.cuda.is_available() | True |
| GPU | Radeon RX 7900 XT |
| SDPA 后端 | MATH |
| 矩阵乘相对 L2 误差（对 CPU FP64） | 4.05e-07 |
| SDPA FP32 最大绝对误差 | 3.58e-07 |
| bf16 相对 L2 误差（对 FP32） | 0.320% |
| bf16 前反向有限值 | True |
| 禁用依赖安装数量 / Triton 被导入 | 0 / False |

只证明这些小算子的数值可用，不等于后续 JEPA bf16 长时间训练已稳定。首次矩阵乘含初始化，不作为稳态性能基准。

## 问题、回退与修正

- PyPI 旧 MetaDrive 依赖 Gym 0.19，Python 3.12 构建失败；按官方建议改用固定官方源码。
- 默认 GLX 能输出但连接 X，不计入 headless；原生 EGL 的 16× MSAA 超过驱动 8× 上限，失败记录保留。
- 软件首试 SIGSEGV；强制 software framebuffer 又误识别 llvmpipe；移除该标志并选设备 2、限制 4× 后可运行。
- 曾误判未注册 main_camera 图像会影响 RGB 挂载并中断一次运行。补充实测确认内部主相机仍存在，原生 hook 已正确挂载，旧/新同帧逐像素差为 0；早期判断有误。最终采用重新从 seed 0 开始的完整 100 episode，并保留显式挂载与断言。
- 软件 EGL llvmpipe + 4× MSAA：1 episode，120 步；step-only 9.66 步/s，含双图提取 9.29 步/s，端到端 4.41 步/s。仅作回退 smoke test，未跑 100 episode。
- 单元测试 4/4 通过。`pip check` 唯一缺项是按用户约束有意省略的 pytorch-triton-rocm；不隐藏此元数据限制。
- 正式运行代码 commit：`a49856bc060360f8ee6cfaa43dbcf4c1c88c7d3a`；MetaDrive：`fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1`。报告/文档之后独立提交，避免自引用 commit。

## 建议的下一步（尚未执行）

用户确认后进入阶段 2：先核实地图几何栅格化，制作 64×64 三通道 uint8 自车 BEV；实现纯追踪 + PID 专家及噪声恢复；固定地图/种子划分；只采 5k 帧，报告专家完成率、好坏比例、轨迹分布与叠加回放后再次停下。
