# Stage 7 独立复查记录

Relay/Gemini 实现到文件、Codex执行与复查；用户随后授权 Codex 自行完成阶段7。以下缺陷均保留原失败产物，不将工具 exit0 当作完成。

| 问题 | 具体证据/影响 | 处理 |
|---|---|---|
| Gemini 终端调用被拒绝但返回空 SUCCESS | native CLI `accept-edits` / `RunCommand` soft deny；`pwd` 最小探针也 approved=false | 记录原日志，不验收空结果；用户自行排查 Relay，Codex恢复执行 |
| 配置引用不存在的阶段6检查点 | 原路径无法加载 | 使用已验收 conditioned_20261019/best.pt，保留旧模型哈希 |
| 风险预采集没有越界阳性 | preflight_001 0/26 off-road | 只用 train-only 路沿前缀探针；冻结13条显式轨迹命令，右转预采集6/26阳性，无执行期隐藏方向盘干预 |
| Flow 掩码与无效目标污染 | 无效坐标可能参与插值、损失分母不匹配4个通道 | 插值前替换无效目标，按实际有效坐标计数，针对性回归测试 |
| CPU smoke 调用 torch.nullcontext | PyTorch无此成员，阻止非 autocast 测试 | 改 contextlib.nullcontext，原失败保留 |
| collector 保存了尚未计算的零 ego 历史 | context['ego']全0，但实测速度约5–6m/s | 正式风险训练前发现；192上下文独立完整前缀重放，核对 BEV/全场景/状态，另存5观测状态恢复4帧速度/转向；原数据不改 |
| 未执行候选的未来真值不可能获得 | 部署中会继续重新规划，实际未来不能当作其他候选的反事实结果 | Rerun target可缺失；闭环实际 latent 明确标成 subsequent replanning；专家 ADE 与执行差异单列 |
| 非反事实的规则 STOP 容易误计为生成成功 | 几何 smoke fallback=1.0 | selected_index=-1、来源为规则刹停，所有候选/评分/拒绝原因同屏显示 |
| 初始化校准影响可复现性 | 测试校准不应消耗正式初始化/优化器/RNG | deepcopy + fork_rng，校准前后初始化哈希核对；world/zero 同初始化/数据顺序 |
| 未使用的 config 架构字段 | flow_num_layers/num_heads 没有实际构造 Transformer | 如实记录：实际为时序线性融合＋MLP；以检查点 model_arch/冻结源码和实际参数计数为准 |
| 浏览器测试的伪失败 | Marimo shadow DOM 文本不在 body.innerText；Radix tab需真实 pointer 事件；测试脚本一次转义错误 | 保留4次失败脚本/JSON/截图；用CDP实际鼠标与键盘、shadow-root DOM断言，滑块图像变化与评分表/回放验证通过 |

实验失败不等同实现缺陷。Flow 的精度/运动学门槛已经失败，没有使用测试结果重训、改采样器或调评分规则。风险标签目前的原生 crash 类型只有路沿接触，不能据此推断车辆动态避碰。最终科学结论以阶段报告为准。
