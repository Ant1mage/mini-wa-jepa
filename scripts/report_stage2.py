"""Publish measured stage 2 results to Marimo and an explicit acceptance report."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mini_wa_jepa.provenance import provenance,write_json
from configs.stage2 import SPLITS


def load(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/restart_stage2'))
    args=p.parse_args();root=args.root
    data=load(root/'data_stats.json');training=load(root/'baseline/report.json')
    expert=load(root/'expert_test/report.json');baseline=load(root/'baseline_eval/report.json')
    if not all(x.get('completed') for x in (training,expert,baseline)):raise ValueError('Incomplete runs cannot be reported as accepted')
    expected_seeds={SPLITS['test']['seed_start']+i*10000+9000+j
                    for i in range(len(SPLITS['test']['maps'])) for j in range(10)}
    for report in (expert,baseline):
        seeds=[row['seed'] for row in report['episodes']]
        if (report['split']!='test' or report.get('max_steps') is not None
                or len(seeds)!=20 or set(seeds)!=expected_seeds):
            raise ValueError('Acceptance requires the same 20 full held-out episodes for both policies')
    keyboard_path=root/'keyboard_desktop/report.json'
    keyboard=load(keyboard_path) if keyboard_path.exists() else {}
    desktop_path=root/'rerun_desktop_probe.json'
    desktop=load(desktop_path) if desktop_path.exists() else {}
    metrics=dict(visualization_schema_version=1,spec_version=2,stage=2,**provenance(dict(
        training=training['config'],data=data['config'],expert=expert['config'],baseline_eval=baseline['config'])),
        seeds=sorted({row['seed'] for report in (expert,baseline) for row in report['episodes']}),runs=[])
    for report in [expert,baseline]:
        policy=report['policy']
        for episode in report['episodes']:
            metrics['runs'].append(dict(run_id=f"{policy}-{episode['map_spec']}-{episode['seed']}",model=policy,
                ablation='no observation degradation; no action noise',split='test',seed=episode['seed'],metrics=dict(
                    route_completion=episode['route_completion'],collision_rate=float(episode['collision']),
                    out_of_road_rate=float(episode['out_of_road']),lateral_error_m=episode['lateral_error_m'],
                    inference_ms=episode.get('inference_ms'))))
    metrics['runs'].append(dict(run_id='baseline-open-loop',model='ResNet18+MLP',ablation='single BEV + ego motion',
        split='test',seed=training['config']['seed'],metrics={key:baseline['open_loop'][key] for key in ['ade_m','fde_m']}))
    write_json(root/'metrics.json',metrics)
    latency=baseline['latency_ms_p50_p95_p99'];pipeline=baseline['pipeline_ms_p50_p95_p99']
    peak=max(row['peak_allocated_gib'] for row in training['epochs'])
    gates=dict(training_frames=data['splits']['train']['frames']>=100000,
               parameters=15e6<=training['parameter_count']<=30e6,training_vram=peak<=14,
               decision_latency_p95=latency[1]<=25,full_pipeline_p95_40hz=pipeline[1]<=25,
               heldout_episodes=len(baseline['episodes'])==20,
               desktop_keyboard_integration=bool(keyboard.get('passed')),
               rerun_desktop_rendering=bool(desktop.get('rendered')))
    outcome=dict(gates=gates,expert=expert['metrics'],baseline=baseline['metrics'],
        open_loop=baseline['open_loop'],latency_ms_p50_p95_p99=latency,pipeline_ms_p50_p95_p99=pipeline,
        peak_training_allocated_gib=peak,best_epoch=training['best_epoch'],
        keyboard_desktop=keyboard,rerun_desktop=desktop,
        checkpoint_sha256=hashlib.sha256(Path(baseline['checkpoint']).read_bytes()).hexdigest(),
        source_reports={name:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for name,path in dict(data=root/'data_stats.json',training=root/'baseline/report.json',
                expert=root/'expert_test/report.json',baseline=root/'baseline_eval/report.json').items()})
    write_json(root/'acceptance.json',outcome)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    epoch=training['epochs']
    axes[0].plot([r['epoch'] for r in epoch],[r['validation']['ade_m'] for r in epoch],label='validation ADE')
    axes[0].set_xlabel('epoch');axes[0].set_ylabel('ADE (m)');axes[0].legend()
    for report,label in [(expert,'expert'),(baseline,'ResNet18 + MLP')]:
        axes[1].bar(label,report['metrics']['route_completion'])
    axes[1].set_ylabel('route completion');axes[1].set_ylim(0,1.05)
    raw=np.load(root/'baseline_eval/latency_samples_ms.npz')
    axes[2].hist(raw['pipeline_ms'],bins=60,range=(0,max(30,float(np.percentile(raw['pipeline_ms'],99.5)))))
    axes[2].axvline(25,color='red',linestyle='--',label='40 Hz budget');axes[2].set_xlabel('full pipeline (ms)');axes[2].legend()
    fig.tight_layout();fig.savefig(root/'stage2_results.png',dpi=150);plt.close(fig)
    lines=['# 阶段 2：四通道专家数据与行为克隆 Baseline','',
        '本报告只覆盖阶段 2；阶段 3 未启动。所有统计来自原始数据及固定留出地图评测。','',
        f"训练帧 {data['splits']['train']['frames']:,}；验证/测试各 {data['splits']['val']['frames']:,} / {data['splits']['test']['frames']:,}。",
        f"ResNet-18 + MLP {training['parameter_count']:,} 参数，从零训练 {len(epoch)} epochs，按验证 ADE 选择第 {training['best_epoch']} epoch。",'',
        '| 指标 | 专家 | Baseline |','|---|---:|---:|']
    for label,key in [('到达成功率','success_rate'),('路线完成率','route_completion'),('碰撞率','collision_rate'),('越界率','out_of_road_rate'),('平均横向误差 (m)','lateral_error_m')]:
        lines.append(f"| {label} | {expert['metrics'][key]:.4f} | {baseline['metrics'][key]:.4f} |")
    lines.extend(['',f"开环测试 ADE/FDE：{baseline['open_loop']['ade_m']:.3f} / {baseline['open_loop']['fde_m']:.3f} m（完整 2s 轨迹）。",
        f"决策延迟 P50/P95/P99：{latency[0]:.2f} / {latency[1]:.2f} / {latency[2]:.2f} ms。",
        f"完整管线 P50/P95/P99：{pipeline[0]:.2f} / {pipeline[1]:.2f} / {pipeline[2]:.2f} ms；控制周期 20ms（50Hz）。",
        f"训练峰值分配显存 {peak:.3f} GiB；另外保留各 epoch RSS/DRM 与 allocator reserved 统计。",'',
        '## 门槛与结论','',*[f"- {'PASS' if passed else 'FAIL'}: {key}" for key,passed in gates.items()], '',
        '阶段 2 的路线分数是基线锚点，没有预设成功率下限；不能把得到基线分数写成世界模型已验证。',
        '若存在失败门槛或闭环失败，保留原始结果并在进入下一阶段前明确说明。','',
        '## 数据监督统计','',
        '| split | 帧数 | 扰动帧 | 恢复帧 | 参与动作监督的扰动/恢复帧 | 终止失败帧 |',
        '|---|---:|---:|---:|---:|---:|'])
    for split in ('train','val','test'):
        stat=data['splits'][split]
        lines.append(f"| {split} | {stat['frames']} | {stat['noise_frames']} | {stat['recovery_frames']} | {stat['noisy_action_frames']} / {stat['recovery_action_frames']} | {stat['terminal_failure_frames']} |")
    lines.extend(['',
        '本批数据的终止碰撞/全车离道失败帧为 0；扰动和恢复监督已保留，但这批数据没有提供终止失败的正样本，不能据此验证后续安全过滤头或盲区避障。',
        '路线完成率按 MetaDrive 原始 info 统计；到达终点条件可在 route_completion 略小于 1 时满足，因此到达成功率另列。','',
        '## 桌面与交互实测','',
        f"- 独立桌面键盘与 EGL 集成：{'通过' if keyboard.get('passed') else '未通过/未产出'}。使用 XSendEvent 将按键送到键盘子进程回传的实际 SDL 客户端窗口，核验暂停、单步、重置、手动模式、WASD 与退出；这不是人工按键测试。",
        '- 初轮自动探测误选同标题的窗口装饰层，事件未传到 SDL；失败日志与按键 MRE 保留。最终按子进程回传的窗口 ID 定向发送，不向桌面其他窗口注入全局按键。',
        f"- Rerun 原生窗口显示：{'已记录截图' if desktop.get('rendered') else '尚未验证'}；截图自动退出：{'正常' if desktop.get('clean_exit') else '失败，保留原始日志/MRE'}。",
        '- 旧同进程 Panda3D/Pygame 路径仍未修复；本次使用进程隔离，不重启旧路径。',
        '- Rerun 0.22.1 `--screenshot-to` 在生成图像后触发 `Failed to take store hub from the Viewer` panic；已记录可见内容与失败退出，不能据此宣称查看器完全稳定。','',
        '## 数据与评测口径','',
        '- 4×128×128 CHW uint8、64m 方形原生 EGL BEV；visibility=255，未实现阶段 5 退化器。',
        '- 10×4 `(x,y,v,heading)`，间隔 0.2s；0.1s 数据决策，闭环控制 0.02s，物理积分仍为 0.02s。',
        '- 扰动事件概率为每个 eligible 决策 10%，持续 0.4–0.8s，之后至少 0.8s 专家满权重恢复。实际 noisy-frame 比例另见 data_stats.json。',
        '- 动作损失保留扰动/恢复与大偏差片段，仅终止失败观察/结构上无控制或无未来的元素不参与相应监督；世界和安全标签保留。',
        '- 原生默认中心越界已改为中心和四角均离开车道持续 1s。修改前采集的完整片段没有越界事件，因此轨迹不受终止规则修正影响；各来源 hash 与续采链记录在 manifest。',
        '- 全地图几何 hash 与 seed 集合互斥；测试闭环 seed 为每张测试地图保留的 +9000 区间，没有用于训练或模型选择。',
        '- 两个独立 EGL 采集进程按地图固定配额，合并时重新编号 episode，所有分片/计数与标签经校验。',
        '- Rerun 录制、视频、PNG、TensorBoard 分别保留；延迟不含磁盘/可视化编码和 reset，reset 耗时单列。',
        '- 逐步计时去掉每 episode 前 20 步；决策时间也包含状态读取和仅用于评分的导航横向误差计算，模型输入没有导航信息。',
        '- 本次完整重跑 20 episodes；此前主动中断的 15/20 原报告、逐步计时及录制保存在 baseline_eval_interrupted_001，不混入最终统计。',
        '- 只训练一个预先固定 seed 的基线；20 个闭环地图/seed episode 的碰撞率不是无限运行安全保证。','',
        '## 证据','',
        '- [原始数据统计](data_stats.json) / [训练记录](baseline/report.json) / [TensorBoard](baseline/tensorboard)',
        '- [专家闭环](expert_test/report.json) / [基线闭环与开环](baseline_eval/report.json) / [逐步延迟](baseline_eval/latency_samples_ms.npz)',
        '- [Marimo 指标源](metrics.json) / [Marimo 导出](dashboard.html) / [结果图](stage2_results.png) / [阶段 1 验收样本仍保留](../restart_stage1/SUMMARY.md)',
        '- [桌面键盘与 EGL 集成](keyboard_desktop/report.json) / [手动驾驶录制](keyboard_desktop/manual.rrd)',
        '- [Rerun 桌面截图](rerun_desktop_completion.png) / [截图退出失败日志](rerun_desktop_completion.log) / [桌面探测与 MRE 命令](rerun_desktop_probe.json) / [86KB 最小录制](rerun_screenshot_mre/minimal.rrd)',
        '- [本次 26 项测试](tests_completion.log) / [中断现场](baseline_eval_interrupted_001/report.json)',
        '- [旧数据清理清单](old_data_cleanup.json) / [迁移前源代码](pre_migration_source.tar.gz)',''])
    for report in [expert,baseline]:
        for row in report['episodes']:
            if row['video']:lines.append(f"- {report['policy']} / {row['map_spec']} / seed {row['seed']}: [video](../../{row['video']})")
    if keyboard.get('video'):lines.append(f"- 桌面按键驱动 EGL 手动驾驶：[video](../../{keyboard['video']})")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(outcome,indent=2))


if __name__=='__main__':main()
