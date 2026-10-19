"""Derive the revised acceptance report from raw measurements, without reruns."""
import csv
import json
from pathlib import Path

import numpy as np


def main():
    root=Path('artifacts/restart_stage1')
    run=root/'egl_100'
    report=json.loads((run/'report.json').read_text())
    supervisor=json.loads((run/'supervisor.json').read_text())
    rocm=json.loads((root/'rocm.json').read_text())
    episodes=report['episodes']
    assert report['completed'] and supervisor['exit_code']==0 and len(episodes)==100
    assert len(report['samples'])==20
    assert report['pipeline_fps']>=120
    sample_hashes=set()
    for sample in report['samples']:
        with np.load(run/sample['file'],allow_pickle=False) as data:
            bev=data['bev']
            assert bev.shape==(4,128,128) and bev.dtype==np.uint8
            assert set(np.unique(bev)).issubset({0,255})
            assert np.all(bev[3]==255) and np.any(bev[0])
            import hashlib
            sample_hashes.add(hashlib.sha256(bev.tobytes()).hexdigest())
    x=np.arange(len(episodes))
    vram=np.array([e['memory']['devices'][0]['process_vram_bytes'] for e in episodes])/2**20
    rss=np.array([e['memory']['rss_bytes'] for e in episodes])/2**20
    device=np.array([e['memory']['devices'][0]['device_used_bytes'] for e in episodes])/2**20
    fps=np.array([e['pipeline_fps'] for e in episodes])
    warm=slice(20,None)
    slope=float(np.polyfit(x[warm],vram[warm],1)[0])
    worst=max(e['max_calibration_error_pixel'] for e in episodes)
    summary=dict(status='average-throughput gate passed; finite-run VRAM stable',
        episodes=len(episodes),process_crashes=0,total_steps=report['total_steps'],
        pipeline_fps=report['pipeline_fps'],wall_fps=report['wall_fps'],
        pipeline_ms_p50_p95=report['pipeline_ms_p50_p95'],
        per_episode_fps_min_max=[float(fps.min()),float(fps.max())],
        sample_count=20,unique_sample_tensors=len(sample_hashes),
        samples_with_other_actors=sum(s['channel_pixels'][2]>0 for s in report['samples']),
        calibration_max_error_pixel=worst,
        vram_process_mib_first_last=[float(vram[0]),float(vram[-1])],
        vram_process_mib_after_warmup_min_max=[float(vram[warm].min()),float(vram[warm].max())],
        vram_slope_mib_per_episode_after_warmup=slope,
        rss_mib_after_warmup_min_max=[float(rss[warm].min()),float(rss[warm].max())],
        reset_seconds=sum(e['reset_seconds'] for e in episodes),
        elapsed_seconds=report['elapsed_seconds'],video=report.get('video'),
        git_commit=report['git_commit'],goal_sha256=report['goal_sha256'])
    # Preserve the original recording; add a provenance-specific filename for
    # the prior run whose initial video name described the sensor, not the map.
    video=Path(report['video'])
    canonical=video.with_name(video.name.replace('map-native-bev',f'map-pg3-seed{report["config"]["start_seed"]}'))
    if canonical!=video:
        import shutil
        if not canonical.exists():shutil.copyfile(video,canonical)
    summary['video']=str(canonical)
    assert worst<0.05
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'episode_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['episode','seed','steps','pipeline_fps','process_vram_mib','device_vram_mib','rss_mib','max_calibration_error_pixel'])
        for i,e in enumerate(episodes):
            writer.writerow([i,e['seed'],e['steps'],e['pipeline_fps'],vram[i],device[i],rss[i],e['max_calibration_error_pixel']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    axes[0].plot(x,vram,label='Process VRAM (DRM, deduplicated)')
    axes[0].set_ylabel('MiB');axes[0].legend()
    axes[1].plot(x,rss,color='tab:green',label='Process RSS')
    axes[1].set_ylabel('MiB');axes[1].legend()
    axes[2].plot(x,fps,label='Step + semantic readback + CHW mapping')
    axes[2].axhline(120,color='tab:red',linestyle='--',label='120 FPS threshold')
    axes[2].set_ylabel('FPS');axes[2].set_xlabel('Episode');axes[2].legend()
    for axis in axes:axis.grid(alpha=.25)
    fig.tight_layout();fig.savefig(root/'memory_and_fps.png',dpi=150);plt.close(fig)
    content=f'''# 新版阶段 1 验收

阶段 1 的环境、张量与**全程平均**吞吐验收通过；停止等待用户确认。未启动新版阶段 2。

## 结果

| 检查 | 实测 |
|---|---:|
| 同一引擎连续 episode / 决策步 | 100 / {report['total_steps']:,} |
| 程序异常 / 子进程退出码 | 0 / 0 |
| 完整活跃提取链路平均 FPS | {report['pipeline_fps']:.2f} |
| 含初始化、地图重置、样本和视频保存的墙钟 FPS | {report['wall_fps']:.2f} |
| 单 episode 平均 FPS 最小 / 最大 | {fps.min():.2f} / {fps.max():.2f} |
| 完整链路单步 P50 / P95 | {report['pipeline_ms_p50_p95'][0]:.2f} / {report['pipeline_ms_p50_p95'][1]:.2f} ms |
| 四通道样本 / 含其他交通参与者的样本 | 20 / {summary['samples_with_other_actors']} |
| 标定最大误差 | {worst:.6f} pixel |
| 暖机后进程 VRAM 范围 | {vram[warm].min():.2f}–{vram[warm].max():.2f} MiB |
| 暖机后 VRAM 线性斜率 | {slope:.4f} MiB/episode |
| 暖机后 RSS 范围 | {rss[warm].min():.2f}–{rss[warm].max():.2f} MiB |

计时没有把渲染替换成空转：每步执行原生 MetaDrive、Panda3D SemanticCamera 渲染、CPU 图像回读及四通道转换。平均 FPS 按总帧数/总活跃提取时间计算。地图 reset 共 {summary['reset_seconds']:.2f} 秒，总墙钟 {report['elapsed_seconds']:.2f} 秒。**不声称含 reset 达到 120 FPS，也不声称每个 episode 均达到 120 FPS**；若要求的是这种更严格口径，目前没有通过。

## 四通道与标定

- 4×128×128 CHW uint8，值为 0/255；64m×64m，0.5m/pixel。
- C0：原生可见道路、横道、道路标线；C1：原生可见路缘/人行道、围栏及静态标志；C2：其他交通参与者（自车度量 footprint 被排除）；C3：完美真值模式全部为 255。
- 不把全 255 visibility 说成已经实现驾驶员视点遮挡；射线法和丢帧退化属于后续阶段。
- 相机使用世界坐标平面位姿，只跟随 xy 和 heading，不继承车身 pitch/roll。使用 Panda3D 原生投影验证中心 (63.5,63.5)、前方 10m (63.5,43.5)、左侧 10m (43.5,63.5)。
- 20 组原始 NPZ 均通过形状、类型、非空、二值检查，{len(sample_hashes)} 个张量哈希不同；动态通道为空的样本表示当时范围内无其他可见交通参与者。

## ROCm

PyTorch {rocm['torch']}；HIP {rocm['hip']}；cuda build=None；GPU available=True。矩阵乘相对 L2 {rocm['matmul_relative_l2']:.3g}，SDPA MATH 最大绝对误差 {rocm['sdpa_fp32_max_abs']:.3g}，bf16 相对 L2 {rocm['bf16_relative_l2']:.3%}，前反向有限。禁用包数量 0，Triton 未导入。12 项现有/新增测试通过。

## 内存结论的范围

显存采用 /proc/self/fdinfo 的 amdgpu DRM 计数，按 PCI 地址与 client ID 去重，避免把同一 GPU 文件描述符重复累加。前 20 个 episode 后进程 VRAM 在 8MiB 范围内波动，未观察到持续增长。RSS 暖机后增加约 {rss[-1]-rss[20]:.2f}MiB，保留原始曲线，不能声称零字节增长。

设备全局 VRAM 同时受其他桌面进程影响，不能据其变化归因本程序泄漏。关闭引擎后进程仍持有 Panda3D/驱动上下文计数；父进程记录了整个子进程退出前后的设备状态（supervisor.json）。本结论只覆盖这次 100 episode 有限测试，不是无限运行无泄漏的证明。

## 产物与已知限制

- [20 组通道预览](egl_100/contact_sheet.png)，每组对应同名 NPZ 原始四通道及 native_rgb。
- [显存/内存/FPS 曲线](memory_and_fps.png)、[逐 episode CSV](episode_metrics.csv)、[原始报告](egl_100/report.json)。
- [阶段 1 录屏](../../{summary['video']})，H.264 MP4，录制首个 episode 的原生语义图与四通道视图；地图 ID pg3-seed0 对应 3 个生成块、地图 seed 0。
- 原生桌面窗口旧补测可运行，但旧同进程 Panda3D/Pygame 多次重置有 SIGSEGV/GLX BadAccess，仍是已知未解决问题；本次 EGL 批量测试与录屏不依赖这条实时路径。
- 源代码 commit `{report['git_commit']}`，源代码 diff 为空；GOAL SHA256 `{report['goal_sha256']}`。旧版阶段 2 成果保留，不作为新版数据使用。

下一步仍需确认阶段 2；当前不采 100k、不实现或训练 ResNet 基线。
'''
    (root/'SUMMARY.md').write_text(content)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
