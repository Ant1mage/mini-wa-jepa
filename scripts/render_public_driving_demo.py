"""把已记录的真实驾驶片段转成中文讲解视频；不运行模型或模拟器。"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mini_wa_jepa.provenance import write_json


OUTPUT = Path('artifacts/public_driving_demo_002')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FPS = 10


@lru_cache(maxsize=None)
def font(size):
    return ImageFont.truetype(FONT, size)


def text(draw, xy, value, size=22, color='#e2e8f0'):
    draw.text(xy, value, font=font(size), fill=color)


def card(title, lines):
    canvas=Image.new('RGB',(1280,720),'#0f172a');draw=ImageDraw.Draw(canvas)
    text(draw,(70,95),title,38,'#f8fafc')
    for index,line in enumerate(lines):
        text(draw,(70,205+index*62),line,25)
    text(draw,(70,650),'来自已有真实驾驶记录 · 无需打开开发工具',18,'#94a3b8')
    return np.asarray(canvas)


def render_frame(original, state, completion, sim_time, occluded, plan, target, valid):
    canvas=Image.new('RGB',(1280,720),'#0f172a');draw=ImageDraw.Draw(canvas)
    text(draw,(24,15),'车是怎样一边预测、一边往前开的？',32,'#f8fafc')
    badge='第二段：前方一块信息被遮住' if occluded else '第一段：信息完整'
    text(draw,(24,64),badge,22,'#fbbf24' if occluded else '#86efac')
    text(draw,(935,67),f'模拟时间 {sim_time:4.1f} 秒 · 1倍速',18,'#94a3b8')
    text(draw,(24,103),'现场画面：供人观察，模型不直接看这张图',18,'#cbd5e1')
    text(draw,(804,103),'模型看到的简化地图与路线',18,'#cbd5e1')
    # 原始视频由 RerunLogger 同源记录生成；只裁掉技术性标题与状态栏。
    camera=original.crop((0,40,720,445)).resize((736,414),Image.Resampling.LANCZOS)
    bev=original.crop((748,40,1260,552)).resize((420,420),Image.Resampling.NEAREST)
    canvas.paste(camera,(24,130));canvas.paste(bev,(804,130))
    draw=ImageDraw.Draw(canvas)
    # 原视频两线几乎重合；用同一真实坐标重画“粗蓝线 + 细橙虚线”，不偏移位置。
    from mini_wa_jepa.geometry import ego_to_pixels
    def screen_points(trajectory):
        return (ego_to_pixels(trajectory[:,:2],128,64)+.5)*(420/128)+np.array([804,130])
    predicted=screen_points(plan);realized=screen_points(target[valid])
    draw.line([tuple(p) for p in predicted],fill=(40,220,250),width=6)
    for point in predicted[::2]:
        draw.ellipse((point[0]-3,point[1]-3,point[0]+3,point[1]+3),fill=(40,220,250))
    for start,end in zip(realized[:-1],realized[1:]):
        length=float(np.linalg.norm(end-start))
        for offset in np.arange(0,length,7):
            a=start+(end-start)*(offset/max(length,1e-8))
            b=start+(end-start)*(min(offset+4,length)/max(length,1e-8))
            draw.line([tuple(a),tuple(b)],fill='orange',width=2)
    text(draw,(24,560),'反复进行：看当前场景 → 预测路线 → 执行一小步 → 再预测',19,'#e2e8f0')
    draw.line((804,570,832,570),fill=(40,220,250),width=4)
    text(draw,(840,555),'蓝线：接下来想走的路线',17)
    draw.line((804,596,832,596),fill='orange',width=4)
    text(draw,(840,581),'橙虚线：后来实际走出的路线',17)
    state_label='输入：前方局部遮挡' if occluded else '输入：完整地图'
    fields=[(state_label,'#fbbf24' if occluded else '#86efac'),
            (f'当前车速：{float(state[3])*3.6:.1f} 公里/小时','#e2e8f0'),
            (f'路线进度：{completion*100:.1f}%','#93c5fd')]
    for index,(label,color) in enumerate(fields):
        left=24+index*416
        draw.rounded_rectangle((left,620,left+400,672),radius=10,fill='#1e293b')
        text(draw,(left+14,632),label,21,color)
    text(draw,(24,689),'当前版本每轮预测一条路线；橙线是回放时追加的真实对照。此片不含多候选选择。',16,'#94a3b8')
    return np.asarray(canvas)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);video=OUTPUT/'driving_demo.mp4'
    if video.exists():raise FileExistsError(video)
    output=imageio_ffmpeg.write_frames(str(video),(1280,720),fps=FPS,codec='libx264',quality=8,
                                     macro_block_size=16,output_params=['-movflags','+faststart'])
    output.send(None);sources=[];frames=0
    def emit(frame, count=1):
        nonlocal frames
        for _ in range(count):output.send(np.ascontiguousarray(frame));frames+=1
    try:
        emit(card('看懂这段视频，只需要认识两条线',[
            '左边：真实模拟器里的车和道路。',
            '右边：模型看到的信息，以及它预测的路线。',
            '蓝线是准备走的路；橙线是后来实际走出的路。',
            '先看信息完整，再看前方一块信息被遮住。']),2*FPS)
        for name,occluded in [('joint__clean',False),('joint__occlusion_l3',True)]:
            folder=Path('artifacts/restart_stage5/closed_loop')/name
            report=json.loads((folder/'report.json').read_text())
            episode=next(row for row in report['episodes'] if row['evidence'])
            evidence=episode['evidence'];actual=np.load(evidence['raw'])
            rollout=np.load(folder/f'episode_{episode["seed"]}.npz')
            states=actual['states'];steps=actual['steps'];route=rollout['route_completion']
            # 两段使用相同种子、相同12~22秒模拟时间窗，按各自真实记录对齐。
            start,stop=12.,22.
            reader=imageio_ffmpeg.read_frames(evidence['video'],pix_fmt='rgb24')
            metadata=next(reader);source_fps=metadata['fps']
            width,height=metadata['size'];written=0
            try:
                for index,buffer in enumerate(reader):
                    if index>=len(steps):raise ValueError('Video/raw recording frame mismatch')
                    step=int(steps[index]);sim_time=step*report['control_dt_s']
                    if sim_time<start:continue
                    if sim_time>=stop:break
                    if abs(index/source_fps-sim_time)>1e-6:raise ValueError('Recording timestamp mismatch')
                    source=Image.fromarray(np.frombuffer(buffer,np.uint8).reshape(height,width,3))
                    frame=render_frame(source,states[step],float(route[step]),sim_time,occluded,
                                       actual['prediction'][index],actual['target'][index],actual['valid'][index])
                    emit(frame);written+=1
                    if written==40:
                        Image.fromarray(frame).save(OUTPUT/('occluded.png' if occluded else 'clean.png'))
            finally:reader.close()
            if written!=100:raise ValueError(f'Expected 100 actual recorded frames, got {written}')
            sources.append(dict(condition=report['condition'],seed=episode['seed'],model=report['policy'],
                                source_video=evidence['video'],source_rrd=evidence['rerun'],
                                source_raw=evidence['raw'],source_sha256=hashlib.sha256(Path(evidence['video']).read_bytes()).hexdigest(),
                                sim_time_window_s=[start,stop],frames=written,arrived_in_full_episode=episode['arrive_dest']))
        emit(card('现在能看到什么，下一步再增加什么',[
            '已实现：模型看当前信息，预测一条路线，车辆持续执行。',
            '本片两段都来自真实记录，未重新训练或补造未来画面。',
            '待实现：同时生成多条候选，展示评分、选择和再规划。',
            '两段仅是固定种子的一次演示，不代表所有场景。']),3*FPS)
    finally:output.close()
    write_json(OUTPUT/'manifest.json',dict(type='public_explainer_of_existing_measured_rollouts',
        video=str(video),fps=FPS,frames=frames,duration_s=frames/FPS,sources=sources,
        candidate_selection_present=False,model_rerun=False,new_simulation=False,
        processing='Cropped original camera/BEV panels; redrew recorded prediction/realized trajectories at identical coordinates with distinct line styles; added Chinese explanations and actual speed/route telemetry.'))
    print(video)


if __name__=='__main__':main()
