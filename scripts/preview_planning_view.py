"""只生成规划主视图的布局示意，不调用模型，不伪造推理、评分或仿真。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from chapters.learning import figure  # 注册系统中文字体；无训练/模拟器副作用。


def main():
    output=Path('artifacts/planning_view_mockup_001');output.mkdir(parents=True,exist_ok=True)
    t=np.arange(1,11)*.2
    # 这些曲线仅用于演示 UI；不是训练模型的候选输出。
    paths={'A':(10*t,np.zeros_like(t)),
           'B':(10*t,3*np.sin(np.pi*t/2)**2),
           'C':(6*(1-np.exp(-1.2*t)),np.zeros_like(t))}
    fig=plt.figure(figsize=(14,8),layout='constrained',facecolor='#f8fafc')
    grid=fig.add_gridspec(1,2,width_ratios=(1.05,1))
    ax=fig.add_subplot(grid[0]);info=fig.add_subplot(grid[1]);info.axis('off')
    ax.set_facecolor('#15202c')
    ax.add_patch(Rectangle((-6,-4),12,33,color='#32414d'))
    for lateral in (-6,6):ax.plot([lateral,lateral],[-4,29],color='#94a3b8',lw=2)
    ax.plot([0,0],[-4,29],color='#94a3b8',ls='--',alpha=.35)
    ax.add_patch(Rectangle((-1,10),2,4,color='#ef4444'))
    ax.text(1.4,12,'前方车辆\n（示意）',color='#fecaca',fontsize=10,va='center')
    colors={'A':'#a78bfa','B':'#22c55e','C':'#fbbf24'}
    labels={'A':'A：保持当前路线','B':'B：向左绕行 · 示例选中','C':'C：减速'}
    for key,(forward,left) in paths.items():
        ax.plot(-left,forward,'o-',color=colors[key],lw=4 if key=='B' else 1.8,
                ms=4,label=labels[key],alpha=1 if key=='B' else .65)
        ax.text(-left[-1]-.25,forward[-1]+.65,key,color=colors[key],fontsize=13,weight='bold')
    ax.add_patch(Rectangle((-.9,-2),1.8,4,facecolor='#e2e8f0',edgecolor='white'))
    ax.annotate('',xy=(0,2.4),xytext=(0,.3),arrowprops=dict(arrowstyle='->',color='#0f172a',lw=2))
    ax.text(1.1,0,'当前自车',color='white',va='center')
    # 虚线车框只是选中曲线上的示意位置，不代表发生过真实执行。
    pos=2;next_forward=paths['B'][0][pos];next_left=paths['B'][1][pos]
    ax.add_patch(Rectangle((-next_left-.9,next_forward-2),1.8,4,fill=False,edgecolor='#22c55e',lw=2,ls='--'))
    ax.text(-next_left-1.3,next_forward,'下一步位置\n（示意）',ha='right',va='center',color='#bbf7d0',fontsize=9)
    ax.set(xlim=(-9,9),ylim=(-4,29),xlabel='横向位置（米，左侧对应自车左方）',ylabel='前向位置（米）',
           title='一张 BEV 展示全部候选 + 高亮所选轨迹')
    ax.set_aspect('equal');ax.legend(loc='upper left',facecolor='#0f172a',labelcolor='white',fontsize=9)
    info.text(0,1,'本轮候选与选择',fontsize=19,weight='bold',va='top',color='#0f172a')
    table=info.table(cellText=[['A','保持路线','仍然显示'],['B','向左绕行','加粗高亮'],['C','减速','仍然显示']],
                     colLabels=['候选 ID','方案','画面状态'],cellLoc='left',bbox=[0,.68,1,.23])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for col in range(3):table[(2,col)].set_facecolor('#dcfce7')
    info.text(0,.61,'真实运行时，这里显示',fontsize=15,weight='bold',color='#0f172a')
    info.text(0,.56,'各候选的未来预测与实际评分分量\n选中的 ID、选择来源和原因\n执行动作与随后实际走出的轨迹',
              fontsize=12,va='top',linespacing=1.8,color='#334155')
    info.text(0,.34,'接下来继续播放',fontsize=15,weight='bold',color='#0f172a')
    info.text(0,.29,'车辆执行选中的方案\n进入下一帧，再生成全部候选\n重新评分、选择、执行……',
              fontsize=12,va='top',linespacing=1.8,color='#334155')
    info.text(0,.05,'界面示意：候选手工构造，B 为预设高亮。\n没有模型评分或真实仿真执行结果。',
              fontsize=11,color='#b45309',va='bottom',bbox=dict(boxstyle='round,pad=.7',fc='#fff7ed',ec='#fed7aa'))
    fig.suptitle('你要看到的是：所有候选 → 选择一条 → 执行 → 下一轮再规划',fontsize=18,weight='bold')
    fig.savefig(output/'main.png',dpi=120);plt.close(fig)
    print(output/'main.png')


if __name__=='__main__':main()
