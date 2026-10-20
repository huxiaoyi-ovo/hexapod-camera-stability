"""Create a conceptual Fig 1 with no results. All boxes are research proposal blocks."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

fig, ax = plt.subplots(figsize=(11.6, 5.7), dpi=160)
fig.patch.set_facecolor('white'); ax.set_xlim(0, 11.6); ax.set_ylim(0, 5.7); ax.axis('off')

def box(x,y,w,h,title,desc='',color='#EEF3F7',edge='#C3D0DA'):
    obj=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=.12',facecolor=color,edgecolor=edge,lw=1.25)
    ax.add_patch(obj)
    ax.text(x+w/2,y+h*.68,title,ha='center',va='center',fontsize=12,weight='bold',color='#182733')
    if desc: ax.text(x+w/2,y+h*.29,desc,ha='center',va='center',fontsize=9.5,color='#445865')

def arrow(p,q):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=13,lw=1.6,color='#667C8B',connectionstyle='arc3'))

ax.text(.34,5.32,'Compensation-aware gait adaptation',fontsize=18,weight='bold',color='#172C3A')
ax.text(.36,4.95,'Select gait by predicted residual IMAGE motion after feasible body correction',fontsize=11,color='#4B6575')
box(.4,3.45,2.9,1.03,'Gait candidates','frequency / duty / touchdown')
box(.4,1.70,2.9,1.03,'Terrain & contact','slope / roughness / support phase')
box(4.03,2.54,3.04,1.26,'Camera disturbance','predicted for each candidate',color='#ECF4FB',edge='#9DBBD0')
box(7.91,3.42,3.02,1.04,'Feasible body correction','Roll / Pitch / Z, limited IK',color='#EFF7F5',edge='#95C0B9')
box(7.91,1.58,3.02,1.04,'Residual pixel motion','image Jacobian + constraints',color='#F3EDF9',edge='#B5A2D1')
box(4.11,.42,3.1,.96,'Gait selection + execution','keep progress, clearance, stability',color='#FBF3E9',edge='#D7BA91')
arrow((3.45,3.87),(3.90,3.32));arrow((3.45,2.21),(3.90,2.78))
arrow((7.25,3.20),(7.73,3.75));arrow((7.25,2.83),(7.73,2.15));arrow((9.42,3.26),(9.42,2.78))
arrow((7.90,1.77),(7.36,1.11)); arrow((5.62,1.61),(5.62,1.49))
ax.text(8.26,.54,'not a gimbal',color='#657987',fontsize=9,ha='center')
ax.text(.42,.29,'Conceptual method diagram — no experimental data',fontsize=8.5,color='#6B7F89')
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
out=Path(__file__).parent
fig.savefig(out/'Fig1_concept.png',dpi=220,facecolor='white')
fig.savefig(out/'Fig1_concept.svg',facecolor='white')
print('Saved Fig1 files')
