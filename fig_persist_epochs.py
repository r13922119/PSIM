import csv, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
repo = sys.argv[1]
psim={0:1,1:0,2:2,3:0,4:0}; clean={0:0,1:2,2:1,3:2,4:1}
rows={}
def load(f,kf):
    for r in csv.DictReader(open(f"{repo}/logs/{f}")):
        fs=int(re.search(r"seed(\d)_ep20",r["name"]).group(1))
        if fs>2: continue
        k=kf(r["name"])
        if k: rows[(k[0],k[1],fs)]=100*float(r["asr_mean_last5"])
def mk(pat,emap=None):
    def f(n):
        m=re.search(pat,n)
        if not m: return None
        s=int(m.group(1)); return (s, emap[s] if emap else int(m.group(2)))
    return f
load("baseline_20261003_071718_parsed.csv", mk(r"pz-badnet_s(\d)$",psim))
load("persist_cleandev_20261022_025758_parsed.csv", mk(r"pz-cleandev_n1500_s(\d)$",clean))
load("persist_epochs_20261022_163329_parsed.csv", mk(r"pz-n1500_s(\d)_ep(\d)$"))
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#a3a29c"; GRID="#e6e5e0"; S1="#2a78d6"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":GRID,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"text.color":INK})
fig,axes=plt.subplots(1,5,figsize=(13,3.6),sharey=True,facecolor=SURF)
for s,ax in enumerate(axes):
    ax.set_facecolor(SURF)
    for e in range(3):
        v=[rows[(s,e,f)] for f in range(3)]
        ax.scatter([e+0.16]*3,v,s=36,color=MUTED,edgecolor=SURF,linewidth=1.5,zorder=2)
    m=[sum(rows[(s,e,f)] for f in range(3))/3 for e in range(3)]
    ax.plot(range(3),m,color=S1,lw=2,marker="o",ms=7,mec=SURF,mew=1.5,zorder=3)
    for e in range(3): ax.annotate(f"{m[e]:.1f}",(e,m[e]),xytext=(-9,0),textcoords="offset points",ha="right",va="center",fontsize=9,color=INK,zorder=4,bbox=dict(boxstyle="round,pad=0.15",fc=SURF,ec="none"))
    labs=[]
    for e in range(3):
        t=[]
        if psim[s]==e: t.append("PSIM")
        if clean[s]==e: t.append("clean-dev")
        if e==2: t.append("last")
        labs.append(f"ep {e}\n"+"/".join(t))
    ax.set_xticks(range(3)); ax.set_xticklabels(labs,fontsize=8.5); ax.set_xlim(-0.55,2.45)
    ax.set_ylim(0,100); ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"PZ seed {s}",fontsize=10,color=INK)
axes[0].set_ylabel("last-5-epoch ASR after clean LoRA (%)")
fig.suptitle("Persistence by poisoned-pretraining epoch (n = 1,500; grey = finetune seeds 0-2, blue = their mean)",fontsize=10.5,color=INK,x=0.01,ha="left")
fig.tight_layout(rect=(0,0,1,0.93))
fig.savefig("persist_epochs.png",dpi=160,facecolor=SURF)
print("ok")