"""Make a standalone scientific summary; panels use different labeled metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();s=json.loads((a.results/'summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(11.4,4.0),layout='constrained')
    colors=['#3274a1','#e1812c']
    series=[([.6435,s['esc50']['accuracy']],['HF helper','Training formula'],'ESC-50 classification\n2,000 clips / 5 folds','Accuracy'),
            ([s['qa'][m]['mmau_sound']['accuracy_or_original_answer_retention'] for m in ('qwen','moss')],['Qwen2-Audio','MOSS-8B'],'MMAU sound reasoning\n333 questions / 311 audio files','Accuracy'),
            ([next(r['harmonic_auc_source_target_pauc'] for r in s['asd2026']['aggregate'] if r['variant']=='beam_rdp4' and r['model']==m) for m in ('BEATs_iter3','dcase2026_nabeats_seed0')],['BEATs','NA-BEATs'],'DCASE2026 noise-aware ASD\n7,000 train / 1,400 test','Harmonic AUC / pAUC')]
    for ax,(values,labels,title,ylabel) in zip(axes,series):
        ax.bar([0,1],values,color=colors,width=.6)
        ax.set(xticks=[0,1],xticklabels=labels,ylim=(0,1.08),title=title,ylabel=ylabel)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        for x,v in enumerate(values):ax.text(x,v+.02,f'{100*v:.2f}%',ha='center',fontsize=11)
    for i,m in enumerate(('qwen','moss')):
        r=s['qa'][m]['mmau_sound'];v=r['accuracy_or_original_answer_retention'];lo,hi=r['audio_cluster_bootstrap95']
        axes[1].errorbar(i,v,yerr=[[v-lo],[hi-v]],color='black',capsize=3,fmt='none')
        axes[1].texts[i].set_y(hi+.02)
    u=s['asd2026']['uncertainty']
    for i,key in enumerate(('beats_ci95','nabeats_ci95')):
        v=series[2][0][i];lo,hi=u[key]
        axes[2].errorbar(i,v,yerr=[[max(0,v-lo)],[max(0,hi-v)]],color='black',capsize=3,fmt='none')
        axes[2].texts[i].set_y(hi+.02)
    fig.suptitle('Released-checkpoint evaluations — task-specific metrics',fontsize=14)
    fig.savefig(a.results/'evaluation_summary.png',dpi=180)
    fig.savefig(a.results/'evaluation_summary.pdf')
    plt.close(fig)


if __name__=='__main__':main()
