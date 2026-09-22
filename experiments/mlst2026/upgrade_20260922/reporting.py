"""Figures and source tables generated only after scientific export validation."""
import csv
import numpy as np
from .protocol import write

def render(out,freeze,results,reduction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    if freeze['config']['study']=='c1':
        for r in results:
            for record in r['result']['records']:rows.append({**r['cell'],**record})
        fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained');ax=axes[0];values=list(reduction['seed_scores'].values());ax.scatter(range(len(values)),values);ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Independent confirmation seed',ylabel='Normalized MAE advantage',title='(a) Held-out directional prediction')
        groups=reduction['estimated'];labels=list(groups);occupancy=[groups[k]['in_domain']/groups[k]['n'] for k in labels]
        axes[1].scatter(range(len(labels)),occupancy,s=8);axes[1].set(xlabel='Dimension/spectrum/method/calibration cell',ylabel='Local-domain fraction',ylim=(-.05,1.05),title='(b) Finite-calibration domain occupancy')
        write(out/'domain_occupancy_index.json',{str(i):key for i,key in enumerate(labels)})
    else:
        for r in results:
            for name,v in r['result']['metrics'].items():rows.append({**r['cell'],'method':name,'amplitude_mse':float(np.mean(v['amplitude_mse'])),'amplitude_rmse':float(np.sqrt(np.mean(v['amplitude_mse']))),'unique_noise_records':r['result']['unique_noise_records'],'scope':r['result']['scope']})
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for i,(name,v) in enumerate(reduction['channels'].items()):
            ax.scatter(v['mean'],i,color='black')
            if v['interval'] is not None:ax.plot(v['interval'],[i,i],color='black')
        ax.set_yticks(range(len(reduction['channels'])),list(reduction['channels']));ax.axvline(0,color='gray',lw=.7);ax.set(xlabel='Weighted GLS − PCA GLS amplitude MSE',ylabel='Run/channel',title='Channel effects; simultaneous intervals where qualified')
    write(out/'source_table.json',rows);fig.savefig(out/'summary.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fields=sorted(set().union(*(set(r) for r in rows)))
    with (out/'source_table.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
