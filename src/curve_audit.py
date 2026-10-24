"""Independently audit plotted metrics against UCI and freshly fitted forests.

No original data, CSV, model, split or plot is overwritten. This audit bypasses
the project's model/split builders and recomputes published checkpoints using
RandomForestClassifier.predict/predict_proba, plus a direct log-loss formula.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from .common import write_json

ROOT = Path(__file__).resolve().parents[1]
UCI_URL = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"


def direct_scores(model, features, target):
    probabilities = model.predict_proba(features)
    prediction = model.predict(features)
    np.testing.assert_array_equal(prediction,model.classes_[probabilities.argmax(axis=1)])
    true_column = np.searchsorted(model.classes_,target)
    eps = np.finfo(probabilities.dtype).eps
    true_probability = probabilities[np.arange(len(target)),true_column]
    return {
        "accuracy":float(np.mean(prediction==target)),
        "log_loss":float(-np.log(np.clip(true_probability,eps,1-eps)).mean()),
        "zero_true_probability_samples":int((true_probability==0).sum())}


def audit_curves():
    base=ROOT/"results"
    protected=[p for p in base.rglob('*') if p.is_file() and
               (p.parent==base or p.parent.name in {'models','tree_diagnostics'})]
    protected += list((ROOT/'figures/tree_diagnostics').glob('*.png'))
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    selection=json.loads((base/'selection.json').read_text(encoding='utf-8'))
    cfg=json.loads((base/'run_manifest.json').read_text(encoding='utf-8'))['config']
    split=json.loads((base/'split_indices.json').read_text(encoding='utf-8'))
    published=pd.read_csv(base/'tree_diagnostics/metrics_per_fold.csv')
    published_summary=pd.read_csv(base/'tree_diagnostics/metrics_summary.csv')
    digits=load_digits()
    with urllib.request.urlopen(UCI_URL,timeout=60) as response:
        raw_zip=response.read()
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        member=next(n for n in archive.namelist() if Path(n).name=='optdigits.tes')
        raw_csv=archive.read(member)
    official=np.loadtxt(io.BytesIO(raw_csv),delimiter=',',dtype=np.int64)
    np.testing.assert_array_equal(official[:,:64],digits.data)
    np.testing.assert_array_equal(official[:,64],digits.target)
    dev,test=train_test_split(np.arange(len(digits.target)),test_size=cfg['test_size'],
                              random_state=cfg['split_seed'],stratify=digits.target)
    np.testing.assert_array_equal(dev,split['development'])
    np.testing.assert_array_equal(test,split['test'])
    assert not set(dev)&set(test)
    x,y=digits.data[dev],digits.target[dev]
    folds=list(StratifiedKFold(n_splits=cfg['cv_folds'],shuffle=True,random_state=cfg['cv_seed']).split(x,y))
    params=dict(selection['selected_params'])
    params['max_features']={'sqrt':'sqrt','log2':'log2','half':.5,'all':1.0}[params['max_features']]
    counts=sorted(published.trees.unique().tolist())
    dense,refitted,controls=[],[],[]
    differences=[]
    for fold,(train,validation) in enumerate(folds):
        np.testing.assert_array_equal(dev[train],split['cv_folds_global'][fold]['train'])
        np.testing.assert_array_equal(dev[validation],split['cv_folds_global'][fold]['validation'])
        assert not set(train)&set(validation)
        inputs={'train':train,'validation':validation}
        for count in counts:
            own_params={**params,'n_estimators':int(count)}
            model=RandomForestClassifier(**own_params,random_state=cfg['model_seed'],n_jobs=1).fit(x[train],y[train])
            for name,ids in inputs.items():
                actual=direct_scores(model,x[ids],y[ids])
                saved=published[(published.fold==fold)&(published.trees==count)&(published.split==name)]
                assert len(saved)==1
                for metric in ['accuracy','log_loss']:
                    difference=abs(actual[metric]-float(saved.iloc[0][metric]))
                    differences.append(difference)
                    if difference>1e-12:
                        raise AssertionError(f'Published {metric} mismatch at fold={fold},trees={count},split={name}')
                refitted.append({'fold':fold,'trees':count,'split':name,**actual})
        # Retain the final independently refitted forest for dense 1..100 curves.
        assert counts[-1]==params['n_estimators']
        sums={name:np.zeros((len(ids),len(model.classes_))) for name,ids in inputs.items()}
        for count,tree in enumerate(model.estimators_,start=1):
            for name,ids in inputs.items():
                sums[name]+=tree.predict_proba(x[ids])
                probabilities=sums[name]/count
                true_column=np.searchsorted(model.classes_,y[ids])
                eps=np.finfo(probabilities.dtype).eps
                dense.append({'fold':fold,'trees':count,'split':name,
                              'accuracy':float(np.mean(model.classes_[probabilities.argmax(axis=1)]==y[ids])),
                              'log_loss':float(-np.log(np.clip(probabilities[np.arange(len(ids)),true_column],eps,1-eps)).mean())})
        # Explicit negative control on a COPY of the fold's training labels.
        # Original labels/arrays and all primary results are unchanged.
        shuffled=np.random.default_rng(1000+fold).permutation(y[train])
        control=RandomForestClassifier(**params,random_state=cfg['model_seed'],n_jobs=1).fit(x[train],shuffled)
        controls.append({'fold':fold,'purpose':'negative control with shuffled copied training labels',
                         'validation_accuracy':float(np.mean(control.predict(x[validation])==y[validation]))})
        print(f'Audit fold {fold+1}/{len(folds)}: all {len(counts)} independent refits agree.',flush=True)
    fresh=pd.DataFrame(refitted)
    for r in published_summary.itertuples():
        subset=fresh[(fresh.split==r.split)&(fresh.trees==r.trees)]
        for metric in ['accuracy','log_loss']:
            for stat in ['mean','std']:
                actual=getattr(subset[metric],stat)()
                assert np.isclose(actual,getattr(r,f'{metric}_{stat}'),rtol=0,atol=1e-12)
    dense=pd.DataFrame(dense)
    # Dense prefix scores must also match the independently refitted checkpoints.
    overlap=dense.merge(fresh,on=['fold','trees','split'],suffixes=('_prefix','_refit'))
    for metric in ['accuracy','log_loss']:
        np.testing.assert_allclose(overlap[f'{metric}_prefix'],overlap[f'{metric}_refit'],rtol=0,atol=1e-12)
    validation=dense[dense.split=='validation']
    summary=validation.groupby('trees').agg(accuracy=('accuracy','mean'),log_loss=('log_loss','mean'))
    decreases=int((summary.accuracy.diff().dropna() < -1e-12).sum())
    loss_increases=int((summary.log_loss.diff().dropna() > 1e-12).sum())
    fold_zero=fresh[(fresh.split=='validation')&(fresh.trees==1)]
    control_accuracy=float(np.mean([r['validation_accuracy'] for r in controls]))
    assert control_accuracy<.3,'Unexpectedly high shuffled-label control: investigate leakage.'
    assert before=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    output=base/'curve_audit'
    output.mkdir(exist_ok=True)
    fresh.to_csv(output/'independent_refits.csv',index=False)
    dense.to_csv(output/'all_tree_metrics_per_fold.csv',index=False)
    pd.DataFrame(controls).to_csv(output/'negative_control.csv',index=False)
    findings={
        'status':'passed','official_source':UCI_URL,'official_member':member,
        'official_zip_sha256':hashlib.sha256(raw_zip).hexdigest(),
        'official_member_sha256':hashlib.sha256(raw_csv).hexdigest(),
        'official_data_and_labels_match_sklearn_exactly':True,
        'split_and_folds_recreated_without_project_split_helpers':True,
        'independent_forest_refits':len(counts)*len(folds),
        'published_rows_checked':len(fresh),'max_absolute_metric_difference':max(differences),
        'all_published_means_and_sample_sds_reproduced':True,
        'original_results_and_plots_unchanged':True,'test_evaluations':0,
        'plot_display':{'old_tree_checkpoints':counts,'new_tree_checkpoints':'every integer from 1 to 100',
                        'averaging':'arithmetic mean over 3 fixed CV folds','spline_or_filter_smoothing':False,
                        'old_loss_full_range_hides_small_late_variations':True},
        'validation_accuracy_downward_steps_in_dense_mean':decreases,
        'validation_log_loss_upward_steps_in_dense_mean':loss_increases,
        'validation_samples_with_zero_true_class_probability_at_one_tree':fold_zero.zero_true_probability_samples.tolist(),
        'negative_control_mean_validation_accuracy':control_accuracy,
        'protected_file_sha256_before_and_after':before,
        'caveat':'CV was also used during configuration selection. These are development diagnostics, not an independent new test estimate.'}
    write_json(output/'audit_checks.json',findings)
    draw_raw_curves(dense)
    print(json.dumps({k:v for k,v in findings.items() if k not in {'protected_file_sha256_before_and_after','plot_display'}},indent=2),flush=True)


def draw_raw_curves(dense):
    dest=ROOT/'figures/curve_audit'
    dest.mkdir(parents=True,exist_ok=True)
    colors=['#176B87','#D65A31','#00865A']
    plt.rcParams.update({'font.size':11,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    for ax,name,title in zip(axes,['train','validation'],['Training accuracy','Validation accuracy']):
        values=dense[dense.split==name]
        for fold,color in enumerate(colors):
            points=values[values.fold==fold]
            ax.plot(points.trees,points.accuracy,marker='.',ms=2,lw=.8,alpha=.75,color=color,label=f'Fold {fold+1}')
        mean=values.groupby('trees').accuracy.mean()
        ax.plot(mean.index,mean.values,color='black',lw=1.6,label='Mean of 3 folds')
        ax.set(title=title,xlabel='Number of trees',ylim=(.65,1.01))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='lower right')
    axes[0].set_ylabel('Accuracy')
    fig.suptitle('Raw scores at every tree count; no smoothing or checkpoint omission',fontsize=12)
    fig.tight_layout()
    fig.savefig(dest/'accuracy_all_folds.png')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,minimum,title in zip(axes,[1,20],['Full range: 1 to 100 trees','Zoom: 20 to 100 trees']):
        for name,color in [('train','#176B87'),('validation','#D65A31')]:
            values=dense[(dense.split==name)&(dense.trees>=minimum)]
            for fold in range(3):
                points=values[values.fold==fold]
                ax.plot(points.trees,points.log_loss,color=color,lw=.5,alpha=.3,marker='.',ms=2)
            mean=values.groupby('trees').log_loss.mean()
            ax.plot(mean.index,mean.values,color=color,lw=1.6,label=f'{name.capitalize()} mean')
        ax.set(title=title,xlabel='Number of trees',ylabel='Log loss')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylim(bottom=0)
    fig.suptitle('Thin lines: individual folds; bold lines: arithmetic mean; no smoothing',fontsize=12)
    fig.tight_layout()
    fig.savefig(dest/'log_loss_full_and_zoom.png')
    plt.close(fig)


if __name__=='__main__':
    audit_curves()
