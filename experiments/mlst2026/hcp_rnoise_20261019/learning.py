"""Array-level learning: whole-block draws, unchanged R1 learners and scores."""
import json
import numpy as np
from experiments.mlst2026.meg_z_20261013.fit import fit_pair, metric, Evaluation, score_fit
from experiments.mlst2026.meg_z_20261013.estimators import arm_estimate
from experiments.mlst2026.meg_z_20261013.protocol import PACKAGE as OLD_PACKAGE
from .protocol import load
from .windows import sample_blocks
from .controls import rng


def fit_cell(roles, model, Sref, N, m, arm, date_index, replicate_index):
    """Pure array API. No real-data CLI is enabled by this kernel."""
    C=model.C; T=model.T
    if arm not in ('reference','lw','kron','isotropic'):
        raise ValueError('unknown arm')
    gen,_=rng(8,100,date_index,replicate_index)
    train,train_indices=sample_blocks(*roles['train'],N,gen)
    clean_train,_=model.draw(N,gen)
    eval_gen,_=rng(8,101,date_index,replicate_index)
    clean_eval,alpha_eval=model.draw(len(roles['eval'][0]),eval_gen)
    if arm=='reference': cov=Sref
    elif arm=='isotropic': cov=np.eye(C*T)
    else:
        cal_gen,_=rng(8,102,date_index,replicate_index)
        calibration,_=sample_blocks(*roles['cal'],m,cal_gen)
        cov,_=arm_estimate(calibration,C,T,arm)
    learners=json.loads((OLD_PACKAGE/'protocol_dev.json').read_text())['learners']
    ev=Evaluation(clean_eval,roles['eval'][0],alpha_eval,model.templates(),model.variances,metric(Sref),model.second_moment())
    # A positive scalar leaves fitted maps/objectives unchanged after the
    # existing optimizer's energy normalization. Normalize W to avoid T25's
    # absolute QR rank cutoff rejecting Tesla-scale reference eigenvectors.
    W=metric(cov);metric_scale=float(np.trace(W)/(C*T));W=W/metric_scale
    fitted=fit_pair(clean_train+train,N,W,C,T,2,3,learners,('hcp-rnoise',date_index,replicate_index,float(model.theta),N))
    return {'date_index':date_index,'replicate':replicate_index,'theta':model.theta,'N':N,'m':m,'arm':arm,
            'sampled_training_indices':train_indices.tolist(),'fit_metric_scalar':metric_scale,**score_fit(fitted,ev)}
