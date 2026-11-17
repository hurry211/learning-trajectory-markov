"""Fixed Q1 parameter replay plus IID/second-/third-order observable baselines.

All extra baselines are reported; none is selected on 2021 results. These are
post-hoc robustness checks on an already used temporal test, not a fresh test.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "paper_output/code/modeling"))
import run_q1_full as q1
from sequence_models import TransparentObservationEncoder


def fit_order(sequences, order, alpha=.5):
    counts=np.full((4**order,4),alpha,dtype=float)
    for seq in sequences:
        if order:
            windows=np.lib.stride_tricks.sliding_window_view(seq,order+1)
            context=windows[:,:-1] @ (4**np.arange(order-1,-1,-1))
            target=windows[:,-1]
        else:
            context=np.zeros(len(seq),dtype=int)
            target=seq
        np.add.at(counts,(context,target),1.)
    return counts/counts.sum(1,keepdims=True)


def score_order(seq, probabilities, order, landmark=42):
    target=seq[landmark:]
    if order:
        windows=np.lib.stride_tricks.sliding_window_view(seq,order+1)
        context=(windows[:,:-1] @ (4**np.arange(order-1,-1,-1)))[landmark-order:]
    else:
        context=np.zeros(len(target),dtype=int)
    return np.log(probabilities[context,target]).sum()


def main():
    print("Q1: replaying archived fixed HMM; fitting observable history baselines",flush=True)
    panel=q1.load_panel()
    development=panel.loc[panel.cohort.isin(["1819","1920"])]
    encoder=TransparentObservationEncoder(q1.encoder_config()).fit(development)
    train=q1.make_bundle(development,encoder,"development")
    outer=q1.make_bundle(panel.loc[panel.cohort.eq("2021")],encoder,"2021")
    params=json.loads((ROOT / "paper_output/results/q1_full/q1_full_model_parameters.json").read_text(encoding="utf-8"))
    hp=params["categorical_hmm_final"]
    model=CategoricalHMM(n_components=hp["k"],n_features=4,init_params="",params="")
    model.startprob_=np.array(hp["start_probability"])
    model.transmat_=np.array(hp["transition_probability"])
    model.emissionprob_=np.array(hp["emission_probability"])
    scores=q1.conditional_hmm_sequence_scores(model,outer,42).rename(columns={"conditional_loglik":"hmm"})
    scores["sequence_hash"]=scores.registration_id.map(lambda v:hashlib.sha256(f"q1-full|{v}".encode()).hexdigest()[:16])
    archived_path=ROOT / "paper_output/tables/q1_full/q1_full_outer_sequence_scores.csv"
    max_error=None
    if archived_path.exists():
        archived=pd.read_csv(archived_path)
        check=scores.merge(archived,on="sequence_hash",validate="one_to_one")
        max_error=float(abs(check.hmm-check.conditional_loglik_hmm).max())
        if max_error>1e-7:
            raise AssertionError(f"fixed HMM replay differs from archived scores: {max_error}")
    order_params={order:fit_order(train.observations,order) for order in [0,1,2,3]}
    saved_mc=np.array(params["observable_markov_chain_final"]["transition_probability"])
    mc_error=float(abs(saved_mc-order_params[1]).max())
    if mc_error>1e-10:
        raise AssertionError(f"first-order baseline mismatch: {mc_error}")
    for order,p in order_params.items():
        values=[score_order(seq,p,order) for seq in outer.observations]
        if len(values)!=len(scores):
            raise AssertionError("bundle score alignment mismatch")
        scores[f"order_{order}"]=values
    scores.to_parquet(ROOT / "private/q1_baseline_sequence_scores.parquet",index=False)
    rows=[]
    lengths=scores.scored_observations.to_numpy()
    for order in order_params:
        delta=(scores.hmm-scores[f"order_{order}"]).to_numpy()
        rng=np.random.default_rng(20261116)
        boot=np.empty(5000)
        for b in range(5000):
            ix=rng.integers(len(scores),size=len(scores))
            boot[b]=delta[ix].sum()/lengths[ix].sum()
        rows.append({"observable_order":order,"baseline":"IID" if order==0 else f"observable-MC-order-{order}",
            "baseline_loglik_per_observation":float(scores[f"order_{order}"].sum()/lengths.sum()),
            "hmm_loglik_per_observation":float(scores.hmm.sum()/lengths.sum()),
            "hmm_minus_baseline_nat_per_observation":float(delta.sum()/lengths.sum()),
            "ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975)),
            "bootstrap_unit":"scoped registration, not linkable cross-course individual",
            "fixed_hmm_replay_max_abs_loglik_error":max_error,"mc1_parameter_replay_max_abs_error":mc_error,
            "replications":5000,"dirichlet_alpha":.5,"outer_status":"already-used temporal test; post-hoc diagnostic"})
    pd.DataFrame(rows).to_csv(ROOT / "results/q1_stronger_baselines.csv",index=False)
    print(pd.DataFrame(rows)[["baseline","hmm_minus_baseline_nat_per_observation","ci_low","ci_high"]].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
