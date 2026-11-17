"""Same-model DP/MC, paired MCSE, profile/threshold/kernel sensitivities.

Observed-future covariate paths are retrospective conditional scenarios, NOT
real-time forecasts. Frozen-history is an explicit alternative, not an intervention.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_output/code/modeling"))
import run_q34_scenarios as q
from q34_helpers import FeatureTransformer, ObservableNHMC, categorical_draw, max_true_run, epsilon_nondominated
from experiments.risk_core import run_risk, row_tv_bound, paired_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=100000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.paths < 30000:
        raise ValueError("at least 30000 paths required for the archived-prefix check")
    print("Q34: reconstructing strictly lagged features", flush=True)
    panel = q.load_panel()
    panel, checks = q.reconstruct_strict_history(panel)
    panel, encoding = q.fit_observation_encoder(panel)
    checkpoints = ROOT / "private/q34_fit.joblib"
    records = q.build_sequence_records(panel, "1819") + q.build_sequence_records(panel, "1920")
    transformer = FeatureTransformer(q.RAW_FEATURE_NAMES, q.CONTINUOUS_INDICES, q.MISSING_VALUE_INDICES)
    transformer.fit(np.concatenate([r["raw_transition"] for r in records]))
    obs, x = q.design_sequences(records, transformer)
    if checkpoints.exists():
        model = joblib.load(checkpoints)
    else:
        print("Q34: refitting the archived ridge=1 observable NHMC, no outer model selection", flush=True)
        model = ObservableNHMC.fit(obs, x, n_states=4, ridge=1.)
        joblib.dump(model, checkpoints)
    if not all(a["success"] for a in model.optimization_audit):
        raise RuntimeError("NHMC optimization failed; do not report scenario results")
    landmarks = q.build_landmark_records(panel)
    base, calibrator, calibration, calibration_meta = q.fit_completion_calibration(landmarks)
    calibration.to_csv(ROOT / "results/q34_recreated_calibration.csv", index=False)
    outer = landmarks.loc[landmarks.cohort.eq("2021")].reset_index(drop=True)
    scenarios, support = q.scenario_contract(panel.loc[panel.cohort.isin(["1819", "1920"])], outer)
    support.to_csv(ROOT / "results/q34_scenario_support.csv", index=False)
    histories = np.stack(outer.history_observations)
    starts = np.eye(4)[histories[:, -1]]
    history_shares = np.column_stack([(histories == k).mean(1) for k in range(4)])
    raw = np.stack(outer.raw_forecast_transitions)
    # KU pseudonyms are course/year scoped, not persistent student identities.
    # Each registration is the recoverable cluster; never merge bare USER_IDs.
    outer_users = outer.registration_id.to_numpy()
    scenario_ids = list(scenarios)
    ref = scenario_ids[0]
    validation_rows, paired_rows, sensitivity_rows, epsilon_rows, estimates, prefix_rows = [], [], [], [], [], []
    eps_values = [0., .001, .002, .005, .01, .02]
    npz_payload = {}
    for context in ["observed_future_context", "frozen_landmark_history"]:
        print("Q34: kernels and exact DP", context, flush=True)
        raw_context = raw.copy()
        if context == "frozen_landmark_history":
            raw_context[:, :, :6] = raw_context[:, :1, :6]
        kernels, exact = {}, {}
        for key, contract in scenarios.items():
            modified = q.modify_scenario_raw(raw_context, contract)
            design = transformer.transform(modified.reshape(-1, modified.shape[-1]))
            kernels[key] = model.transition_matrices(design).reshape(len(outer), 28, 4, 4)
            exact[key] = {m: run_risk(kernels[key], starts, m) for m in [3, 5, 7, 10, 14]}
            for strength in [.005, .02, .05]:
                for target in ["uniform", "inactive"]:
                    target_p = np.full_like(kernels[key], .25) if target == "uniform" else np.zeros_like(kernels[key])
                    if target == "inactive":
                        target_p[:, :, :, 0] = 1.
                    changed = (1-strength)*kernels[key]+strength*target_p
                    perturbed = run_risk(changed, starts, 7)
                    difference = np.abs(perturbed-exact[key][7])
                    bound = row_tv_bound(kernels[key], changed)
                    if not np.all(difference <= bound+1e-11):
                        raise AssertionError("empirical-kernel path-event TV bound violated")
                    sensitivity_rows.append({"context": context, "scenario_id": key,
                        "mixture_strength": strength, "perturbation_target": target,
                        "risk": float(perturbed.mean()), "baseline_risk": float(exact[key][7].mean()),
                        "max_anchor_abs_change": float(difference.max()), "mean_coupling_bound": float(bound.mean()),
                        "bound_violations": int((difference > bound+1e-11).sum()),
                        "scope": "synthetic kernel stress test, not a fitted-parameter confidence region"})
        rng = np.random.default_rng(20260822+7001)
        metrics = {key: {"completion": [], "sustained": [], "max_run": []} for key in scenarios}
        indices = []
        for batch_start in range(0, args.paths, 10000):
            size = min(10000, args.paths-batch_start)
            anchor = rng.integers(0, len(outer), size=size)
            rng.random(size)  # preserve original shared random-number stream (u_start)
            uniforms = rng.random((size, 28))
            rng.random((size, 28))  # original u_emission (unused for observable MC)
            indices.append(anchor)
            for key in scenarios:
                state = histories[anchor, -1].copy()
                future = np.empty((size, 28), dtype=np.int64)
                for t in range(28):
                    state = categorical_draw(kernels[key][anchor,t,state,:], uniforms[:,t])
                    future[:,t] = state
                runs = max_true_run(future == 0)
                active = (future[:, -7:] != 0).mean(1)
                future_shares = np.column_stack([(future==k).mean(1) for k in range(4)])
                features = np.column_stack([outer.course_global.to_numpy()[anchor], history_shares[anchor],
                    future_shares, runs/28, active])
                metrics[key]["completion"].append(q.calibrated_completion_probability(base,calibrator,features))
                metrics[key]["sustained"].append((active>0).astype(float))
                metrics[key]["max_run"].append(runs)
        indices = np.concatenate(indices)
        for key in metrics:
            metrics[key] = {m: np.concatenate(v) for m,v in metrics[key].items()}
            metrics[key]["pi_risk"] = (metrics[key]["max_run"] >= 7).astype(float)
            for name in ["completion", "sustained", "pi_risk"]:
                values = metrics[key][name]
                estimates.append({"context": context, "scenario_id": key, "metric": name,
                    "mean": float(values.mean()), "mcse": float(values.std(ddof=1)/np.sqrt(args.paths)),
                    "paths": args.paths})
                prefix_rows.append({"context": context, "scenario_id": key, "metric": name,
                    "first_30000_mean": float(values[:30000].mean())})
                if key != ref:
                    paired_rows.append({"context": context, "scenario_id": key, "reference": ref,
                        "metric": name, **paired_summary(values, metrics[ref][name])})
                npz_payload[f"{context}__{key}__{name}"] = values
            for m in [3,5,7,10,14]:
                indicator = (metrics[key]["max_run"] >= m).astype(float)
                exact_on_sample = exact[key][m][indices]
                residual = indicator-exact_on_sample
                residual_se = residual.std(ddof=1)/np.sqrt(args.paths)
                error = residual.mean()
                validation_rows.append({"context": context, "scenario_id": key, "run_length": m,
                    "risk_dp_equal_anchor": float(exact[key][m].mean()), "risk_mc": float(indicator.mean()),
                    "risk_dp_sampled_anchor": float(exact_on_sample.mean()), "mc_minus_sampled_dp": float(error),
                    "conditional_mcse": float(residual_se),
                    "absolute_standardized_error": float(abs(error)/residual_se) if residual_se else 0.,
                    "within_five_mcse": bool(abs(error)<=5*residual_se+1e-12),
                    "event_scope": "entirely within next 28 days, historical run counter reset"})
        # Epsilon comparisons use all original alternatives, never sequential pruning.
        completion = np.array([metrics[k]["completion"].mean() for k in scenario_ids])
        risk = np.array([exact[k][7].mean() for k in scenario_ids])
        _, user_inverse = np.unique(outer_users, return_inverse=True)
        n_users = user_inverse.max()+1
        # Equal-anchor estimand with registration-cluster resampling. At each anchor,
        # estimate completion association by its mean over assigned MC paths.
        counts = np.bincount(indices, minlength=len(outer))
        if (counts == 0).any():
            raise RuntimeError("some anchors have no simulations")
        anchor_completion = np.column_stack([np.bincount(indices, weights=metrics[k]["completion"],
            minlength=len(outer))/counts for k in scenario_ids])
        anchor_risk = np.column_stack([exact[k][7] for k in scenario_ids])
        b_rng=np.random.default_rng(20261116)
        bc, br = np.empty((args.bootstrap,4)), np.empty((args.bootstrap,4))
        for b in range(args.bootstrap):
            w=b_rng.multinomial(n_users,np.full(n_users,1/n_users))[user_inverse]
            bc[b]=np.average(anchor_completion,axis=0,weights=w)
            br[b]=np.average(anchor_risk,axis=0,weights=w)
        for ec in eps_values:
            for er in eps_values:
                point=epsilon_nondominated(anchor_completion.mean(0),risk,epsilon_completion=ec,epsilon_risk=er)
                boot=np.array([epsilon_nondominated(bc[b],br[b],epsilon_completion=ec,epsilon_risk=er)
                    for b in range(args.bootstrap)])
                for j,key in enumerate(scenario_ids):
                    epsilon_rows.append({"context":context,"scenario_id":key,"epsilon_completion":ec,
                        "epsilon_risk":er,"point_nondominated":bool(point[j]),
                        "registration_cluster_nondominated_frequency":float(boot[:,j].mean()),
                        "registration_clusters":int(n_users),"bootstrap_replicates":args.bootstrap,
                        "uncertainty_scope":"fixed kernel and calibration; cluster variability and finite-path anchor means only"})
        npz_payload[f"{context}__anchor_index"]=indices
    for name, rows in [("q34_dp_mc",validation_rows),("q34_paired_mcse",paired_rows),
        ("q34_kernel_sensitivity",sensitivity_rows),("q34_epsilon_sensitivity",epsilon_rows),
        ("q34_estimates",estimates),("q34_original_stream_prefix",prefix_rows)]:
        pd.DataFrame(rows).to_csv(ROOT / f"results/{name}.csv",index=False)
    np.savez_compressed(ROOT / "private/q34_paired_paths.npz",**npz_payload)
    result={"kind":"locked_observable_NHMC_reproduction_and_sensitivity_not_causal",
        "paths_per_scenario_per_context":args.paths,"contexts":2,"scenarios":scenarios,
        "anchors":len(outer),"scoped_pseudonyms":int(pd.Series(outer_users).nunique()),
        "optimization_audit":model.optimization_audit,"feature_transformer":transformer.metadata(),
        "coefficients":model.coefficients.tolist(),"calibration":calibration_meta,
        "all_dp_mc_within_five_mcse":all(r["within_five_mcse"] for r in validation_rows),
        "kernel_bound_violations":sum(r["bound_violations"] for r in sensitivity_rows),
        "scope_cautions":["Observed-future covariates are oracle conditioning, not a deployable day-28 forecast.",
            "Frozen history holds behavioral covariates fixed and assumes the fitted conditional model remains applicable.",
            "The archived NH-HMM/NHMC outer comparison influenced fallback; no new untouched-holdout claim.",
            "Kernel perturbations are stress tests, not empirically identified student interventions.",
            "Completion values are association scores with poor cross-year calibration, not pass probabilities or effects.",
            "KU pseudonyms cannot link students across courses/years; bootstrap units are registrations, not verified unique people."],
        "elapsed_seconds":time.perf_counter()-started}
    (ROOT / "results/q34_extension_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["anchors","scoped_pseudonyms","all_dp_mc_within_five_mcse","kernel_bound_violations","elapsed_seconds"]}),flush=True)


if __name__ == "__main__":
    main()
