"""Recreate locked Q2 models; retain local paired predictions; publish aggregates only."""
import argparse
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_output/code/modeling"))
import run_q2_oulad as q2
from sequence_models import _fit_categorical_hmm_once, canonicalize_hmm


def replay_archived_winner(panel, partitions):
    """Replay the recorded final winner, not a new grid search or test selection."""
    archived = pd.read_csv(ROOT / "paper_output/tables/q2/q2_nested_model_selection.csv")
    row = archived.loc[archived.scope.eq("main_presentation_nested:fixed_k_refit")].iloc[0]
    k, seed = int(row.k), int(row.selected_start_seed)
    development = pd.concat([partitions["inner_train"], partitions["inner_validation"]],
        ignore_index=True).drop_duplicates("sequence_id")
    samples = [q2.stable_capacity_sample(partitions[name], cap_per_stratum=30,
        seed=20260822+i, partition=name)[0] for i,name in enumerate(["inner_train", "inner_validation"])]
    sample = pd.concat(samples,ignore_index=True).drop_duplicates("sequence_id")
    encoder = q2.TransparentObservationEncoder(q2.encoder_config()).fit(q2.history_frame(panel,development,28))
    bundle = q2.bundle(q2.history_frame(panel,sample,28),encoder,"archived-final-refit-replay")
    model, warnings = _fit_categorical_hmm_once(bundle,n_states=k,seed=seed,
        n_iter=300,tol=.001,prior_strength=.1)
    model, _, _ = canonicalize_hmm(model,bundle.observation_labels)
    ll = float(model.score(bundle.X,lengths=bundle.lengths))
    if abs(ll-float(row.train_loglik)) > 1e-7 or model.monitor_.iter >= 300:
        raise AssertionError("archived winner did not reproduce; do not use outcome comparisons")
    result = q2.FittedRepresentation(encoder,q2.fit_dirichlet_markov_chain(bundle,alpha=.5),
        model,"CategoricalHMM",k,.05388127762422956,"archived_selection_replayed","archived_fixed_k_seed_replayed")
    q2.write_json(ROOT / "results/q2_winner_replay.json",{
        "mode":"one recorded winning seed; original full-grid tables retained as provenance, not rerun",
        "k":k,"seed":seed,"train_loglik":ll,"archived_train_loglik":float(row.train_loglik),
        "absolute_loglik_difference":abs(ll-float(row.train_loglik)),"iterations":int(model.monitor_.iter),
        "n_training_sequences":bundle.n_sequences,"n_training_observations":bundle.n_observations,"warnings":warnings})
    return result


def paired_metrics(y, p0, p1, cluster, reps, seed):
    _, ix = np.unique(cluster, return_inverse=True)
    n = ix.max() + 1
    rng = np.random.default_rng(seed)
    def metrics(w):
        return np.array([roc_auc_score(y, p1, sample_weight=w) - roc_auc_score(y, p0, sample_weight=w),
            average_precision_score(y, p1, sample_weight=w) - average_precision_score(y, p0, sample_weight=w),
            np.average((p1-y)**2 - (p0-y)**2, weights=w)])
    point = metrics(np.ones(len(y)))
    draws = []
    for _ in range(reps):
        w = rng.multinomial(n, np.full(n, 1/n))[ix]
        if np.unique(y[w > 0]).size == 2:
            draws.append(metrics(w))
    draws = np.asarray(draws)
    return [{"metric": name, "hmm_minus_observable": float(point[i]),
        "ci_low": float(np.quantile(draws[:, i], .025)),
        "ci_high": float(np.quantile(draws[:, i], .975)),
        "higher_is_better": name != "Brier", "valid_replicates": len(draws), "students": int(n)}
        for i, name in enumerate(["AUC", "average_precision", "Brier"])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--refit-grid", action="store_true", help="rerun the full archived nested grid instead of replaying its recorded winner")
    args = parser.parse_args()
    start = time.perf_counter()
    print("Q2: loading registry and first 56 days", flush=True)
    registry = q2.load_registry()
    panel = q2.load_history_panel(56)
    partitions, partition_meta = q2.main_partition(registry, 28)
    checkpoint = ROOT / "private/q2_representation.joblib"
    if checkpoint.exists():
        representation = joblib.load(checkpoint)
        print("Q2: restored release-local model checkpoint", flush=True)
    else:
        if args.refit_grid:
            print("Q2: refitting original nested grid K2-5, 10 starts, 300 iterations", flush=True)
            representation, comparisons, starts, sampling, warnings = q2.fit_nested_representation(
                panel, partitions, cap_per_stratum=30, seed=20260822, k_values=[2,3,4,5],
                n_starts=10, n_iter=300, tol=.001, scope="main_presentation_nested")
            comparisons.to_csv(ROOT / "results/q2_recreated_model_selection.csv", index=False)
            starts.to_csv(ROOT / "results/q2_recreated_random_starts.csv", index=False)
        else:
            print("Q2: replaying the archived winning K and seed; no new selection", flush=True)
            representation = replay_archived_winner(panel, partitions)
        joblib.dump(representation, checkpoint)
        print("Q2 model", representation.chosen_kind, representation.selected_k, flush=True)
    rows, performance = [], []
    for landmark in [14, 28, 56]:
        print(f"Q2: paired outcomes at landmark {landmark}", flush=True)
        perf, pred = q2.evaluate_landmark(panel, registry, representation, landmark=landmark,
            bootstrap_replications=0, seed=20260822)
        performance.append(perf)
        pred.to_parquet(ROOT / f"private/q2_predictions_lm{landmark}.parquet", index=False)
        for target, frame in pred.groupby("target"):
            pivot = frame.pivot(index=["sequence_id", "id_student", target],
                columns="representation", values="predicted_probability").reset_index()
            if "Observable+numbered-HMM-history" not in pivot:
                rows.append({"landmark": landmark, "target": target, "status": "no_HMM_passed_original_gate"})
                continue
            result = paired_metrics(pivot[target].to_numpy(), pivot["Observable-history"].to_numpy(),
                pivot["Observable+numbered-HMM-history"].to_numpy(), pivot.id_student.to_numpy(),
                args.replicates, 20261116 + landmark)
            rows.extend(dict(landmark=landmark, target=target, registrations=len(pivot), **r) for r in result)
            pd.DataFrame(rows).to_csv(ROOT / "results/q2_paired_differences.csv", index=False)
    all_perf = pd.concat(performance, ignore_index=True)
    all_perf.to_csv(ROOT / "results/q2_recreated_landmark_performance.csv", index=False)
    old = pd.read_csv(ROOT / "paper_output/tables/q2/q2_landmark_performance.csv")
    keys = ["landmark_days", "target", "representation"]
    comparison = all_perf.merge(old[keys + ["auc", "brier"]], on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    result = {"kind": "reproduction_and_fixed_model_paired_bootstrap_not_new_holdout",
        "selected_k": representation.selected_k, "chosen_kind": representation.chosen_kind,
        "partition": partition_meta, "seed": 20260822, "bootstrap_seed_base": 20261116,
        "paired_cluster": "id_student", "replicates": args.replicates,
        "max_abs_auc_difference_from_archived": float(abs(comparison.auc_new-comparison.auc_old).max()),
        "max_abs_brier_difference_from_archived": float(abs(comparison.brier_new-comparison.brier_old).max()),
        "elapsed_seconds": time.perf_counter()-start,
        "uncertainty_scope": "Paired test-student cluster bootstrap conditional on fixed fitted models; no model-selection uncertainty."}
    (ROOT / "results/q2_extension_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
