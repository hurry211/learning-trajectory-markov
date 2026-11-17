"""Scoped-ID audit and registration bootstrap on fixed Q1 predictions.

KU pseudonyms cannot link students across courses/years. A bare USER_ID
cluster would conflate unrelated people; true cross-course student clustering
cannot be recovered from the released data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    meta = pd.read_parquet(ROOT / "paper_output/data_cleaned/ku_daily_panel.parquet",
        columns=["registration_id", "USER_ID", "cohort", "COURSE_ID"]).drop_duplicates("registration_id")
    meta["sequence_hash"] = meta.registration_id.map(
        lambda x: hashlib.sha256(f"q1-full|{x}".encode()).hexdigest()[:16])
    archived = ROOT / "paper_output/tables/q1_full/q1_full_outer_sequence_scores.csv"
    if archived.exists():
        scores = pd.read_csv(archived)
    else:
        scores = pd.read_parquet(ROOT / "private/q1_baseline_sequence_scores.parquet")
        scores["delta_loglik_hmm_minus_mc"] = scores.hmm - scores.order_1
    joined = scores.merge(meta[["sequence_hash", "USER_ID"]], on="sequence_hash", validate="one_to_one")
    if joined.USER_ID.isna().any() or len(joined) != len(scores):
        raise ValueError("outer score mapping incomplete")
    grouped = joined[["delta_loglik_hmm_minus_mc", "scored_observations"]].to_numpy()
    rng = np.random.default_rng(20261116)
    draws = np.empty(args.replicates)
    for b in range(args.replicates):
        sample = grouped[rng.integers(len(grouped), size=len(grouped))].sum(0)
        draws[b] = sample[0] / sample[1]
    cohorts = sorted(meta.cohort.astype(str).unique())
    overlap = []
    for i, a in enumerate(cohorts):
        for b in cohorts[i + 1:]:
            overlap.append({"cohort_a": a, "cohort_b": b,
                "shared_released_user_ids": len(set(meta.loc[meta.cohort.astype(str).eq(a), "USER_ID"]) &
                                                set(meta.loc[meta.cohort.astype(str).eq(b), "USER_ID"]))})
    result = {"kind": "fixed_archived_prediction_reanalysis_not_new_holdout", "seed": 20261116,
        "registrations": len(joined), "scoped_pseudonyms": len(grouped), "bootstrap_cluster": "registration",
        "replicates": args.replicates,
        "delta_nat_per_observation": float(grouped[:, 0].sum() / grouped[:, 1].sum()),
        "ci_95_percentile": np.quantile(draws, [.025, .975]).tolist(), "numeric_id_collisions_not_identity_overlap": overlap,
        "id_scope_caution": "Pseudonyms cannot identify the same learner across courses and academic years. Bare numeric collisions are NOT evidence of student overlap. True cross-course/academic-year student-disjointness cannot be verified.",
        "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13066476/",
        "uncertainty_scope": "Fixed fitted model; excludes selection/refitting uncertainty."}
    (ROOT / "results/q1_scoped_id_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
