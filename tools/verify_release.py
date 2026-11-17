"""Audit aggregate evidence, source/input hashes and the publication allowlist.

No individual rows, credentials, tokens or model checkpoints are printed.
"""
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    commands=[["-m","unittest","discover","-s","tests","-v"],
              ["-m","unittest","discover","-s","paper_output/code/modeling/tests","-v"]]
    import sys
    tests=[]
    for command in commands:
        run=subprocess.run([sys.executable,"-X","utf8",*command],cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
        if run.returncode:
            print(run.stdout,run.stderr)
            raise SystemExit("unit tests failed")
        match=re.search(r"Ran (\d+) tests?",run.stderr)
        tests.append({"command":"python "+" ".join(command),"tests":int(match.group(1)),"passed":True})
    required=["q1_scoped_id_audit.json","q1_stronger_baselines.csv","q2_winner_replay.json",
        "q2_extension_manifest.json","q2_paired_differences.csv","q34_extension_manifest.json",
        "q34_dp_mc.csv","q34_kernel_sensitivity.csv","q34_epsilon_sensitivity.csv","reproduction_checks.json"]
    if any(not (ROOT/"results"/name).is_file() for name in required):
        raise SystemExit("required empirical evidence missing")
    dp=pd.read_csv(ROOT/"results/q34_dp_mc.csv")
    kernel=pd.read_csv(ROOT/"results/q34_kernel_sensitivity.csv")
    pairs=pd.read_csv(ROOT/"results/q2_paired_differences.csv")
    if len(dp)!=40 or not dp.within_five_mcse.all() or kernel.bound_violations.sum()!=0:
        raise SystemExit("risk checks failed")
    if len(pairs)!=18 or not pairs.valid_replicates.eq(2000).all():
        raise SystemExit("paired bootstrap incomplete")
    # Prevent accidentally publishing individual rows even if ignore rules change.
    git=subprocess.run(["git","-c",f"safe.directory={ROOT.as_posix()}","ls-files","-z"],
        cwd=ROOT,capture_output=True)
    tracked=[p for p in git.stdout.decode("utf-8").split("\0") if p] if git.returncode==0 else []
    forbidden=[]
    for name in tracked:
        path=ROOT/name
        if (name.startswith(("private/","crawled_data/","data/","paper_output/data_cleaned/"))
            or path.suffix.lower() in {".parquet",".npz",".joblib",".zip",".pdf",".pem",".key"}
            or ".env" in Path(name).parts):
            forbidden.append(name)
            continue
        if path.suffix.lower() in {".csv",".json"}:
            content=path.read_text(encoding="utf-8-sig")
            has_ids = (bool(set(pd.read_csv(path,nrows=0).columns) & {"id_student","USER_ID","registration_id","sequence_id","sequence_hash"})
                if path.suffix.lower()==".csv" else bool(re.search(r'"(?:id_student|USER_ID|registration_id|sequence_id|sequence_hash)"\s*:',content)))
            if has_ids:
                forbidden.append(name)
        if path.is_file() and path.stat().st_size>2_000_000:
            forbidden.append(name)
    if forbidden:
        raise SystemExit("publication allowlist failed: "+", ".join(forbidden))
    sources=list((ROOT/"experiments").glob("*.py"))+list((ROOT/"tools").glob("*.py"))+list((ROOT/"tests").glob("*.py"))+list((ROOT/"paper_output/code").rglob("*.py"))
    sources=[p for p in sources if p.name!="github_repository.py"]
    inputs=[ROOT/"paper_output/data_cleaned"/f for f in ["ku_daily_panel.parquet","oulad_daily.parquet","oulad_out_of_window_sensitivity.parquet"]]
    old_manifest_names={"q1_full":"q1_full_execution_manifest.json","q2":"q2_execution_manifest.json","q3":"q34_execution_manifest.json"}
    source_checks=[]
    for q,name in old_manifest_names.items():
        path=ROOT/"paper_output/results"/q/name
        if not path.exists():
            continue
        original=json.loads(path.read_text(encoding="utf-8"))
        for key in ["source_code", "dependencies"]:
            records=original.get(key,[])
            if isinstance(records,dict):
                records=[records]
            for record in records:
                if not isinstance(record,dict) or "path" not in record or "sha256" not in record:
                    continue
                candidate=ROOT/record["path"]
                if candidate.is_file():
                    actual=digest(candidate)
                    source_checks.append({"path":record["path"],"archived_sha256":record["sha256"],"current_sha256":actual,"match":actual==record["sha256"]})
    if any(not r["match"] for r in source_checks):
        raise SystemExit("archived source hash mismatch")
    q2manifest=json.loads((ROOT/"results/q2_extension_manifest.json").read_text(encoding="utf-8"))
    manifest={"generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"reproduction and post-hoc diagnostics; not proof of all paper claims",
        "python":platform.python_version(),"platform":platform.platform(),
        "packages":{p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","hmmlearn","matplotlib","pyarrow","polars","joblib"]},
        "tests":tests,"source_hash_checks":source_checks,
        "source_files":[{"path":p.relative_to(ROOT).as_posix(),"sha256":digest(p)} for p in sorted(sources)],
        "inputs":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":digest(p),"uploaded":False} for p in inputs],
        "q2_replay_caveat":{"auc_max_abs_difference":q2manifest["max_abs_auc_difference_from_archived"],
            "note":"Small day-14 ranking differences retained; day-28/56 AUC match archived values. Old individual predictions unavailable."},
        "privacy_scan":{"tracked_files":len(tracked),"forbidden_files":forbidden,"scope":"path and field-name checks plus curated staging, not a formal privacy guarantee"},
        "claim_gates":{"numerical_checks":True,"clinical_or_cognitive_state_validity":False,
            "causal_strategy_effects":False,"new_untouched_external_test":False,"hsmm_implemented":False}}
    (ROOT/"results/release_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"unit_tests_passed":sum(x["tests"] for x in tests),"empirical_dp_mc_checks":len(dp),
        "paired_metric_rows":len(pairs),"tracked_files_reviewed":len(tracked),"archived_source_hash_matches":len(source_checks)},ensure_ascii=False))


if __name__=="__main__":
    main()
