"""Download and register the public datasets used by the paper.

The script deliberately separates three evidential roles:

1. KU Leuven clickstream: fine-grained HMM/HSMM model development.
2. OULAD: day-level, leave-course-out external validation.
3. ISILA: retained for data-quality audit and possible cross-sectional exploration.

Simulation is not downloaded here because it will be generated from an explicit
data-generating process and stored separately under ``crawled_data/simulated``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CRAWLED_ROOT = PROJECT_ROOT / "crawled_data"
RAW_ROOT = CRAWLED_ROOT / "raw"
LEGACY_OULAD_ROOT = PROJECT_ROOT / "data"

ZENODO_RECORDS = {
    "ku_leuven": {
        "record_id": "17087849",
        "selected_files": [
            "dataset.zip",
            "course_info.json",
            "final_column_names.json",
        ],
        "role": (
            "day-level state discovery and HMM/HSMM estimation; second-level timestamps "
            "are used for 30-minute session and time-regularity features"
        ),
    },
    "isila": {
        "record_id": "18793674",
        "selected_files": ["isila_all.csv", "isila_weekly.csv"],
        "role": (
            "data-quality audit and exploratory cross-platform mapping; the downloaded "
            "weekly file has a defective Week field and is excluded from the main analysis"
        ),
    },
}

OULAD_EXPECTED_FILES = [
    "courses.csv",
    "assessments.csv",
    "vle.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "studentVle.csv",
]


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(url: str, retries: int = 4) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "learning-state-paper/1.0 (academic use)"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def download_file(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected_size:
        print(f"[skip] {destination.name}: existing size matches")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "learning-state-paper/1.0 (academic use)"},
    )
    print(f"[download] {destination.name} ({expected_size / 1024 / 1024:.1f} MiB)")
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    if partial.stat().st_size != expected_size:
        raise IOError(
            f"size mismatch for {destination.name}: "
            f"{partial.stat().st_size} != {expected_size}"
        )
    os.replace(partial, destination)


def verify_zenodo_checksum(path: Path, checksum_value: str) -> None:
    algorithm, expected = checksum_value.split(":", 1)
    if algorithm.lower() != "md5":
        raise ValueError(f"unsupported Zenodo checksum: {algorithm}")
    digest = hashlib.md5()  # nosec B303 - used only for archive integrity verification
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest().lower() != expected.lower():
        raise IOError(f"checksum mismatch for {path.name}")


def register_zenodo_dataset(slug: str, download: bool) -> dict[str, Any]:
    config = ZENODO_RECORDS[slug]
    record_id = config["record_id"]
    metadata = request_json(f"https://zenodo.org/api/records/{record_id}")
    by_name = {item["key"]: item for item in metadata["files"]}
    destination_root = RAW_ROOT / slug
    selected: list[dict[str, Any]] = []

    for filename in config["selected_files"]:
        item = by_name[filename]
        destination = destination_root / filename
        if download:
            download_file(item["links"]["self"], destination, int(item["size"]))
            verify_zenodo_checksum(destination, item["checksum"])
        file_record: dict[str, Any] = {
            "name": filename,
            "source_url": item["links"]["self"],
            "bytes": int(item["size"]),
            "zenodo_checksum": item["checksum"],
            "local_path": str(destination.relative_to(PROJECT_ROOT)),
        }
        if destination.exists():
            file_record["sha256"] = sha256(destination)
        selected.append(file_record)

    metadata_block = metadata["metadata"]
    return {
        "title": metadata_block["title"],
        "record_url": f"https://zenodo.org/records/{record_id}",
        "doi": metadata_block.get("doi"),
        "license": (metadata_block.get("license") or {}).get("id"),
        "publication_date": metadata_block.get("publication_date"),
        "role_in_paper": config["role"],
        "files": selected,
    }


def register_local_oulad() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for filename in OULAD_EXPECTED_FILES:
        path = LEGACY_OULAD_ROOT / filename
        if not path.exists():
            raise FileNotFoundError(
                f"OULAD file missing: {path}. Download anonymisedData.zip from the official page."
            )
        files.append(
            {
                "name": filename,
                "bytes": path.stat().st_size,
                "local_path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
            }
        )
    archive = LEGACY_OULAD_ROOT / "oulad.zip"
    if archive.exists():
        files.append(
            {
                "name": archive.name,
                "bytes": archive.stat().st_size,
                "local_path": str(archive.relative_to(PROJECT_ROOT)),
                "sha256": sha256(archive),
            }
        )
    return {
        "title": "Open University Learning Analytics Dataset (OULAD)",
        "official_page": "https://research.stem.open.ac.uk/ouanalyse/dataset/",
        "download_url": "https://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip",
        "data_paper_doi": "10.1038/sdata.2017.171",
        "license": "CC-BY-4.0",
        "role_in_paper": "day-level leave-course-out and outcome-linked external validation",
        "granularity_warning": (
            "studentVle is aggregated by day; do not pool its states directly with "
            "the 15/30-minute KU Leuven states"
        ),
        "files": files,
    }


def write_manifest(datasets: dict[str, Any]) -> None:
    CRAWLED_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "evidence_policy": {
            "observed_data": "supports empirical model estimation and validation",
            "simulated_data": "supports parameter recovery and conditional scenarios only",
            "causal_claims": "not permitted without an identified intervention design",
        },
        "datasets": datasets,
    }
    target = CRAWLED_ROOT / "sources.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="register sources without downloading Zenodo files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(ZENODO_RECORDS),
        default=sorted(ZENODO_RECORDS),
    )
    args = parser.parse_args()

    datasets: dict[str, Any] = {"oulad": register_local_oulad()}
    for slug in args.datasets:
        datasets[slug] = register_zenodo_dataset(slug, download=not args.metadata_only)
    write_manifest(datasets)


if __name__ == "__main__":
    main()
