"""
Run full Human-AI corpus pipeline (Phases 1-5).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCRIPTS = BASE / "scripts"


def run(label: str, script: str):
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    path = SCRIPTS / script
    result = subprocess.run([sys.executable, str(path)], cwd=str(BASE))
    if result.returncode != 0:
        raise SystemExit(f"Failed: {script}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sample", action="store_true",
                        help="Skip recreating sample corpus (use existing CSVs)")
    args = parser.parse_args()

    if not args.skip_sample:
        run("Phase 1: Create sample corpus", "create_sample_corpus.py")
    else:
        print("[INFO] Using existing corpus files")

    run("Phase 2: Feature engineering", "feature_engineering.py")
    run("Phase 3: Clustering", "cluster_analysis.py")
    run("Phase 4: Classification validation", "classify_validate.py")
    run("Phase 5: Analysis report", "analyze_results.py")

    print("\nALL PHASES COMPLETE")
    print("Corpus : data/processed/corpus.csv")
    print("Features: data/processed/feature_matrix.csv")
    print("Report : reports/project_report.md")


if __name__ == "__main__":
    main()
