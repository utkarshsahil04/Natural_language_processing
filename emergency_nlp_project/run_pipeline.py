"""
run_pipeline.py
Runs all 5 phases end-to-end.
Usage:  python run_pipeline.py
        python run_pipeline.py --skip-download   (if Kaggle not configured)
"""
import os, sys, subprocess, argparse

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SCRIPTS  = os.path.join(BASE_DIR, "scripts")

def run(label, script, **kwargs):
    print(f"\n{'#'*60}")
    print(f"  {label}")
    print(f"{'#'*60}")
    result = subprocess.run([sys.executable, os.path.join(SCRIPTS, script)], **kwargs)
    if result.returncode != 0:
        print(f"\nFAILED at: {label}")
        sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--skip-download", action="store_true",
                    help="Use sample corpus instead of downloading from Kaggle")
args = parser.parse_args()

if args.skip_download:
    print("[INFO] Using sample corpus (no Kaggle download).")
    run("Phase 1b: Creating Sample Corpus", "create_sample_corpus.py")
else:
    run("Phase 1: Download Kaggle Corpus", "download_data.py")

run("Phase 2: BIO Annotation",        "annotate_bio.py")
run("Phase 3: Feature Engineering",   "feature_engineering.py")
run("Phase 4: Train Models",          "train_models.py")
run("Phase 5: Analysis & Report",     "analyze_results.py")

print("\n" + "="*60)
print("  ALL PHASES COMPLETE!")
print("="*60)
print("  Corpus    : data/processed/corpus.csv")
print("  Annotated : data/processed/annotated_bio.csv")
print("  Features  : data/processed/features.csv")
print("  Models    : models/*.pkl")
print("  Report    : reports/project_report.md")
print("  Plots     : reports/*.png")
print("\n  Start UI  : python app/app.py")
