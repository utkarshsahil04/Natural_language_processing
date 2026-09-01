"""
Phase 1 - Corpus Download from Kaggle
======================================
Two modes:
  A. Competition dataset (requires accepting rules + kaggle.json)
  B. Public dataset upload (no rules needed, just kaggle.json)

SETUP (one-time, 2 minutes):
  1. Go to https://www.kaggle.com/settings
     API section -> "Create New Token"  ->  kaggle.json downloads
  2. Place kaggle.json at:
     Windows: C:/Users/<YourName>/.kaggle/kaggle.json

THEN for Mode A (competition):
  3. Go to https://www.kaggle.com/competitions/nlp-getting-started/rules
     Click "I Understand and Accept" button  (REQUIRED, free & instant)

THEN run:
  python scripts/download_data.py
"""

import os, sys, zipfile, shutil
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(RAW_DIR,  exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# Check project root first, then ~/.kaggle/
_project_json = os.path.join(BASE_DIR, "kaggle.json")
_default_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
KAGGLE_JSON = _project_json if os.path.exists(_project_json) else _default_json

# If found in project root, copy to ~/.kaggle/ so the kaggle CLI can find it
if os.path.exists(_project_json) and _project_json == KAGGLE_JSON:
    import shutil as _sh
    os.makedirs(os.path.dirname(_default_json), exist_ok=True)
    _sh.copy(_project_json, _default_json)
    os.chmod(_default_json, 0o600)

# ── Public dataset slug (no competition acceptance needed) ───────────────────
# Same data, uploaded as a regular dataset by the community
PUBLIC_DATASET = "vbmokin/nlp-with-disaster-tweets-cleaning-data"
COMPETITION    = "nlp-getting-started"

HINGLISH_MESSAGES = [
    "Bahut tez barish ho rahi hai Yamuna bridge ke paas paani aa gaya help chahiye",
    "Flood in Sector 12 Noida please send boats urgently log fase mein atke hain",
    "SOS Gandhi Nagar mein ghar ke andar paani aa gaya rescue team bhejo please",
    "Yamuna overflowing near Kalindi Kunj need immediate evacuation DelhiFloods",
    "Chambal river flooding Morena district villages near Joura cut off need army",
    "Barish ne rasta band kar diya ambulance nahi aa sakti medical help needed Kota",
    "Brahmaputra flood in Majuli island 5000 people stranded boats aur food chahiye",
    "Rescue needed at Vaishali Nagar Guwahati houses submerged AssamFloods",
    "Need helicopter rescue from Bageshwar Uttarakhand road completely washed away",
    "Srinagar Dal Lake area flooding send NDRF team to Hazratbal immediately",
    "Bhookamp aa gaya Chamoli mein buildings gir gayi NDRF ko bulao jaldi",
    "Tremors felt in Delhi NCR Noida Sector 62 mein cracks in buildings evacuating",
    "People trapped under rubble in Joshimath after quake heavy equipment needed now",
    "Factory mein aag lag gayi Okhla Industrial Area Phase 2 fire brigade call karo",
    "Building fire in Dharavi slum people stuck on upper floors ladder truck needed",
    "Massive fire in Anaj Mandi Delhi multiple casualties blood O+ urgently needed",
    "Forest fire spreading near Nainital villages in Ramgarh at risk evacuation needed",
    "Cyclone Biparjoy approaching Kutch coast fishing villages need immediate evacuation",
    "Very heavy rain in Chennai Velachery area waterlogged need boats urgently",
    "Storm in Odisha coast Puri district fishermen stuck at sea coast guard needed",
    "Road accident on NH-8 near Gurgaon toll 3 injured need ambulance urgently",
    "Child fell in borewell near Hisar Haryana NDRF team needed immediately",
    "Landslide blocked NH-44 near Ramban JK 200 vehicles stranded need bulldozers",
    "Multiple people injured after wall collapse in Bhiwandi send rescue immediately",
    "Train derailment near Balasore Odisha many casualties blood donors needed urgently",
    "Heatwave in Rajasthan 48 deg C in Churu elderly people collapsing medical camps",
    "Gas leak in BHEL plant Haridwar workers evacuating fire brigade medical on way",
    "Missing child 8 yr old lost near Kumbh Mela Prayagraj police help needed",
    "Jal bhar gaya basement mein RK Puram sector 4 pump chalao please urgently",
    "Nainital mein landslide road band water supply cut rescue teams bhejo immediately",
    "Bus accident near Manali hill road 15 injured hospital 50km away send helicopter",
    "Gas explosion near Sector 5 Noida industrial area injured people need ambulance",
    "Earthquake in Manipur buildings collapsed near Imphal West send rescue teams",
    "Cyclone warning coastal villages near Kakinada not yet evacuated need buses now",
    "Stampede at Kumbh Mela Prayagraj many injured near Sangam ghat send medical",
]

# ── Step 1: Check credentials ─────────────────────────────────────────────────
def check_kaggle_json():
    if not os.path.exists(KAGGLE_JSON):
        print("\n" + "="*60)
        print("  kaggle.json NOT FOUND")
        print("="*60)
        print("""
  To fix this (takes ~2 minutes):

  1. Open: https://www.kaggle.com/settings
  2. Scroll to "API" section
  3. Click "Create New Token"
  4. A file called kaggle.json will download
  5. Move it to:
       """ + KAGGLE_JSON + """

  Then run this script again.
""")
        return False
    print("[OK] kaggle.json found")
    return True

# ── Step 2: Try competition download, fall back to public dataset ─────────────
def download_data():
    train_path = os.path.join(RAW_DIR, "train.csv")
    if os.path.exists(train_path):
        print("[1/4] train.csv already exists - skipping download.")
        return True

    print("[1/4] Attempting competition download ...")
    ret = os.system(
        f'kaggle competitions download -c {COMPETITION} -p "{RAW_DIR}" --quiet'
    )

    zip_path = os.path.join(RAW_DIR, f"{COMPETITION}.zip")

    if ret != 0 or not os.path.exists(zip_path):
        print("\n  Competition download failed (403?).")
        print("  --> You may need to accept the rules first:")
        print("      https://www.kaggle.com/competitions/nlp-getting-started/rules")
        print("\n  Trying public dataset fallback ...")

        ret2 = os.system(
            f'kaggle datasets download -d {PUBLIC_DATASET} -p "{RAW_DIR}" --quiet --unzip'
        )

        # Look for any CSV that looks like training data
        for fname in os.listdir(RAW_DIR):
            if "train" in fname.lower() and fname.endswith(".csv"):
                src = os.path.join(RAW_DIR, fname)
                dst = os.path.join(RAW_DIR, "train.csv")
                if src != dst:
                    shutil.copy(src, dst)
                print(f"  Found: {fname} -> using as train.csv")
                return True

        if ret2 != 0:
            print("\n  Public dataset also failed.")
            print("  Falling back to sample corpus instead ...")
            return False

    # Extract competition zip
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        print(f"  Extracted to {RAW_DIR}")

    return os.path.exists(train_path)

# ── Step 3: Load + filter ─────────────────────────────────────────────────────
def load_and_filter():
    train_path = os.path.join(RAW_DIR, "train.csv")
    print("[2/4] Loading train.csv ...")
    df = pd.read_csv(train_path)

    # Handle both dataset formats (competition has 'target', public may differ)
    if "target" in df.columns:
        disaster_df = df[df["target"] == 1].copy()
    else:
        disaster_df = df.copy()   # public dataset may already be filtered

    print(f"  Total rows     : {len(df)}")
    print(f"  Disaster rows  : {len(disaster_df)}")

    disaster_df = disaster_df.rename(columns={"id": "message_id", "text": "raw_text"})
    disaster_df["message_id"] = disaster_df["message_id"].astype(str).apply(
        lambda x: f"KG_{int(x):05d}" if x.isdigit() else x
    )
    disaster_df["source"] = "kaggle_nlp_disaster_tweets"
    return disaster_df[["message_id", "raw_text", "source"]]

# ── Step 4: Add Hinglish ──────────────────────────────────────────────────────
def add_hinglish(disaster_df):
    print("[3/4] Adding Hinglish/Indian context messages ...")
    rows = [{"message_id": f"HI_{i+1:04d}", "raw_text": m, "source": "manual_hinglish"}
            for i, m in enumerate(HINGLISH_MESSAGES)]
    combined = pd.concat([disaster_df, pd.DataFrame(rows)], ignore_index=True)
    print(f"  Total after merge: {len(combined)}")
    return combined

# ── Step 5: Clean + save ──────────────────────────────────────────────────────
def clean_and_save(df):
    print("[4/4] Cleaning and saving ...")
    df["raw_text"] = df["raw_text"].astype(str).str.strip()
    df = df[df["raw_text"].str.len() > 10].drop_duplicates(subset=["raw_text"])
    df = df.reset_index(drop=True)

    out = os.path.join(PROC_DIR, "corpus.csv")
    df[["message_id", "raw_text", "source"]].to_csv(out, index=False, encoding="utf-8")
    print(f"\n  Saved  : {out}")
    print(f"  Total  : {len(df)} messages")
    print(f"\nDone! Next: python scripts/annotate_bio.py")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not check_kaggle_json():
        sys.exit(1)

    ok = download_data()

    if not ok:
        # Full fallback: create sample corpus and continue pipeline
        print("\n  Generating sample corpus as fallback ...")
        import subprocess
        subprocess.run([sys.executable,
                        os.path.join(BASE_DIR, "scripts", "create_sample_corpus.py")])
        print("  Sample corpus ready. Pipeline can continue.")
        sys.exit(0)

    disaster_df = load_and_filter()
    combined    = add_hinglish(disaster_df)
    clean_and_save(combined)

