"""
Phase 1 - Corpus Collection Script
===================================
Downloads the "NLP with Disaster Tweets" dataset from Kaggle,
filters for real disaster messages, adds Hinglish/Indian-context
samples, and outputs a clean CSV:
    data/processed/corpus.csv  ->  message_id, raw_text, source, is_disaster

SETUP (one-time):
  1. Go to https://www.kaggle.com/settings  -> API -> Create New Token
  2. Place the downloaded kaggle.json in:   C:/Users/<YourName>/.kaggle/kaggle.json
  3. Run:  python scripts/download_data.py
"""

import os
import sys
import zipfile
import shutil
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
KAGGLE_COMP = "nlp-getting-started"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 – Download from Kaggle
# ---------------------------------------------------------------------------
def download_kaggle_data():
    print("[1/4] Downloading dataset from Kaggle ...")
    try:
        import kaggle                                   # noqa – triggers auth check
        os.system(
            f'kaggle competitions download -c {KAGGLE_COMP} -p "{RAW_DIR}"'
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Make sure kaggle.json is placed in C:/Users/<You>/.kaggle/")
        sys.exit(1)

    zip_path = os.path.join(RAW_DIR, f"{KAGGLE_COMP}.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        print(f"  Extracted to {RAW_DIR}")
    else:
        print("  Zip not found – check download path.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2 – Load & filter
# ---------------------------------------------------------------------------
def load_and_filter():
    print("[2/4] Loading train.csv ...")
    train_path = os.path.join(RAW_DIR, "train.csv")
    df = pd.read_csv(train_path)

    print(f"  Total rows      : {len(df)}")
    print(f"  Disaster (1)    : {df['target'].sum()}")
    print(f"  Non-disaster (0): {(df['target'] == 0).sum()}")

    # Keep only real disaster messages
    disaster_df = df[df["target"] == 1].copy()
    disaster_df = disaster_df[["id", "text", "keyword", "location"]].rename(
        columns={"id": "message_id", "text": "raw_text"}
    )
    disaster_df["source"] = "kaggle_nlp_disaster_tweets"
    disaster_df["message_id"] = disaster_df["message_id"].astype(str)

    print(f"  Kept {len(disaster_df)} disaster messages")
    return disaster_df

# ---------------------------------------------------------------------------
# Step 3 – Add Hinglish / Indian-context messages
# ---------------------------------------------------------------------------
HINGLISH_MESSAGES = [
    # Floods
    "Bahut tez barish ho rahi hai, Yamuna bridge ke paas paani aa gaya, help chahiye!",
    "Flood in Sector 12, Noida. Please send boats urgently. Log phase mein atke hain.",
    "SOS - Gandhi Nagar mein ghar ke andar paani aa gaya. Rescue team bhejo please.",
    "Yamuna overflowing near Kalindi Kunj. Need immediate evacuation. #DelhiFloods",
    "Chambal river flooding Morena district. Villages near Joura cut off. Need army.",
    "Barish ne rasta band kar diya, ambulance nahi aa sakti, medical help needed at Kota.",
    "Brahmaputra flood in Majuli island. 5000 people stranded. Boats aur food chahiye.",
    "Rescue needed at Vaishali Nagar, Guwahati. Houses submerged. #AssamFloods",
    "Need helicopter rescue from Bageshwar, Uttarakhand - road completely washed away.",
    "Srinagar Dal Lake area flooding. Send NDRF team to Hazratbal immediately.",
    # Earthquakes
    "Earthquake in Manipur, buildings collapsed near Imphal West. Send rescue teams!",
    "Bhookamp aa gaya! Chamoli mein buildings gir gayi. NDRF ko bulao jaldi.",
    "Tremors felt in Delhi NCR. Noida Sector 62 mein cracks in buildings. Evacuating.",
    "Earthquake 6.2 magnitude hit Uttarkashi. Mountain villages unreachable. Need help.",
    "People trapped under rubble in Joshimath after quake. Heavy equipment needed now.",
    # Fires
    "Factory mein aag lag gayi, Okhla Industrial Area Phase 2. Fire brigade call karo!",
    "Building fire in Dharavi slum. People stuck on upper floors. Ladder truck needed.",
    "Massive fire in Anaj Mandi Delhi. Multiple casualties. Blood O+ urgently needed.",
    "Forest fire spreading near Nainital. Villages in Ramgarh at risk. Evacuation needed.",
    "Aag Chandni Chowk market mein. Water supply cut off. More fire engines needed urgently.",
    # Cyclones / Storms
    "Cyclone Biparjoy approaching Kutch coast. Fishing villages need immediate evacuation.",
    "Tauktae cyclone impact in Raigad district. Trees uprooted, roads blocked. Send help.",
    "Very heavy rain in Chennai due to depression. Velachery area waterlogged. Need boats.",
    "Storm in Odisha coast, Puri district. Fishermen stuck at sea. Coast Guard needed!",
    "Cyclone warning - coastal villages near Kakinada not yet evacuated. Need buses now.",
    # General emergencies
    "Medical emergency - insulin patient fainted at Connaught Place metro. Ambulance please!",
    "Road accident on NH-8 near Gurgaon toll. 3 injured. Need ambulance urgently.",
    "Child fell in borewell near Hisar, Haryana. NDRF team needed immediately. #SaveHim",
    "Landslide blocked NH-44 near Ramban J&K. 200 vehicles stranded. Need bulldozers.",
    "Multiple people injured after wall collapse in Bhiwandi. Send rescue immediately.",
    "Bus accident near Manali, Himachal Pradesh - hill road. 15 injured, hospital far.",
    "Train derailment near Balasore, Odisha. Many casualties. Blood donors needed urgently.",
    "Gas leak in BHEL plant Haridwar. Workers evacuating. Fire brigade and medical on way.",
    "Missing child - 8 yr old boy lost near Kumbh Mela Prayagraj. Police help needed.",
    "Heatwave in Rajasthan - 48 deg C in Churu. Elderly people collapsing. Medical camps!",
]

def add_hinglish_messages(disaster_df):
    print("[3/4] Adding Hinglish / Indian-context messages ...")
    hinglish_rows = []
    for i, msg in enumerate(HINGLISH_MESSAGES, start=1):
        hinglish_rows.append({
            "message_id" : f"HI_{i:04d}",
            "raw_text"   : msg,
            "keyword"    : None,
            "location"   : None,
            "source"     : "manual_hinglish_indian_context",
        })
    hinglish_df = pd.DataFrame(hinglish_rows)
    combined = pd.concat([disaster_df, hinglish_df], ignore_index=True)
    print(f"  Total messages after merge: {len(combined)}")
    return combined

# ---------------------------------------------------------------------------
# Step 4 – Clean & save
# ---------------------------------------------------------------------------
def clean_and_save(df):
    print("[4/4] Cleaning and saving corpus ...")

    # Basic cleaning
    df["raw_text"] = df["raw_text"].astype(str).str.strip()
    df = df[df["raw_text"].str.len() > 10].copy()          # drop near-empty rows
    df = df.drop_duplicates(subset=["raw_text"]).copy()     # drop exact duplicates
    df = df.reset_index(drop=True)
    df["message_id"] = df.apply(
        lambda r: r["message_id"] if str(r["message_id"]).startswith("HI_")
                  else f"KG_{int(r['message_id']):05d}",
        axis=1
    )

    out_path = os.path.join(PROC_DIR, "corpus.csv")
    df[["message_id", "raw_text", "source"]].to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n  Corpus saved  : {out_path}")
    print(f"  Total messages: {len(df)}")
    print("\nDone! Proceed to Phase 2 - Annotation.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(os.path.join(RAW_DIR, "train.csv")):
        download_kaggle_data()
    else:
        print("[1/4] train.csv already exists - skipping download.")

    disaster_df = load_and_filter()
    combined    = add_hinglish_messages(disaster_df)
    clean_and_save(combined)
