"""
Flask backend for Emergency NLP Testing UI
Endpoint: POST /api/analyze   {text, model}  -> {tokens, labels}
Endpoint: GET  /api/status    -> model status
Endpoint: POST /api/train     -> runs full pipeline
"""

import os, sys, pickle, subprocess
import numpy as np
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, template_folder="templates", static_folder="static")

LOC_PREPS = {"in","at","near","by","from","around","along","across","beside",
             "outside","inside","behind","beneath","between","within","onto","into","towards"}
ACT_VERBS = {"need","send","require","help","rescue","evacuate","dispatch","deploy",
             "bring","call","request","provide","save","assist","supply","get",
             "airlift","mobilize","respond","relief"}
LOC_WORDS = {"area","district","region","zone","sector","block","ward","village","town",
             "city","road","bridge","river","colony","nagar","marg","lane","chowk",
             "ghat","station","coast","highway","street","avenue","park","island",
             "valley","mountain","hill","beach","shore","dam","camp","shelter",
             "junction","crossing","flyover","border","port"}

# ── State ────────────────────────────────────────────────────────────────────
models_state = {"loaded": False, "models": {}, "enc": {}}
spacy_state  = {"loaded": False, "nlp": None}

def try_load_models():
    enc_path = os.path.join(MODEL_DIR, "encoders.pkl")
    if not os.path.exists(enc_path):
        return False
    try:
        with open(enc_path, "rb") as f:
            models_state["enc"] = pickle.load(f)
        m = {}
        for name in ["decision_tree","random_forest","adaboost"]:
            p = os.path.join(MODEL_DIR, f"{name}.pkl")
            if os.path.exists(p):
                with open(p,"rb") as f:
                    m[name] = pickle.load(f)
        models_state["models"] = m
        models_state["loaded"] = len(m) > 0
        return models_state["loaded"]
    except Exception as e:
        print(f"Model load error: {e}")
        return False

def try_load_spacy():
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            import importlib; importlib.reload(spacy)
            nlp = spacy.load("en_core_web_sm")
        spacy_state["nlp"]    = nlp
        spacy_state["loaded"] = True
        return True
    except Exception as e:
        print(f"spaCy load error: {e}")
        return False

# Load on startup
try_load_models()
try_load_spacy()

# ── Feature extraction (mirrors feature_engineering.py) ──────────────────────
def extract_features(text):
    nlp = spacy_state["nlp"]
    enc = models_state["enc"]
    if nlp is None or not enc:
        return None, None

    doc      = nlp(str(text)[:500])
    pos_le   = enc.get("pos_le")
    dep_le   = enc.get("dep_le")

    ent_map = {}
    for ent in doc.ents:
        if ent.label_ in ("GPE","LOC","FAC"):
            for i, tok in enumerate(ent):
                ent_map[tok.i] = i

    tokens_info, features = [], []
    n = len(doc)

    for i, tok in enumerate(doc):
        prev_tok  = doc[i-1] if i > 0   else None
        next_tok  = doc[i+1] if i < n-1 else None
        prev_pos  = prev_tok.pos_ if prev_tok else "X"
        prev_lem  = prev_tok.lemma_.lower() if prev_tok else ""
        next_pos  = next_tok.pos_ if next_tok else "X"

        def safe_enc(le, val):
            if le is not None and val in le.classes_:
                return int(le.transform([val])[0])
            return 0

        feat = [
            safe_enc(pos_le, tok.pos_),
            safe_enc(dep_le, tok.dep_),
            safe_enc(pos_le, prev_pos),
            safe_enc(pos_le, next_pos),
            int(prev_lem in LOC_PREPS),
            int(next_pos in ("NOUN","PROPN")),
            int(tok.lemma_.lower() in ACT_VERBS),
            int(tok.lemma_.lower() in LOC_WORDS),
            int(len(tok.text) > 0 and tok.text[0].isupper()),
            int(tok.i in ent_map),
            round(i / max(n-1,1), 4),
            round(n / 50.0, 4),
            int(str(tok.text).replace(".","").replace(",","").isdigit()),
        ]
        features.append(feat)
        tokens_info.append({
            "token": tok.text,
            "pos"  : tok.pos_,
            "dep"  : tok.dep_,
            "is_ent": int(tok.i in ent_map),
        })

    return np.array(features, dtype=float), tokens_info

TAG_CSS = {"B-LOC":"loc","I-LOC":"loc","B-ACT":"act","I-ACT":"act","O":"other"}

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           models_ready = models_state["loaded"],
                           spacy_ready  = spacy_state["loaded"],
                           model_list   = list(models_state["models"].keys()))

@app.route("/api/status")
def status():
    return jsonify({
        "models_loaded"   : models_state["loaded"],
        "spacy_ready"     : spacy_state["loaded"],
        "available_models": list(models_state["models"].keys()),
    })

@app.route("/api/reload")
def reload_models():
    ok = try_load_models()
    return jsonify({"success": ok, "available_models": list(models_state["models"].keys())})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data       = request.get_json(force=True)
    text       = data.get("text","").strip()
    model_name = data.get("model","random_forest")

    if not text:
        return jsonify({"error": "Please enter some text."}), 400
    if not models_state["loaded"]:
        return jsonify({"error": "Models not trained yet. Click 'Run Pipeline' first."}), 503
    if not spacy_state["loaded"]:
        return jsonify({"error": "spaCy not loaded."}), 503
    if model_name not in models_state["models"]:
        model_name = list(models_state["models"].keys())[0]

    X, tokens_info = extract_features(text)
    if X is None or len(X) == 0:
        return jsonify({"error": "Feature extraction failed."}), 500

    bio_le = models_state["enc"]["bio_le"]
    clf    = models_state["models"][model_name]
    y_pred = clf.predict(X)
    labels = bio_le.inverse_transform(y_pred)

    result_tokens = []
    for info, label in zip(tokens_info, labels):
        result_tokens.append({
            "token"  : info["token"],
            "pos"    : info["pos"],
            "dep"    : info["dep"],
            "label"  : label,
            "css"    : TAG_CSS.get(label,"other"),
        })

    # Summary stats
    loc_spans = [t["token"] for t in result_tokens if t["css"] == "loc"]
    act_spans = [t["token"] for t in result_tokens if t["css"] == "act"]

    return jsonify({
        "tokens"     : result_tokens,
        "model_used" : model_name,
        "loc_tokens" : loc_spans,
        "act_tokens" : act_spans,
    })

@app.route("/api/train", methods=["POST"])
def train():
    """Runs the full pipeline in background."""
    use_sample = request.get_json(force=True).get("use_sample", True)
    flag = "--skip-download" if use_sample else ""
    cmd  = [sys.executable, os.path.join(BASE_DIR, "run_pipeline.py")]
    if use_sample:
        cmd.append("--skip-download")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        ok = result.returncode == 0
        try_load_models()
        return jsonify({
            "success": ok,
            "stdout" : result.stdout[-3000:],
            "stderr" : result.stderr[-1000:],
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Pipeline timed out (>5 min)."}), 504
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print(f"Models loaded : {models_state['loaded']} — {list(models_state['models'].keys())}")
    print(f"spaCy ready   : {spacy_state['loaded']}")
    print("Open browser  : http://localhost:5000")
    app.run(debug=False, port=5000, host="0.0.0.0")
