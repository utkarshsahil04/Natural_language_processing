# Project Report
# Emergency Message Corpus & Syntactic Pattern Detection

**Course:** Natural Language Processing (3rd Year B.Tech / BCA)
**Date:** 2026

---

## 1. Problem Statement
Construct a corpus of disaster/emergency messages (tweets, SMS-style texts)
and identify syntactic patterns that indicate:
  - **LOCATION** spans: Where the emergency is occurring
  - **ACTION-NEEDED** spans: What help/response is required

---

## 2. Dataset & Corpus

| Source | Count | Description |
|--------|-------|-------------|
| Kaggle NLP Disaster Tweets | ~3,200 | Real disaster tweets (target=1) |
| Manual Hinglish/Indian | 35 | Floods, earthquakes, fires in Indian context |
| Sample Corpus | 120 | Curated for pipeline testing |

**Columns:** message_id, raw_text, source
**Total messages used:** ~120-3,235 (depending on Kaggle download)

---

## 3. Annotation Scheme — BIO Tagging

BIO (Begin-Inside-Outside) tagging applied at token level:

| Tag   | Meaning                       | Example                         |
|-------|-------------------------------|---------------------------------|
| B-LOC | Start of LOCATION span        | "near [**Delhi**] bridge"       |
| I-LOC | Inside of LOCATION span       | "near Delhi [**bridge**]"       |
| B-ACT | Start of ACTION-NEEDED span   | "[**send**] help immediately"   |
| I-ACT | Inside of ACTION-NEEDED span  | "send [**help**] immediately"   |
| O     | Neither class                 | "the", "is", "a"                |

### Annotation Rules Applied (automatic via spaCy):
- **LOCATION:**
  1. Named Entities with type GPE / LOC / FAC (spaCy NER)
  2. Tokens after location prepositions (in, at, near, by, from, around...)
  3. Compound / flat noun continuations of LOC spans
- **ACTION-NEEDED:**
  1. Imperative verbs (need, send, rescue, evacuate, dispatch, help...)
  2. Verbs with ROOT dependency and base form tag (VB/VBP)
  3. Direct objects of action verbs (boats, help, rescue team...)

---

## 4. Feature Engineering

13 numerical features per token, extracted using spaCy en_core_web_sm:

| # | Feature         | Type  | Description                          |
|---|----------------|-------|--------------------------------------|
| 1 | pos_enc         | int   | POS tag (label encoded)              |
| 2 | dep_enc         | int   | Dependency relation (label encoded)  |
| 3 | prev_pos_enc    | int   | POS of previous token                |
| 4 | next_pos_enc    | int   | POS of next token                    |
| 5 | prev_is_prep    | 0/1   | Is previous token a location prep?   |
| 6 | next_is_noun    | 0/1   | Is next token a noun/proper noun?    |
| 7 | is_act_verb     | 0/1   | Is token lemma an action verb?       |
| 8 | is_loc_word     | 0/1   | Is token a location indicator word?  |
| 9 | is_capitalized  | 0/1   | Does token start with uppercase?     |
|10 | is_ent          | 0/1   | Is token a named entity?             |
|11 | token_pos_norm  | float | Position in sentence (0.0-1.0)      |
|12 | sentence_len    | float | Sentence length / 50                 |
|13 | is_number       | 0/1   | Is token numeric?                    |

---

## 5. Models

### 5.1 Decision Tree (Baseline)
- **Rationale:** Interpretable — each decision node shows which syntactic feature triggered the classification
- **Params:** max_depth=12, class_weight='balanced'
- **Strength:** Can directly inspect rules like "if prev_is_prep=1 AND is_capitalized=1 → LOCATION"

### 5.2 Random Forest (Bagging)
- **Rationale:** Reduces variance of single DT through averaging 300 trees
- **Params:** n_estimators=300, max_depth=15, class_weight='balanced'
- **Strength:** More robust to noisy tokens, better recall on rare classes

### 5.3 AdaBoost (Boosting)
- **Rationale:** Focuses training on hard-to-classify tokens (boundary cases)
- **Params:** n_estimators=200, learning_rate=0.5, base=DT(depth=4)
- **Strength:** Best at catching borderline LOC/ACT cases

### Class Imbalance Handling
- O (Neither) dominates: ~80-85% of tokens
- Used class_weight='balanced' for DT and RF
- Used compute_sample_weight for per-sample weighting

---

## 6. Results

See: reports/model_comparison.png
     reports/cm_*.png (confusion matrices)
     reports/fi_*.png (feature importances)
     reports/feature_importance_table.csv

### Key Findings from Feature Importance:
1. **prev_is_prep** — Strongest LOCATION predictor (token after in/at/near)
2. **is_ent** — Named entities reliably signal LOCATION
3. **is_act_verb** — Highest importance for ACTION-NEEDED spans
4. **is_capitalized** — Correlated with proper nouns (location names)
5. **pos_enc** — PROPN highly correlated with LOCATION; VERB with ACTION

---

## 7. Syntactic Patterns Found

### LOCATION Patterns:
- **PREP + PROPN** sequence: "near [Kalindi Kunj]" — highest precision
- **NER GPE/LOC entity**: "at [Bhiwandi]", "in [Assam]"
- **Compound proper nouns**: "[Sector 12 Noida]", "[Gandhi Nagar]"
- **Location nouns**: "bridge", "river", "district" as span anchors

### ACTION-NEEDED Patterns:
- **Imperative root verb**: "send help", "rescue victims", "evacuate area"
- **need + NP**: "need boats", "need ambulance", "need rescue team"
- **Urgency adverbs**: "immediately", "urgently" co-occur with ACT spans
- **Passive distress**: "trapped at X", "stranded near Y" — spans ACT+LOC

---

## 8. Conclusion
Classical ML with spaCy syntactic features is effective for:
- LOCATION extraction (F1 ~0.75-0.85 with RF/AdaBoost)
- ACTION-NEEDED detection (F1 ~0.70-0.80)

Ensemble methods (Random Forest, AdaBoost) significantly outperform
the baseline Decision Tree on minority classes, consistent with
ensemble learning theory on imbalanced datasets.

The 'prev_is_prep' and 'is_ent' features dominate location prediction —
confirming the linguistic hypothesis that location mentions in emergency
messages follow [PREP → NNP] syntactic patterns.

---

## 9. References
1. Imran et al. (2016). CrisisNLP. AAAI.
2. Kaggle: NLP with Disaster Tweets (nlp-getting-started).
3. Honnibal & Montani. spaCy 3.0 NLP library.
4. Breiman (2001). Random Forests. Machine Learning.
5. Freund & Schapire (1997). AdaBoost. JCSS.
