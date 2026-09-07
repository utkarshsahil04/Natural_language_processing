# Corpus of Human-AI Conversation with Synthetic Annotation

## 1. Problem Statement

Create a corpus of human-AI interactions and analyze differences in sentence structure between human-generated responses and AI-generated responses.

This project uses **classic ML only** (clustering, decision trees, boosting).

## 2. Corpus

- Total responses: **360**
- Human: **180**
- AI: **180**
- Prompt categories: advice, factual, creative, opinion
- Format: `response_id, prompt_id, prompt, response_text, source_type`

## 3. Method

### Phase 1 - Collection
Paired prompts answered by human-style and AI-style responses.

### Phase 2 - Linguistic features
Extracted with spaCy:

- avg sentence length, parse tree depth, type-token ratio
- passive ratio, hedging rate, discourse marker rate
- POS percentages (noun/verb/adj/adv/pron)
- word length and punctuation ratio

### Phase 3 - Clustering
K-Means and Hierarchical clustering on standardized features (labels hidden).
Optimal k checked via elbow + silhouette. PCA/t-SNE used for visualization.

### Phase 4 - Classification validation
Decision Tree, AdaBoost, and Gradient Boosting predict human vs AI.

## 4. Results

### Feature means by source

```
source_type              AI   human
avg_sentence_length   9.500   6.470
num_sentences         3.067   2.256
num_words            29.050  14.300
parse_tree_depth      4.344   3.448
type_token_ratio      0.947   0.952
passive_ratio         0.094   0.053
hedge_count           0.000   0.244
hedge_rate            0.000   0.019
discourse_count       0.128   0.189
discourse_rate        0.004   0.012
avg_word_length       5.849   4.590
punct_ratio           0.155   0.157
pct_noun              0.328   0.196
pct_verb              0.153   0.168
pct_adj               0.162   0.126
pct_adv               0.072   0.134
pct_pron              0.021   0.117
```

### Clustering summary

```
              method  silhouette  ARI_vs_true
           KMeans_k2    0.301907     1.000000
     Hierarchical_k2    0.297040     0.881183
best_k_by_silhouette    0.384153     6.000000
```

Figures: `elbow_silhouette.png`, `kmeans_pca.png`, `kmeans_tsne.png`, `true_labels_pca.png`

### Classification metrics

```
           model  accuracy  precision  recall  f1  cv_f1_mean
    DecisionTree       1.0        1.0     1.0 1.0         1.0
        AdaBoost       1.0        1.0     1.0 1.0         1.0
GradientBoosting       1.0        1.0     1.0 1.0         1.0
```

### Top distinguishing features (Gradient Boosting)

```
         Unnamed: 0   importance
          num_words 1.000000e+00
   type_token_ratio 3.529494e-16
    avg_word_length 2.092087e-16
   parse_tree_depth 1.509348e-16
avg_sentence_length 7.794687e-17
           pct_verb 7.717082e-17
        punct_ratio 8.691806e-18
      passive_ratio 0.000000e+00
```

### Did clustering align with human/AI?

- K-Means ARI vs true labels: **1.000**
- Hierarchical ARI vs true labels: **0.881**

ARI close to 1 means clusters recovered the human/AI split well. Lower ARI means linguistic styles overlap more than expected.

## 5. Linguistic interpretation

- **Syntactic layer:** parse-tree depth and sentence length often differ; AI replies tend to be more uniformly structured.
- **Lexical/morphological layer:** type-token ratio and average word length capture vocabulary style differences.
- **Discourse/pragmatic layer:** discourse markers and hedging reflect politeness/certainty patterns.
- **POS distribution:** noun/verb/adj balance indicates informational vs conversational style.

## 6. Conclusion

Human and AI responses can be compared using transparent linguistic features and classic ML. Clustering checks whether structure alone separates sources; supervised models quantify separability and highlight which features matter most.

## 7. How to reproduce

```bash
python run_pipeline.py
```

## Suggested submission structure

1. Introduction & problem statement
2. Related linguistic background
3. Corpus design and collection
4. Feature engineering
5. Clustering experiments
6. Classification validation
7. Discussion of human vs AI differences
8. Limitations & future work
9. Conclusion