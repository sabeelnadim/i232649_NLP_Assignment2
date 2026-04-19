# i232649_NLP_Assignment2
# Sabeel Nadeem
# 23i 2649

# CS-4063 NLP Assignment 2 — Neural NLP Pipeline
**BBC Urdu Corpus | PyTorch from Scratch**

---

## Repository Structure

```
i23-XXXX_Assignment2/
├── i23-XXXX_Assignment2.ipynb     # Main notebook (all cells executed)
├── report.pdf                     # 2-3 page report
├── README.md                      # This file
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   └── word2idx.json
├── models/
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   └── transformer_cls.pt
└── data/
    ├── pos_train.conll
    ├── pos_test.conll
    ├── ner_train.conll
    └── ner_test.conll
```

---

## Requirements

```
Python 3.8+
PyTorch
NumPy
scikit-learn
matplotlib
seaborn
```

Install all dependencies:
```bash
pip install torch numpy scikit-learn matplotlib seaborn
```

---

## Input Files Required

Place these two files in the same directory as the notebook:

| File | Purpose |
|------|---------|
| `cleaned.txt` | Primary training corpus (all parts) |
| `raw.txt` | Raw corpus (used in Part 1 ablation C2) |

> **Note:** `Metadata.json` is NOT required. Topic labels are assigned automatically using keyword matching on the corpus text.

---

## How to Reproduce

### Step 1 — Open the notebook
Open `i23-XXXX_Assignment2.ipynb` in Kaggle or Jupyter.

### Step 2 — Enable GPU (Kaggle)
Go to **Settings → Accelerator → GPU T4 x2** before running.

### Step 3 — Upload input files
Upload `cleaned.txt` and `raw.txt` to the notebook's working directory.

### Step 4 — Run all cells in order
Run cells from top to bottom. Each part depends on variables from previous parts.

> **Important:** The corpus file `cleaned.txt` may be a single line. The code automatically splits on the Urdu sentence-ending character `۔` to extract individual sentences.

---

## Part Descriptions

### Part 1 — Word Embeddings
- Builds TF-IDF weighted term-document matrix (vocab capped at 10,000)
- Builds PPMI weighted word-word co-occurrence matrix (window k=5)
- Trains Skip-gram Word2Vec from scratch (d=100, K=10 negatives, 5 epochs)
- Evaluates nearest neighbours and analogy tests
- Compares four conditions: PPMI, SG-raw, SG-clean, SG-d200

### Part 2 — Sequence Labeling
- Annotates 500 sentences with POS (12 tags) and NER (BIO scheme, 9 tags)
- Trains 2-layer BiLSTM with dropout=0.5 and CRF+Viterbi for NER
- Evaluates frozen vs fine-tuned embeddings
- Runs 4 ablation studies (A1-A4)

### Part 3 — Transformer Encoder
- Implements full Transformer encoder from scratch (no nn.Transformer)
- 4 encoder blocks, 4 attention heads, CLS token classification
- 5-class topic classification on the BBC Urdu corpus
- Compares BiLSTM vs Transformer performance

---

## Output Files Generated

| File | Description |
|------|-------------|
| `tfidf_matrix.npy` | TF-IDF weighted term-document matrix |
| `ppmi_matrix.npy` | PPMI weighted co-occurrence matrix |
| `embeddings_w2v.npy` | Averaged Skip-gram embeddings (V+U)/2 |
| `word2idx.json` | Vocabulary index mapping |
| `bilstm_pos.pt` | Trained BiLSTM POS tagger weights |
| `bilstm_ner.pt` | Trained BiLSTM-CRF NER model weights |
| `transformer_cls.pt` | Trained Transformer classifier weights |
| `data/pos_train.conll` | POS training annotations (CoNLL format) |
| `data/pos_test.conll` | POS test annotations |
| `data/ner_train.conll` | NER training annotations (BIO CoNLL format) |
| `data/ner_test.conll` | NER test annotations |

---

## Notes
- All models implemented from scratch in PyTorch — no HuggingFace, no Gensim
- `nn.Transformer`, `nn.MultiheadAttention`, `nn.TransformerEncoder` are NOT used
- GPU is automatically detected and used if available
