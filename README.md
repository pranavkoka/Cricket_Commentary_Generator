# Cricket Commentary Generation with Decoder-Only Transformers

Given a short cricket commentary label (e.g. *"FOUR"*, *"1 run"*, *"OUT"*), the model generates a plausible long-form ball-by-ball commentary sentence. Built from scratch using a decoder-only transformer architecture.

---

## Why This Project?

Ball-by-ball cricket commentary is rich, structured, and domain-specific. This repo tests whether a transformer trained purely on commentary data can pick up enough statistical patterns to generate coherent, cricket-appropriate sentences without any pre-trained weights or external language models.

---

## Dataset

[Cricket Scorecard and Commentary Dataset](https://www.kaggle.com/datasets/raghuvansht/cricket-scorecard-and-commentary-dataset/data) on Kaggle. The task is sequence-to-sequence generation: a short commentary label goes in, a full commentary sentence comes out. The dataset covers ball-by-ball commentary from Test matches.

---

## Repository Structure

```
├── commentary.ipynb           # Baseline transformer model (v1)
├── commentary_improved.ipynb  # Improved transformer model (v2)
└── README.md
```

---

## Models

### v1 — Baseline (`commentary.ipynb`)

A decoder-only transformer built entirely from scratch.

**Architecture**

```
Word Embeddings
    → Positional Encoding
    → [ Layer Norm → Self-Attention → Residual Connection
        → Layer Norm → Feed-Forward Network ] × 3
    → MLP Head
```

| Detail | Value |
|---|---|
| Embedding dimension | 200 |
| Transformer blocks | 3 |
| Tokenization | Byte Pair Encoding (BPE) |
| Optimizer | Mini-batch SGD |
| Cross-entropy loss | ~4.3 |
| Random baseline loss | ~8.5 |

Positional encoding, self-attention, feed-forward, and layer norm are all implemented from scratch with no framework modules.

Generated sentences are grammatically structured but don't make much cricketing sense.

---

### v2 — Improved (`commentary_improved.ipynb`)

The same architecture, refactored with AI assistance and scaled up.

**Changes from v1**

- Embedding dimension increased from 200 to 512
- Transformer blocks increased from 3 to 6
- Migrated to `torch.nn.Module` for all components
- Self-attention and feed-forward layers use `torch.nn.Linear` instead of raw weight matrices
- Residual dropout added after each sub-layer
- Optimizer switched from SGD to AdamW

| Detail | Value |
|---|---|
| Embedding dimension | 512 |
| Transformer blocks | 6 |
| Tokenization | Byte Pair Encoding (BPE) |
| Optimizer | AdamW |
| Cross-entropy loss | ~2.4 |

Generated sentences are well-formed and largely make cricketing sense. Results by outcome type:

| Outcome | Generation quality |
|---|---|
| No run | Good |
| 1 run | Good |
| 2 runs | Good |
| FOUR | Good |
| 3 runs | Inconsistent |
| SIX | Inconsistent |
| OUT | Inconsistent |
| 5 wides | Inconsistent |

The inconsistent outcomes are also the rarest in the training data. Test matches don't produce many boundaries, wickets, or extras relative to dot balls and singles, so the model has seen far fewer examples of these to learn from.

---

## Results Summary

| Model | Loss | Random baseline | Quality |
|---|---|---|---|
| v1 (baseline) | 4.3 | 8.5 | Structured English, poor cricket |
| v2 (improved) | 2.4 | 8.5 | Coherent commentary for common outcomes |

---

## Future Work

**Class imbalance.** Rare outcomes like SIX, OUT, 3 runs, and wides are underrepresented in Test match data. Oversampling or a class-balanced sampling strategy during training should meaningfully improve generation quality for these cases.

**Scale.** Larger embedding dimensions and more blocks would likely push the loss down further.

**Conditional generation.** Feeding in match context (over number, current score, pitch conditions) alongside the outcome label could improve semantic coherence.

**Evaluation.** Cross-entropy loss is a blunt instrument here. BLEU scores or domain-specific human evaluation would give a clearer picture of commentary quality.

---

## Acknowledgements

Dataset provided by [raghuvansht on Kaggle](https://www.kaggle.com/datasets/raghuvansht/cricket-scorecard-and-commentary-dataset/data).
