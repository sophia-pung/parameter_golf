# Dual-Representation JEPA Architecture

A proposed architecture for the Parameter Golf challenge that combines a dual-input embedding pipeline with a JEPA-style joint training objective.

---

## Run

```bash
# 1. Create and activate a virtual environment
python3 -m venv pyvenv
source pyvenv/bin/activate

# 2. Install dependencies
pip install -e .

# 3. Download a sample training corpus (~3MB of text → data/corpus.txt)
python src/data.py

# 4. Train the model (checkpoints saved to checkpoints/ every 5000 steps)
python src/train.py --config configs/base.yaml

# 5. Compress the trained model to a ≤16MB submission artifact
python src/compress.py --checkpoint checkpoints/final.pt --output artifact/model.pt.gz
```

## Runs

### 03/24/2026 — 9 PM

| Setting | Value |
|---|---|
| Corpus | Project Gutenberg *War and Peace* (~3.2M chars, 853,923 tokens) |
| Tokenizer | tiktoken gpt2 (vocab_size = 50,257) |
| Model | 10,904,384 parameters (d_model=128, n_layers=6, n_heads=4, d_sym=64) |
| Device | Apple MPS (Apple Silicon) |
| Batch size | 4 sequences × 256 tokens |
| Max LR | 3e-4 (500-step linear warmup + cosine decay) |
| λ_JEPA | 0.1 |

**Loss curve (sampled):**

| Step | LR | LM loss | bpt |
|---|---|---|---|
| 100 | 5.94e-05 | 27.45 | 39.6 |
| 500 | 2.99e-04 | 14.42 | 20.8 |
| 700 | 3.00e-04 | 8.88 | 12.8 |
| 1100 | 3.00e-04 | 6.18 | 8.9 |
| 1500 | 3.00e-04 | 5.83 | 8.4 |
| 5300 | 2.94e-04 | 4.40 | 6.3 |
| 8000 | 2.85e-04 | 4.69 | 6.8 |

**Notes:**
- Loss crossed below the 15.6 bpt random-baseline at step ~700
- bpt plateaued in the 6.1–7.1 range from step ~5000 onward, indicating the model began oscillating around the optimum for this corpus size
- JEPA auxiliary loss remained stable at 0.003–0.006 throughout — no interference with the LM objective
- Corpus is too small (~853k tokens) for 10k steps without overfitting; a larger dataset is the main lever for improvement

---

## Overview

Standard language models map token IDs to a single embedding stream before passing through a transformer. This design introduces a **second symbolic stream** (Stream B) that runs in parallel with the normal embedding lookup (Stream A), then merges both representations before the transformer.

The model is then trained with two simultaneous objectives:
- **LM Head (A):** next-token prediction (the standard compression objective)
- **JEPA Head (B):** future-state prediction, encouraging the model to learn structured, predictive representations beyond surface-level token distributions

---

## Runs



## Pipeline

```
Text
 ↓
Tokenizer (or byte-level later)
 ↓
Token IDs
 ↓
┌─────────────────────┬──────────────────────┐
│ Stream A            │ Stream B             │
│ Normal embedding    │ Symbol generator     │
└─────────────────────┴──────────────────────┘
            ↓
          Combine
            ↓
      Positional encoding  ← see note below
            ↓
        Transformer
            ↓
       Hidden states (H)
         /        \
        /          \
 LM Head (A)   JEPA Head (B)
      ↓               ↓
 next-token      future-state
 prediction       prediction
```

---

## Streams

### Stream A — Normal Embedding
Standard learned embedding lookup: each token ID maps to a dense vector of dimension `d_model`. Equivalent to the embedding table in a conventional transformer.

### Stream B — Symbol Generator
A learned module that generates auxiliary symbolic features for each token. The intent is to inject structured, discrete-like signals (e.g. morphological, syntactic, or positional priors) alongside the soft embeddings from Stream A.

Stream B outputs a vector of the same dimension as Stream A. The two streams are combined (e.g. via addition or concatenation followed by a projection) before positional encoding.

---

## Positional Encoding

The diagram uses an explicit positional encoder block after the stream merge.

> **Note from Gokul Srinivasan:** *"This is usually handled by RoPE in the Q,K layers in modern LLMs."*

RoPE (Rotary Position Embedding) applies a rotation to the query and key vectors at each attention layer rather than adding positional information to the residual stream up front. This has several practical advantages:
- No dedicated positional parameters that count against the 16MB limit
- Naturally length-generalizable
- Compatible with grouped-query attention (GQA / KV head reduction)

**Recommendation:** Replace the standalone positional encoder block with RoPE applied inside each attention layer's Q and K projections. This is parameter-free in terms of learned weights and is how modern efficient transformers (LLaMA, Gemma, etc.) handle sequence order.

---

## Training Objective

The model is optimized with a joint loss:

```
L = L_LM + λ · L_JEPA
```

- **L_LM** — cross-entropy next-token prediction loss (directly optimizes the challenge metric, bits per byte)
- **L_JEPA** — future-state prediction loss in representation space (encourages the hidden states to capture predictive, structured information about upcoming tokens rather than just predicting the next surface token)
- **λ** — a scalar weight controlling the JEPA regularization strength

The intuition is that L_JEPA acts as an auxiliary signal that shapes the representation toward more compressible, structured features — potentially improving L_LM beyond what standard autoregressive training achieves.

---

## Parameter Budget Considerations

With a hard 16MB compressed artifact limit:

| Component | Notes |
|-----------|-------|
| Embedding table (Stream A) | Tied with output head to save parameters |
| Symbol generator (Stream B) | Lightweight MLP or lookup; keep small |
| Positional encoding | Use RoPE (zero learned params) |
| Transformer layers | Primary capacity, tune depth × width |
| LM Head | Tied to Stream A embedding table |
| JEPA Head | Small projection head; discarded at eval if not needed |

The JEPA Head is only used during training. If it is discarded before serialization, it contributes zero bytes to the submission artifact.

---

## Open Questions

- What architecture should Stream B use? Options: a small learned embedding table on token IDs, a character/byte-level CNN, or a fixed rule-based feature extractor.
- What is the right target for the JEPA Head — next hidden state, a future window of hidden states, or an aggregated summary?
- How sensitive is L_LM to λ? A too-large λ may hurt compression performance.
- Does the dual stream help at the extreme vocabulary sizes (1024 tokens) used in this challenge, or does it only matter at large vocabularies?
