# NLP & Transformers

> **Week:** 8 · **Status:** ✅ written · **From-scratch build:** no (conceptual; a numpy attention sketch is in [§4](#4-self-attention-from-scratch-numpy-sketch))
> **One-line:** Represent tokens as dense **embeddings**, then use **self-attention** to let every token look at every other token and pull in context — replacing the sequential bottleneck of RNNs with a fully **parallel**, content-based mixing operation. Stack attention + an MLP into a **transformer block**, and that block is the engine behind every modern LLM.

## Table of contents
- [1. Intuition](#1-intuition)
- [2. Tokenization & embeddings](#2-tokenization--embeddings)
- [3. The attention mechanism](#3-the-attention-mechanism)
  - [3.1 Scaled dot-product attention](#31-scaled-dot-product-attention)
  - [3.2 Multi-head attention](#32-multi-head-attention)
  - [3.3 Self- vs. cross-attention; masking](#33-self--vs-cross-attention-masking)
- [4. Self-attention from scratch (numpy sketch)](#4-self-attention-from-scratch-numpy-sketch)
- [5. The transformer block](#5-the-transformer-block)
- [6. Architectures: encoder / decoder / both](#6-architectures-encoder--decoder--both)
- [7. Why transformers beat RNNs](#7-why-transformers-beat-rnns)
- [8. With libraries](#8-with-libraries)
- [9. When to use / not use](#9-when-to-use--not-use)
- [10. Pitfalls & gotchas](#10-pitfalls--gotchas)
- [11. Interview questions](#11-interview-questions)
- [12. Connections](#12-connections)

---

## 1. Intuition

The core problem in NLP is **context**: the meaning of "bank" depends on whether the sentence mentions a river or money. Older models read left-to-right (RNNs/LSTMs), compressing all prior context into a single fixed-size hidden state — a bottleneck that forgets long-range information and can't be parallelized (step *t* needs step *t−1*).

**Self-attention** solves both at once. Instead of a running summary, every token directly computes a **weighted average of all other tokens' representations**, where the weights are how relevant each other token is to it. "It" can look straight at "the dog" three words back; "bank" can look at "river." There's no sequential dependency, so a whole sequence is processed in **one parallel matrix operation** — which is exactly what made training on internet-scale data feasible. ⭐

> Mental model: attention is a **soft, differentiable dictionary lookup**. Each token emits a **query** ("what am I looking for?"); every token offers a **key** ("what do I contain?") and a **value** ("what I'll contribute if attended to"). You match queries against keys to get attention weights, then return the weighted sum of values.

## 2. Tokenization & embeddings

- **Tokenization** splits text into units. Word-level has huge vocabularies and chokes on unknown words; character-level makes sequences too long. The modern standard is **subword** tokenization — **BPE** / WordPiece / SentencePiece — which learns a vocabulary of frequent character chunks (e.g. "token", "##ization"). It handles rare/unseen words by composing pieces and keeps the vocabulary bounded (~30k–100k). ⭐
- **Embeddings** map each token id to a dense vector in $\mathbb{R}^{d}$. Static embeddings (word2vec, GloVe) gave each word **one** vector learned so that *words in similar contexts get similar vectors* (the distributional hypothesis) — enough to capture analogies (king − man + woman ≈ queen). Their limitation: one vector per word regardless of context. **Transformers produce contextual embeddings** — the representation of "bank" changes with its sentence. ⭐
- **Positional encoding:** attention is **permutation-invariant** (it's a weighted sum — it has no inherent notion of order). So we **add position information** to the token embeddings — original sinusoidal encodings, or learned/rotary (RoPE) positions in modern LLMs. Without it, "dog bites man" = "man bites dog." ⭐

## 3. The attention mechanism

### 3.1 Scaled dot-product attention

Project the input $X \in \mathbb{R}^{n\times d}$ into queries, keys, values via learned matrices: $Q = XW_Q$, $K = XW_K$, $V = XW_V$. Then:

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Step by step:
1. **$QK^\top$** — every query dotted with every key → an $n\times n$ matrix of raw relevance scores (token *i* vs token *j*).
2. **$/\sqrt{d_k}$** — scale by the square root of the key dimension. Without it, dot products grow with $d_k$, pushing softmax into saturated regions where gradients vanish. This is *why* it's "scaled" dot-product attention. ⭐
3. **softmax (row-wise)** — turn each row of scores into a probability distribution: token *i*'s attention weights over all tokens.
4. **$\times V$** — weighted sum of value vectors → each token's new, context-mixed representation.

### 3.2 Multi-head attention

One attention is one "relationship lens." **Multi-head attention** runs $h$ attentions in parallel, each with its own $W_Q, W_K, W_V$ projecting into a smaller subspace ($d/h$), then concatenates and projects back. Different heads specialize — one tracks syntax, another coreference, another nearby tokens. It's the attention analogue of having multiple filters in a CNN. ⭐

### 3.3 Self- vs. cross-attention; masking

- **Self-attention:** Q, K, V all come from the **same** sequence — tokens attend to each other within one input.
- **Cross-attention:** queries from one sequence, keys/values from **another** (e.g. the decoder attending to the encoder's output in translation).
- **Causal (masked) attention:** in a decoder/generative LM, a token must not see the future. Set the upper-triangular scores to $-\infty$ **before** softmax so position *i* only attends to positions $\le i$. This is what makes GPT-style autoregressive generation valid. ⭐

## 4. Self-attention from scratch (numpy sketch)

No `fit`/`predict` build for this week, but the core operation is short — being able to write it cold is the interview target:

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def self_attention(X, Wq, Wk, Wv, causal=False):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv          # (n, d_k) each
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)           # (n, n) relevance, SCALED
    if causal:                                # block attention to the future
        n = scores.shape[0]
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        scores[mask] = -np.inf
    weights = softmax(scores, axis=-1)        # rows sum to 1
    return weights @ V                        # (n, d_v) context-mixed output
```

The four lines to memorize: **project to Q/K/V**, **QKᵀ/√dₖ**, **softmax**, **×V**. Everything else (multi-head, the block) wraps this.

## 5. The transformer block

A transformer layer stacks two sublayers, each wrapped in a **residual connection + layer norm**:

```
x = x + MultiHeadAttention(LayerNorm(x))      # mix information ACROSS tokens
x = x + FeedForward(LayerNorm(x))             # process EACH token independently
```

- **Attention sublayer** mixes information *between* tokens.
- **Feed-forward sublayer** is a small per-token MLP (Linear → GELU → Linear, usually 4× wider) — it adds nonlinear capacity *within* each token's representation. (This is literally the [Week 7 MLP](07-neural-networks.md).)
- **Residual connections** let gradients flow through deep stacks (the [vanishing-gradient](07-neural-networks.md#24-backpropagation) fix); **layer norm** stabilizes training (batch-size-independent, unlike batch norm — right for variable-length sequences). ⭐

Stack $N$ of these blocks (12 for BERT-base, dozens-to-100+ for large LLMs) and you have a transformer.

## 6. Architectures: encoder / decoder / both

- **Encoder-only (BERT):** bidirectional self-attention (every token sees the whole sentence). Pretrained with **masked language modeling** (predict masked-out tokens). Great for **understanding** tasks — classification, NER, embeddings/retrieval. Not generative.
- **Decoder-only (GPT, Llama):** causal/masked self-attention; pretrained to **predict the next token**. Generative; this is the architecture of modern LLMs. ⭐
- **Encoder–decoder (T5, original Transformer):** encoder reads the input, decoder generates output attending to it via **cross-attention**. Natural for **seq2seq** — translation, summarization.

## 7. Why transformers beat RNNs

| | RNN / LSTM | Transformer |
|---|---|---|
| **Computation** | Sequential (step *t* needs *t−1*) | **Fully parallel** across positions |
| **Long-range deps** | Decays through many steps; vanishing gradients | **Direct** O(1) path between any two tokens |
| **Context** | Fixed-size hidden state bottleneck | Every token attends to all tokens |
| **Cost** | O(n) time, O(1) attention | O(n²·d) attention — **quadratic in sequence length** |

The win is parallelism + direct long-range connections; the price is the **O(n²) attention cost** that makes long contexts expensive (motivating FlashAttention, sparse/linear attention, etc.). ⭐

## 8. With libraries

```python
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("bert-base-uncased")     # subword tokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tok("a transformer reads the whole sentence", return_tensors="pt")
out = model(**inputs)
out.last_hidden_state        # (1, n_tokens, hidden) contextual embeddings

# Sentence embeddings for retrieval / similarity:
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("all-MiniLM-L6-v2").encode(["hello", "world"])
```

The practical workflow is almost always **use a pretrained model**: take embeddings (encoder) or generate/finetune (decoder) — see [Week 9](09-llms-agentic.md).

## 9. When to use / not use

**Use transformers when:** text/sequence/language tasks (classification, NER, QA, summarization, translation, retrieval, generation), and increasingly vision (ViT), audio, and multimodal. When you have a pretrained model to leverage (almost always), they're the default.

**Avoid / overkill when:** tiny datasets or simple text problems where a **TF-IDF + logistic regression / linear SVM** baseline is faster, cheaper, interpretable, and competitive; latency/compute-constrained settings; tabular data (use a [GBM](05-ensembles-boosting.md)). Always state a cheap baseline first. ⭐

## 10. Pitfalls & gotchas
- **Forgetting positional encoding** → the model is order-blind (attention is permutation-invariant). ⭐
- **Forgetting the $\sqrt{d_k}$ scaling** → large dot products saturate softmax, gradients vanish, training stalls. ⭐
- **No causal mask in a generative model** → the model "cheats" by seeing future tokens during training, then fails at inference. ⭐
- **Quadratic memory blow-up** on long sequences — attention is O(n²); long documents need chunking, sparse/linear attention, or FlashAttention.
- **Confusing self- and cross-attention** — self = within one sequence; cross = decoder queries attending to encoder keys/values.
- **Treating static embeddings as contextual** — word2vec/GloVe give one vector per word; transformers give context-dependent vectors.
- **Using BERT to *generate* text** — it's an encoder (understanding), not autoregressive; use a decoder (GPT) for generation.
- **Tokenization mismatch** — using a different tokenizer than the model was trained with corrupts everything; always pair them.
- **Ignoring sequence-length limits / truncation** — inputs over the context window get silently truncated.

## 11. Interview questions
Must answer cold (⭐):
- ⭐ **Explain self-attention / the Q-K-V mechanism.** Each token emits a query, key, and value. Attention weights = softmax(QKᵀ/√dₖ); the output is those weights times V — a content-weighted average of all tokens' values. (§3.1)
- ⭐ **Why divide by √dₖ?** Dot products grow with dimension; without scaling, softmax saturates and gradients vanish. The scaling keeps variance controlled. (§3.1)
- ⭐ **Why do transformers need positional encoding?** Attention is permutation-invariant (a weighted sum has no order); position info must be added so the model knows token order. (§2)
- ⭐ **Multi-head attention — what and why?** Run several attentions in parallel in different subspaces and concatenate; each head learns a different relationship (syntax, coreference, locality). (§3.2)
- ⭐ **Transformers vs. RNNs?** Transformers are parallel and give every token a direct path to every other (no long-range decay); RNNs are sequential with a fixed-size bottleneck. Trade-off: attention is O(n²). (§7)
- ⭐ **What is a causal mask and when do you use it?** In a generative/decoder LM, set future-position scores to −∞ before softmax so a token only attends to earlier tokens — required for valid autoregressive generation. (§3.3)
- ⭐ **Encoder-only vs. decoder-only vs. encoder–decoder?** BERT (bidirectional, understanding, MLM), GPT (causal, generation, next-token), T5/seq2seq (cross-attention, translation/summarization). (§6)
- ⭐ **What's in a transformer block?** Multi-head self-attention + a per-token feed-forward MLP, each wrapped in a residual connection and layer norm. (§5)
- **What is subword tokenization and why use it (BPE)?** Splits text into frequent character chunks — bounded vocabulary, handles rare/unseen words by composing pieces. (§2)
- **Static vs. contextual embeddings?** word2vec/GloVe give one fixed vector per word; transformers produce a different vector per occurrence depending on context. (§2)
- **What's the complexity of self-attention and why does it matter?** O(n²·d) in sequence length — the bottleneck for long contexts. (§7)
- **Why layer norm instead of batch norm in transformers?** It normalizes per token across features, independent of batch size — right for variable-length sequences. (§5)
- **Why residual connections?** They give gradients a direct path through deep stacks, preventing vanishing gradients. (§5, ties to [Week 7](07-neural-networks.md#24-backpropagation))

## 12. Connections
- **Built on [Week 7 neural nets](07-neural-networks.md)** — the feed-forward sublayer *is* an MLP; residuals and layer norm are the deep-learning-stability tools from §6 there; softmax + the $\hat y - y$ gradient power next-token training.
- **Embeddings → retrieval** underpins **RAG** in [Week 9](09-llms-agentic.md) — encoder embeddings + nearest-neighbor search (ties to [KNN, Week 2](02-knn.md) and [ANN serving](../system-design/00-framework.md)).
- **Decoder-only LM** is the substrate for everything in [Week 9: LLMs & Agentic AI](09-llms-agentic.md) — pretraining is next-token prediction over this architecture.
- **Cheap baseline first** — TF-IDF + logistic regression ([Foundations](00-foundations.md)) before reaching for a transformer.
- Flashcards: [flashcards/08-nlp-transformers.md](../flashcards/08-nlp-transformers.md).

---
### Sources
- Vaswani et al. (2017), *Attention Is All You Need* — the original transformer.
- Jay Alammar, *The Illustrated Transformer* — the clearest visual walkthrough.
- Géron, *Hands-On ML* — NLP with RNNs and Attention; Transformers chapter.
- Devlin et al. (2018) *BERT*; Radford et al. *GPT* — the encoder vs. decoder pretraining objectives.
