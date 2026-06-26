# Flashcards: NLP & Transformers

> One `Q:`/`A:` pair per card, atomic answers. Tags in `[brackets]` for later Anki export.
> ✅ Seeded from the topic note's "Interview questions" section.

---

**Q:** Explain self-attention (the Q-K-V mechanism).
**A:** Each token emits a query, key, and value. Attention weights = softmax(QKᵀ/√dₖ); the output is those weights × V — a content-weighted average of all tokens' value vectors. It's a soft, differentiable dictionary lookup. `[transformers][attention]`

---

**Q:** Why divide attention scores by √dₖ?
**A:** Dot products grow with the key dimension; without scaling, softmax saturates into low-gradient regions. Dividing by √dₖ keeps the variance controlled so gradients flow. `[transformers][attention]`

---

**Q:** Why do transformers need positional encoding?
**A:** Attention is permutation-invariant (a weighted sum has no notion of order), so position information must be added to the embeddings — otherwise "dog bites man" = "man bites dog." `[transformers][positional]`

---

**Q:** What is multi-head attention and why use it?
**A:** Run several attentions in parallel, each projecting into a smaller subspace with its own Wq/Wk/Wv, then concatenate. Each head specializes (syntax, coreference, locality) — multiple relationship "lenses." `[transformers][attention]`

---

**Q:** Transformers vs. RNNs/LSTMs?
**A:** Transformers process all positions in parallel and give every token a direct O(1) path to every other (no long-range decay); RNNs are sequential with a fixed-size hidden-state bottleneck. Trade-off: attention is O(n²) in sequence length. `[transformers][rnn]`

---

**Q:** What is causal (masked) attention and when is it used?
**A:** In a decoder/generative LM, set future-position scores to −∞ before softmax so a token attends only to earlier tokens. Required for valid autoregressive (next-token) generation. `[transformers][decoder]`

---

**Q:** Encoder-only vs. decoder-only vs. encoder–decoder?
**A:** BERT = encoder, bidirectional, masked-LM pretraining, for understanding. GPT/Llama = decoder, causal, next-token, for generation. T5/original Transformer = encoder–decoder with cross-attention, for seq2seq (translation, summarization). `[transformers][architecture]`

---

**Q:** What's inside a transformer block?
**A:** Multi-head self-attention (mixes info across tokens) + a per-token feed-forward MLP (adds capacity within each token), each wrapped in a residual connection and layer norm. `[transformers][block]`

---

**Q:** What is subword tokenization (BPE) and why use it?
**A:** It splits text into frequent character chunks, learned from data. Keeps the vocabulary bounded and handles rare/unseen words by composing pieces — avoiding both huge word vocabularies and overly long character sequences. `[nlp][tokenization]`

---

**Q:** Static vs. contextual embeddings?
**A:** word2vec/GloVe give one fixed vector per word regardless of context; transformers produce a different vector per occurrence depending on the surrounding sentence (so "bank" differs by context). `[nlp][embeddings]`

---

**Q:** What is the computational complexity of self-attention and why does it matter?
**A:** O(n²·d) in sequence length n — the QKᵀ matrix is n×n. It's the bottleneck for long contexts, motivating sparse/linear attention and FlashAttention. `[transformers][complexity]`

---

**Q:** Why layer norm instead of batch norm in transformers?
**A:** Layer norm normalizes per token across features, independent of batch size and sequence length — right for variable-length sequences, where batch statistics would be unstable. `[transformers][normalization]`

---

**Q:** Why do transformer blocks use residual connections?
**A:** They give gradients a direct path through the deep stack of blocks, preventing vanishing gradients and enabling very deep models. `[transformers][residual]`

---

**Q:** Self-attention vs. cross-attention?
**A:** Self-attention: Q, K, V all from the same sequence (tokens attend within one input). Cross-attention: queries from one sequence, keys/values from another (e.g., decoder attending to encoder output). `[transformers][attention]`

---

**Q:** What does the distributional hypothesis say, and how do embeddings use it?
**A:** Words appearing in similar contexts have similar meanings. Embeddings are trained so context-similar words get nearby vectors, which is why analogies like king − man + woman ≈ queen emerge. `[nlp][embeddings]`

---

**Q:** When would you NOT reach for a transformer?
**A:** Tiny datasets or simple text tasks where TF-IDF + logistic regression / linear SVM is faster, cheaper, interpretable, and competitive; latency/compute-constrained settings; tabular data (use a GBM). `[transformers][model-selection]`
