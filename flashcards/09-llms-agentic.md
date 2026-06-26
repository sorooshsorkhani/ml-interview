# Flashcards: LLMs & Agentic AI

> One `Q:`/`A:` pair per card, atomic answers. Tags in `[brackets]` for later Anki export.
> ✅ Seeded from the topic note's "Interview questions" section.

---

**Q:** Walk through how a modern chat LLM is trained.
**A:** Pretrain (next-token prediction on web-scale text → knowledge) → supervised fine-tuning on instruction/response pairs (assistant behavior) → RLHF or DPO on human preferences (alignment: helpful/honest/harmless). `[llm][training]`

---

**Q:** Fine-tuning vs. RAG — when do you use each?
**A:** RAG for *knowledge* (fresh, proprietary, citable facts) injected at inference; fine-tuning for *behavior/style/format/skill* baked into weights. Complementary, not either/or — RAG over a fine-tuned model is common. `[llm][rag][finetuning]`

---

**Q:** What is RAG and why use it?
**A:** Retrieve relevant chunks (embed the query → ANN search over a vector DB) and place them in the prompt so the LLM answers grounded in your data. Fixes knowledge cutoff, private data, and hallucination; enables citations; needs no retraining. `[llm][rag]`

---

**Q:** What is RLHF (and how does DPO simplify it)?
**A:** Train a reward model on human preference comparisons, then RL-tune (PPO) the LLM to maximize reward with a KL penalty to the SFT model. DPO optimizes directly on preference pairs with a classification-style loss, skipping the separate reward model and RL loop. `[llm][alignment]`

---

**Q:** What is an agent / tool use?
**A:** An LLM in a reason→act→observe loop (ReAct) that calls external tools — search, calculator, code, APIs — letting it accomplish multi-step tasks against real systems beyond its frozen weights. Risks: compounding errors, loops, latency, safety. `[llm][agents]`

---

**Q:** What is LoRA / PEFT?
**A:** Parameter-efficient fine-tuning: freeze the base model and train small low-rank adapter matrices (~0.1% of parameters). Cheap, fast, and swappable per task. `[llm][finetuning]`

---

**Q:** What is hallucination and how do you reduce it?
**A:** Fluent, confident, but false output. Reduce via RAG/grounding + citations, lower temperature, allowing "I don't know," and verification of high-stakes outputs. `[llm][safety]`

---

**Q:** How do you evaluate an LLM / RAG system?
**A:** No single ground truth → LLM-as-judge with rubrics, human/A-B eval, and task benchmarks. For RAG, separately measure retrieval (recall@k) and generation faithfulness/groundedness + answer relevance. `[llm][evaluation]`

---

**Q:** What is in-context (few-shot) learning?
**A:** The model adapts to a task from examples placed directly in the prompt, with no weight updates. `[llm][prompting]`

---

**Q:** What is chain-of-thought prompting?
**A:** Prompting the model to reason step by step before answering, which improves accuracy on multi-step / reasoning tasks. `[llm][prompting]`

---

**Q:** What is prompt injection and why does it matter?
**A:** Malicious instructions hidden in user or retrieved content that hijack the model's behavior — the core security risk for RAG and agents. Treat such text as untrusted and give tools least privilege. `[llm][safety]`

---

**Q:** What limits the context window?
**A:** The O(n²) attention cost in sequence length. KV caching speeds decoding, but the window is still bounded — the constraint behind chunking and RAG. `[llm][inference]`

---

**Q:** What are scaling laws?
**A:** Empirically, loss decreases predictably with more parameters, data, and compute — the basis for building ever-larger models. `[llm][pretraining]`

---

**Q:** What is the adaptation "ladder" for using an LLM on your task?
**A:** Prompt engineering / few-shot → RAG → parameter-efficient fine-tuning (LoRA) → full fine-tuning. Climb only as far as needed; each rung costs more. `[llm][adaptation]`

---

**Q:** What is temperature in decoding?
**A:** It controls output randomness: 0 = greedy/deterministic, higher = more diverse/creative. Paired with top-k / top-p (nucleus) sampling to restrict the candidate set. `[llm][inference]`

---

**Q:** When would you NOT use an LLM?
**A:** A narrow task with ample labels, tabular prediction (use a GBM), strict latency/cost budgets, or when you need determinism/guarantees — a small fine-tuned model or classical ML is cheaper and more reliable. `[llm][model-selection]`
