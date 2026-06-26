# LLMs & Agentic AI

> **Week:** 9 · **Status:** ✅ written · **From-scratch build:** no (conceptual; this is the "discuss intelligently" frontier topic)
> **One-line:** An **LLM** is a large decoder-only transformer ([Week 8](08-nlp-transformers.md)) **pretrained** to predict the next token on internet-scale text, then **aligned** via instruction-tuning + RLHF. Around that core sit the practitioner's toolkit: **prompting**, **RAG** (ground answers in retrieved documents), **fine-tuning** (incl. parameter-efficient LoRA), and **agents** (LLMs that plan and call **tools** in a loop). The interview goal here is breadth and good judgment about *when to use which*, not depth.

## Table of contents
- [1. The LLM lifecycle](#1-the-llm-lifecycle)
  - [1.1 Pretraining](#11-pretraining)
  - [1.2 Supervised fine-tuning (instruction tuning)](#12-supervised-fine-tuning-instruction-tuning)
  - [1.3 RLHF / preference alignment](#13-rlhf--preference-alignment)
- [2. Adapting an LLM to your task — the ladder](#2-adapting-an-llm-to-your-task--the-ladder)
- [3. RAG (Retrieval-Augmented Generation)](#3-rag-retrieval-augmented-generation)
- [4. Agents and tool use](#4-agents-and-tool-use)
- [5. Inference: decoding & context](#5-inference-decoding--context)
- [6. Evaluating LLM systems](#6-evaluating-llm-systems)
- [7. Failure modes & safety](#7-failure-modes--safety)
- [8. When to use / not use](#8-when-to-use--not-use)
- [9. Pitfalls & gotchas](#9-pitfalls--gotchas)
- [10. Interview questions](#10-interview-questions)
- [11. Connections](#11-connections)

---

## 1. The LLM lifecycle

### 1.1 Pretraining
Take a decoder-only transformer and train it with one self-supervised objective: **predict the next token** over trillions of tokens of web text/code. The label is free (the next word in the corpus), so no human annotation is needed at this scale. The model learns grammar, facts, reasoning patterns, and world knowledge as a side effect of getting good at next-token prediction. This is the expensive step (millions of GPU-hours) and yields a **base/foundation model** — a powerful next-token predictor that is *not yet* good at following instructions. **Scaling laws** say loss falls predictably with more parameters, data, and compute, which is why models kept getting bigger. ⭐

### 1.2 Supervised fine-tuning (instruction tuning)
The base model completes text; it doesn't naturally "answer questions." **SFT** fine-tunes it on curated `(instruction, ideal response)` pairs so it learns the *assistant* behavior — follow instructions, adopt a helpful format. Far cheaper than pretraining (thousands, not trillions, of examples).

### 1.3 RLHF / preference alignment
SFT teaches the form; **alignment** teaches preferences (helpful, honest, harmless). Classic **RLHF**:
1. Collect human **preference data**: show two model responses, a human picks the better.
2. Train a **reward model** to predict that human preference.
3. **RL** (PPO) fine-tunes the LLM to maximize the reward model's score, with a **KL penalty** keeping it close to the SFT model (so it doesn't drift into gibberish that games the reward).

**DPO** (Direct Preference Optimization) is the modern simplification: optimize directly on preference pairs with a classification-style loss, skipping the separate reward model and RL loop. ⭐

> Memory hook for the lifecycle: **Pretrain (knowledge) → SFT (instruction-following) → RLHF/DPO (preferences/alignment).**

## 2. Adapting an LLM to your task — the ladder

Climb only as far as you need — each rung costs more:

1. **Prompt engineering** — clear instructions, examples (**few-shot / in-context learning**), and **chain-of-thought** ("think step by step") for reasoning. Zero training. Always start here. ⭐
2. **RAG** — inject *your* relevant data at inference time (see [§3](#3-rag-retrieval-augmented-generation)). The right tool when the problem is **missing/changing knowledge**, not missing skill.
3. **Fine-tuning** — update weights on your task data. Use when you need a **consistent style/format**, a narrow specialized behavior, or to distill a smaller cheaper model. **PEFT / LoRA** freezes the base model and trains tiny low-rank adapter matrices — ~0.1% of the parameters, cheap and swappable. ⭐
4. **Full fine-tuning / continued pretraining** — rare, expensive; for deep domain shifts.

> **The canonical interview decision:** "Fine-tune vs. RAG?" → **RAG** for *knowledge* (facts that change, are proprietary, or must be cited), **fine-tuning** for *behavior/skill/format*. They're complementary, not either/or — RAG over a fine-tuned model is common. ⭐

## 3. RAG (Retrieval-Augmented Generation)

LLMs have a **knowledge cutoff**, **can't see private data**, and **hallucinate**. RAG fixes all three by retrieving relevant documents and putting them in the prompt as grounding context.

**Pipeline:**
1. **Index (offline):** chunk your documents → embed each chunk with an embedding model ([Week 8](08-nlp-transformers.md)) → store vectors in a **vector database** (FAISS, Pinecone, pgvector).
2. **Retrieve (online):** embed the user query → **nearest-neighbor search** (ANN — approximate, for speed) over the vector store → pull the top-k most similar chunks. (This is [KNN, Week 2](02-knn.md) at scale.)
3. **Generate:** stuff the retrieved chunks + the question into the prompt; the LLM answers **grounded** in them, ideally with citations.

**Why it's the default for knowledge tasks:** fresh/proprietary data with no retraining, **source attribution** (trust + auditability), and far less hallucination. Key knobs: chunk size/overlap, embedding model quality, k, and **reranking** (a cross-encoder re-scores the top candidates). **Hybrid search** (dense embeddings + sparse BM25 keyword) usually beats either alone. ⭐

## 4. Agents and tool use

An **agent** is an LLM placed in a **loop** where it can take actions, not just emit text. The pattern (**ReAct** = Reason + Act):

```
loop:
    THINK   — the LLM reasons about the goal and what to do next
    ACT     — it calls a TOOL (search, calculator, code-runner, API, DB query)
    OBSERVE — the tool's result is fed back into the context
until the goal is met → final answer
```

- **Tool use / function calling:** the LLM outputs a structured call (tool name + JSON args); your harness executes it and returns the result. This breaks the model out of its frozen weights — live data, exact arithmetic, real side effects.
- **Planning & memory:** decompose a goal into steps; keep scratchpad/episodic memory (often itself a RAG store) to persist beyond the context window.
- **Multi-agent:** specialized agents (planner, coder, critic) collaborate.

**Why agents:** they turn a passive text predictor into something that can *accomplish multi-step tasks* against real systems. **Why they're hard:** errors **compound** over steps, they can loop or go off-track, tool calls add latency/cost, and acting in the real world raises safety/permission concerns. Good agent design = tight tools, guardrails, step limits, and human-in-the-loop for risky actions. ⭐

## 5. Inference: decoding & context
- **Autoregressive decoding:** generate one token at a time, feeding each back in. **Temperature** controls randomness (0 = greedy/deterministic, higher = more diverse); **top-k / top-p (nucleus)** sampling restrict the candidate set.
- **Context window:** the max tokens the model can attend to (prompt + output). Bounded by the O(n²) attention cost ([Week 8](08-nlp-transformers.md)); the constraint behind chunking and RAG.
- **KV cache:** cache past keys/values so each new token is O(n) not O(n²) — the main inference-speed trick.
- **Cost/latency** scale with tokens — a real production constraint (prompt length matters).

## 6. Evaluating LLM systems
Open-ended generation has **no single ground truth**, so evaluation is genuinely hard:
- **Reference-based:** BLEU/ROUGE (overlap with a reference — weak for open-ended), exact-match/F1 for QA.
- **Benchmarks:** MMLU, HumanEval (code), etc. — useful but gameable / contaminated.
- **LLM-as-a-judge:** use a strong model to score responses against a rubric — scalable, the modern default, but watch for biases (position, verbosity, self-preference). ⭐
- **Human eval / A/B:** the gold standard for product quality.
- **RAG-specific:** measure **retrieval** (recall@k, are the right chunks fetched?) and **generation** (**faithfulness/groundedness** — is the answer supported by the retrieved context? answer relevance?) separately.
- **Guardrails:** hallucination rate, toxicity, latency, cost.

## 7. Failure modes & safety
- **Hallucination** — fluent, confident, wrong. Mitigate with RAG/grounding, citations, "say I don't know," lower temperature.
- **Prompt injection** — malicious instructions hidden in retrieved/user content hijack the model; the central security risk for RAG/agents. Treat retrieved text as untrusted; sandbox tools; least privilege. ⭐
- **Jailbreaks**, **bias/toxicity** inherited from pretraining data, **knowledge cutoff** staleness, **sycophancy**, and **non-determinism** (hard to test/reproduce).

## 8. When to use / not use
**Use an LLM when:** open-ended language understanding/generation, few labels (zero/few-shot), fast prototyping, or tasks needing broad world knowledge — summarization, extraction, classification, chat, coding assistance, RAG over docs.

**Avoid / be cautious when:** you have **plenty of labeled data and a narrow task** (a fine-tuned small model or even logistic regression is cheaper, faster, more reliable); **tabular** prediction (GBM); strict **latency/cost** budgets; or you need **guarantees/determinism** (correctness-critical logic). "Don't use a 70B model where a logistic regression would do" is a strong, credible take. ⭐

## 9. Pitfalls & gotchas
- **Fine-tuning when you needed RAG** (or vice versa) — fine-tuning won't inject fresh facts; RAG won't fix a missing *skill*/format. ⭐
- **Skipping the prompt-engineering rung** — many "we need to fine-tune" problems are solved by a better prompt + few-shot examples.
- **Treating LLM output as ground truth** — hallucination is the default failure; always ground and verify high-stakes outputs.
- **Ignoring prompt injection** — retrieved/agent inputs are untrusted; never give tools more permission than necessary. ⭐
- **No retrieval evaluation in RAG** — if the right chunks aren't retrieved, the LLM can't answer; debug retrieval (recall@k) and generation separately.
- **Bad chunking** — too large dilutes relevance and wastes context; too small loses coherence.
- **Unbounded agent loops** — set step/iteration limits and budgets; errors compound.
- **Over-trusting BLEU/ROUGE** for open-ended tasks — they correlate poorly with quality; prefer human / LLM-judge eval.
- **Forgetting cost/latency scale with tokens** — long prompts and big contexts are expensive in production.

## 10. Interview questions
Must answer cold (⭐):
- ⭐ **Walk me through how a modern chat LLM is trained.** Pretrain (next-token prediction on web-scale text → knowledge) → supervised fine-tuning on instruction/response pairs (assistant behavior) → RLHF or DPO on human preferences (alignment: helpful/honest/harmless). (§1)
- ⭐ **Fine-tuning vs. RAG — when each?** RAG for *knowledge* (fresh, proprietary, citable facts) injected at inference; fine-tuning for *behavior/style/format/skill* baked into weights. Complementary, not either/or. (§2)
- ⭐ **What is RAG and why use it?** Retrieve relevant chunks (embed query → ANN over a vector DB) and put them in the prompt so the LLM answers grounded in your data — fixes knowledge cutoff, private data, and hallucination, and enables citations, with no retraining. (§3)
- ⭐ **What is RLHF (and DPO)?** Train a reward model on human preference comparisons, then RL-tune (PPO) the LLM to maximize reward with a KL penalty to the SFT model. DPO optimizes preferences directly with a classification loss, skipping the reward model + RL. (§1.3)
- ⭐ **What is an agent / tool use?** An LLM in a reason→act→observe loop that calls external tools (search, code, APIs), letting it do multi-step tasks against real systems beyond its frozen weights. Risks: compounding errors, loops, latency, safety. (§4)
- ⭐ **What is LoRA / PEFT?** Parameter-efficient fine-tuning: freeze the base model, train small low-rank adapter matrices (~0.1% of params) — cheap, fast, swappable. (§2)
- ⭐ **What is hallucination and how do you reduce it?** Fluent but false output; reduce via RAG/grounding + citations, lower temperature, "say I don't know," and verification. (§7)
- ⭐ **How do you evaluate an LLM / RAG system?** No single ground truth → LLM-as-judge with rubrics, human/A-B eval, task benchmarks; for RAG, separately measure retrieval (recall@k) and generation faithfulness/groundedness. (§6)
- **What is in-context / few-shot learning?** The model adapts from examples placed in the prompt, without weight updates. (§2)
- **What is chain-of-thought prompting?** Asking the model to reason step by step, which improves multi-step/reasoning accuracy. (§2)
- **What is prompt injection?** Malicious instructions embedded in user/retrieved content that hijack the model — the core security risk for RAG/agents; treat such text as untrusted and limit tool permissions. (§7)
- **What limits the context window?** The O(n²) attention cost in sequence length; KV caching speeds decoding but the window is still bounded. (§5)
- **What are scaling laws?** Loss decreases predictably with more parameters/data/compute — the empirical basis for building ever-larger models. (§1.1)
- **When would you NOT use an LLM?** Narrow task with ample labels, tabular data, strict latency/cost or determinism needs — a small fine-tuned model or classical ML is better. (§8)

## 11. Connections
- **Built directly on [Week 8: Transformers](08-nlp-transformers.md)** — an LLM is a scaled decoder-only transformer; pretraining is next-token prediction over it.
- **RAG = embeddings ([Week 8](08-nlp-transformers.md)) + nearest-neighbor search ([KNN, Week 2](02-knn.md))** at scale, with ANN for low-latency serving ([system-design framework §7](../system-design/00-framework.md)).
- **RLHF's reward model** is a learned preference scorer — a classification/ranking model ([evaluation](../evaluation/)); the KL penalty is a [regularizer](01-regularization-and-optimization.md) keeping it near the base policy.
- **Evaluation is the hard part** — ties to the [evaluation notes](../evaluation/) and the [system-design](../system-design/00-framework.md) offline/online split; LLM-judge is the scalable proxy.
- **Serving constraints** (tokens → cost/latency, context limits) feed straight into [ML system design](../system-design/00-framework.md).
- Flashcards: [flashcards/09-llms-agentic.md](../flashcards/09-llms-agentic.md).

---
### Sources
- Ouyang et al. (2022), *InstructGPT* — SFT + RLHF lifecycle.
- Rafailov et al. (2023), *Direct Preference Optimization (DPO)*.
- Lewis et al. (2020), *Retrieval-Augmented Generation*.
- Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting*.
- Hu et al. (2021), *LoRA: Low-Rank Adaptation*.
- Chip Huyen, *Designing Machine Learning Systems* & writing on LLM evaluation / production.
