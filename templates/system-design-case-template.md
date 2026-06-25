# System Design Case: <Name>

> **Prompt:** "<the one-line interview prompt, e.g. 'Design a fraud detection system for an e-commerce checkout.'>"

Use the [framework spine](../system-design/00-framework.md). Each section is a checklist of what to say out loud.

## 1. Clarify the problem & success metric
- Business goal vs. ML objective:
- Constraints (latency, scale, cost, regulatory):
- Online vs. batch:

## 2. Frame as an ML task
- Inputs / outputs / labels:
- Where do labels come from? Delay? Noise?

## 3. Data
- Sources:
- Labeling strategy:
- Leakage risks:
- Splits (time-based?):

## 4. Features
- Engineering & encoding:
- Real-time vs. precomputed:

## 5. Model
- Baseline first:
- Candidate models & tradeoffs:

## 6. Evaluation
- Offline metrics:
- Online (A/B, guardrails):

## 7. Serving
- Latency budget, batch vs. real-time, infra sketch:

## 8. Monitoring
- Drift, retraining cadence, failure modes, feedback loops:

## Likely follow-ups
- <interviewer curveballs and how to answer>
