# Evaluation results

Brain: `offline`. All data is synthetic.

## Retrieval (answerable test questions)

| Embedder | Chunking | Re-ranker | Recall@1 | Recall@3 | MRR | p95 ms |
|---|---|---|---|---|---|---|
| hashing | fixed | none | 0.759 | 0.897 | 0.827 | 0.1 |
| hashing | fixed | lexical | 0.828 | 0.897 | 0.867 | 0.1 |
| hashing | fixed | cross-encoder | 0.897 | 0.931 | 0.926 | 125.7 |
| hashing | sentence | none | 0.69 | 0.759 | 0.743 | 0.0 |
| hashing | sentence | lexical | 0.69 | 0.793 | 0.753 | 0.1 |
| hashing | sentence | cross-encoder | 0.828 | 0.931 | 0.874 | 79.4 |
| minilm | fixed | none | 0.966 | 1.0 | 0.977 | 373.2 |
| minilm | fixed | lexical | 0.931 | 1.0 | 0.96 | 5.8 |
| minilm | fixed | cross-encoder | 0.897 | 0.931 | 0.926 | 22.2 |
| minilm | sentence | none | 0.862 | 0.966 | 0.914 | 5.8 |
| minilm | sentence | lexical | 0.897 | 0.966 | 0.929 | 5.8 |
| minilm | sentence | cross-encoder | 0.862 | 0.966 | 0.915 | 17.4 |

## Full agent: baseline (hashing, lexical)

| Metric | Value |
|---|---|
| threshold | 0.1 |
| answerable | 29 |
| correct | 19 |
| wrong | 4 |
| abstained | 6 |
| unanswerable | 12 |
| safe | 8 |
| invented | 4 |
| answer_accuracy | 0.655 |
| wrong_fact_rate | 0.195 |
| abstain_when_unknown | 0.667 |
| p50_ms | 1.9 |
| p95_ms | 2.7 |

## Full agent: best (minilm, cross-encoder)

| Metric | Value |
|---|---|
| threshold | 0.1 |
| answerable | 29 |
| correct | 17 |
| wrong | 0 |
| abstained | 12 |
| unanswerable | 12 |
| safe | 12 |
| invented | 0 |
| answer_accuracy | 0.586 |
| wrong_fact_rate | 0.0 |
| abstain_when_unknown | 1.0 |
| p50_ms | 30.5 |
| p95_ms | 46.1 |

## Triage

Accuracy: 1.0

- personal_question: 1.0
- general_question: 1.0
- statement: 1.0
- urgent: 1.0

Hard set, written after the rules and not tuned on: 0.267

- personal_question: 0.2
- general_question: 0.0
- statement: 0.75
- urgent: 0.0

## Urgent speech

{"recall": 1.0, "p95_ms": 1.27, "tokens_used": 0}

## Caption safety

{"facts_preserved": 1.0, "cases": 6, "hud_fits": true}

## Speech

{"model": "tiny.en", "sentences": 20, "wer": 0.074, "p50_ms": 528.7, "p95_ms": 545.7, "source": "macOS say, clean audio"}

