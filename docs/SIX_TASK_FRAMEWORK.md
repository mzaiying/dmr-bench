# Six-Task Evaluation Framework

## Design Rationale

The six-task framework decomposes the government IT investment audit decision chain into cognitively distinct sub-tasks, enabling researchers to precisely identify where LLMs fail rather than simply reporting aggregate performance.

## Task Definitions

| Task | Name | Input | Output | Metric | Cognitive Type |
|------|------|-------|--------|--------|----------------|
| T1 | Reduction Direction | Item description + RAG evidence | Binary: reduce/not-reduce | Macro-F1 ↑ | Semantic judgment |
| T2 | Reduction Rate | Item description + RAG evidence | Reduction percentage (0-100%) | MAE ↓ | Quantitative estimation |
| T3 | Reduction Amount | Item description + RAG evidence | Final audit amount (¥) | PRED25 ↑ | Multi-step calculation |
| T4 | Reduction Category | Item description + RAG evidence | Category (4 classes) | Macro-F1 ↑ | Meta-cognitive classification |
| T5 | Anomaly Detection | Mixed item list (with noise) | Anomalous items (F1) | F1 ↑ | Symbolic reasoning |
| T6 | Rationale Generation | Item description + RAG evidence | Professional audit rationale | ROUGE-L, BERTScore ↑ | Text generation |

## Metrics Explained

### T1: Macro-F1
- Binary classification: "核减" (reduce) vs "不核减" (not reduce)
- Macro-averaged to handle class imbalance (57.5% positive)
- Statistical baseline: 0.575 (majority class prior)

### T2: MAE (Mean Absolute Error)
- Predicts reduction rate as percentage
- Output clipped to [0, 100]%
- Statistical baseline: 21.89 (mean blind prediction)

### T3: PRED25
- PRED25 = proportion of predictions within ±25% of ground truth
- Tolerates absolute error within 25% of true audit amount
- More appropriate than MAE for highly skewed amount distributions

## RAG Layers

| Layer | Description | Knowledge Source |
|-------|-------------|-----------------|
| L1 | Zero-knowledge | No retrieval, model prior only |
| L_BM25 | Sparse retrieval | BM25 keyword matching |
| L_Dense | Dense retrieval | Sentence-BERT semantic search |
| L2 | Oracle upper bound | Expert-annotated complete evidence |
| L3 | HTG-Align | Dual-path graph-augmented retrieval |

## Key Experimental Findings

1. **Cognitive Decoupling**: LLMs show strong T1 performance (best Macro-F1=0.544) but catastrophic T2 failure (zero-knowledge MAE=72-79, >3× baseline)
2. **BM25 Advantage**: In vocabulary-closed government domains, BM25 outperforms Dense retrieval (P@1: 0.330 vs 0.196)
3. **Oracle Paradox**: Providing complete expert evidence *reduces* T2 performance (Qwen-Plus: BM25 MAE=22.51, Oracle MAE=35.66)
4. **ML Superiority on T2**: GBM with 12 structural features achieves MAE=11.07, outperforming all LLM+RAG (best: 21.36)
