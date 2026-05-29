# DMR-Bench: A Decision-Making Retrieval Benchmark for Government IT Investment Auditing

[![Paper](https://img.shields.io/badge/Paper-软件学报-blue)](https://www.jos.org.cn)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-green)](LICENSE)
[![Data](https://img.shields.io/badge/Data-Controlled%20Access-orange)](docs/DATA_ACCESS.md)
[![CCGP](https://img.shields.io/badge/CCGP--PriceBench-Public-brightgreen)](data/ccgp_pricebench/)

> **DMR-Bench** (Decision-Making Retrieval Benchmark) is an LLM evaluation benchmark for government IT investment auditing, featuring 802 expert-annotated samples from real government informatization projects. This repository accompanies the paper *"DMR-Bench: 面向政务信息化投资审核的大语言模型评测基准"* published in the **Journal of Software (软件学报)**.

---

## 📋 Overview

Government IT investment auditing requires expert judgment across multiple decision types: determining whether budget items should be reduced (T1), estimating reduction rates (T2), predicting final audit amounts (T3), and more. DMR-Bench systematically evaluates LLMs on this **closed-domain, decision-making vertical task**.

### Key Contributions

| Component | Description |
|-----------|-------------|
| **DMR-Bench Dataset** | 802 expert-annotated samples, controlled access via DUA |
| **Six-Task Framework** | T1 (direction) → T2 (rate) → T3 (amount) → T4 (category) → T5 (anomaly) → T6 (rationale) |
| **6DQV Framework** | Six-dimensional quality validation (annotation consistency, IRT discrimination, difficulty gradient, coverage uniformity, task feasibility, robustness) |
| **HTG-Align** | Heterogeneous Text Graph alignment for evidence linking in feasibility reports |
| **CCGP-PriceBench** | **Fully public** cross-domain dataset: 185 IT hardware procurement samples (seed=42, reproducible) |

---

## 🗂️ Repository Structure

```
dmr-bench/
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   └── ccgp_pricebench/          # Fully public cross-domain dataset
│       ├── ccgp_pricebench.json  # 185 test samples with T1/T2/T3 labels
│       ├── ccgp_rag_corpus.json  # 555 RAG knowledge base entries
│       ├── dataset_stats.json    # Dataset statistics
│       └── eval_ccgp.py          # Evaluation script for CCGP-PriceBench
│
├── evaluation/
│   ├── six_tasks/                # Six-task evaluation scripts
│   │   ├── eval_t1_t2_t3.py     # Main T1/T2/T3 evaluation (5-layer RAG)
│   │   ├── eval_fewshot.py      # Few-shot experiment (T1/T2)
│   │   ├── eval_ml_baseline.py  # Traditional ML baselines (Ridge, RF, GBM)
│   │   ├── eval_oracle_paradox.py # Oracle paradox mechanism analysis
│   │   └── eval_t6_generation.py  # T6 rationale generation (ROUGE-L + BERTScore)
│   │
│   ├── 6dqv/                    # Six-Dimensional Quality Validation framework
│   │   └── benchmark_validation_6dqv.py
│   │
│   └── htg_align/               # HTG-Align evidence linking algorithm
│       ├── build_htg.py         # Build Heterogeneous Text Graph from feasibility reports
│       ├── embed_nodes.py       # Node embedding with domain-specific encoder
│       └── align_evidence.py   # Evidence alignment and retrieval
│
├── figures/                     # Paper figures (reproducible)
│   └── redraw_figures.py        # Script to regenerate all paper figures
│
└── docs/
    ├── DATA_ACCESS.md           # DMR-Bench DUA application instructions
    ├── DATASET_CARD.md          # Dataset documentation (Croissant format)
    ├── SIX_TASK_FRAMEWORK.md    # Six-task framework design rationale
    └── 6DQV_FRAMEWORK.md        # 6DQV quality validation methodology
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/mzaiying/dmr-bench.git
cd dmr-bench
pip install -r requirements.txt
```

### 2. Evaluate on CCGP-PriceBench (Public, No DUA Required)

```bash
# Set your API key
export OPENAI_API_KEY="your-key"   # or DEEPSEEK_API_KEY, etc.

# Run evaluation on CCGP-PriceBench
python data/ccgp_pricebench/eval_ccgp.py \
    --model deepseek-chat \
    --rag bm25 \
    --tasks T1 T2 T3
```

### 3. Replicate Paper Results (DMR-Bench, requires DUA)

```bash
# After DUA approval, place data in data/dmr_bench/
python evaluation/six_tasks/eval_t1_t2_t3.py \
    --data_path data/dmr_bench/ \
    --model qwen-plus \
    --rag_layer L_BM25 \
    --tasks T1 T2 T3
```

---

## 📊 Main Results

### T1: Reduction Direction Prediction (Macro-F1)

| Model | L1 (Zero-Knowledge) | L_BM25 | L_Dense | L2 (Oracle) | L3 |
|-------|--------------------:|-------:|--------:|------------:|---:|
| DeepSeek V3 | 0.401 | 0.412 | **0.435** | 0.419 | 0.415 |
| Qwen-Plus | 0.364 | **0.480** | 0.475 | 0.432 | 0.440 |
| GLM-4-Plus | **0.479** | 0.399 | 0.379 | 0.437 | 0.363 |
| Doubao-1.5-Pro | **0.544** | 0.525 | 0.539 | 0.521 | 0.518 |
| *Statistical Baseline* | *0.575* | — | — | — | — |

### T2: Reduction Rate Estimation (MAE ↓)

| Model | L1 | L_BM25 | Best ML (GBM) |
|-------|----|--------|---------------|
| DeepSeek V3 | 25.81 | **21.36** | — |
| Qwen-Plus | 79.22 | **22.51** | — |
| GBM (12 features, no text) | — | — | **11.07** |

> **Key Finding**: GBM using only 12 structural features (no text) achieves MAE=11.07, outperforming all LLM+RAG configurations (best: 21.36). We recommend a "ML for estimation, LLM for explanation" hybrid architecture.

### Cross-Domain Validation: CCGP-PriceBench

| Task | BM25 Mean | Oracle Mean | Finding |
|------|-----------|-------------|---------|
| T1 Macro-F1 ↑ | **0.789** | 0.809 | BM25 competitive |
| T2 MAE ↓ | **11.30** | 13.40 | Oracle paradox replicated |

All three key findings (cognitive decoupling, BM25 advantage, Oracle paradox) **replicate across domains**.

---

## 🔒 Data Access

### DMR-Bench (Controlled Access)

DMR-Bench contains desensitized but sensitive government procurement data. Following the practice of PhysioNet (medical) and LegalBench (legal), we adopt a **controlled access** model:

> **Apply for access**: See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) for the Data Use Agreement (DUA) and application process.

### CCGP-PriceBench (Fully Public)

CCGP-PriceBench is **fully open** without any DUA requirement:
- 185 test samples with complete T1/T2/T3 labels
- 555 RAG knowledge base entries
- Generated with `seed=42` for full reproducibility
- Based on publicly available IT hardware market prices from ZOL and IT168

```
data/ccgp_pricebench/
```

---

## 📖 Citation

If you use DMR-Bench or CCGP-PriceBench in your research, please cite:

```bibtex
@article{ma2026dmrbench,
  title   = {DMR-Bench: 面向政务信息化投资审核的大语言模型评测基准},
  author  = {马在营 and 杜小勇 and 张峰},
  journal = {软件学报},
  year    = {2026},
  url     = {https://github.com/mzaiying/dmr-bench}
}
```

---

## 📄 License

- **Code**: [MIT License](LICENSE)
- **CCGP-PriceBench Data**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **DMR-Bench Data**: Controlled access, see [DATA_ACCESS.md](docs/DATA_ACCESS.md)

---

## 🙏 Acknowledgments

This work was supported by [funding information]. We thank the two domain experts who provided annotation for DMR-Bench.
