# 6DQV: Six-Dimensional Quality Validation Framework

## Overview

The **6DQV** (Six-Dimensional Quality Validation) framework provides a domain-agnostic methodology for validating the quality of closed-domain LLM evaluation benchmarks. All six dimensions use domain-independent statistical methods and can be directly applied to other decision-making vertical domains without modification.

## Six Dimensions

| Dimension | Code | Method | Threshold | DMR-Bench Result |
|-----------|------|--------|-----------|-----------------|
| Annotation Consistency | D1 | Spearman-Brown reliability | ≥ 0.80 | **SB = 0.881** ✅ |
| Discriminative Validity | D2 | IRT discrimination parameter | a_min ≥ 1.5 | **a_min = 2.24** ✅ |
| Difficulty Gradient | D3 | Kruskal-Wallis η² | ≥ 0.20 | **η² = 0.348** ✅ |
| Coverage Uniformity | D4 | Entropy-based uniformity | ≥ 0.60 | **U = 0.767** ✅ |
| Task Feasibility | D5 | Best model vs. random baseline | significant | **Δ >> 0** ✅ |
| Robustness | D6 | Cross-split variance | σ ≤ 0.02 | **σ = 0.0088** ✅ |

All six thresholds passed.

## Dimension Details

### D1: Annotation Consistency (标注一致性)
- **Method**: Spearman-Brown inter-rater reliability between two independent expert annotators
- **What it measures**: Agreement on T1 binary labels across 802 samples
- **Interpretation**: SB=0.881 indicates high inter-rater agreement, validating annotation reliability

### D2: Discriminative Validity (区分效度)
- **Method**: Item Response Theory (IRT) 2-parameter logistic model
- **What it measures**: Whether each item can distinguish high-ability from low-ability models
- **Interpretation**: a_min=2.24 (all items highly discriminating), no items should be removed

### D3: Difficulty Gradient (难度梯度)
- **Method**: Kruskal-Wallis test with effect size η² across L1-L4 difficulty levels
- **What it measures**: Whether labeled difficulty levels correspond to actual performance differences
- **Interpretation**: η²=0.348 (large effect), difficulty labeling is meaningful

### D4: Coverage Uniformity (类型覆盖度)
- **Method**: Shannon entropy-based uniformity score across expense categories
- **What it measures**: Whether the dataset uniformly covers different types of audit items
- **Interpretation**: U=0.767 (good coverage), no severe category imbalance

### D5: Task Feasibility (任务可行性)
- **Method**: Compare best model performance vs. random baseline with statistical significance
- **What it measures**: Whether the tasks are neither too easy (ceiling) nor too hard (floor)
- **Interpretation**: T1 Macro-F1=0.544 > random, T2 MAE clearly improvable with RAG

### D6: Robustness (鲁棒性)
- **Method**: Cross-split performance variance across 5 random 80/20 splits
- **What it measures**: Whether benchmark results are stable across different data splits
- **Interpretation**: σ=0.0088 (very low variance), results are reproducible

## Usage

```python
from evaluation.6dqv.benchmark_validation_6dqv import run_6dqv_validation

results = run_6dqv_validation(
    data_path="data/dmr_bench/bench_data.jsonl",
    annotation_path="data/dmr_bench/annotation_metadata.json"
)
results.print_report()
```

## Cross-Domain Application

The 6DQV framework can be applied to any closed-domain decision-making benchmark:

```python
# Example: Applying 6DQV to a new domain
results = run_6dqv_validation(
    data_path="your_domain_data.jsonl",
    thresholds={
        "D1_spearman_brown": 0.80,   # Adjust if needed
        "D2_irt_a_min": 1.5,
        "D3_kruskal_eta2": 0.20,
        "D4_uniformity": 0.60,
        "D6_sigma": 0.02
    }
)
```
