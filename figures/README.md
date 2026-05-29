# Paper Figures

This directory contains all paper figures in both PDF and PNG formats.

## Regenerating Figures

All figures can be reproduced using the scripts in this repository:

```bash
# Regenerate experiment result figures (fig04-fig14)
python figures/redraw_figures.py

# Regenerate pipeline figure (fig01)
python figures/make_fig01_pipeline.py

# Regenerate architecture figure
python figures/make_architecture_fig.py
```

## Figure Index

| File | Description | Section |
|------|-------------|---------|
| fig01_pipeline.pdf | DMR-Bench data construction pipeline | §3 |
| fig_architecture.pdf | System architecture overview | §1 |
| fig02_distribution.pdf | Sample distribution by category | §3 |
| fig03_kde_distribution.pdf | KDE distribution of reduction rates | §3 |
| fig04_t1_macroF1.pdf | T1 Macro-F1 by model and RAG layer | §6.2 |
| fig05_t2_MAE.pdf | T2 MAE by model and RAG layer | §6.3 |
| fig06_t3_PRED25.pdf | T3 PRED25 by model and RAG layer | §6.4 |
| fig07_retrieval_quality.pdf | Retrieval quality comparison | §6.5 |
| fig08_ml_vs_llm.pdf | ML baseline vs LLM T2 comparison | §6.8 |
| fig09_decoupling.pdf | Cognitive decoupling visualization | §6.7 |
| fig10_oracle_paradox.pdf | Oracle paradox: MAE by RAG layer | §6.9 |
| fig11_vocabulary_closure.pdf | Vocabulary closure analysis | §6.8 |
| fig12_fewshot.pdf | Few-shot effect on T1/T2 | §6.6 |
| fig13_ablation_heatmap.pdf | Retrieval ablation heatmap | §6.5 |
| fig14_task_strategy.pdf | Task-strategy matching matrix | §6.10 |
