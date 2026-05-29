# DMR-Bench Data Access

## DMR-Bench Dataset (Controlled Access)

DMR-Bench contains 802 expert-annotated samples derived from real government informatization investment audit records. Although fully desensitized (project names, vendor names, and amounts have been replaced or perturbed), the dataset touches on government budget allocation patterns and is therefore released under **controlled access**.

This policy follows the established practice in sensitive-domain AI benchmarks:
- **PhysioNet** (medical ICU data): requires signed DUA before access
- **LegalBench** (legal NLP): subset tasks restricted due to copyright/privacy
- **AutoIF** (government): tiered access strategy

---

## How to Apply

### Step 1: Prepare Your Application

Please prepare the following information:

| Field | Description |
|-------|-------------|
| Name | Full name of the principal researcher |
| Affiliation | University / Research Institute / Company |
| Position | PhD student / Postdoc / Faculty / Industry researcher |
| Research Purpose | Brief description (≤200 words) of your intended use |
| Publication Plan | Whether you plan to publish results citing DMR-Bench |

### Step 2: Sign the Data Use Agreement (DUA)

Download and sign the DUA: [DMR-Bench-DUA.pdf](DMR-Bench-DUA.pdf)

Key terms of the DUA:
1. **Non-commercial use only**: The dataset may only be used for non-commercial academic research.
2. **No redistribution**: You may not share, re-host, or redistribute the dataset or any derivative.
3. **Citation required**: Any publication using DMR-Bench must cite the original paper.
4. **No re-identification**: You agree not to attempt re-identification of any entity in the dataset.
5. **Reporting**: You agree to notify us of any publications using DMR-Bench.

### Step 3: Submit Application

Send the signed DUA and your application information to:

📧 **mazaiying@ruc.edu.cn** 

Subject: `[DMR-Bench DUA] <Your Name> - <Affiliation>`

### Step 4: Review and Access

- Applications are typically reviewed within **5-7 business days**.
- Approved applicants will receive a secure download link valid for 30 days.
- The dataset will be provided as a password-protected archive.

---

## What You Get

Upon approval, you will receive:

```
dmr_bench_v1.0/
├── README_DATA.md          # Data documentation
├── bench_data.jsonl        # 802 annotated samples (T1-T6 labels)
├── rag_corpus/             # Evidence corpus from feasibility reports
│   ├── corpus_bm25.json    # Pre-indexed BM25 corpus
│   └── corpus_dense.npy    # Pre-computed dense embeddings
├── splits/
│   ├── train_80pct.jsonl   # 80% training split (for few-shot sampling)
│   └── test_20pct.jsonl    # 20% test split
└── annotation_metadata.json # Difficulty levels, category labels, IRT parameters
```

---

## Public Alternative: CCGP-PriceBench

If you need immediate access for methods testing, **CCGP-PriceBench** is fully public with no DUA required:

```bash
# Directly available in this repository
ls data/ccgp_pricebench/
```

CCGP-PriceBench shares the same evaluation framework as DMR-Bench and all three key findings replicate across domains (see paper §6.14).

---

## Contact

For questions about data access, please open a GitHub Issue with the label `data-access`.
