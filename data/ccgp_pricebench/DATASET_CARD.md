# CCGP-PriceBench Dataset Card

## Dataset Overview

**CCGP-PriceBench** (China Government Procurement Price Benchmark) is a fully public evaluation dataset for assessing LLMs' capability in IT hardware price reasonableness auditing in government procurement contexts.

- **Size**: 185 test samples + 555 RAG knowledge base entries
- **Access**: Fully public, no DUA required
- **Reproducibility**: Generated with `seed=42`, fully reproducible
- **License**: CC BY 4.0

---

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total test samples | 185 |
| RAG corpus entries | 555 |
| Samples requiring reduction (T1=1) | 93 (50.3%) |
| Samples not requiring reduction (T1=0) | 92 (49.7%) |
| Mean T2 reduction rate | 39.5% |
| Std T2 reduction rate | 14.5% |
| Hardware categories | 6 |
| Random seed | 42 |

### Hardware Categories

| Category | Count |
|----------|-------|
| Server (服务器) | ~31 |
| Desktop PC (台式机) | ~31 |
| Laptop (笔记本电脑) | ~31 |
| Printer (打印机) | ~31 |
| Network Switch (网络交换机) | ~31 |
| Storage Device (存储设备) | ~30 |

---

## Data Format

### Test Samples (`ccgp_pricebench.json`)

Each sample contains:

```json
{
  "id": "ccgp_001",
  "category": "server",
  "item_name": "Dell PowerEdge R750 服务器",
  "declared_unit_price": 85000.0,
  "declared_quantity": 2,
  "declared_total": 170000.0,
  "market_ref_price_low": 58000.0,
  "market_ref_price_high": 68000.0,
  "threshold_pct": 20.0,
  "label_t1": 1,
  "label_t2": 32.5,
  "label_t3": 136000.0,
  "difficulty": "L2"
}
```

### RAG Knowledge Base (`ccgp_rag_corpus.json`)

```json
{
  "id": "rag_001",
  "category": "server",
  "source": "zol.com.cn",
  "item_spec": "Dell PowerEdge R750 (Xeon Gold 6330 x2, 256GB RAM, 3.84TB SSD x4)",
  "market_price_range": "55000-70000",
  "reference_date": "2024-Q4",
  "text": "Dell PowerEdge R750服务器市场参考价区间为55,000至70,000元..."
}
```

---

## Task Definitions

Following DMR-Bench's six-task framework:

| Task | Input | Output | Metric |
|------|-------|--------|--------|
| T1 | Item details + RAG evidence | Reduce (1) or Not (0) | Macro-F1 ↑ |
| T2 | Item details + RAG evidence | Reduction rate (%) | MAE ↓ |
| T3 | Item details + RAG evidence | Reasonable total price (¥) | PRED25 ↑ |

**T1 criterion**: Declared unit price exceeds market reference price by >20%.

---

## Data Sources

Market reference prices were collected from publicly available sources:
- **ZOL (中关村在线)**: `zol.com.cn`
- **IT168**: `it168.com`
- **Government procurement records**: CCGP portal (`ccgp.gov.cn`)

All prices reflect Q4 2024 market conditions.

---

## Citation

```bibtex
@article{ma2026dmrbench,
  title   = {DMR-Bench: 面向政务信息化投资审核的大语言模型评测基准},
  author  = {马再英 and 杜晓勇 and 张峰},
  journal = {软件学报},
  year    = {2026},
  url     = {https://github.com/dmr-bench/dmr-bench}
}
```
