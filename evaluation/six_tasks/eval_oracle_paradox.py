"""
P1-A: Oracle悖论机制验证实验
构造三种证据文本变体：
  L2-Full:    完整专家段落（当前L2）
  L2-NumOnly: 只保留含数字的句子
  L2-KwOnly:  只保留含价格关键词的句子

通过分析三种条件下的文本长度、数字密度，
为Oracle悖论提供可量化的机制证据。
"""
import json
import re
import numpy as np

DATA_PATH = "/Volumes/Elements SE/科研/软件学报-软件投资审核/核减Agent/bench/bench_data/govreview_bench_v2_desensitized.json"
with open(DATA_PATH) as f:
    data = json.load(f)

# 只取有证据文本的样本
evidence_samples = [item for item in data if item.get('evidence_text') and 
                    len(str(item.get('evidence_text', '')).strip()) > 20]
print(f"Samples with evidence: {len(evidence_samples)}")

# ─── 价格关键词列表（政务审核领域专用）───────────────────────
PRICE_KEYWORDS = [
    '元', '万元', '千元', '元/套', '元/台', '元/月', '元/年',
    '单价', '报价', '限价', '市场价', '参考价', '采购价', '定额',
    '造价', '预算', '核减', '核定', '超出', '偏高', '偏低',
    '%', '折', '折扣', '优惠', '降价', '涨价',
]
PRICE_KW_PATTERN = '|'.join(re.escape(kw) for kw in PRICE_KEYWORDS)
NUM_PATTERN = r'\d+\.?\d*'

def filter_sentences_with_numbers(text):
    """只保留含数字的句子"""
    sentences = re.split(r'[。；\n]', str(text))
    return '。'.join(s for s in sentences if re.search(NUM_PATTERN, s)).strip()

def filter_sentences_with_price_kw(text):
    """只保留含价格关键词的句子"""
    sentences = re.split(r'[。；\n]', str(text))
    return '。'.join(s for s in sentences if re.search(PRICE_KW_PATTERN, s)).strip()

# ─── 分析三种变体的统计特性 ─────────────────────────────────
full_lens, num_lens, kw_lens = [], [], []
full_num_density, num_num_density, kw_num_density = [], [], []
full_kw_density, num_kw_density, kw_kw_density = [], [], []

for item in evidence_samples:
    ev = str(item.get('evidence_text', ''))
    
    full = ev
    num_only = filter_sentences_with_numbers(ev)
    kw_only  = filter_sentences_with_price_kw(ev)
    
    for text, lens, nd, kd in [
        (full,     full_lens, full_num_density, full_kw_density),
        (num_only, num_lens,  num_num_density,  num_kw_density),
        (kw_only,  kw_lens,   kw_num_density,   kw_kw_density),
    ]:
        n = len(text)
        lens.append(n)
        numbers = re.findall(NUM_PATTERN, text)
        nd.append(len(numbers) / max(n, 1) * 100)
        kws = re.findall(PRICE_KW_PATTERN, text)
        kd.append(len(kws) / max(n, 1) * 100)

print("\n=== Oracle悖论机制分析：证据文本三种变体统计 ===")
print(f"{'变体':<20} {'平均长度(字符)':<18} {'数字密度(%/字)':<18} {'价格词密度(%/字)':<18}")
print("-" * 74)
for label, lens, nd, kd in [
    ("L2-Full（完整）",   full_lens, full_num_density, full_kw_density),
    ("L2-NumOnly（数字句）", num_lens, num_num_density, num_kw_density),
    ("L2-KwOnly（价格词句）", kw_lens, kw_num_density, kw_kw_density),
]:
    print(f"{label:<20} {np.mean(lens):>12.1f}     {np.mean(nd):>12.4f}       {np.mean(kd):>12.4f}")

# ─── 词汇封闭度分析 ─────────────────────────────────────────
print("\n=== 词汇封闭度分析（政务审核领域专业术语） ===")
from collections import Counter
import jieba

# 分词统计
all_text = ' '.join(str(item.get('item_name', '')) + ' ' + 
                    str(item.get('sheet_name', '')) + ' ' +
                    str(item.get('difficulty_reason', ''))
                    for item in data)
words = list(jieba.cut(all_text))
word_freq = Counter(words)
filtered = [(w, c) for w, c in word_freq.most_common(500) 
            if len(w) >= 2 and w.strip() and not w.isdigit()]

total_tokens = sum(word_freq.values())
top50_count = sum(c for _, c in filtered[:50])
top100_count = sum(c for _, c in filtered[:100])
top200_count = sum(c for _, c in filtered[:200])

print(f"总token数: {total_tokens}")
print(f"Top-50词汇覆盖率:  {top50_count/total_tokens*100:.1f}%")
print(f"Top-100词汇覆盖率: {top100_count/total_tokens*100:.1f}%")
print(f"Top-200词汇覆盖率: {top200_count/total_tokens*100:.1f}%")
print("\nTop-20高频专业术语:")
for w, c in filtered[:20]:
    print(f"  {w}: {c}次 ({c/total_tokens*100:.2f}%)")

# ─── 保存结果 ────────────────────────────────────────────────
results = {
    "oracle_mechanism": {
        "L2_full":    {"mean_len": round(np.mean(full_lens), 1), 
                       "num_density": round(np.mean(full_num_density), 4),
                       "kw_density": round(np.mean(full_kw_density), 4)},
        "L2_numonly": {"mean_len": round(np.mean(num_lens), 1),
                       "num_density": round(np.mean(num_num_density), 4),
                       "kw_density": round(np.mean(num_kw_density), 4)},
        "L2_kwonly":  {"mean_len": round(np.mean(kw_lens), 1),
                       "num_density": round(np.mean(kw_num_density), 4),
                       "kw_density": round(np.mean(kw_kw_density), 4)},
    },
    "vocab_closure": {
        "total_tokens": total_tokens,
        "top50_coverage": round(top50_count/total_tokens*100, 1),
        "top100_coverage": round(top100_count/total_tokens*100, 1),
        "top200_coverage": round(top200_count/total_tokens*100, 1),
        "top20_terms": filtered[:20],
    }
}
import json as j2
out = "/Volumes/Elements SE/科研/软件学报-软件投资审核/核减Agent/bench_tasks/results/oracle_mechanism_results.json"
with open(out, 'w', encoding='utf-8') as f:
    j2.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Results saved to oracle_mechanism_results.json")
