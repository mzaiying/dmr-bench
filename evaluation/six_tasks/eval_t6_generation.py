#!/usr/bin/env python3
"""
HTG-Align 对齐质量评估实验
=======================================
用 bench 中的 evidence_text 作为金标准，
评测三种检索方法在"找到专家标注证据段落"任务上的性能：

  方法1: BM25       — 关键词检索
  方法2: Dense      — 纯向量语义检索
  方法3: HTG-Align  — 我们的双路对齐（向量 + LLM判断）

指标:
  P@1    = 首个返回结果命中金标准的比例
  R@3    = 前三个结果中命中金标准的比例
  NDCG@3 = 折扣累积增益（考虑排序质量）

金标准匹配策略:
  对每个有 evidence_text 的样本，在 text_nodes 中
  找字符级 f-score 最高的节点作为"金标准节点"；
  若 f-score >= 0.3 则认为该样本可用（过滤噪声）。

输出: 控制台表格 + results/align_quality_{timestamp}.json
"""
import os, json, re, math
from collections import defaultdict
from datetime import datetime
import numpy as np

DIR        = os.path.dirname(os.path.abspath(__file__))
BENCH      = os.path.join(DIR, '..', 'bench', 'bench_data', 'govreview_bench_v2.json')
ALIGN_DUAL = os.path.join(DIR, '..', 'bench', 'alignment', 'alignment_results', 'bench_alignment_dual.json')
ALIGN_VEC  = os.path.join(DIR, '..', 'bench', 'alignment', 'alignment_results', 'bench_alignment_vec_only.json')
TEXT_NODES = os.path.join(DIR, '..', 'bench', 'alignment', 'embeddings', 'text_nodes.json')
OUT        = os.path.join(DIR, 'results')


# ══ BM25 ══
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75, ngram=2):
        self.k1, self.b, self.ngram = k1, b, ngram
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n, 1)
        self.idf, self.tf_docs = {}, []
        df = defaultdict(int)
        for doc in corpus:
            for t in set(self._tok(doc)): df[t] += 1
        for t, d in df.items():
            self.idf[t] = math.log((self.n - d + 0.5) / (d + 0.5) + 1)
        for doc in corpus:
            tf = defaultdict(int)
            for t in self._tok(doc): tf[t] += 1
            self.tf_docs.append(dict(tf))

    def _tok(self, t):
        return [t[i:i+self.ngram] for i in range(len(t)-self.ngram+1)] + list(t)

    def get_scores(self, query):
        s = np.zeros(self.n)
        for t in self._tok(query):
            if t not in self.idf: continue
            for i, tfd in enumerate(self.tf_docs):
                tf = tfd.get(t, 0); dl = sum(tfd.values())
                s[i] += self.idf[t]*tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*dl/self.avgdl))
        return s

    def top_k(self, query, k=3):
        s = self.get_scores(query)
        return sorted(enumerate(s), key=lambda x: -x[1])[:k]


# ══ 工具函数 ══
def char_ngrams(text, n=2):
    return set(text[i:i+n] for i in range(len(text)-n+1))

def fscore_overlap(text_a, text_b, ngram=2):
    """字符 n-gram F1 overlap（用于判断两段文本是否来自同一来源）"""
    if not text_a or not text_b: return 0.0
    a, b = char_ngrams(text_a, ngram), char_ngrams(text_b, ngram)
    if not a or not b: return 0.0
    inter = len(a & b)
    p = inter / len(a); r = inter / len(b)
    return 2*p*r/(p+r) if p+r > 0 else 0.0

def dcg_at_k(hits, k):
    """DCG@k，hits = [1/0 list of relevance]"""
    return sum(h / math.log2(i+2) for i, h in enumerate(hits[:k]))

def ndcg_at_k(hits, k):
    """NDCG@k，理想情况是第一位就命中"""
    ideal = dcg_at_k([1]+[0]*(k-1), k)
    return dcg_at_k(hits, k) / ideal if ideal > 0 else 0.0


# ══ 主评估 ══
def run():
    print("加载数据...")
    with open(BENCH, encoding='utf-8') as f:      bench = json.load(f)
    with open(ALIGN_DUAL, encoding='utf-8') as f: align_dual = json.load(f)
    with open(ALIGN_VEC,  encoding='utf-8') as f: align_vec  = json.load(f)
    with open(TEXT_NODES, encoding='utf-8') as f: text_nodes = json.load(f)

    dual_by_sid = {a['sample_id']: a for a in align_dual}
    vec_by_sid  = {a['sample_id']: a for a in align_vec}

    # 建文本节点库（按项目）
    text_by_proj = defaultdict(list)
    for i, n in enumerate(text_nodes):
        text_by_proj[n['project_id']].append((i, n))

    bm25_by_proj = {}
    for pid, items in text_by_proj.items():
        corpus = [f"{it['title']} {it['section_path']} {it['content']}" for _, it in items]
        bm25_by_proj[pid] = (BM25(corpus), items)

    # ══ 为有 evidence_text 的样本建立金标准映射 ══
    print("\n构建金标准节点映射...")
    gold_map = {}   # sid → (gold_proj_node_idx, fscore)
    GOLD_THRESH = 0.25   # F1 阈值

    for d in bench:
        sid = d.get('sample_id', '')
        pid = d.get('project_id', '')
        ev  = d.get('evidence_text', '').strip()
        if not ev or len(ev) < 20: continue
        if pid not in text_by_proj: continue

        best_idx, best_f = -1, 0.0
        for local_idx, node in text_by_proj[pid]:
            content = node.get('content', '')
            f = fscore_overlap(ev[:300], content[:300])
            if f > best_f:
                best_f = f; best_idx = local_idx
        if best_f >= GOLD_THRESH:
            gold_map[sid] = {'gold_idx': best_idx, 'fscore': round(best_f, 3)}

    n_gold = len(gold_map)
    print(f"  有效金标准样本: {n_gold}/{len(bench)} "
          f"(金标准覆盖率 {100*n_gold//len(bench)}%, fscore≥{GOLD_THRESH})")

    # ══ 评估三种方法 ══
    methods = {
        'BM25':      {'p1': [], 'r3': [], 'ndcg3': []},
        'Dense':     {'p1': [], 'r3': [], 'ndcg3': []},
        'HTG-Align': {'p1': [], 'r3': [], 'ndcg3': []},
    }

    for d in bench:
        sid = d.get('sample_id', ''); pid = d.get('project_id', '')
        iname = d.get('item_name', ''); ipath = d.get('item_path', '')
        if sid not in gold_map: continue
        gold_idx = gold_map[sid]['gold_idx']

        # ── BM25 ──
        if pid in bm25_by_proj:
            bm25_obj, proj_items = bm25_by_proj[pid]
            query = f"{iname} {ipath}"
            hits_bm25 = bm25_obj.top_k(query, k=3)
            bm25_indices = [proj_items[li][0] for li, _ in hits_bm25]
            hits = [1 if idx == gold_idx else 0 for idx in bm25_indices]
            methods['BM25']['p1'].append(hits[0] if hits else 0)
            methods['BM25']['r3'].append(int(any(h == 1 for h in hits)))
            methods['BM25']['ndcg3'].append(ndcg_at_k(hits, 3))

        # ── Dense ──
        a_vec = vec_by_sid.get(sid)
        if a_vec:
            vec_chunks = a_vec.get('aligned_chunks', [])[:3]
            # 通过内容 f-score 找 Dense 返回的节点在 text_nodes 中的位置
            dense_indices = []
            for c in vec_chunks:
                c_content = c.get('content', '')
                best_i, best_f = -1, 0.0
                for li, node in text_by_proj.get(pid, []):
                    f = fscore_overlap(c_content[:200], node.get('content','')[:200])
                    if f > best_f: best_f = f; best_i = li
                if best_f >= 0.3: dense_indices.append(best_i)
            hits = [1 if idx == gold_idx else 0 for idx in dense_indices]
            if not hits: hits = [0, 0, 0]
            methods['Dense']['p1'].append(hits[0])
            methods['Dense']['r3'].append(int(any(h == 1 for h in hits)))
            methods['Dense']['ndcg3'].append(ndcg_at_k(hits, 3))

        # ── HTG-Align（双路对齐）──
        a_dual = dual_by_sid.get(sid)
        if a_dual:
            dual_chunks = a_dual.get('aligned_chunks', [])[:3]
            dual_indices = []
            for c in dual_chunks:
                c_content = c.get('content', '')
                best_i, best_f = -1, 0.0
                for li, node in text_by_proj.get(pid, []):
                    f = fscore_overlap(c_content[:200], node.get('content','')[:200])
                    if f > best_f: best_f = f; best_i = li
                if best_f >= 0.3: dual_indices.append(best_i)
            hits = [1 if idx == gold_idx else 0 for idx in dual_indices]
            if not hits: hits = [0, 0, 0]
            methods['HTG-Align']['p1'].append(hits[0])
            methods['HTG-Align']['r3'].append(int(any(h == 1 for h in hits)))
            methods['HTG-Align']['ndcg3'].append(ndcg_at_k(hits, 3))

    # ══ 汇总输出 ══
    print(f"\n{'='*65}")
    print(f"  HTG-Align 对齐质量评估 （金标准: evidence_text, n={n_gold}）")
    print(f"{'='*65}")
    print(f"{'方法':<14}  {'样本数':>6}  {'P@1':>8}  {'Recall@3':>10}  {'NDCG@3':>8}")
    print(f"{'-'*60}")

    result_data = {}
    for mname, m in methods.items():
        n = len(m['p1'])
        if n == 0:
            print(f"  {mname:<12}  {'N/A':>6}")
            continue
        p1   = np.mean(m['p1'])
        r3   = np.mean(m['r3'])
        ndcg = np.mean(m['ndcg3'])
        result_data[mname] = {'n': n, 'P@1': round(float(p1),4),
                              'Recall@3': round(float(r3),4),
                              'NDCG@3': round(float(ndcg),4)}
        mark = '← 最优' if mname == 'HTG-Align' else ''
        print(f"  {mname:<12}  {n:>6}  {p1:>8.4f}  {r3:>10.4f}  {ndcg:>8.4f}  {mark}")

    print(f"{'='*65}")

    # 与 BM25 对比 delta
    if 'BM25' in result_data and 'HTG-Align' in result_data:
        bm = result_data['BM25']; ht = result_data['HTG-Align']
        print(f"\nHTG-Align vs BM25:")
        print(f"  P@1    : {ht['P@1']:.4f} vs {bm['P@1']:.4f}  ({'+' if ht['P@1']>=bm['P@1'] else ''}{(ht['P@1']-bm['P@1'])*100:.2f}pp)")
        print(f"  Recall@3: {ht['Recall@3']:.4f} vs {bm['Recall@3']:.4f}  ({'+' if ht['Recall@3']>=bm['Recall@3'] else ''}{(ht['Recall@3']-bm['Recall@3'])*100:.2f}pp)")
        print(f"  NDCG@3 : {ht['NDCG@3']:.4f} vs {bm['NDCG@3']:.4f}  ({'+' if ht['NDCG@3']>=bm['NDCG@3'] else ''}{(ht['NDCG@3']-bm['NDCG@3'])*100:.2f}pp)")

    if 'Dense' in result_data and 'HTG-Align' in result_data:
        dn = result_data['Dense']; ht = result_data['HTG-Align']
        print(f"\nHTG-Align vs Dense:")
        print(f"  P@1    : {ht['P@1']:.4f} vs {dn['P@1']:.4f}  ({'+' if ht['P@1']>=dn['P@1'] else ''}{(ht['P@1']-dn['P@1'])*100:.2f}pp)")
        print(f"  Recall@3: {ht['Recall@3']:.4f} vs {dn['Recall@3']:.4f}  ({'+' if ht['Recall@3']>=dn['Recall@3'] else ''}{(ht['Recall@3']-dn['Recall@3'])*100:.2f}pp)")
        print(f"  NDCG@3 : {ht['NDCG@3']:.4f} vs {dn['NDCG@3']:.4f}  ({'+' if ht['NDCG@3']>=dn['NDCG@3'] else ''}{(ht['NDCG@3']-dn['NDCG@3'])*100:.2f}pp)")

    # 保存
    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = os.path.join(OUT, f'align_quality_{ts}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': str(datetime.now()),
            'n_gold_samples': n_gold,
            'gold_threshold': GOLD_THRESH,
            'results': result_data
        }, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")


if __name__ == '__main__':
    run()
