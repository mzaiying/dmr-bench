#!/usr/bin/env python3
"""
03_eval.py — CCGP价格合理性审核 LLM评测（5层RAG对比）

完全复用 GovReview-Bench 的评测框架：
  L1:     Zero-Knowledge（无证据基线）
  L_BM25: BM25关键词检索
  L_Dense:向量语义检索（FAISS）
  L2:     Oracle（直接提供参考价格，上界）
  L3:     BM25+价格统计混合（本域最优方法）

用法:
    python3 03_eval.py Qwen
    python3 03_eval.py DeepSeek
    python3 03_eval.py --all  # 跑全部4款模型
"""
import os, sys, json, re, asyncio, math, time
import numpy as np
from datetime import datetime
from collections import defaultdict
from openai import AsyncOpenAI

DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_FILE  = os.path.join(DIR, "ccgp_bench.json")
CORPUS_FILE = os.path.join(DIR, "ccgp_rag_corpus.json")
OUT_DIR     = os.path.join(DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 与原始实验相同的模型配置 ──────────────────────────────
MODELS = {
    "Qwen": {
        "async_client": lambda: AsyncOpenAI(
            api_key="sk-029357b2dcc14b14a78d6a5532416c3d",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "model": "qwen-plus",
    },
    "DeepSeek": {
        "async_client": lambda: AsyncOpenAI(
            api_key="sk-029357b2dcc14b14a78d6a5532416c3d",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "model": "deepseek-v3",
    },
    "Doubao": {
        "async_client": lambda: AsyncOpenAI(
            api_key="0e1e8476-22ab-401d-8c2c-49151d9e1160",
            base_url="https://ark.cn-beijing.volces.com/api/v3/"),
        "model": "ep-20260321200826-t2vvd",
    },
    "GLM4": {
        "async_client": lambda: AsyncOpenAI(
            api_key="f180f8e7dc5449d78bc14bcae7307954.pF4C5ubuu7IU7d3R",
            base_url="https://open.bigmodel.cn/api/paas/v4/"),
        "model": "glm-4-plus",
    },
}


# ══════════════════════════════════════════════
# BM25（与原始实验相同的实现）
# ══════════════════════════════════════════════

class BM25:
    """字符N-gram BM25，与 llm_eval_rag_comparison.py 保持一致"""
    def __init__(self, corpus, k1=1.5, b=0.75, ngram=2):
        self.k1, self.b, self.ngram = k1, b, ngram
        self.corpus = corpus
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n, 1)
        self.idf, self.tf_docs = {}, []
        df = defaultdict(int)
        for doc in corpus:
            for t in set(self._tok(doc)):
                df[t] += 1
        for t, d in df.items():
            self.idf[t] = math.log((self.n - d + 0.5) / (d + 0.5) + 1)
        for doc in corpus:
            tf = defaultdict(int)
            for t in self._tok(doc):
                tf[t] += 1
            self.tf_docs.append(dict(tf))

    def _tok(self, text):
        tokens = [text[i:i+self.ngram] for i in range(len(text)-self.ngram+1)]
        tokens += list(text)
        return tokens

    def get_scores(self, query):
        scores = np.zeros(self.n)
        for t in self._tok(query):
            if t not in self.idf:
                continue
            idf_t = self.idf[t]
            for i, tf_doc in enumerate(self.tf_docs):
                tf = tf_doc.get(t, 0)
                dl = sum(tf_doc.values())
                scores[i] += idf_t * tf * (self.k1+1) / (tf + self.k1*(1-self.b+self.b*dl/self.avgdl))
        return scores

    def top_k(self, query, k=3):
        scores = self.get_scores(query)
        idxs = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idxs if scores[i] > 0.1]


# ══════════════════════════════════════════════
# Dense检索（FAISS向量检索）
# ══════════════════════════════════════════════

def build_dense_index(corpus_texts: list[str], client, model: str):
    """构建FAISS向量索引（使用与原实验相同的embedding模型）"""
    try:
        import faiss
        from openai import OpenAI
        sync_client = OpenAI(
            api_key="sk-029357b2dcc14b14a78d6a5532416c3d",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        DIM = 1024
        all_emb = []
        bs = 6
        print(f"  构建Dense索引({len(corpus_texts)}条)...")
        for i in range(0, len(corpus_texts), bs):
            batch = corpus_texts[i:i+bs]
            try:
                resp = sync_client.embeddings.create(
                    model="text-embedding-v3", input=batch, dimensions=DIM)
                for d in resp.data:
                    all_emb.append(d.embedding)
            except Exception as e:
                for _ in batch:
                    all_emb.append([0.0] * DIM)
            time.sleep(0.3)
        emb = np.array(all_emb, dtype="float32")
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(DIM)
        index.add(emb)
        return index, DIM
    except ImportError:
        print("  [跳过] faiss未安装，跳过Dense检索层")
        return None, None


# ══════════════════════════════════════════════
# Prompts（与领域对应的新版本）
# ══════════════════════════════════════════════

def mk_t1_ccgp(sample: dict, ev_text: str) -> str:
    """T1：判断该政府采购价格是否虚高"""
    spec = sample.get("spec", "")[:150] if sample.get("spec") else ""
    evidence_block = f"\n\n## 同类历史中标价格参考\n{ev_text[:600]}\n" if ev_text else ""
    return (
        f"你是一位政府采购价格合理性审核专家。请根据以下信息判断该采购单价是否虚高。\n\n"
        f"品目名称: {sample.get('item_name', '')}\n"
        f"规格参数: {spec}\n"
        f"申报单价: {sample.get('original_unit', 0):.4f} 万元\n"
        f"采购数量: {sample.get('quantity', 1)}\n"
        f"申报总价: {sample.get('original_total', 0):.4f} 万元\n"
        f"{evidence_block}"
        f"请仅回答'核减'（价格虚高）或'不核减'（价格合理），不要解释。"
    )


def mk_t2_ccgp(sample: dict, ev_text: str) -> str:
    """T2：估算该采购价格的虚高比例（核减率%）"""
    spec = sample.get("spec", "")[:150] if sample.get("spec") else ""
    evidence_block = f"\n\n## 同类历史中标价格参考\n{ev_text[:600]}\n" if ev_text else ""
    return (
        f"你是一位政府采购价格合理性审核专家。请估算该采购单价的虚高比例。\n\n"
        f"品目名称: {sample.get('item_name', '')}\n"
        f"规格参数: {spec}\n"
        f"申报单价: {sample.get('original_unit', 0):.4f} 万元\n"
        f"{evidence_block}"
        f"请仅回答一个数字（0-100之间的虚高百分比），不要解释。例如: 25\n"
        f"如果认为价格合理，请回答: 0"
    )


def mk_t3_ccgp(sample: dict, ev_text: str) -> str:
    """T3：估算该采购的合理总价（万元）"""
    spec = sample.get("spec", "")[:150] if sample.get("spec") else ""
    evidence_block = f"\n\n## 同类历史中标价格参考\n{ev_text[:600]}\n" if ev_text else ""
    return (
        f"你是一位政府采购价格合理性审核专家。请估算该采购的合理总价。\n\n"
        f"品目名称: {sample.get('item_name', '')}\n"
        f"规格参数: {spec}\n"
        f"申报单价: {sample.get('original_unit', 0):.4f} 万元\n"
        f"采购数量: {sample.get('quantity', 1)}\n"
        f"申报总价: {sample.get('original_total', 0):.4f} 万元\n"
        f"{evidence_block}"
        f"请仅回答一个数字（合理总价，单位万元），不要解释。例如: 85.50"
    )


# ══════════════════════════════════════════════
# 解析和指标（与原始实验完全一致）
# ══════════════════════════════════════════════

def parse_t1(r: str) -> int | None:
    if not r or "ERROR" in r:
        return None
    if "不核减" in r or "合理" in r or "保留" in r:
        return 0
    if "核减" in r or "虚高" in r:
        return 1
    return None


def parse_num(r: str, cap=None) -> float | None:
    if not r or "ERROR" in r:
        return None
    nums = re.findall(r"[\d.]+", r)
    if nums:
        v = float(nums[0])
        return min(v, cap) if cap else v
    return None


def calc_clf(yt: list, yp: list) -> dict:
    valid = [(t, p) for t, p in zip(yt, yp) if p is not None]
    if not valid:
        return {"n": 0}
    yt2, yp2 = zip(*valid)
    acc = sum(t == p for t, p in zip(yt2, yp2)) / len(yt2)
    f1s = []
    for c in [0, 1]:
        tp = sum(t == c and p == c for t, p in zip(yt2, yp2))
        fp = sum(t != c and p == c for t, p in zip(yt2, yp2))
        fn = sum(t == c and p != c for t, p in zip(yt2, yp2))
        pr = tp / (tp + fp) if tp + fp > 0 else 0
        rc = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0
        f1s.append(f1)
    return {
        "n": len(yt2),
        "Accuracy": round(acc, 4),
        "Macro-F1": round(sum(f1s) / len(f1s), 4),
    }


def calc_reg(yt: list, yp: list, yo: list) -> dict:
    valid = [(t, p, o) for t, p, o in zip(yt, yp, yo) if p is not None]
    if not valid:
        return {"n": 0}
    yt2 = np.array([v[0] for v in valid])
    yp2 = np.array([v[1] for v in valid])
    yo2 = np.array([v[2] for v in valid])
    mae = float(np.mean(np.abs(yp2 - yt2)))
    mre = np.abs(yp2 - yt2) / np.maximum(yo2, 0.01)
    return {
        "n": len(valid),
        "MAE": round(mae, 4),
        "PRED25": round(float(np.mean(mre <= 0.25)), 4),
    }


# ══════════════════════════════════════════════
# 主评测流程
# ══════════════════════════════════════════════

async def eval_layer(
    client, model: str, bench: list, corpus: list,
    bm25: BM25, dense_index, layer_name: str,
    concurrency: int = 8
) -> dict:
    sem = asyncio.Semaphore(concurrency)
    results = {}

    def get_evidence(sample: dict, layer: str) -> str:
        cat = sample.get("category", "")
        item_name = sample.get("item_name", "")
        spec = sample.get("spec", "")
        query = f"{cat} {item_name} {spec}"

        if layer == "L1":
            return ""

        elif layer == "L_BM25":
            hits = bm25.top_k(query, k=5)
            parts = []
            for idx, score in hits:
                c = corpus[idx]
                parts.append(
                    f"- {c['item_name']} 单价{c['unit_price_wan']:.2f}万 "
                    f"({c.get('date','')}) 供应商:{c.get('supplier','')[:20]}"
                )
            return "\n".join(parts)

        elif layer == "L_Dense" and dense_index is not None:
            try:
                import faiss
                from openai import OpenAI
                sync_c = OpenAI(
                    api_key="sk-029357b2dcc14b14a78d6a5532416c3d",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                resp = sync_c.embeddings.create(
                    model="text-embedding-v3", input=[query], dimensions=1024)
                vec = np.array([resp.data[0].embedding], dtype="float32")
                faiss.normalize_L2(vec)
                _, idxs = dense_index.search(vec, 5)
                parts = []
                for i in idxs[0]:
                    if 0 <= i < len(corpus):
                        c = corpus[i]
                        parts.append(
                            f"- {c['item_name']} 单价{c['unit_price_wan']:.2f}万 "
                            f"({c.get('date','')})"
                        )
                return "\n".join(parts)
            except:
                return ""

        elif layer == "L2":
            # Oracle: 直接告知参考价格（上界）
            ref = sample.get("market_ref_unit", 0)
            return (
                f"同类商品历史中标单价参考（Q25分位数）: {ref:.4f} 万元/件\n"
                f"数据来源: 政府采购网历史中标记录"
            )

        elif layer == "L3":
            # BM25 + 统计摘要（本域最优组合）
            hits = bm25.top_k(query, k=5)
            parts = []
            prices = []
            for idx, score in hits:
                c = corpus[idx]
                parts.append(
                    f"- {c['item_name']} 单价{c['unit_price_wan']:.2f}万 "
                    f"({c.get('date','')})"
                )
                prices.append(c["unit_price_wan"])
            summary = ""
            if prices:
                summary = (
                    f"\n统计摘要: 检索到{len(prices)}条同类记录, "
                    f"均价{np.mean(prices):.2f}万, 最低{min(prices):.2f}万"
                )
            return "\n".join(parts) + summary

        return ""

    # T1
    print(f"  [{layer_name} T1]...")
    preds1 = [None] * len(bench)
    done = [0]

    async def t1_one(i, d):
        async with sem:
            ev = get_evidence(d, layer_name)
            for attempt in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": mk_t1_ccgp(d, ev)}],
                        temperature=0.1, max_tokens=30)
                    preds1[i] = parse_t1(r.choices[0].message.content.strip())
                    done[0] += 1
                    return
                except:
                    await asyncio.sleep(2 ** (attempt + 1))
            done[0] += 1

    await asyncio.gather(*[t1_one(i, d) for i, d in enumerate(bench)])
    labels1 = [1 if d["direction"] == "核减" else 0 for d in bench]
    results[f"T1_{layer_name}"] = calc_clf(labels1, preds1)
    print(f"    → Macro-F1={results[f'T1_{layer_name}'].get('Macro-F1')}")

    # T2
    print(f"  [{layer_name} T2]...")
    preds2 = [None] * len(bench)
    done[0] = 0

    async def t2_one(i, d):
        async with sem:
            ev = get_evidence(d, layer_name)
            for attempt in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": mk_t2_ccgp(d, ev)}],
                        temperature=0.1, max_tokens=30)
                    preds2[i] = parse_num(r.choices[0].message.content.strip(), 100)
                    done[0] += 1
                    return
                except:
                    await asyncio.sleep(2 ** (attempt + 1))
            done[0] += 1

    await asyncio.gather(*[t2_one(i, d) for i, d in enumerate(bench)])
    labels2 = [d.get("ground_truth_rate", 0) for d in bench]
    orig2    = [d.get("original_total", 1) for d in bench]
    results[f"T2_{layer_name}"] = calc_reg(labels2, preds2, orig2)
    print(f"    → MAE={results[f'T2_{layer_name}'].get('MAE')}")

    # T3
    valid_t3 = [d for d in bench if d.get("original_total", 0) > 0 and d.get("ground_truth_total") is not None]
    print(f"  [{layer_name} T3] ({len(valid_t3)})...")
    preds3 = [None] * len(valid_t3)
    done[0] = 0

    async def t3_one(i, d):
        async with sem:
            ev = get_evidence(d, layer_name)
            for attempt in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": mk_t3_ccgp(d, ev)}],
                        temperature=0.1, max_tokens=30)
                    preds3[i] = parse_num(r.choices[0].message.content.strip())
                    done[0] += 1
                    return
                except:
                    await asyncio.sleep(2 ** (attempt + 1))
            done[0] += 1

    await asyncio.gather(*[t3_one(i, d) for i, d in enumerate(valid_t3)])
    labels3 = [d["ground_truth_total"] for d in valid_t3]
    orig3   = [d["original_total"]     for d in valid_t3]
    results[f"T3_{layer_name}"] = calc_reg(labels3, preds3, orig3)
    print(f"    → MAE={results[f'T3_{layer_name}'].get('MAE')} PRED25={results[f'T3_{layer_name}'].get('PRED25')}")

    return results


async def run_model(model_name: str):
    cfg = MODELS[model_name]
    client = cfg["async_client"]()
    model  = cfg["model"]

    # 加载数据
    with open(BENCH_FILE, encoding="utf-8") as f:
        bench = json.load(f)
    with open(CORPUS_FILE, encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"\n{'='*65}")
    print(f"  CCGP-PriceBench 跨域验证实验")
    print(f"  模型: {model_name} | 测试集: {len(bench)}条 | 知识库: {len(corpus)}条")
    print(f"{'='*65}")

    # 构建BM25索引
    corpus_texts = [c.get("search_text", f"{c['item_name']} {c.get('spec','')}") for c in corpus]
    bm25 = BM25(corpus_texts)
    print(f"BM25索引构建完成 ({len(corpus_texts)}条)")

    # 构建Dense索引（如果faiss可用）
    dense_index = None
    if "--no-dense" not in sys.argv:
        dense_index, _ = build_dense_index(corpus_texts, client, model)

    all_results = {}

    layers = [
        "L1",
        "L_BM25",
        "L_Dense",
        "L2",
        "L3",
    ]

    for layer_name in layers:
        if layer_name == "L_Dense" and dense_index is None:
            print(f"\n── {layer_name} [跳过，faiss不可用] ──")
            continue
        print(f"\n── {layer_name} ─────────────────────────────────────────")
        r = await eval_layer(
            client, model, bench, corpus, bm25, dense_index, layer_name
        )
        all_results.update(r)

    # 保存
    output = {
        "model": model_name,
        "model_id": model,
        "dataset": "CCGP-PriceBench",
        "n_bench": len(bench),
        "n_corpus": len(corpus),
        "timestamp": str(datetime.now()),
        "results": all_results,
    }
    fname = f"ccgp_eval_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {fpath}")

    # 打印对比表
    print(f"\n{'='*75}")
    print(f"  {model_name}  CCGP-PriceBench RAG方法对比\n")
    layer_labels = ["L1(无证据)", "BM25-RAG", "Dense-RAG", "L2(Oracle)", "L3(BM25+统计)"]
    layer_keys   = ["L1", "L_BM25", "L_Dense", "L2", "L3"]

    for task, metric, higher_better in [
        ("T1", "Macro-F1", True),
        ("T2", "MAE", False),
        ("T3", "PRED25", True)
    ]:
        vals = [all_results.get(f"{task}_{k}", {}).get(metric, float("nan")) for k in layer_keys]
        row = f"  {task}  {metric:10s}" + "".join(
            f"  {v:>12.4f}" if not math.isnan(v) else f"  {'N/A':>12}" for v in vals
        )
        valid_vals = [(i, v) for i, v in enumerate(vals) if not math.isnan(v)]
        if valid_vals:
            best = max(valid_vals, key=lambda x: x[1] if higher_better else -x[1])
            print(row + f"  ← {layer_labels[best[0]]}")
        else:
            print(row)

    print(f"{'='*75}")
    return all_results


async def main():
    if "--all" in sys.argv:
        all_model_results = {}
        for model_name in ["Qwen", "DeepSeek", "GLM4", "Doubao"]:
            try:
                r = await run_model(model_name)
                all_model_results[model_name] = r
            except Exception as e:
                print(f"[错误] {model_name}: {e}")

        # 保存汇总
        summary_path = os.path.join(OUT_DIR, f"ccgp_all_models_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_model_results, f, ensure_ascii=False, indent=2)
        print(f"\n汇总保存: {summary_path}")
    else:
        model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen"
        await run_model(model_name)


if __name__ == "__main__":
    asyncio.run(main())
