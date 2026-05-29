#!/usr/bin/env python3
"""
表文异构对齐 - 针对 bench 样本的双路对齐
只对 802 条 bench 样本运行 LLM 语义判断 (节省 API 调用)
对每个 bench 样本检索 top-5 可研段落，然后用 LLM 判断相关性

输出: bench 中每条样本的对齐结果 (新的 evidence_chunks)
"""

import os, json, sys, time, re
import numpy as np
from openai import OpenAI
from collections import defaultdict

DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(DIR, 'embeddings')
BENCH_DIR = os.path.join(DIR, '..', 'bench_data')
OUTPUT_DIR = os.path.join(DIR, 'alignment_results')

CLIENT = OpenAI(
    api_key="sk-029357b2dcc14b14a78d6a5532416c3d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
CHAT_MODEL = "qwen-plus"
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024

# 超参数
LAMBDA = 0.6
TAU = 0.35
TOP_K_VEC = 5   # 向量检索 top-k
TOP_K_FINAL = 3  # 最终保留


def load_all():
    """加载所有数据"""
    with open(os.path.join(EMBED_DIR, 'text_nodes.json')) as f:
        text_nodes = json.load(f)
    with open(os.path.join(EMBED_DIR, 'table_nodes.json')) as f:
        table_nodes = json.load(f)
    text_emb = np.load(os.path.join(EMBED_DIR, 'text_embeddings.npy'))
    table_emb = np.load(os.path.join(EMBED_DIR, 'table_embeddings.npy'))

    # 归一化
    for emb in [text_emb, table_emb]:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb /= norms

    with open(os.path.join(BENCH_DIR, 'govreview_bench_v2.json')) as f:
        bench = json.load(f)

    return text_nodes, table_nodes, text_emb, table_emb, bench


def embed_query(text):
    """将查询文本向量化"""
    resp = CLIENT.embeddings.create(model=EMBED_MODEL, input=[text], dimensions=EMBED_DIM)
    vec = np.array([resp.data[0].embedding], dtype='float32')
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (vec / norms)[0]


def llm_judge(item_name, item_spec, text_title, text_content):
    """LLM 判断可研段落是否支撑投资条目"""
    prompt = f"""判断以下可行性研究报告段落是否为该投资条目提供了功能说明或依据。

投资条目: {item_name}
{'规格: ' + item_spec[:100] if item_spec else ''}

可研报告段落:
标题: {text_title}
内容: {text_content[:300]}

该段落是否与该投资条目直接相关？请只回答一个0-1之间的数字（0=无关，1=直接相关）。"""

    try:
        resp = CLIENT.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "只输出0-1之间的数字。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0, max_tokens=10
        )
        txt = resp.choices[0].message.content.strip()
        match = re.search(r'([01]\.?\d*)', txt)
        return min(1.0, max(0.0, float(match.group(1)))) if match else 0.0
    except Exception as e:
        return 0.0


def align_bench_sample(sample, text_nodes, text_emb, text_by_proj, use_llm=True):
    """对一条 bench 样本执行双路对齐"""
    pid = sample['project_id']
    item_name = sample['item_name']
    item_spec = sample.get('item_spec', '')

    # 构建查询向量
    query_text = f"设备/项目: {item_name}"
    if item_spec:
        query_text += f" 规格: {item_spec[:80]}"
    if sample.get('sheet_name'):
        query_text += f" 费用类别: {sample['sheet_name']}"

    query_vec = embed_query(query_text)

    # 取同项目的可研段落
    proj_indices = text_by_proj.get(pid, [])
    if not proj_indices:
        return []

    proj_indices = np.array(proj_indices)
    proj_emb = text_emb[proj_indices]

    # 余弦相似度
    sims = np.dot(proj_emb, query_vec)
    top_k_idx = np.argsort(sims)[::-1][:TOP_K_VEC]

    results = []
    for rank, idx in enumerate(top_k_idx):
        global_idx = proj_indices[idx]
        sim_score = float(sims[idx])
        text_node = text_nodes[global_idx]

        if use_llm and sim_score >= 0.15:
            llm_score = llm_judge(
                item_name, item_spec,
                text_node.get('title', ''),
                text_node.get('content', '')
            )
            combined = LAMBDA * sim_score + (1 - LAMBDA) * llm_score
        else:
            llm_score = 0.0
            combined = sim_score

        if combined >= TAU:
            results.append({
                'section_path': text_node.get('section_path', ''),
                'title': text_node.get('title', ''),
                'content': text_node.get('content', '')[:500],
                'sim_score': round(sim_score, 4),
                'llm_score': round(llm_score, 4),
                'combined_score': round(combined, 4),
                'relevance_score': round(combined * 10, 1),  # 兼容 bench 格式
            })

    # 排序并保留 top-k
    results.sort(key=lambda x: -x['combined_score'])
    return results[:TOP_K_FINAL]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    use_llm = '--no-llm' not in sys.argv
    mode = '双路对齐' if use_llm else '仅向量'
    print(f"模式: {mode}")

    print("加载数据...")
    text_nodes, table_nodes, text_emb, table_emb, bench = load_all()
    print(f"  V_text: {len(text_nodes)}, bench: {len(bench)}")

    # 索引: 按项目分组 text_nodes
    text_by_proj = defaultdict(list)
    for i, n in enumerate(text_nodes):
        text_by_proj[n['project_id']].append(i)

    # 逐条对齐
    results = []
    n_with_evidence = 0

    for i, sample in enumerate(bench):
        aligned = align_bench_sample(sample, text_nodes, text_emb, text_by_proj, use_llm=use_llm)

        result = {
            'sample_id': sample['sample_id'],
            'project_id': sample['project_id'],
            'item_name': sample['item_name'],
            'n_aligned': len(aligned),
            'aligned_chunks': aligned,
        }
        results.append(result)

        if aligned:
            n_with_evidence += 1

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(bench)} (有证据: {n_with_evidence})")

        time.sleep(0.3)  # API rate limit

    # 保存
    suffix = 'dual' if use_llm else 'vec_only'
    output_file = os.path.join(OUTPUT_DIR, f'bench_alignment_{suffix}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"完成! ({mode})")
    print(f"  总样本: {len(bench)}")
    print(f"  有对齐证据: {n_with_evidence} ({n_with_evidence/len(bench)*100:.1f}%)")
    print(f"  输出: {output_file}")

    # 示例
    print(f"\n=== 示例 ===")
    for r in results[:3]:
        print(f"\n  {r['item_name'][:30]} (对齐 {r['n_aligned']} 条)")
        for c in r['aligned_chunks'][:2]:
            print(f"    ←→ {c['title'][:30]} sim={c['sim_score']:.3f} llm={c['llm_score']:.3f} combined={c['combined_score']:.3f}")


if __name__ == '__main__':
    main()
