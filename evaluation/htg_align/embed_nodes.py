#!/usr/bin/env python3
"""
表文异构对齐 - 第1步：节点向量化
将可研报告语义块和投资表条目编码为向量，为跨模态对齐做准备。

V_text: 可研报告中的语义块 (来自 docx_chunks/)
V_table: 投资表中的条目 (来自 xlsx_trees/)

使用 Qwen text-embedding-v3 模型进行编码。
"""

import os, json, sys, time
import numpy as np
from openai import OpenAI

DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(DIR, '..', 'bench_data')
CHUNKS_DIR = os.path.join(BENCH_DIR, 'docx_chunks')
TREES_DIR = os.path.join(BENCH_DIR, 'xlsx_trees')
OUTPUT_DIR = os.path.join(DIR, 'embeddings')

# API 配置 (复用已有配置)
CLIENT = OpenAI(
    api_key="sk-029357b2dcc14b14a78d6a5532416c3d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024
BATCH_SIZE = 6


def get_embeddings(texts, batch_size=BATCH_SIZE):
    """批量获取文本向量"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = CLIENT.embeddings.create(
                model=EMBED_MODEL, input=batch, dimensions=EMBED_DIM
            )
            for d in resp.data:
                all_embeddings.append(d.embedding)
        except Exception as e:
            print(f"  [ERROR] batch {i//batch_size}: {e}")
            for _ in batch:
                all_embeddings.append([0.0] * EMBED_DIM)
            time.sleep(2)
        if i + batch_size < len(texts):
            time.sleep(0.3)
        if (i // batch_size) % 20 == 0:
            print(f"  嵌入进度: {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.array(all_embeddings, dtype='float32')


# ──────────────────────────────────────────────
# V_text: 可研报告语义块
# ──────────────────────────────────────────────

def chunk_to_text(chunk):
    """将可研报告语义块转为用于编码的文本"""
    parts = []
    if chunk.get('section_path'):
        parts.append(f"[章节: {chunk['section_path']}]")
    if chunk.get('title'):
        parts.append(f"[标题: {chunk['title']}]")
    content = chunk.get('content', '')
    if content:
        parts.append(content[:500])  # 截断到500字
    return ' '.join(parts)


def extract_text_nodes():
    """从 docx_chunks/ 提取所有可研报告语义块"""
    nodes = []
    chunk_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.json')])

    for cf in chunk_files:
        proj_id = cf.replace('.json', '')
        with open(os.path.join(CHUNKS_DIR, cf), 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)

        for doc in chunk_data.get('documents', []):
            doc_type = doc.get('doc_type', '')
            if doc_type not in ('feasibility_report', 'evaluation_report'):
                continue

            for i, chunk in enumerate(doc.get('chunks', [])):
                content = chunk.get('content', '').strip()
                if not content or len(content) < 20:
                    continue  # 跳过过短的块

                node = {
                    'node_id': f"{proj_id}_text_{doc_type}_{i}",
                    'node_type': 'text',
                    'project_id': proj_id,
                    'doc_type': doc_type,
                    'section_path': chunk.get('section_path', ''),
                    'title': chunk.get('title', ''),
                    'content': content,
                    'char_count': len(content),
                    'embed_text': chunk_to_text(chunk),
                }
                nodes.append(node)

    return nodes


# ──────────────────────────────────────────────
# V_table: 投资表条目
# ──────────────────────────────────────────────

def item_to_text(item, sheet_name=''):
    """将投资表条目转为用于编码的文本"""
    parts = []
    if sheet_name:
        parts.append(f"[费用类别: {sheet_name}]")
    if item.get('path'):
        parts.append(f"[路径: {item['path']}]")
    parts.append(f"设备/项目: {item['name']}")
    if item.get('spec'):
        parts.append(f"规格: {item['spec'][:100]}")
    if item.get('unit'):
        parts.append(f"单位: {item['unit']}")
    if item.get('quantity') is not None:
        parts.append(f"数量: {item['quantity']}")
    if item.get('original_total') is not None:
        parts.append(f"金额: {item['original_total']}万元")
    return ' '.join(parts)


def flatten_items(items, parent_path='', depth=0):
    """将层级树展平"""
    result = []
    for item in items:
        path = f"{parent_path} > {item['name']}" if parent_path else item['name']
        item_flat = {
            'path': path.strip(),
            'depth': depth,
            'seq': item.get('seq', ''),
            'name': item['name'],
            'spec': item.get('spec', ''),
            'unit': item.get('unit', ''),
            'quantity': item.get('quantity'),
            'unit_price': item.get('unit_price'),
            'original_total': item.get('original_total'),
            'adjusted_total': item.get('adjusted_total'),
            'diff': item.get('diff'),
            'has_children': len(item.get('children', [])) > 0,
            'children': item.get('children', []),
        }
        result.append(item_flat)
        for child in item.get('children', []):
            result.extend(flatten_items([child], path, depth + 1))
    return result


def extract_table_nodes():
    """从 xlsx_trees/ 提取所有投资表条目"""
    nodes = []
    tree_files = sorted([f for f in os.listdir(TREES_DIR) if f.endswith('.json')])

    for tf in tree_files:
        proj_id = tf.replace('.json', '')
        with open(os.path.join(TREES_DIR, tf), 'r', encoding='utf-8') as f:
            tree_data = json.load(f)

        for sheet in tree_data.get('sheets', []):
            sheet_name = sheet.get('sheet_name', '')
            items = flatten_items(sheet.get('items', []))

            for i, item in enumerate(items):
                if not item['name'] or item['name'].strip() in ('', '合计', '小计', '总计'):
                    continue

                node = {
                    'node_id': f"{proj_id}_table_{sheet_name[:10]}_{item['seq']}",
                    'node_type': 'table',
                    'project_id': proj_id,
                    'sheet_name': sheet_name,
                    'item_name': item['name'],
                    'item_path': item['path'],
                    'depth': item['depth'],
                    'spec': item.get('spec', ''),
                    'quantity': item.get('quantity'),
                    'unit_price': item.get('unit_price'),
                    'original_total': item.get('original_total'),
                    'adjusted_total': item.get('adjusted_total'),
                    'has_children': item.get('has_children', False),
                    'embed_text': item_to_text(item, sheet_name),
                }
                nodes.append(node)

    return nodes


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 提取节点
    print("=" * 60)
    print("提取可研报告语义块 (V_text)...")
    text_nodes = extract_text_nodes()
    print(f"  共 {len(text_nodes)} 个语义块")

    print("\n提取投资表条目 (V_table)...")
    table_nodes = extract_table_nodes()
    print(f"  共 {len(table_nodes)} 个条目")

    # 按项目统计
    from collections import Counter
    text_by_proj = Counter(n['project_id'] for n in text_nodes)
    table_by_proj = Counter(n['project_id'] for n in table_nodes)
    print(f"\n{'项目':>6s} {'V_text':>8s} {'V_table':>8s}")
    for pid in sorted(set(list(text_by_proj.keys()) + list(table_by_proj.keys()))):
        print(f"  {pid:>4s} {text_by_proj.get(pid,0):>8d} {table_by_proj.get(pid,0):>8d}")

    # 保存节点元数据
    with open(os.path.join(OUTPUT_DIR, 'text_nodes.json'), 'w', encoding='utf-8') as f:
        json.dump(text_nodes, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'table_nodes.json'), 'w', encoding='utf-8') as f:
        json.dump(table_nodes, f, ensure_ascii=False, indent=2)

    # 向量化
    print(f"\n{'='*60}")
    print(f"向量化 V_text ({len(text_nodes)} 个)...")
    text_texts = [n['embed_text'] for n in text_nodes]
    text_embeddings = get_embeddings(text_texts)
    np.save(os.path.join(OUTPUT_DIR, 'text_embeddings.npy'), text_embeddings)
    print(f"  保存: text_embeddings.npy ({text_embeddings.shape})")

    print(f"\n向量化 V_table ({len(table_nodes)} 个)...")
    table_texts = [n['embed_text'] for n in table_nodes]
    table_embeddings = get_embeddings(table_texts)
    np.save(os.path.join(OUTPUT_DIR, 'table_embeddings.npy'), table_embeddings)
    print(f"  保存: table_embeddings.npy ({table_embeddings.shape})")

    print(f"\n{'='*60}")
    print(f"完成！输出目录: {OUTPUT_DIR}")
    print(f"  text_nodes.json + text_embeddings.npy: {len(text_nodes)} 个可研语义块")
    print(f"  table_nodes.json + table_embeddings.npy: {len(table_nodes)} 个投资条目")


if __name__ == '__main__':
    main()
