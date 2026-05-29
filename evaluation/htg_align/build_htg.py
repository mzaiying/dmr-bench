#!/usr/bin/env python3
"""
第三步：异构图谱构建
定义 G = (V, E):
  V_text:  可研报告语义块 (2578个)
  V_table: 投资表条目 (2065个)
  E_align: 跨模态对齐边 (第二步输出)
  E_hier:  投资表层级边 (父→子)

+ 子图召回算法：给定投资条目 → 返回"报价+证据+上下文"局部子图
"""

import os, json, sys
import numpy as np
from collections import defaultdict

DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(DIR, 'embeddings')
ALIGN_DIR = os.path.join(DIR, 'alignment_results')
BENCH_DIR = os.path.join(DIR, '..', 'bench_data')
TREES_DIR = os.path.join(BENCH_DIR, 'xlsx_trees')
OUTPUT_DIR = os.path.join(DIR, 'graph')


# ──────────────────────────────────────
# 节点构建
# ──────────────────────────────────────

def build_text_nodes():
    """V_text: 可研报告语义块"""
    with open(os.path.join(EMBED_DIR, 'text_nodes.json')) as f:
        raw = json.load(f)
    nodes = {}
    for n in raw:
        nodes[n['node_id']] = {
            'id': n['node_id'],
            'type': 'text',
            'project_id': n['project_id'],
            'doc_type': n.get('doc_type', ''),
            'section_path': n.get('section_path', ''),
            'title': n.get('title', ''),
            'content': n.get('content', ''),
            'char_count': n.get('char_count', 0),
        }
    return nodes


def build_table_nodes():
    """V_table: 投资表条目"""
    with open(os.path.join(EMBED_DIR, 'table_nodes.json')) as f:
        raw = json.load(f)
    nodes = {}
    for n in raw:
        nodes[n['node_id']] = {
            'id': n['node_id'],
            'type': 'table',
            'project_id': n['project_id'],
            'sheet_name': n.get('sheet_name', ''),
            'item_name': n.get('item_name', ''),
            'item_path': n.get('item_path', ''),
            'depth': n.get('depth', 0),
            'spec': n.get('spec', ''),
            'quantity': n.get('quantity'),
            'unit_price': n.get('unit_price'),
            'original_total': n.get('original_total'),
            'adjusted_total': n.get('adjusted_total'),
            'has_children': n.get('has_children', False),
        }
    return nodes


# ──────────────────────────────────────
# 边构建
# ──────────────────────────────────────

def build_alignment_edges(text_nodes, table_nodes):
    """E_align: 跨模态对齐边 (来自 bench_alignment_dual.json)"""
    with open(os.path.join(ALIGN_DIR, 'bench_alignment_dual.json')) as f:
        alignments = json.load(f)

    edges = []
    # 建立 title→node_id 索引
    text_by_proj_title = defaultdict(dict)
    for nid, n in text_nodes.items():
        text_by_proj_title[n['project_id']][n['title']] = nid

    for a in alignments:
        pid = a['project_id']
        item_name = a['item_name']
        # 找到对应的 table_node
        t_node_id = None
        for tnid, tn in table_nodes.items():
            if tn['project_id'] == pid and tn['item_name'] == item_name:
                t_node_id = tnid
                break

        if not t_node_id:
            continue

        for chunk in a.get('aligned_chunks', []):
            title = chunk.get('title', '')
            txt_node_id = text_by_proj_title.get(pid, {}).get(title)
            if txt_node_id:
                edges.append({
                    'type': 'align',
                    'source': t_node_id,     # table → text
                    'target': txt_node_id,
                    'sim_score': chunk.get('sim_score', 0),
                    'llm_score': chunk.get('llm_score', 0),
                    'combined_score': chunk.get('combined_score', 0),
                })

    return edges


def build_hierarchy_edges(table_nodes):
    """E_hier: 投资表层级边 (父→子)"""
    # 从 xlsx_trees 中提取父子关系
    edges = []
    tree_files = sorted([f for f in os.listdir(TREES_DIR) if f.endswith('.json')])

    for tf in tree_files:
        proj_id = tf.replace('.json', '')
        with open(os.path.join(TREES_DIR, tf)) as f:
            tree_data = json.load(f)

        for sheet in tree_data.get('sheets', []):
            sheet_name = sheet.get('sheet_name', '')
            _build_hier_recursive(sheet.get('items', []),
                                  proj_id, sheet_name, '', table_nodes, edges)

    return edges


def _build_hier_recursive(items, proj_id, sheet_name, parent_path, table_nodes, edges):
    """递归构建层级边"""
    for item in items:
        name = item['name']
        path = f"{parent_path} > {name}" if parent_path else name

        # 找当前节点
        curr_id = _find_table_node(proj_id, sheet_name, name, item.get('seq', ''), table_nodes)

        if curr_id and parent_path:
            # 找父节点
            parent_name = parent_path.split(' > ')[-1]
            parent_id = _find_table_node(proj_id, sheet_name, parent_name, '', table_nodes)
            if parent_id and parent_id != curr_id:
                edges.append({
                    'type': 'hierarchy',
                    'source': parent_id,  # 父 → 子
                    'target': curr_id,
                })

        for child in item.get('children', []):
            _build_hier_recursive([child], proj_id, sheet_name, path, table_nodes, edges)


def _find_table_node(proj_id, sheet_name, item_name, seq, table_nodes):
    """查找对应的 table node ID"""
    for nid, n in table_nodes.items():
        if (n['project_id'] == proj_id and
            n['item_name'] == item_name and
            n['sheet_name'] == sheet_name):
            return nid
    return None


# ──────────────────────────────────────
# 子图召回
# ──────────────────────────────────────

class HeteroGraph:
    """异构图谱 + 子图召回"""

    def __init__(self, text_nodes, table_nodes, align_edges, hier_edges):
        self.text_nodes = text_nodes
        self.table_nodes = table_nodes
        self.align_edges = align_edges
        self.hier_edges = hier_edges

        # 建立邻接表
        self.adj = defaultdict(list)  # node_id → [(edge_type, neighbor_id, edge_data)]
        for e in align_edges:
            self.adj[e['source']].append(('align', e['target'], e))
            self.adj[e['target']].append(('align', e['source'], e))
        for e in hier_edges:
            self.adj[e['source']].append(('child', e['target'], e))
            self.adj[e['target']].append(('parent', e['source'], e))

    def get_node(self, node_id):
        if node_id in self.text_nodes:
            return self.text_nodes[node_id]
        if node_id in self.table_nodes:
            return self.table_nodes[node_id]
        return None

    def subgraph_recall(self, table_node_id, max_hops=2, max_text=5, max_siblings=3):
        """
        子图召回：给定投资表条目，返回局部子图
        包含：条目本身 + 对齐的可研段落 + 父/子条目 + 兄弟条目
        """
        result = {
            'anchor': table_node_id,
            'anchor_node': self.table_nodes.get(table_node_id),
            'text_evidence': [],     # 对齐的可研段落
            'parent_items': [],      # 父条目
            'child_items': [],       # 子条目
            'sibling_items': [],     # 兄弟条目
        }

        if not result['anchor_node']:
            return result

        # 1. 收集对齐的可研段落
        for etype, neighbor, edata in self.adj.get(table_node_id, []):
            if etype == 'align' and neighbor in self.text_nodes:
                result['text_evidence'].append({
                    'node': self.text_nodes[neighbor],
                    'score': edata.get('combined_score', edata.get('sim_score', 0)),
                })

        # 按分数排序，取 top-k
        result['text_evidence'].sort(key=lambda x: -x['score'])
        result['text_evidence'] = result['text_evidence'][:max_text]

        # 2. 收集父/子/兄弟条目
        parent_ids = []
        for etype, neighbor, edata in self.adj.get(table_node_id, []):
            if etype == 'parent' and neighbor in self.table_nodes:
                parent_ids.append(neighbor)
                result['parent_items'].append(self.table_nodes[neighbor])
            elif etype == 'child' and neighbor in self.table_nodes:
                result['child_items'].append(self.table_nodes[neighbor])

        # 兄弟 = 共享父节点的其他子节点
        for pid in parent_ids:
            for etype, neighbor, edata in self.adj.get(pid, []):
                if etype == 'child' and neighbor != table_node_id:
                    if neighbor in self.table_nodes:
                        result['sibling_items'].append(self.table_nodes[neighbor])

        result['sibling_items'] = result['sibling_items'][:max_siblings]

        return result

    def subgraph_to_context(self, subgraph):
        """将子图转为 LLM prompt 中可用的文本上下文"""
        parts = []
        anchor = subgraph['anchor_node']
        if not anchor:
            return ""

        parts.append(f"## 当前审核条目")
        parts.append(f"- 名称: {anchor['item_name']}")
        parts.append(f"- 路径: {anchor['item_path']}")
        parts.append(f"- 费用类别: {anchor['sheet_name']}")
        if anchor.get('original_total') is not None:
            parts.append(f"- 申报金额: {anchor['original_total']}万元")

        if subgraph['text_evidence']:
            parts.append(f"\n## 可研报告相关证据")
            for i, ev in enumerate(subgraph['text_evidence']):
                n = ev['node']
                parts.append(f"\n### 证据{i+1} (相关度: {ev['score']:.2f})")
                if n.get('section_path'):
                    parts.append(f"章节: {n['section_path']}")
                if n.get('title'):
                    parts.append(f"标题: {n['title']}")
                parts.append(f"内容: {n['content'][:300]}")

        if subgraph['parent_items'] or subgraph['sibling_items']:
            parts.append(f"\n## 同类参考条目")
            for p in subgraph['parent_items']:
                parts.append(f"- [上级] {p['item_name']}: {p.get('original_total', '?')}万元")
            for s in subgraph['sibling_items']:
                parts.append(f"- [同级] {s['item_name']}: {s.get('original_total', '?')}万元")

        return '\n'.join(parts)

    def stats(self):
        """图谱统计"""
        return {
            'n_text_nodes': len(self.text_nodes),
            'n_table_nodes': len(self.table_nodes),
            'n_total_nodes': len(self.text_nodes) + len(self.table_nodes),
            'n_align_edges': len(self.align_edges),
            'n_hier_edges': len(self.hier_edges),
            'n_total_edges': len(self.align_edges) + len(self.hier_edges),
        }


# ──────────────────────────────────────
# 主流程
# ──────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 构建节点
    print("构建节点...")
    text_nodes = build_text_nodes()
    table_nodes = build_table_nodes()
    print(f"  V_text: {len(text_nodes)}")
    print(f"  V_table: {len(table_nodes)}")

    # 构建边
    print("\n构建对齐边...")
    align_edges = build_alignment_edges(text_nodes, table_nodes)
    print(f"  E_align: {len(align_edges)}")

    print("\n构建层级边...")
    hier_edges = build_hierarchy_edges(table_nodes)
    print(f"  E_hier: {len(hier_edges)}")

    # 构建图
    graph = HeteroGraph(text_nodes, table_nodes, align_edges, hier_edges)
    s = graph.stats()

    print(f"\n{'='*50}")
    print(f"异构图谱 G = (V, E) 构建完成")
    print(f"  节点: {s['n_total_nodes']} (V_text={s['n_text_nodes']}, V_table={s['n_table_nodes']})")
    print(f"  边: {s['n_total_edges']} (E_align={s['n_align_edges']}, E_hier={s['n_hier_edges']})")

    # 保存图谱
    graph_data = {
        'text_nodes': text_nodes,
        'table_nodes': table_nodes,
        'align_edges': align_edges,
        'hier_edges': hier_edges,
        'stats': s,
    }
    with open(os.path.join(OUTPUT_DIR, 'hetero_graph.json'), 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False)
    print(f"\n  保存: graph/hetero_graph.json")

    # 子图召回示例
    print(f"\n{'='*50}")
    print(f"子图召回示例")
    sample_ids = list(table_nodes.keys())[:3]
    for tid in sample_ids:
        subgraph = graph.subgraph_recall(tid)
        anchor = subgraph['anchor_node']
        print(f"\n  📦 {anchor['item_name'][:30]} ({anchor['project_id']})")
        print(f"     对齐证据: {len(subgraph['text_evidence'])} 条")
        for ev in subgraph['text_evidence'][:2]:
            print(f"       ← {ev['node']['title'][:30]} (score={ev['score']:.3f})")
        print(f"     父条目: {len(subgraph['parent_items'])}")
        print(f"     子条目: {len(subgraph['child_items'])}")
        print(f"     兄弟: {len(subgraph['sibling_items'])}")

    # 输出为 prompt 上下文示例
    print(f"\n{'='*50}")
    print(f"上下文生成示例")
    if sample_ids:
        subgraph = graph.subgraph_recall(sample_ids[0])
        context = graph.subgraph_to_context(subgraph)
        print(context[:500])


if __name__ == '__main__':
    main()
