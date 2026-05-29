#!/usr/bin/env python3
"""
一次性跑完一个模型的 L1 + L2 + L3 三层评测 (T1/T2/T3)
用法: python3 llm_eval_all_layers.py GLM4
      python3 llm_eval_all_layers.py Doubao
"""
import os, sys, json, re, asyncio
import numpy as np
from datetime import datetime
from collections import defaultdict
from openai import AsyncOpenAI

DIR = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(DIR, '..', 'bench', 'bench_data', 'govreview_bench_v2.json')
ALIGN = os.path.join(DIR, '..', 'bench', 'alignment', 'alignment_results', 'bench_alignment_dual.json')
GRAPH = os.path.join(DIR, '..', 'bench', 'alignment', 'graph', 'hetero_graph.json')
OUT = os.path.join(DIR, 'results')

MODELS = {
    'GLM4': {
        'async_client': lambda: AsyncOpenAI(
            api_key='f180f8e7dc5449d78bc14bcae7307954.pF4C5ubuu7IU7d3R',
            base_url='https://open.bigmodel.cn/api/paas/v4/'),
        'model': 'glm-4-plus',
    },
    'Doubao': {
        'async_client': lambda: AsyncOpenAI(
            api_key='0e1e8476-22ab-401d-8c2c-49151d9e1160',
            base_url='https://ark.cn-beijing.volces.com/api/v3/'),
        'model': 'ep-20260321200826-t2vvd',
    },
}


# ══════════════════════════
# 数据加载
# ══════════════════════════

def load_all():
    with open(BENCH) as f: bench = json.load(f)
    with open(ALIGN) as f: align_data = json.load(f)
    with open(GRAPH) as f: graph = json.load(f)

    table_nodes = graph['table_nodes']
    hier_edges = graph['hier_edges']
    parents = defaultdict(list)
    children = defaultdict(list)
    for e in hier_edges:
        parents[e['target']].append(e['source'])
        children[e['source']].append(e['target'])
    table_lookup = {}
    for nid, n in table_nodes.items():
        table_lookup[(n['project_id'], n['item_name'])] = nid

    # 构建三种 evidence map
    l2_full = {}; l2_signal = {}          # L2: 预对齐
    l3_full = {}; l3_signal = {}          # L3: 对齐 + 图谱层级增强

    for a in align_data:
        sid = a['sample_id']; pid = a['project_id']; iname = a['item_name']
        chunks = a.get('aligned_chunks', [])
        if not chunks: continue

        ev_parts = []; total_len = 0
        for c in chunks[:3]:
            t, s, ct = c.get('title',''), c.get('section_path',''), c.get('content','')[:200]
            ev_parts.append(f"[{s}] {t}: {ct}")
            total_len += len(c.get('content',''))

        top = chunks[0]; n_ev = len(chunks[:3])
        avg_sc = sum(c.get('combined_score', c.get('sim_score',0)) for c in chunks[:3])/n_ev
        scope = '复杂(多功能模块)' if total_len>500 else ('中等' if total_len>200 else '简单(描述较少)')

        # L2
        l2_full[sid] = '\n'.join(ev_parts)
        l2_signal[sid] = {'n_evidence':n_ev,'relevance':round(avg_sc,2),'scope':scope,
                          'top_section':top.get('section_path','')}

        # L3 = L2 + hierarchy
        hier_ctx = ""
        tnid = table_lookup.get((pid, iname))
        if tnid:
            pnames = []; snames = []
            for pnid in parents.get(tnid,[]):
                pn = table_nodes.get(pnid)
                if pn: pnames.append(f"{pn['item_name']}({pn.get('original_total','?')}万)")
            for pnid in parents.get(tnid,[]):
                for cnid in children.get(pnid,[]):
                    if cnid != tnid:
                        cn = table_nodes.get(cnid)
                        if cn: snames.append(f"{cn['item_name']}({cn.get('original_total','?')}万)")
            snames = snames[:3]
            if pnames or snames:
                hp = []
                if pnames: hp.append(f"上级条目: {', '.join(pnames[:2])}")
                if snames: hp.append(f"同级条目: {', '.join(snames)}")
                hier_ctx = '\n'.join(hp)

        l3_ev = '\n'.join(ev_parts)
        if hier_ctx: l3_ev += f"\n\n## 投资表层级参考\n{hier_ctx}"
        l3_full[sid] = l3_ev
        l3_signal[sid] = {**l2_signal[sid], 'has_hierarchy': bool(hier_ctx), 'hier_context': hier_ctx}

    return bench, l2_full, l2_signal, l3_full, l3_signal


# ══════════════════════════
# Prompt 构建函数
# ══════════════════════════

def mk_t1_prompt(d, ev_text):
    block = f"\n\n## 可行性研究报告相关证据\n{ev_text[:500]}\n" if ev_text else ""
    return (f"你是一位政务信息化投资审核专家。请根据以下信息判断该设备是否应该被核减。\n\n"
            f"设备名称: {d.get('item_name','')}\n申报价格: {d.get('original_total',0):.2f} 万元\n"
            f"{block}请仅回答'核减'或'不核减'，不要解释。")

def mk_t2_prompt(d, sig_info):
    if sig_info:
        sig = (f"\n审核参考信息:\n- 可研报告依据: 有 ({sig_info['n_evidence']}条, 相关度{sig_info['relevance']})\n"
               f"- 功能范围: {sig_info['scope']}\n- 所属模块: {sig_info['top_section']}\n")
        if sig_info.get('hier_context'): sig += f"- {sig_info['hier_context']}\n"
    else:
        sig = "\n审核参考信息:\n- 可研报告依据: 无\n"
    return (f"你是一位政务信息化投资审核专家。请根据以下信息预测该设备的核减率（百分比）。\n\n"
            f"设备名称: {d.get('item_name','')}\n申报价格: {d.get('original_total',0):.2f} 万元\n"
            f"{sig}\n请仅回答一个数字（0-100之间的百分比），不要解释。例如: 25")

def mk_t3_prompt(d, sig_info):
    if sig_info:
        sig = (f"\n审核参考信息:\n- 可研报告依据: 有 ({sig_info['n_evidence']}条, 相关度{sig_info['relevance']})\n"
               f"- 功能范围: {sig_info['scope']}\n- 所属模块: {sig_info['top_section']}\n")
        if sig_info.get('hier_context'): sig += f"- {sig_info['hier_context']}\n"
    else:
        sig = "\n审核参考信息:\n- 可研报告依据: 无\n"
    return (f"你是一位政务信息化投资审核专家。请根据以下信息预测该设备经过审核后的合理价格。\n\n"
            f"设备名称: {d.get('item_name','')}\n申报价格: {d.get('original_total',0):.2f} 万元\n"
            f"{sig}\n请仅回答一个数字（万元），不要解释。例如: 85.50")


# ══════════════════════════
# 解析 & 指标
# ══════════════════════════

def parse_t1(r):
    if 'ERROR' in r: return None
    if '不核减' in r or '不需要' in r or '保留' in r: return 0
    if '核减' in r: return 1
    return None

def parse_num(r, cap=None):
    if 'ERROR' in r: return None
    nums = re.findall(r'[\d.]+', r)
    if nums:
        v = float(nums[0])
        return min(v, cap) if cap else v
    return None

def calc_clf(yt, yp):
    valid = [(t,p) for t,p in zip(yt,yp) if p is not None]
    if not valid: return {}
    yt2,yp2 = zip(*valid)
    acc = sum(t==p for t,p in zip(yt2,yp2))/len(yt2)
    f1s = []
    for c in [0,1]:
        tp=sum(t==c and p==c for t,p in zip(yt2,yp2))
        fp=sum(t!=c and p==c for t,p in zip(yt2,yp2))
        fn=sum(t==c and p!=c for t,p in zip(yt2,yp2))
        pr=tp/(tp+fp) if tp+fp>0 else 0; rc=tp/(tp+fn) if tp+fn>0 else 0
        f1s.append(2*pr*rc/(pr+rc) if pr+rc>0 else 0)
    return {'n':len(yt2),'Accuracy':round(acc,4),'Macro-F1':round(sum(f1s)/len(f1s),4)}

def calc_reg(yt, yp, yo=None):
    valid = [(t,p,o) for t,p,o in zip(yt,yp,yo or yt) if p is not None]
    if not valid: return {}
    yt2,yp2,yo2 = zip(*valid)
    yt2,yp2,yo2 = np.array(yt2),np.array(yp2),np.array(yo2)
    mae = np.mean(np.abs(yp2-yt2))
    mre = np.abs(yp2-yt2)/np.maximum(yo2,0.01)
    return {'n':len(yt2),'MAE':round(float(mae),4),'PRED25':round(float(np.mean(mre<=0.25)),4)}


# ══════════════════════════
# 异步执行
# ══════════════════════════

async def run_batch(client, model, data, prompt_fn, parse_fn, concurrency=8, max_tokens=100):
    sem = asyncio.Semaphore(concurrency)
    preds = [None]*len(data)
    done = [0]
    async def one(i,d,ctx):
        async with sem:
            prompt = prompt_fn(d, ctx)
            for attempt in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=model, messages=[{'role':'user','content':prompt}],
                        temperature=0.1, max_tokens=max_tokens)
                    preds[i] = parse_fn(r.choices[0].message.content.strip())
                    done[0]+=1
                    if done[0]%200==0: print(f'      [{done[0]}/{len(data)}]')
                    return
                except: await asyncio.sleep(2**(attempt+1))
            done[0]+=1
    return preds, one


async def eval_one_layer(client, model, bench, layer_name,
                         t1_ctx, t2_ctx, t3_ctx):
    """评测一层 (T1/T2/T3)"""
    results = {}

    # T1
    print(f'  [{layer_name} T1]...')
    preds, one = await run_batch(client, model, bench, mk_t1_prompt, parse_t1)
    tasks = [one(i,d,t1_ctx.get(d.get('sample_id',''),'')) for i,d in enumerate(bench)]
    await asyncio.gather(*tasks)
    labels = [1 if d.get('direction')=='核减' else 0 for d in bench]
    results[f'T1_{layer_name}'] = calc_clf(labels, preds)
    print(f'    → {results[f"T1_{layer_name}"]}')

    # T2
    print(f'  [{layer_name} T2]...')
    preds2 = [None]*len(bench); done=[0]
    sem = asyncio.Semaphore(8)
    async def t2_one(i,d):
        async with sem:
            ctx = t2_ctx.get(d.get('sample_id',''))
            prompt = mk_t2_prompt(d, ctx)
            for attempt in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=model, messages=[{'role':'user','content':prompt}],
                        temperature=0.1, max_tokens=50)
                    preds2[i] = parse_num(r.choices[0].message.content.strip(), 100)
                    done[0]+=1; 
                    if done[0]%200==0: print(f'      [{done[0]}/{len(bench)}]')
                    return
                except: await asyncio.sleep(2**(attempt+1))
            done[0]+=1
    await asyncio.gather(*[t2_one(i,d) for i,d in enumerate(bench)])
    labels2 = [d.get('ground_truth_rate',0) or 0 for d in bench]
    results[f'T2_{layer_name}'] = calc_reg(labels2, preds2, labels2)
    print(f'    → {results[f"T2_{layer_name}"]}')

    # T3
    valid_t3 = [d for d in bench if (d.get('original_total') or 0)>0 and d.get('ground_truth_total') is not None]
    print(f'  [{layer_name} T3] ({len(valid_t3)})...')
    preds3 = [None]*len(valid_t3); done[0]=0
    async def t3_one(i,d):
        async with sem:
            ctx = t3_ctx.get(d.get('sample_id',''))
            prompt = mk_t3_prompt(d, ctx)
            for attempt in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=model, messages=[{'role':'user','content':prompt}],
                        temperature=0.1, max_tokens=50)
                    preds3[i] = parse_num(r.choices[0].message.content.strip())
                    done[0]+=1
                    if done[0]%200==0: print(f'      [{done[0]}/{len(valid_t3)}]')
                    return
                except: await asyncio.sleep(2**(attempt+1))
            done[0]+=1
    await asyncio.gather(*[t3_one(i,d) for i,d in enumerate(valid_t3)])
    labels3 = [d['ground_truth_total'] for d in valid_t3]
    orig3 = [d['original_total'] for d in valid_t3]
    results[f'T3_{layer_name}'] = calc_reg(labels3, preds3, orig3)
    print(f'    → {results[f"T3_{layer_name}"]}')

    return results


async def main():
    model_name = sys.argv[1] if len(sys.argv)>1 else 'GLM4'
    if model_name not in MODELS:
        print(f'Unknown: {model_name}'); return

    cfg = MODELS[model_name]
    client = cfg['async_client']()
    model = cfg['model']
    bench, l2_full, l2_signal, l3_full, l3_signal = load_all()

    print('='*60)
    print(f'  {model_name} 三层评测 (L1+L2+L3) | {len(bench)} samples')
    print('='*60)

    all_results = {}

    # L1: 无证据
    empty_full = {}; empty_sig = {}
    r = await eval_one_layer(client, model, bench, 'L1',
                              empty_full, empty_sig, empty_sig)
    all_results.update(r)

    # L2: 预对齐证据
    r = await eval_one_layer(client, model, bench, 'L2',
                              l2_full, l2_signal, l2_signal)
    all_results.update(r)

    # L3: GraphRAG 检索 + 层级增强
    r = await eval_one_layer(client, model, bench, 'L3',
                              l3_full, l3_signal, l3_signal)
    all_results.update(r)

    # 保存
    output = {'model':model_name, 'model_id':model, 'n':len(bench),
              'timestamp':str(datetime.now()), 'results':all_results}
    fname = f'all_layers_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(os.path.join(OUT, fname), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 打印对比
    print(f'\n{"="*60}')
    print(f'{model_name} 三层评测汇总:')
    print(f'{"任务":>6}  {"指标":>8}  {"L1":>8}  {"L2":>8}  {"L3":>8}  {"L1→L2":>8}  {"L1→L3":>8}')
    print('-'*60)
    t1l1=all_results.get('T1_L1',{}); t1l2=all_results.get('T1_L2',{}); t1l3=all_results.get('T1_L3',{})
    f1=(t1l1.get('Macro-F1',0),t1l2.get('Macro-F1',0),t1l3.get('Macro-F1',0))
    print(f'  T1    F1      {f1[0]:>8.4f}  {f1[1]:>8.4f}  {f1[2]:>8.4f}  {f1[1]-f1[0]:>+7.4f}  {f1[2]-f1[0]:>+7.4f}')
    
    t2l1=all_results.get('T2_L1',{}); t2l2=all_results.get('T2_L2',{}); t2l3=all_results.get('T2_L3',{})
    m=(t2l1.get('MAE',0),t2l2.get('MAE',0),t2l3.get('MAE',0))
    print(f'  T2    MAE     {m[0]:>8.4f}  {m[1]:>8.4f}  {m[2]:>8.4f}  {m[1]-m[0]:>+7.4f}  {m[2]-m[0]:>+7.4f}')
    
    t3l1=all_results.get('T3_L1',{}); t3l2=all_results.get('T3_L2',{}); t3l3=all_results.get('T3_L3',{})
    p=(t3l1.get('PRED25',0),t3l2.get('PRED25',0),t3l3.get('PRED25',0))
    print(f'  T3    PRED25  {p[0]:>8.4f}  {p[1]:>8.4f}  {p[2]:>8.4f}  {p[1]-p[0]:>+7.4f}  {p[2]-p[0]:>+7.4f}')

if __name__ == '__main__':
    asyncio.run(main())
