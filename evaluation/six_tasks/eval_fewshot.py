#!/usr/bin/env python3
"""
GovReview-Bench 3-Shot 评测 (v2 增量版)
- 基于 govreview_bench_v2.json (802条)
- 仅对 v2 中新增的 258 条发 API 请求
- 旧 544 条中有 per-sample 预测记录的直接复用 (若无则重跑)
- 评测 L1 (zero-knowledge) 条件下的 0-Shot vs 3-Shot 对比

用法: python3 llm_eval_fewshot_v2.py Qwen
      python3 llm_eval_fewshot_v2.py DeepSeek
      python3 llm_eval_fewshot_v2.py Doubao
      python3 llm_eval_fewshot_v2.py GLM4
"""
import os, sys, json, re, asyncio
import numpy as np
from datetime import datetime
from openai import AsyncOpenAI

DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_V2  = os.path.join(DIR, '..', 'bench', 'bench_data', 'govreview_bench_v2.json')
BENCH_V1  = os.path.join(DIR, '..', 'bench', 'bench_data', 'govreview_bench.json')
OUT       = os.path.join(DIR, 'results')

# ── 3-Shot 示例（从v1中挑选的代表性样本，固定不变）──
FEW_SHOT_EXAMPLES = [
    {
        'item_name': 'Web网页服务器',
        'original_total': 15.0,
        'direction': '不核减',
        'ground_truth_rate': 0.0,
        'ground_truth_total': 15.0,
        'reason': '价格合理，符合市场行情'
    },
    {
        'item_name': '数据库软件授权',
        'original_total': 120.0,
        'direction': '核减',
        'ground_truth_rate': 25.0,
        'ground_truth_total': 90.0,
        'reason': '申报价格偏高，按市场价核减'
    },
    {
        'item_name': '防火墙设备',
        'original_total': 45.0,
        'direction': '核减',
        'ground_truth_rate': 15.0,
        'ground_truth_total': 38.25,
        'reason': '同类设备可研报告参考价格较低'
    },
]


MODELS = {
    'Qwen': {
        'async_client': lambda: AsyncOpenAI(
            api_key='sk-029357b2dcc14b14a78d6a5532416c3d',
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'),
        'model': 'qwen-plus',
    },
    'DeepSeek': {
        'async_client': lambda: AsyncOpenAI(
            api_key='sk-029357b2dcc14b14a78d6a5532416c3d',
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'),
        'model': 'deepseek-v3',
    },
    'Doubao': {
        'async_client': lambda: AsyncOpenAI(
            api_key='0e1e8476-22ab-401d-8c2c-49151d9e1160',
            base_url='https://ark.cn-beijing.volces.com/api/v3/'),
        'model': 'ep-20260321200826-t2vvd',
    },
    'GLM4': {
        'async_client': lambda: AsyncOpenAI(
            api_key='f180f8e7dc5449d78bc14bcae7307954.pF4C5ubuu7IU7d3R',
            base_url='https://open.bigmodel.cn/api/paas/v4/'),
        'model': 'glm-4-plus',
    },
}


# ── 构建 Few-Shot 前缀 ──
def make_few_shot_prefix_t1():
    lines = ["以下是几个审核案例供参考:\n"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"案例{i}: 设备名称={ex['item_name']}, 申报价格={ex['original_total']}万元 → 答案: {ex['direction']}")
    lines.append("\n现在请判断以下案例:")
    return '\n'.join(lines)

def make_few_shot_prefix_t2():
    lines = ["以下是几个审核案例供参考:\n"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        r = ex['ground_truth_rate']
        lines.append(f"案例{i}: 设备名称={ex['item_name']}, 申报价格={ex['original_total']}万元 → 核减率: {r:.0f}%")
    lines.append("\n现在请预测以下案例的核减率:")
    return '\n'.join(lines)

def make_few_shot_prefix_t3():
    lines = ["以下是几个审核案例供参考:\n"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        t = ex['ground_truth_total']
        lines.append(f"案例{i}: 设备名称={ex['item_name']}, 申报价格={ex['original_total']}万元 → 审核后价格: {t:.2f}万元")
    lines.append("\n现在请预测以下案例审核后的合理价格:")
    return '\n'.join(lines)


# ── Prompts (0-Shot) ──
def mk_t1_0shot(d):
    return (f"你是一位政务信息化投资审核专家。请根据以下信息判断该设备是否应该被核减。\n\n"
            f"设备名称: {d.get('item_name','')}\n申报价格: {d.get('original_total',0):.2f} 万元\n"
            f"\n请仅回答'核减'或'不核减'，不要解释。")

def mk_t2_0shot(d):
    return (f"你是一位政务信息化投资审核专家。请根据以下信息预测该设备的核减率（百分比）。\n\n"
            f"设备名称: {d.get('item_name','')}\n申报价格: {d.get('original_total',0):.2f} 万元\n"
            f"\n请仅回答一个数字（0-100之间的百分比），不要解释。例如: 25")

def mk_t3_0shot(d):
    return (f"你是一位政务信息化投资审核专家。请根据以下信息预测该设备经过审核后的合理价格。\n\n"
            f"设备名称: {d.get('item_name','')}\n申报价格: {d.get('original_total',0):.2f} 万元\n"
            f"\n请仅回答一个数字（万元），不要解释。例如: 85.50")

# ── Prompts (3-Shot) ──
def mk_t1_3shot(d):
    prefix = make_few_shot_prefix_t1()
    return (f"你是一位政务信息化投资审核专家。\n{prefix}\n\n"
            f"设备名称: {d.get('item_name','')}\n申报价格: {d.get('original_total',0):.2f} 万元\n"
            f"\n请仅回答'核减'或'不核减'，不要解释。")

def mk_t2_3shot(d):
    prefix = make_few_shot_prefix_t2()
    return (f"你是一位政务信息化投资审核专家。\n{prefix}\n\n"
            f"设备名称: {d.get('item_name','')}\n申报价格: {d.get('original_total',0):.2f} 万元\n"
            f"\n请仅回答一个数字（0-100之间的百分比），不要解释。例如: 25")

def mk_t3_3shot(d):
    prefix = make_few_shot_prefix_t3()
    return (f"你是一位政务信息化投资审核专家。\n{prefix}\n\n"
            f"设备名称: {d.get('item_name','')}\n申报价格: {d.get('original_total',0):.2f} 万元\n"
            f"\n请仅回答一个数字（万元），不要解释。例如: 85.50")


# ── Parsers ──
def parse_t1(r):
    if not r or 'ERROR' in r: return None
    if '不核减' in r or '不需要' in r or '保留' in r: return 0
    if '核减' in r: return 1
    return None

def parse_num(r, cap=None):
    if not r or 'ERROR' in r: return None
    nums = re.findall(r'[\d.]+', r)
    if nums:
        v = float(nums[0])
        return min(v, cap) if cap else v
    return None


# ── Metrics ──
def calc_clf(yt, yp):
    valid = [(t, p) for t, p in zip(yt, yp) if p is not None]
    if not valid: return {}
    yt2, yp2 = zip(*valid)
    acc = sum(t == p for t, p in zip(yt2, yp2)) / len(yt2)
    f1s = []
    for c in [0, 1]:
        tp = sum(t == c and p == c for t, p in zip(yt2, yp2))
        fp = sum(t != c and p == c for t, p in zip(yt2, yp2))
        fn = sum(t == c and p != c for t, p in zip(yt2, yp2))
        pr = tp / (tp + fp) if tp + fp > 0 else 0
        rc = tp / (tp + fn) if tp + fn > 0 else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc > 0 else 0)
    return {'n': len(yt2), 'Accuracy': round(acc, 4), 'Macro-F1': round(sum(f1s) / 2, 4)}

def calc_reg(yt, yp, yo=None):
    valid = [(t, p, o) for t, p, o in zip(yt, yp, yo or yt) if p is not None]
    if not valid: return {}
    yt2, yp2, yo2 = zip(*valid)
    yt2, yp2, yo2 = np.array(yt2), np.array(yp2), np.array(yo2)
    mae = np.mean(np.abs(yp2 - yt2))
    mre = np.abs(yp2 - yt2) / np.maximum(yo2, 0.01)
    return {
        'n': len(yt2), 'MAE': round(float(mae), 4),
        'PRED25': round(float(np.mean(mre <= 0.25)), 4),
        'pred_mean': round(float(np.mean(yp2)), 2),
        'gt_mean': round(float(np.mean(yt2)), 2),
    }


# ── 增量 Cache（逐条存，支持断点续跑）──
def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}

def save_cache(cache_path, cache):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── Async Eval ──
async def run_task(client, model, bench_items, prompt_fn, parse_fn, task_label,
                   cache, cache_key, cache_path, concurrency=8, cap=None):
    sem = asyncio.Semaphore(concurrency)
    done = [0]
    preds = {}

    # 先从 cache 加载已有结果
    for d in bench_items:
        sid = d.get('sample_id', '')
        k = f"{cache_key}_{sid}"
        if k in cache:
            preds[sid] = cache[k]

    # 只跑没 cache 的
    todo = [d for d in bench_items if f"{cache_key}_{d.get('sample_id','')}" not in cache]
    print(f"  [{task_label}] cache命中={len(preds)}, 需API调用={len(todo)}")

    async def call_one(d):
        sid = d.get('sample_id', '')
        prompt = prompt_fn(d)
        async with sem:
            for attempt in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=model,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=0.1, max_tokens=50)
                    val = parse_fn(r.choices[0].message.content.strip(), cap) if cap else parse_fn(r.choices[0].message.content.strip())
                    preds[sid] = val
                    cache[f"{cache_key}_{sid}"] = val
                    done[0] += 1
                    if done[0] % 100 == 0:
                        save_cache(cache_path, cache)
                        print(f"    [{done[0]}/{len(todo)}] cache已保存")
                    return
                except Exception as e:
                    await asyncio.sleep(2 ** (attempt + 1))
            preds[sid] = None
            cache[f"{cache_key}_{sid}"] = None
            done[0] += 1

    await asyncio.gather(*[call_one(d) for d in todo])
    save_cache(cache_path, cache)

    # 按 bench_items 顺序返回
    return [preds.get(d.get('sample_id', '')) for d in bench_items]


async def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'Qwen'
    if model_name not in MODELS:
        print(f"未知模型: {model_name}, 可选: {list(MODELS.keys())}")
        sys.exit(1)

    cfg = MODELS[model_name]
    client = cfg['async_client']()
    model  = cfg['model']

    # 加载 v2 bench
    with open(BENCH_V2) as f:
        bench = json.load(f)

    # 加载 v1 bench 的 sample_ids（用于标记哪些是旧样本）
    with open(BENCH_V1) as f:
        v1 = json.load(f)
    v1_ids = set(d.get('sample_id', '') for d in v1)
    new_ids = set(d.get('sample_id', '') for d in bench) - v1_ids

    print('=' * 60)
    print(f'  Few-Shot v2 评测 | {model_name} | {len(bench)} samples')
    print(f'  v1中已有: {len(v1_ids)} unique IDs')
    print(f'  v2新增:   {len(new_ids)} unique IDs')
    print('=' * 60)

    # Cache 文件路径
    cache_path = os.path.join(OUT, f'fewshot_v2_cache_{model_name}.json')
    cache = load_cache(cache_path)

    all_results = {}

    for shot_label, t1_fn, t2_fn, t3_fn in [
        ('0shot', mk_t1_0shot, mk_t2_0shot, mk_t3_0shot),
        ('3shot', mk_t1_3shot, mk_t2_3shot, mk_t3_3shot),
    ]:
        print(f"\n{'─'*40}")
        print(f"  模式: {shot_label.upper()}")
        print(f"{'─'*40}")

        # T1
        preds_t1 = await run_task(
            client, model, bench,
            t1_fn, parse_t1, f"T1 {shot_label}",
            cache, f"{model_name}_{shot_label}_t1", cache_path
        )
        labels_t1 = [1 if d.get('direction') == '核减' else 0 for d in bench]
        r_t1 = calc_clf(labels_t1, preds_t1)
        all_results[f'T1_{shot_label}'] = r_t1
        print(f"    → T1 Macro-F1={r_t1.get('Macro-F1')}")

        # T2
        preds_t2 = await run_task(
            client, model, bench,
            t2_fn, lambda r, cap=100: parse_num(r, cap), f"T2 {shot_label}",
            cache, f"{model_name}_{shot_label}_t2", cache_path
        )
        labels_t2 = [d.get('ground_truth_rate', 0) or 0 for d in bench]
        r_t2 = calc_reg(labels_t2, preds_t2, labels_t2)
        all_results[f'T2_{shot_label}'] = r_t2
        print(f"    → T2 MAE={r_t2.get('MAE')}")

        # T3
        valid_t3 = [d for d in bench if (d.get('original_total') or 0) > 0 and d.get('ground_truth_total') is not None]
        preds_t3 = await run_task(
            client, model, valid_t3,
            t3_fn, parse_num, f"T3 {shot_label}",
            cache, f"{model_name}_{shot_label}_t3", cache_path
        )
        labels_t3 = [d['ground_truth_total'] for d in valid_t3]
        orig_t3   = [d['original_total'] for d in valid_t3]
        r_t3 = calc_reg(labels_t3, preds_t3, orig_t3)
        all_results[f'T3_{shot_label}'] = r_t3
        print(f"    → T3 PRED25={r_t3.get('PRED25')}, MAE={r_t3.get('MAE')}")

    # 保存结果
    output = {
        'model': model_name, 'model_id': model, 'n': len(bench),
        'timestamp': str(datetime.now()),
        'note': '3-Shot增量评测，基于govreview_bench_v2(802条)',
        'results': all_results
    }
    fname = f'fewshot_v2_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    out_path = os.path.join(OUT, fname)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {fname}")

    # 打印对比表
    print(f'\n{"="*60}')
    print(f'{model_name} 0-Shot vs 3-Shot 对比:')
    print(f'{"任务":<6} {"指标":<10} {"0-Shot":>8} {"3-Shot":>8} {"Delta":>8}')
    print('-' * 44)
    for task, metric, _ in [('T1', 'Macro-F1', True), ('T2', 'MAE', False), ('T3', 'PRED25', True)]:
        v0 = all_results.get(f'{task}_0shot', {}).get(metric, 0)
        v3 = all_results.get(f'{task}_3shot', {}).get(metric, 0)
        delta = v3 - v0
        sign = '+' if delta >= 0 else ''
        print(f'{task:<6} {metric:<10} {v0:>8.4f} {v3:>8.4f} {sign}{delta:>7.4f}')


if __name__ == '__main__':
    asyncio.run(main())
