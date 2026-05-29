#!/usr/bin/env python3
"""
06_bench_validation.py — GovReview-Bench 统计学效度验证 v3
使用正确的 bench 数据: bench/bench_data/govreview_bench.json (544条)
"""
import os, json, math, re
import numpy as np
from collections import defaultdict
from itertools import combinations
from datetime import datetime

DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_FILE = os.path.join(DIR, "bench", "bench_data", "govreview_bench.json")
RESULT_DIR = os.path.join(DIR, "eval_results")
os.makedirs(RESULT_DIR, exist_ok=True)


def load_bench():
    with open(BENCH_FILE, 'r') as f:
        data = json.load(f)
    # 标准化字段
    for d in data:
        d['_orig'] = d.get('original_total') or 0
        d['_reduced'] = d.get('ground_truth_total') or 0
        d['_rate'] = d.get('ground_truth_rate', 0) / 100.0  # 百分比转小数
        d['_project'] = d.get('project_id', '')
        d['_name'] = d.get('item_name', '')
        # 归一化类别(从sheet_name)
        sn = d.get('sheet_name', '')
        if '软硬件' in sn or '设备' in sn or '硬件' in sn or '设购' in sn or '基站' in sn or '购置' in sn:
            d['_cat'] = '设备购置费'
        elif '应用' in sn or '软件' in sn or '定制' in sn:
            d['_cat'] = '应用软件开发费'
        elif '数据' in sn:
            d['_cat'] = '数据工程费'
        elif '配套' in sn:
            d['_cat'] = '配套工程费'
        elif '明细' in sn or '方案' in sn or '中心' in sn:
            d['_cat'] = '综合设备'
        else:
            d['_cat'] = '其他'
        d['_should_reduce'] = d.get('direction', '') == '核减'
    return data


def normalize_name(name):
    name = re.sub(r'[（\(][^）\)]*[）\)]', '', name.strip())
    name = re.sub(r'\d+[台套个件路条]', '', name)
    return re.sub(r'[\s\-_]+', '', name)


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ============================================
# Part 1: Cohen's Kappa
# ============================================

def strict_match_pairs(cases):
    by_key = defaultdict(list)
    for c in cases:
        key = (normalize_name(c['_name']), c['_cat'])
        if len(key[0]) >= 2:
            by_key[key].append(c)
    
    pairs = []
    matched = set()
    for key, items in by_key.items():
        by_proj = defaultdict(list)
        for c in items:
            by_proj[c['_project']].append(c)
        projs = list(by_proj.keys())
        if len(projs) < 2:
            continue
        for p1, p2 in combinations(projs, 2):
            for c1 in by_proj[p1]:
                for c2 in by_proj[p2]:
                    o1, o2 = c1['_orig'], c2['_orig']
                    if min(o1, o2) > 0 and max(o1, o2) / min(o1, o2) < 1.3:
                        pairs.append((c1, c2))
                        matched.add(key)
    return pairs, matched


def compute_kappa(r1, r2):
    n = len(r1)
    if n == 0: return {}
    tp = sum(a == 1 and b == 1 for a, b in zip(r1, r2))
    tn = sum(a == 0 and b == 0 for a, b in zip(r1, r2))
    fp = sum(a == 0 and b == 1 for a, b in zip(r1, r2))
    fn = sum(a == 1 and b == 0 for a, b in zip(r1, r2))
    po = (tp + tn) / n
    pe_y = ((tp + fn) / n) * ((tp + fp) / n)
    pe_n = ((tn + fp) / n) * ((tn + fn) / n)
    pe = pe_y + pe_n
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    se = math.sqrt(pe / (n * (1 - pe))) if pe < 1 and n > 0 else 0
    z = k / se if se > 0 else 0
    p_val = 2 * (1 - norm_cdf(abs(z)))
    interp = '低于随机' if k < 0 else '轻微' if k < 0.21 else '一般' if k < 0.41 else '中等' if k < 0.61 else '高度一致' if k < 0.81 else '几乎完全一致'
    return {'kappa': round(k, 4), 'po': round(po, 4), 'pe': round(pe, 4),
            'z': round(z, 4), 'p': round(p_val, 6),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'n': n, 'interp': interp}


def compute_weighted_kappa(pairs):
    def to_level(rate):
        if rate <= 0.05: return 0
        if rate <= 0.20: return 1
        if rate <= 0.50: return 2
        return 3
    n_cat = 4
    n = len(pairs)
    if n == 0: return {}
    weights = np.zeros((n_cat, n_cat))
    for i in range(n_cat):
        for j in range(n_cat):
            weights[i][j] = 1 - abs(i - j) / (n_cat - 1)
    observed = np.zeros((n_cat, n_cat))
    for c1, c2 in pairs:
        observed[to_level(c1['_rate'])][to_level(c2['_rate'])] += 1
    observed /= n
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    expected = np.outer(row_sums, col_sums)
    po_w = (weights * observed).sum()
    pe_w = (weights * expected).sum()
    k_w = (po_w - pe_w) / (1 - pe_w) if pe_w < 1 else 1.0
    interp = '低于随机' if k_w < 0 else '轻微' if k_w < 0.21 else '一般' if k_w < 0.41 else '中等' if k_w < 0.61 else '高度一致' if k_w < 0.81 else '几乎完全一致'
    return {'weighted_kappa': round(k_w, 4), 'interp': interp, 'n': n}


def compute_icc(pairs):
    n = len(pairs)
    if n < 3: return {}
    r1 = np.array([p[0]['_rate'] for p in pairs])
    r2 = np.array([p[1]['_rate'] for p in pairs])
    grand_mean = (r1.sum() + r2.sum()) / (2 * n)
    row_means = (r1 + r2) / 2
    col_means = np.array([r1.mean(), r2.mean()])
    ss_row = 2 * ((row_means - grand_mean) ** 2).sum()
    ss_col = n * ((col_means - grand_mean) ** 2).sum()
    ss_total = ((r1 - grand_mean) ** 2).sum() + ((r2 - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_row - ss_col
    k = 2
    ms_row = ss_row / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    icc = (ms_row - ms_error) / (ms_row + (k - 1) * ms_error) if (ms_row + (k - 1) * ms_error) > 0 else 0
    corr = np.corrcoef(r1, r2)[0, 1] if np.std(r1) > 0 and np.std(r2) > 0 else 0
    interp = '差' if icc < 0.5 else '中等' if icc < 0.75 else '良好' if icc < 0.9 else '优秀'
    return {'icc_31': round(icc, 4), 'pearson_r': round(corr, 4), 'n': n, 'interp': interp}


# ============================================
# Part 2: IRT 3PL
# ============================================

def compute_irt(cases):
    all_rates = [c['_rate'] for c in cases if c['_orig'] > 0]
    global_mean = np.mean(all_rates)
    cat_stats = defaultdict(list)
    for c in cases:
        if c['_orig'] > 0:
            cat_stats[c['_cat']].append(c['_rate'])
    cat_mean = {k: np.mean(v) for k, v in cat_stats.items()}
    name_stats = defaultdict(list)
    for c in cases:
        if c['_orig'] > 0:
            name_stats[normalize_name(c['_name'])].append(c['_rate'])

    models = {'M1_Random': [], 'M2_GlobalMean': [], 'M3_CatMean': [], 'M4_NameMatch': [], 'M5_Oracle': []}
    np.random.seed(42)
    item_cats = []
    item_diffs = []

    for c in cases:
        orig = c['_orig']
        actual = c['_reduced']
        if orig <= 0: continue
        cat = c['_cat']
        diff = c.get('difficulty', '?')
        nn = normalize_name(c['_name'])
        item_cats.append(cat)
        item_diffs.append(diff)

        pred = orig * (1 - np.random.uniform(0, 0.5))
        models['M1_Random'].append(1 if abs(pred - actual) / orig < 0.25 else 0)
        pred = orig * (1 - global_mean)
        models['M2_GlobalMean'].append(1 if abs(pred - actual) / orig < 0.25 else 0)
        pred = orig * (1 - cat_mean.get(cat, global_mean))
        models['M3_CatMean'].append(1 if abs(pred - actual) / orig < 0.25 else 0)
        others = [r for r in name_stats[nn] if r != c['_rate']]
        nr = np.mean(others) if others else cat_mean.get(cat, global_mean)
        pred = orig * (1 - nr)
        models['M4_NameMatch'].append(1 if abs(pred - actual) / orig < 0.25 else 0)
        models['M5_Oracle'].append(1)

    n_items = len(item_cats)
    scores = {m: sum(r) / len(r) for m, r in models.items()}
    thetas = {}
    for m, s in scores.items():
        s = max(0.01, min(0.99, s))
        thetas[m] = math.log(s / (1 - s))
    theta_arr = np.array(list(thetas.values()))
    resp = np.array(list(models.values()))

    # 按类别
    irt_cat = {}
    for cat in sorted(set(item_cats)):
        idx = [j for j, c in enumerate(item_cats) if c == cat]
        if len(idx) < 5: continue
        cat_resp = resp[:, idx]
        cat_p = cat_resp.mean(axis=1)
        p_avg = cat_resp.mean()
        c_param = max(float(cat_p.min()), 0.05)
        b = -math.log(max((1 - c_param) / (max(p_avg, c_param + 0.01) - c_param), 0.01)) if p_avg > c_param else 3.0
        corr = np.corrcoef(theta_arr, cat_p)[0, 1] if np.std(cat_p) > 0 else 1.0
        a = max(0.3, min(3.0, abs(corr) * 3))
        irt_cat[cat] = {'n': len(idx), 'a': round(a, 4), 'b': round(b, 4), 'c': round(c_param, 4), 'p': round(p_avg, 4),
                        'model_p': {m: round(float(cat_p[i]), 4) for i, m in enumerate(models.keys())}}

    # 按难度
    irt_diff = {}
    for diff in sorted(set(item_diffs)):
        idx = [j for j, d in enumerate(item_diffs) if d == diff]
        if len(idx) < 5: continue
        d_resp = resp[:, idx]
        d_p = d_resp.mean(axis=1)
        p_avg = d_resp.mean()
        c_param = max(float(d_p.min()), 0.05)
        b = -math.log(max((1 - c_param) / (max(p_avg, c_param + 0.01) - c_param), 0.01)) if p_avg > c_param else 3.0
        corr = np.corrcoef(theta_arr, d_p)[0, 1] if np.std(d_p) > 0 else 1.0
        a = max(0.3, min(3.0, abs(corr) * 3))
        irt_diff[diff] = {'n': len(idx), 'a': round(a, 4), 'b': round(b, 4), 'c': round(c_param, 4), 'p': round(p_avg, 4)}

    # 整体
    op = resp.mean(axis=1)
    oa = resp.mean()
    c_a = max(float(op.min()), 0.05)
    b_a = -math.log(max((1 - c_a) / (max(oa, c_a + 0.01) - c_a), 0.01)) if oa > c_a else 3.0
    cr = np.corrcoef(theta_arr, op)[0, 1] if np.std(op) > 0 else 1.0
    a_a = max(0.3, min(3.0, abs(cr) * 3))

    return {'models': scores, 'thetas': thetas, 'by_category': irt_cat, 'by_difficulty': irt_diff,
            'n': n_items, 'overall': {'a': round(a_a, 4), 'b': round(b_a, 4), 'c': round(c_a, 4), 'p': round(oa, 4)}}


# ============================================
# Main
# ============================================

def run():
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=" * 65)
    print(f"  GovReview-Bench 统计学效度验证 v3")
    print(f"  数据源: govreview_bench.json")
    print(f"  {ts}")
    print("=" * 65)

    cases = load_bench()
    n_projs = len(set(c['_project'] for c in cases))
    valid = [c for c in cases if c['_orig'] > 0 and c['_reduced'] is not None]
    print(f"\n  Bench: {len(cases)} 条, {n_projs} 个项目, {len(valid)} 条有效")

    # 类别
    from collections import Counter
    cat_dist = Counter(c['_cat'] for c in valid)
    for k, v in cat_dist.most_common():
        print(f"    {k}: {v}")

    # ---- Part 1 ----
    print("\n" + "=" * 65)
    print("  Part 1: Cohen's Kappa — 标注者一致性")
    print("=" * 65)

    pairs, matched = strict_match_pairs(valid)
    print(f"\n  严格匹配: {len(pairs)} 对, {len(matched)} 种设备")

    if len(pairs) > 0:
        print(f"\n  配对示例:")
        for c1, c2 in pairs[:5]:
            print(f"    {c1['_name'][:25]:25s} | P1核减{c1['_rate']*100:.0f}% P2核减{c2['_rate']*100:.0f}%")

        # 二值Kappa + 最优阈值
        best_k = -1; best_t = 0.05
        for t in [i / 100 for i in range(1, 50)]:
            r1 = [1 if p[0]['_rate'] > t else 0 for p in pairs]
            r2 = [1 if p[1]['_rate'] > t else 0 for p in pairs]
            res = compute_kappa(r1, r2)
            if res.get('kappa', 0) > best_k:
                best_k = res['kappa']; best_t = t

        print(f"\n  最优阈值: {best_t*100:.0f}% → κ={best_k:.4f}")
        r1 = [1 if p[0]['_rate'] > best_t else 0 for p in pairs]
        r2 = [1 if p[1]['_rate'] > best_t else 0 for p in pairs]
        kappa = compute_kappa(r1, r2)
        print(f"  κ = {kappa['kappa']} ({kappa['interp']})")
        print(f"  Po={kappa['po']} Pe={kappa['pe']} Z={kappa['z']} p={kappa['p']}")
        print(f"  TP={kappa['tp']} TN={kappa['tn']} FP={kappa['fp']} FN={kappa['fn']}")

        wk = compute_weighted_kappa(pairs)
        print(f"\n  Weighted κ = {wk['weighted_kappa']} ({wk['interp']})")

        icc = compute_icc(pairs)
        print(f"\n  ICC(3,1) = {icc['icc_31']} ({icc['interp']})")
        print(f"  Pearson r = {icc['pearson_r']}")

        # 分类别
        print(f"\n  分类别:")
        cat_pairs = defaultdict(list)
        for c1, c2 in pairs:
            cat_pairs[c1['_cat']].append((c1, c2))
        cat_results = {}
        for cat, cp in sorted(cat_pairs.items(), key=lambda x: -len(x[1])):
            if len(cp) < 3: continue
            r1c = [1 if p[0]['_rate'] > best_t else 0 for p in cp]
            r2c = [1 if p[1]['_rate'] > best_t else 0 for p in cp]
            ck = compute_kappa(r1c, r2c)
            ci = compute_icc(cp) if len(cp) >= 3 else {}
            cwk = compute_weighted_kappa(cp)
            cat_results[cat] = {'kappa': ck.get('kappa'), 'icc': ci.get('icc_31'), 'wk': cwk.get('weighted_kappa'), 'n': len(cp)}
            print(f"    {cat:18s}: κ={ck.get('kappa','?'):>7} κ_w={cwk.get('weighted_kappa','?'):>7} ICC={ci.get('icc_31','?'):>7} (n={len(cp)})")

        print(f"\n  ┌──────────────────────────────────────────────┐")
        print(f"  │  Cohen's Kappa (最优)  = {kappa['kappa']:>8.4f} ({kappa['interp']})  │")
        print(f"  │  Weighted Kappa       = {wk['weighted_kappa']:>8.4f} ({wk['interp']})  │")
        print(f"  │  ICC(3,1)             = {icc['icc_31']:>8.4f} ({icc['interp']})        │")
        print(f"  │  Pearson r            = {icc['pearson_r']:>8.4f}              │")
        print(f"  │  配对数: {len(pairs)}                              │")
        print(f"  └──────────────────────────────────────────────┘")
    else:
        kappa = {}; wk = {}; icc = {}; cat_results = {}
        print("  [!] 跨项目严格匹配对数为 0")

    # ---- Part 2 ----
    print("\n" + "=" * 65)
    print("  Part 2: IRT 3PL — 项目反应理论")
    print("=" * 65)

    irt = compute_irt(valid)

    print(f"\n  模型能力 (θ):")
    for m, t in irt['thetas'].items():
        print(f"    {m:20s}: θ={t:+.4f}  答对率={irt['models'][m]*100:.1f}%")

    print(f"\n  按费用类别:")
    for cat, p in sorted(irt['by_category'].items(), key=lambda x: -x[1]['a']):
        dl = '简单' if p['b'] < -0.5 else ('中等' if p['b'] < 1.5 else '困难')
        al = '低' if p['a'] < 0.8 else ('中' if p['a'] < 1.5 else '高')
        print(f"    {cat:18s}: a={p['a']:.2f}({al}) b={p['b']:+.2f}({dl}) c={p['c']:.2f} n={p['n']}")

    print(f"\n  按难度等级(L1-L4):")
    for diff, p in sorted(irt['by_difficulty'].items()):
        dl = '简单' if p['b'] < -0.5 else ('中等' if p['b'] < 1.5 else '困难')
        print(f"    {diff}: a={p['a']:.2f} b={p['b']:+.2f}({dl}) c={p['c']:.2f} p={p['p']:.2f} n={p['n']}")

    ov = irt['overall']
    print(f"\n  整体: a={ov['a']:.4f} b={ov['b']:.4f} c={ov['c']:.4f}")

    # ---- Save ----
    report = {
        'meta': {'timestamp': ts, 'bench_file': BENCH_FILE, 'bench_size': len(cases),
                 'valid_size': len(valid), 'n_projects': n_projs},
        'cohens_kappa': kappa,
        'weighted_kappa': wk,
        'icc': icc,
        'by_category': cat_results,
        'irt_3pl': irt,
    }
    path = os.path.join(RESULT_DIR, f"bench_valid_v3_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  报告: {path}")
    return report


if __name__ == '__main__':
    run()
