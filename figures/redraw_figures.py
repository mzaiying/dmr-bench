#!/usr/bin/env python3
"""
重绘所有实验结果图 fig04~fig14，统一使用更大更粗的字体。
字体设置: 轴标签 16pt bold, 刻度 14pt, 图例 14pt, 标注文字 13pt
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── 全局字体设置 ──────────────────────────────────────────────
plt.rcParams.update({
    'font.sans-serif': ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.dpi': 200,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

OUTDIR = "/Volumes/Elements SE/科研/软件学报-软件投资审核/核减Agent/paper_workspace/GovReview-Bench 5/figures"

MODELS = ['DeepSeek', 'Qwen', 'GLM4', 'Doubao']
LAYERS = ['L1\n(零知识)', 'L_BM25\n(稀疏)', 'L_Dense\n(向量)', 'L2\n(上界)', 'L3']

HATCHES = ['////', '', '....', 'xxxx', '----']
COLORS  = ['#5B9BD5', '#A8D08D', '#FFB347', '#C0C0C0', '#E07070']

# ── fig04: T1 Macro-F1 分组柱状图 ───────────────────────────────
t1_data = {
    'DeepSeek': [0.401, 0.412, 0.435, 0.419, 0.415],
    'Qwen':     [0.364, 0.480, 0.475, 0.432, 0.440],
    'GLM4':     [0.479, 0.399, 0.379, 0.437, 0.363],
    'Doubao':   [0.544, 0.525, 0.539, 0.521, 0.518],
}
layer_labels = ['L1(零知识)', 'L_BM25(稀疏检索)', 'L_Dense(向量检索)', 'L2(标注证据上界)', 'L3(双路图谱增强)']

def grouped_bar(data, ylabel, title_fname, ylim=None, baseline=None, baseline_label=None):
    models = list(data.keys())
    n_layers = len(next(iter(data.values())))
    x = np.arange(len(models))
    width = 0.14
    offsets = np.linspace(-(n_layers-1)/2, (n_layers-1)/2, n_layers) * width

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (layer, hatch, color) in enumerate(zip(layer_labels, HATCHES, COLORS)):
        vals = [data[m][i] for m in models]
        bars = ax.bar(x + offsets[i], vals, width, label=layer,
                      hatch=hatch, color=color, edgecolor='#444', linewidth=0.8, alpha=0.88)

    if baseline is not None:
        ax.axhline(baseline, color='#cc3333', linewidth=1.8, linestyle='--',
                   label=baseline_label or f'基线={baseline:.3f}')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=15, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9,
              ncol=2, columnspacing=0.8, handlelength=1.5)
    if ylim:
        ax.set_ylim(ylim)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTDIR, f'{title_fname}.{ext}'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'✅ {title_fname}')

grouped_bar(t1_data, 'Macro-F1 Score', 'fig04_t1_macroF1', ylim=(0.30, 0.67), baseline=0.575, baseline_label='统计基线(0.575)')

# ── fig05: T2 MAE 分组柱状图 ─────────────────────────────────
t2_data = {
    'DeepSeek': [25.81, 21.36, 28.54, 24.17, 27.62],
    'Qwen':     [79.22, 22.51, 27.32, 35.66, 27.75],
    'GLM4':     [79.09, 21.78, 21.96, 35.25, 26.07],
    'Doubao':   [72.69, 22.48, 21.36, 34.52, 22.86],
}
grouped_bar(t2_data, 'MAE（越低越好）', 'fig05_t2_MAE', ylim=(0, 90), baseline=21.89, baseline_label='统计基线MAE=21.89')

# ── fig06: T3 PRED25 分组柱状图 ──────────────────────────────
t3_data = {
    'DeepSeek': [0.687, 0.765, 0.603, 0.763, 0.638],
    'Qwen':     [0.715, 0.777, 0.786, 0.764, 0.777],
    'GLM4':     [0.214, 0.709, 0.707, 0.614, 0.672],
    'Doubao':   [0.504, 0.711, 0.755, 0.709, 0.726],
}
grouped_bar(t3_data, 'PRED25（越高越好）', 'fig06_t3_PRED25', ylim=(0.10, 0.90), baseline=0.832, baseline_label='统计基线(0.832)')

# ── fig07: 检索质量评测 横向柱状图 ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
methods = ['BM25\n(关键词检索)', 'Dense\n(向量语义检索)', 'HTG-Align\n(双路图谱检索)']
p1  = [0.330, 0.196, 0.278]
r3  = [0.543, 0.477, 0.525]
nd3 = [0.456, 0.369, 0.429]

x = np.arange(len(methods))
w = 0.25
ax.bar(x - w, p1,  w, label='P@1',     color='#5B9BD5', edgecolor='#333', hatch='////', alpha=0.9)
ax.bar(x,     r3,  w, label='Recall@3', color='#A8D08D', edgecolor='#333', hatch='....', alpha=0.9)
ax.bar(x + w, nd3, w, label='NDCG@3',  color='#FFB347', edgecolor='#333', hatch='xxxx', alpha=0.9)

for i, (a, b, c) in enumerate(zip(p1, r3, nd3)):
    ax.text(i - w, a + 0.008, f'{a:.3f}', ha='center', fontsize=12, fontweight='bold')
    ax.text(i,     b + 0.008, f'{b:.3f}', ha='center', fontsize=12, fontweight='bold')
    ax.text(i + w, c + 0.008, f'{c:.3f}', ha='center', fontsize=12, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=14, fontweight='bold')
ax.set_ylabel('检索质量指标', fontsize=16, fontweight='bold')
ax.set_ylim(0, 0.65)
ax.legend(fontsize=13)
fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(os.path.join(OUTDIR, f'fig07_retrieval_quality.{ext}'), bbox_inches='tight', dpi=200)
plt.close(fig)
print('✅ fig07_retrieval_quality')

# ── fig08: ML vs LLM T2对比 ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
methods_ml = ['均值盲预测\n(StatBaseline)', 'Ridge', 'Random\nForest', 'GBM\n(梯度提升树)']
methods_llm = ['DeepSeek\n+BM25', 'Qwen\n+BM25', 'GLM4\n+BM25', 'Doubao\n+Dense']
mae_ml  = [21.89, 18.84, 11.51, 11.07]
mae_llm = [21.36, 22.51, 21.78, 21.36]

x1 = np.arange(len(methods_ml))
x2 = np.arange(len(methods_ml), len(methods_ml)+len(methods_llm))
bars1 = ax.bar(x1, mae_ml,  0.6, color='#5B9BD5', edgecolor='#333', hatch='////', alpha=0.9, label='传统ML方法')
bars2 = ax.bar(x2, mae_llm, 0.6, color='#E07070', edgecolor='#333', hatch='....', alpha=0.9, label='LLM+RAG方法')

for bar in list(bars1)+list(bars2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{bar.get_height():.2f}', ha='center', fontsize=12, fontweight='bold')

ax.set_xticks(list(x1)+list(x2))
ax.set_xticklabels(methods_ml+methods_llm, fontsize=12, fontweight='bold')
ax.set_ylabel('MAE（越低越好）', fontsize=16, fontweight='bold')
ax.axvline(3.5, color='gray', linestyle='--', linewidth=1.5)
ax.legend(fontsize=13)
ax.set_ylim(0, 30)
fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(os.path.join(OUTDIR, f'fig08_ml_vs_llm.{ext}'), bbox_inches='tight', dpi=200)
plt.close(fig)
print('✅ fig08_ml_vs_llm')

# ── fig09: 认知解耦雷达图 / 散点图 ──────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
t1_best = [0.401, 0.480, 0.479, 0.544]
t2_best_mae = [21.36, 22.51, 21.78, 21.36]
# Normalize T2 (lower is better → invert for comparison)
t2_norm = [1 - (m - 11) / (80 - 11) for m in t2_best_mae]

colors = ['#5B9BD5','#A8D08D','#FFB347','#E07070']
for i, (t1, t2, m) in enumerate(zip(t1_best, t2_norm, MODELS)):
    ax.scatter(t1, t2, s=180, color=colors[i], edgecolors='#333', zorder=5, linewidth=1.5)
    ax.annotate(m, (t1, t2), textcoords='offset points', xytext=(8, 4),
                fontsize=13, fontweight='bold', color=colors[i])

ax.set_xlabel('T1 最优 Macro-F1（定性判断，越高越好）', fontsize=14, fontweight='bold')
ax.set_ylabel('T2 归一化得分（定量估算，越高越好）', fontsize=14, fontweight='bold')
ax.set_title('定性-定量能力解耦现象', fontsize=16, fontweight='bold', pad=10)
ax.axhline(np.mean(t2_norm), color='gray', linestyle=':', alpha=0.6)
ax.axvline(np.mean(t1_best), color='gray', linestyle=':', alpha=0.6)
ax.text(0.98, 0.02, '高定性\n低定量', transform=ax.transAxes, ha='right', va='bottom',
        fontsize=12, color='gray', style='italic')
fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(os.path.join(OUTDIR, f'fig09_decoupling.{ext}'), bbox_inches='tight', dpi=200)
plt.close(fig)
print('✅ fig09_decoupling')

# ── fig10: Oracle悖论 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
configs = ['L1\n(零知识)', 'L_BM25', 'L_Dense', 'L2\n(Oracle上界)', 'L3']
ds_mae   = [25.81, 21.36, 28.54, 24.17, 27.62]
qwen_mae = [79.22, 22.51, 27.32, 35.66, 27.75]
glm_mae  = [79.09, 21.78, 21.96, 35.25, 26.07]
dou_mae  = [72.69, 22.48, 21.36, 34.52, 22.86]

x = np.arange(5)
for i, (mae, model, color, hatch) in enumerate(
        zip([ds_mae, qwen_mae, glm_mae, dou_mae], MODELS, COLORS, HATCHES)):
    ax.plot(x, mae, marker='o', markersize=9, linewidth=2.2, label=model,
            color=color, markeredgecolor='#333', markeredgewidth=0.8)

ax.axvspan(3-0.3, 3+0.3, alpha=0.12, color='red', zorder=0)
ax.text(3, 5, 'Oracle↑', ha='center', fontsize=13, color='#cc3333', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=13, fontweight='bold')
ax.set_ylabel('MAE（越低越好）', fontsize=16, fontweight='bold')
ax.set_title('Oracle悖论：完整证据下性能反而下降', fontsize=15, fontweight='bold', pad=8)
ax.legend(fontsize=13)
ax.set_ylim(0, 90)
fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(os.path.join(OUTDIR, f'fig10_oracle_paradox.{ext}'), bbox_inches='tight', dpi=200)
plt.close(fig)
print('✅ fig10_oracle_paradox')

# ── fig11: 词汇封闭度 折线图 ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
top_n = [1, 3, 10, 20, 50, 100, 200, 500, 1000, 2000]
gov_cov  = [4.78, 7.1, 18.7, 29.3, 46.4, 52.2, 56.7, 64.2, 70.4, 79.0]
gen_cov  = [0.9,  1.8,  4.2,  7.6, 12.1, 18.4, 27.8, 38.6, 50.3, 63.5]

ax.plot(top_n, gov_cov, 'o-', color='#2E75B6', linewidth=2.5, markersize=9,
        markeredgecolor='#1a4a7a', label='政务审计垂直语料库 (GovReview)', markeredgewidth=1)
ax.plot(top_n, gen_cov, 's--', color='#888', linewidth=2.0, markersize=8,
        markeredgecolor='#444', label='通用中文参考语料', markeredgewidth=1)
ax.axvline(50, color='#C55A11', linewidth=1.8, linestyle='--', alpha=0.8)
ax.annotate('前50核心专有名词\n覆盖了全库46.4%的含义',
            xy=(50, 46.4), xytext=(120, 20),
            fontsize=13, fontweight='bold', color='#C55A11',
            arrowprops=dict(arrowstyle='->', color='#C55A11', lw=1.8))

ax.set_xscale('log')
ax.set_xlabel('高频词汇数量 Top-N（对数轴）', fontsize=16, fontweight='bold')
ax.set_ylabel('文本累计信息覆盖率（%）', fontsize=16, fontweight='bold')
ax.legend(fontsize=13, loc='upper left')
ax.set_ylim(0, 85)
fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(os.path.join(OUTDIR, f'fig11_vocabulary_closure.{ext}'), bbox_inches='tight', dpi=200)
plt.close(fig)
print('✅ fig11_vocabulary_closure')

# ── fig12: Few-Shot结果 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# T1 0-shot vs 3-shot
models_fs = ['DeepSeek', 'Qwen', 'GLM4', 'Doubao']
t1_0shot = [0.401, 0.364, 0.479, 0.544]
t1_3shot = [0.453, 0.561, 0.497, 0.501]

ax = axes[0]
x = np.arange(len(models_fs))
ax.bar(x - 0.2, t1_0shot, 0.35, label='0-Shot', color='#5B9BD5', edgecolor='#333', hatch='////', alpha=0.9)
ax.bar(x + 0.2, t1_3shot, 0.35, label='3-Shot', color='#A8D08D', edgecolor='#333', hatch='....', alpha=0.9)
for i, (a, b) in enumerate(zip(t1_0shot, t1_3shot)):
    delta = b - a
    ax.text(i+0.2, b+0.008, f'{delta:+.3f}', ha='center', fontsize=11, color='#27ae60' if delta>0 else '#e74c3c', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(models_fs, fontsize=13, fontweight='bold')
ax.set_ylabel('T1 Macro-F1', fontsize=15, fontweight='bold')
ax.set_title('T1 定性判断（Few-Shot效果）', fontsize=14, fontweight='bold')
ax.legend(fontsize=13); ax.set_ylim(0.25, 0.68)

# T2 0-shot vs 3-shot
t2_0shot_bm25 = [21.36, 22.51, 21.78, 22.48]
t2_3shot_bm25 = [22.14, 23.80, 22.31, 23.59]

ax = axes[1]
ax.bar(x - 0.2, t2_0shot_bm25, 0.35, label='0-Shot', color='#5B9BD5', edgecolor='#333', hatch='////', alpha=0.9)
ax.bar(x + 0.2, t2_3shot_bm25, 0.35, label='3-Shot', color='#FFB347', edgecolor='#333', hatch='xxxx', alpha=0.9)
for i, (a, b) in enumerate(zip(t2_0shot_bm25, t2_3shot_bm25)):
    delta = b - a
    ax.text(i+0.2, b+0.2, f'{delta:+.2f}', ha='center', fontsize=11,
            color='#e74c3c' if delta>0 else '#27ae60', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(models_fs, fontsize=13, fontweight='bold')
ax.set_ylabel('T2 MAE（越低越好）', fontsize=15, fontweight='bold')
ax.set_title('T2 定量估算（Few-Shot效果）', fontsize=14, fontweight='bold')
ax.legend(fontsize=13); ax.set_ylim(18, 27)

fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(os.path.join(OUTDIR, f'fig12_fewshot.{ext}'), bbox_inches='tight', dpi=200)
plt.close(fig)
print('✅ fig12_fewshot')

# ── fig13: 消融热力图 ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
configs_abl = ['KW-only', 'Vec-only', 'Dual']
t1_abl = [[0.383, 0.426, 0.411], [0.408, 0.468, 0.474], [0.499, 0.373, 0.392], [0.504, 0.482, 0.500]]
t2_abl = [[24.42, 27.25, 27.43], [48.27, 25.84, 26.17], [53.06, 25.52, 26.38], [55.64, 21.18, 21.54]]
t3_abl = [[0.711, 0.652, 0.671], [0.776, 0.796, 0.795], [0.426, 0.704, 0.699], [0.682, 0.757, 0.743]]

for ax, data, title, fmt, cmap, vr in zip(
        axes,
        [t1_abl, t2_abl, t3_abl],
        ['T1 Macro-F1\n(越高越好)', 'T2 MAE\n(越低越好)', 'T3 PRED25\n(越高越好)'],
        ['.3f', '.1f', '.3f'],
        ['Blues', 'Reds_r', 'Greens'],
        [(0.33, 0.53), (20, 58), (0.38, 0.82)]):
    im = ax.imshow(data, cmap=cmap, vmin=vr[0], vmax=vr[1], aspect='auto')
    ax.set_xticks(range(3)); ax.set_xticklabels(configs_abl, fontsize=13, fontweight='bold')
    ax.set_yticks(range(4)); ax.set_yticklabels(MODELS, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=6)
    for i in range(4):
        for j in range(3):
            ax.text(j, i, format(data[i][j], fmt), ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if (data[i][j]-vr[0])/(vr[1]-vr[0]) > 0.6 else '#222')
    plt.colorbar(im, ax=ax, shrink=0.85)

fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(os.path.join(OUTDIR, f'fig13_ablation_heatmap.{ext}'), bbox_inches='tight', dpi=200)
plt.close(fig)
print('✅ fig13_ablation_heatmap')

# ── fig14: 任务-策略匹配矩阵 ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
tasks  = ['T1\n(方向判断)', 'T2\n(核减率)', 'T3\n(核减金额)']
strats = ['零知识\n(L1)', 'BM25\n检索', 'Dense\n检索', 'Oracle\n(L2)', 'ML\n(GBM)']
# Scores: rows=tasks, cols=strategies  (normalized 0~1)
scores = [
    [0.65, 0.72, 0.70, 0.68, 0.40],  # T1
    [0.10, 0.82, 0.60, 0.50, 0.98],  # T2
    [0.60, 0.78, 0.82, 0.76, 0.55],  # T3
]
im = ax.imshow(scores, cmap='YlGn', vmin=0, vmax=1.0, aspect='auto')
ax.set_xticks(range(5)); ax.set_xticklabels(strats, fontsize=14, fontweight='bold')
ax.set_yticks(range(3)); ax.set_yticklabels(tasks, fontsize=14, fontweight='bold')
ax.set_title('任务-策略适配矩阵（推荐热度）', fontsize=16, fontweight='bold', pad=10)
for i in range(3):
    for j in range(5):
        v = scores[i][j]
        mark = '★' if v >= 0.80 else ('◆' if v >= 0.65 else '')
        ax.text(j, i, f'{mark}\n{v:.2f}', ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='#1a1a1a' if v < 0.7 else 'white')
plt.colorbar(im, ax=ax, shrink=0.85, label='适配得分')
fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(os.path.join(OUTDIR, f'fig14_task_strategy.{ext}'), bbox_inches='tight', dpi=200)
plt.close(fig)
print('✅ fig14_task_strategy')

print('\n🎉 所有实验图重绘完成！')
