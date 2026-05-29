import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Fonts
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Increase figure width slightly to provide padding on the left and right
fig, ax = plt.subplots(figsize=(15, 3.5))
# Previous xlim might have been tight. Let's add 0.5 margin on both sides.
ax.set_xlim(-0.5, 25.5) 
ax.set_ylim(-1, 5)
ax.axis('off')

# Data for 5 boxes
# colors roughly match the screenshot: blue, orange, green, purple, red
boxes = [
    {"title": "步骤 1\n自动解析\n与数据过滤", "sub": "→ 802个\n有效样本", "color": "#4f81bd", "x": 0.5},
    {"title": "步骤 2\n专家领域标注\n(四维标签)", "sub": "→ 4维度\n校验标签", "color": "#c0504d", "x": 5.5},
    {"title": "步骤 3\n异构图谱对齐\n及证据抽取", "sub": "→ 70.8%\n高覆盖率", "color": "#9bbb59", "x": 10.5},
    {"title": "步骤 4\n隐私深度脱敏\n规范化处理", "sub": "→ 完全合规\n可公开", "color": "#8064a2", "x": 15.5},
    {"title": "GovReview-\nBench\n(802个有效样本)", "sub": "", "color": "#f79646", "x": 20.5}
]

# Adjust colors to better match the screenshot (which has blue, brown/orange, green, purple, red)
boxes[0]['color'] = '#4a7ebb' # blue
boxes[1]['color'] = '#be794f' # brown/orange
boxes[2]['color'] = '#769258' # green
boxes[3]['color'] = '#7a6096' # purple
boxes[4]['color'] = '#be5255' # red

y = 1
w = 4.2
h = 2.5

for i, b in enumerate(boxes):
    # Draw box
    rect = patches.FancyBboxPatch((b['x'], y), w, h, boxstyle="round,pad=0.1,rounding_size=0.1", 
                                  linewidth=2, edgecolor=b['color'], facecolor="white", zorder=3)
    ax.add_patch(rect)
    
    # Draw title
    ax.text(b['x'] + w/2, y + h/2, b['title'], color='black', fontweight='bold', fontsize=12, ha='center', va='center', linespacing=1.5, zorder=4)
    
    # Draw sub text
    if b['sub']:
        ax.text(b['x'] + w/2, y - 0.8, b['sub'], color='#3a5f8a', fontsize=11, ha='center', va='top', zorder=4)
    
    # Draw arrows
    if i < len(boxes) - 1:
        ax.annotate("", xy=(b['x'] + w + 0.8, y + h/2), xytext=(b['x'] + w, y + h/2), 
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5), zorder=4)

# Top left annotation
ax.annotate("", xy=(1.5, y + h), xytext=(1.5, y + h + 1.2), 
            arrowprops=dict(arrowstyle="-|>", color="gray", lw=1), zorder=4)
ax.text(1.5, y + h + 1.4, "原始项目材料\n(16个实际项目,约14,000条记录)", color='gray', fontsize=10, ha='center', va='bottom')

plt.tight_layout()
plt.savefig('GovReview-Bench 2/figures/fig01_pipeline.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('GovReview-Bench 2/figures/fig01_pipeline.png', format='png', bbox_inches='tight', dpi=300)
print("Generated fig01_pipeline!")
