import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Fonts
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 30)
ax.set_ylim(0, 16)
ax.axis('off')

# Color Scheme matches PriceBench Reference
bg_color = "none" 
panel_edge = "#a6a6a6" 
header_bg = "#686868" 
header_fg = "#ffffff"

center_box_edge = "#4f81bd" 
center_box_bg = "#ffffff"
center_title_bg = "#d3dfea" 
center_title_fg = "#1f497d"

right_outer_edge = "#a6a6a6"
right_inner_edge = "#4f81bd"

arrow_gray = "#7f7f7f"
arrow_blue = "#4f81bd"

# Draw background panels
def draw_panel(x, y, w, h, title):
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=panel_edge, facecolor=bg_color, linestyle='--', zorder=0)
    ax.add_patch(rect)
    header = patches.Rectangle((x + 0.2, y + h - 1.2), w - 0.4, 1.4, facecolor=header_bg, edgecolor='none', zorder=1)
    ax.add_patch(header)
    ax.text(x + w/2, y + h - 0.5, title, color=header_fg, fontweight='bold', fontsize=14, ha='center', va='center', zorder=2)

draw_panel(0.2, 0.5, 6, 14.5, "政务信息化项目审核工作流\n(Review Process)")
draw_panel(6.8, 0.5, 9.4, 14.5, "DMR-Bench 数据集\n(Dataset)")
draw_panel(16.8, 0.5, 13, 14.5, "六任务评测体系\n(Evaluation Tasks)")

# Left Process Nodes
def draw_left_node(x, y, text):
    ax.text(x, y, text, color='#333333', fontweight='bold', fontsize=12.5, ha='center', va='center', zorder=4)

draw_left_node(3.2, 12, "申报单位\n(Client Dept.)")
draw_left_node(3.2, 8.8, "初审机构\n(Initial Reviewer)")
draw_left_node(3.2, 5.6, "财政审核专家\n(IT Expert)")
draw_left_node(3.2, 2.4, "审批部门\n(Supervisor)")

# Arrows between left nodes
# Node 1 to 2
ax.annotate("", xy=(3.2, 9.5), xytext=(3.2, 11.3), arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1))
ax.text(3.5, 10.15, "① 提交项目材料", fontsize=11.5, color="#555555", va='center')

# Node 2 to 3
ax.annotate("", xy=(3.2, 6.3), xytext=(3.2, 8.1), arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1))
ax.text(3.5, 7.2, "② 技术评估意见", fontsize=11.5, color="#555555", va='center')

# Node 3 to 4
ax.annotate("", xy=(3.2, 3.1), xytext=(3.2, 4.9), arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1))
ax.text(3.5, 4.15, "③ 定案核减结论", fontsize=11.5, color="#555555", va='center')

# Center Nodes
def draw_center_node(x, y, w, h, title_en, title_cn, content):
    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=center_box_edge, facecolor=center_box_bg, linestyle='--', zorder=3)
    ax.add_patch(rect)
    title_bar = patches.Rectangle((x+0.1, y+h-1.2), w-0.2, 1.1, facecolor=center_title_bg, edgecolor='none', zorder=4)
    ax.add_patch(title_bar)
    ax.text(x+w/2, y+h-0.45, title_en, color=center_title_fg, fontweight='bold', fontsize=13.5, ha='center', va='center', zorder=5)
    ax.text(x+w/2, y+h-0.9, title_cn, color='#7f7f7f', fontsize=11.5, ha='center', va='center', zorder=5)
    ax.text(x+0.5, y+h-1.8, content, color='#333333', fontsize=12.5, ha='left', va='top', zorder=5, linespacing=1.6)

d1_txt = "• 项目元数据: 政务领域、年份等\n• 结构化特征: 12维量化指标\n• 非结构化文本: 建设目标、清单\n• 申报金额: CNY 850,000"
draw_center_node(7.5, 8.2, 8, 4.5, "申报材料 (Input X_i)", "Metadata (Cost Sheet)", d1_txt)

d2_txt = "• 核减决策: 建议核减 (-25%)\n• 核减金额: CNY 212,500\n• 核减理由: 软硬件报价虚高\n• 审定金额: CNY 637,500"
draw_center_node(7.5, 2.2, 8, 4.5, "专家审核结论 (Label Y_gt)", "Review Result", d2_txt)

# Helper for Step Arrows
def draw_step_arrow(x1, y1, x2, y2, color, text, is_dashed, text_offset_y=0.15):
    ls = '--' if is_dashed else '-'
    
    if y1 == y2:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, ls=ls))
        if text:
            mid_x = (x1 + x2) / 2
            ax.text(mid_x, y2 + text_offset_y, text, fontsize=11.5, color=color, va='bottom', ha='center', zorder=6)
    else:
        mid_x = x1 + (x2 - x1) * 0.45
        ax.plot([x1, mid_x], [y1, y1], color=color, lw=1.5, ls=ls)
        ax.plot([mid_x, mid_x], [y1, y2], color=color, lw=1.5, ls=ls)
        ax.annotate("", xy=(x2, y2), xytext=(mid_x, y2), 
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, ls=ls))
        if text:
            ax.text(mid_x + 0.2, y2 + text_offset_y, text, fontsize=11.5, color=color, va='bottom', ha='left', zorder=6)

# Left to Center Arrows
draw_step_arrow(4.2, 10.45, 7.5, 10.45, arrow_gray, "特征提取", is_dashed=True)
draw_step_arrow(4.2, 4.45, 7.5, 4.45, arrow_gray, "金标准标注", is_dashed=True)

# Right Nodes (Tasks)
def draw_task_group(x, y, w, h, title_en, title_cn, t1, t2):
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=right_outer_edge, facecolor="#ffffff", zorder=3)
    ax.add_patch(rect)
    title_bar = patches.Rectangle((x, y+h-1.2), w, 1.2, facecolor=center_title_bg, edgecolor='none', zorder=4)
    ax.add_patch(title_bar)
    ax.text(x+0.3, y+h-0.45, title_en, color=center_title_fg, fontweight='bold', fontsize=12.5, ha='left', va='center', zorder=5)
    ax.text(x+0.3, y+h-0.95, title_cn, color='#555555', fontsize=10.5, ha='left', va='center', zorder=5)
    
    t1_box = patches.FancyBboxPatch((x+0.5, y+1.5), w-1, 1, boxstyle="round,pad=0.1,rounding_size=0.1", 
                                  linewidth=1.5, edgecolor=right_inner_edge, facecolor="#ffffff", zorder=4)
    ax.add_patch(t1_box)
    ax.text(x+w/2, y+2.0, t1, color=center_title_fg, fontsize=12, fontweight='bold', ha='center', va='center', zorder=5)
    
    t2_box = patches.FancyBboxPatch((x+0.5, y+0.2), w-1, 1, boxstyle="round,pad=0.1,rounding_size=0.1", 
                                  linewidth=1.5, edgecolor=right_inner_edge, facecolor="#ffffff", zorder=4)
    ax.add_patch(t2_box)
    ax.text(x+w/2, y+0.7, t2, color=center_title_fg, fontsize=12, fontweight='bold', ha='center', va='center', zorder=5)

tw = 11.5
draw_task_group(17.5, 9.8, tw, 4, "判断层任务 (Classification)", "Determine whether the decision should be Reduce", 
                "T1: 核减方向预测 (Macro-F1)", "T5: 算术异常检测 (F1)")
draw_task_group(17.5, 5.2, tw, 4, "定量层任务 (Regression)", "Calculate exact ratio and final price", 
                "T2: 核减率回归 (MAE)", "T3: 核减金额预测 (PRED25)")
draw_task_group(17.5, 0.6, tw, 4, "认知与生成任务 (Generation)", "Estimate the difficulty and generate reasons", 
                "T4: 难度等级预测 (Macro-F1)", "T6: 核减理由生成 (CharOverlap)")

# Center to Right Arrows - unified bus design
bus_x = 16.5
# Vertical bus line
ax.plot([bus_x, bus_x], [2.6, 11.8], color=arrow_blue, lw=1.5)

# Inputs to bus
ax.plot([15.5, bus_x], [10.45, 10.45], color=arrow_blue, lw=1.5)
ax.plot([15.5, bus_x], [4.45, 4.45], color=arrow_blue, lw=1.5)

# Outputs from bus
ax.annotate("", xy=(17.5, 11.8), xytext=(bus_x, 11.8), arrowprops=dict(arrowstyle="-|>", color=arrow_blue, lw=1.5))
ax.annotate("", xy=(17.5, 7.2), xytext=(bus_x, 7.2), arrowprops=dict(arrowstyle="-|>", color=arrow_blue, lw=1.5))
ax.annotate("", xy=(17.5, 2.6), xytext=(bus_x, 2.6), arrowprops=dict(arrowstyle="-|>", color=arrow_blue, lw=1.5))

plt.tight_layout()
plt.savefig('GovReview-Bench 4/figures/fig_architecture.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('GovReview-Bench 4/figures/fig_architecture.png', format='png', bbox_inches='tight', dpi=300)
print("DMR-Bench architecture figure generated!")
