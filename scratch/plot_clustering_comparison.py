"""
Plot comparison charts for clustering algorithms benchmark:
- Fig 11: Clustering Algorithms Benchmark: OOS Sharpe, Annual Return, Max Drawdown, Calmar Ratio, and Geometry Metrics
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

# Load scorecard
df = pd.read_csv('data/clustering_comparison_scorecard.csv')

# Dark theme palette
BG_DARK = '#090a0f'
BG_PANEL = '#12141c'
BORDER = '#2a2a35'
TEXT_WHITE = '#f8fafc'
TEXT_MUTED = '#94a3b8'
GRID_COLOR = '#1e2230'

fig = plt.figure(figsize=(18, 12), facecolor=BG_DARK)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28, left=0.06, right=0.96, top=0.92, bottom=0.08)

# Color palettes for algorithms
algos = ['K-Means', 'Gaussian Mixture', 'Mean Shift', 'DBSCAN', 'Hierarchical', 'Spectral']
colors = ['#64748b', '#38bdf8', '#a855f7', '#ec4899', '#f59e0b', '#10b981']
color_map = dict(zip(algos, colors))

# ── 1. SUBPLOT 1: OOS SHARPE RATIO (DIRECT vs HMM INIT) ──
ax1 = fig.add_subplot(gs[0, 0], facecolor=BG_PANEL)
df_direct = df[df['Paradigm'] == 'Direct Clustering (No HMM)'].set_index('Algorithm')
df_hmm = df[df['Paradigm'] == 'HMM EM Initializer'].set_index('Algorithm')

x = np.arange(len(algos))
width = 0.36

rects1 = ax1.bar(x - width/2, [df_direct.loc[a, 'OOS_Sharpe'] for a in algos], width, label='Direct Clustering', color='#38bdf8', alpha=0.85, edgecolor='#0284c7')
rects2 = ax1.bar(x + width/2, [df_hmm.loc[f'HMM + {a}', 'OOS_Sharpe'] for a in algos], width, label='HMM + Cluster Init', color='#10b981', alpha=0.85, edgecolor='#059669')

ax1.set_title('Out-of-Sample Sharpe Ratio (27 Folds)', fontsize=12, fontweight='bold', color=TEXT_WHITE, pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(algos, rotation=25, ha='right', fontsize=9, color=TEXT_MUTED)
ax1.set_ylabel('Sharpe Ratio', fontsize=10, color=TEXT_MUTED)
ax1.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
ax1.legend(facecolor='#18181b', edgecolor='#374151', labelcolor=TEXT_WHITE, fontsize=8.5)
ax1.set_ylim(1.2, 2.3)

for rect in rects1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=8, color='#7dd3fc', fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=8, color='#6ee7b7', fontweight='bold')

# ── 2. SUBPLOT 2: OOS ANNUALIZED RETURN (%) ──
ax2 = fig.add_subplot(gs[0, 1], facecolor=BG_PANEL)
rects1_ret = ax2.bar(x - width/2, [df_direct.loc[a, 'OOS_AnnRet_pct'] for a in algos], width, label='Direct Clustering', color='#818cf8', alpha=0.85, edgecolor='#6366f1')
rects2_ret = ax2.bar(x + width/2, [df_hmm.loc[f'HMM + {a}', 'OOS_AnnRet_pct'] for a in algos], width, label='HMM + Cluster Init', color='#f59e0b', alpha=0.85, edgecolor='#d97706')

ax2.set_title('Out-of-Sample Annualized Return (%)', fontsize=12, fontweight='bold', color=TEXT_WHITE, pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(algos, rotation=25, ha='right', fontsize=9, color=TEXT_MUTED)
ax2.set_ylabel('Annual Return (%)', fontsize=10, color=TEXT_MUTED)
ax2.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
ax2.legend(facecolor='#18181b', edgecolor='#374151', labelcolor=TEXT_WHITE, fontsize=8.5)
ax2.set_ylim(18, 32)

for rect in rects1_ret:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + 0.3, f'{h:.1f}%', ha='center', va='bottom', fontsize=8, color='#c7d2fe', fontweight='bold')
for rect in rects2_ret:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + 0.3, f'{h:.1f}%', ha='center', va='bottom', fontsize=8, color='#fde68a', fontweight='bold')

# ── 3. SUBPLOT 3: OOS MAX DRAWDOWN (%) ──
ax3 = fig.add_subplot(gs[0, 2], facecolor=BG_PANEL)
rects1_dd = ax3.bar(x - width/2, [df_direct.loc[a, 'OOS_MaxDD_pct'] for a in algos], width, label='Direct Clustering', color='#f43f5e', alpha=0.85, edgecolor='#e11d48')
rects2_dd = ax3.bar(x + width/2, [df_hmm.loc[f'HMM + {a}', 'OOS_MaxDD_pct'] for a in algos], width, label='HMM + Cluster Init', color='#fb7185', alpha=0.85, edgecolor='#f43f5e')

ax3.set_title('Out-of-Sample Max Drawdown (Lower is Better)', fontsize=12, fontweight='bold', color=TEXT_WHITE, pad=10)
ax3.set_xticks(x)
ax3.set_xticklabels(algos, rotation=25, ha='right', fontsize=9, color=TEXT_MUTED)
ax3.set_ylabel('Max Drawdown (%)', fontsize=10, color=TEXT_MUTED)
ax3.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
ax3.legend(facecolor='#18181b', edgecolor='#374151', labelcolor=TEXT_WHITE, fontsize=8.5)
ax3.set_ylim(-20, 0)

for rect in rects1_dd:
    h = rect.get_height()
    ax3.text(rect.get_x() + rect.get_width()/2., h - 0.9, f'{h:.1f}%', ha='center', va='top', fontsize=8, color='#fda4af', fontweight='bold')
for rect in rects2_dd:
    h = rect.get_height()
    ax3.text(rect.get_x() + rect.get_width()/2., h - 0.9, f'{h:.1f}%', ha='center', va='top', fontsize=8, color='#fecdd3', fontweight='bold')

# ── 4. SUBPLOT 4: CALMAR RATIO (RETURN / MAX DD) ──
ax4 = fig.add_subplot(gs[1, 0], facecolor=BG_PANEL)
rects1_cal = ax4.bar(x - width/2, [df_direct.loc[a, 'OOS_Calmar'] for a in algos], width, label='Direct Clustering', color='#a855f7', alpha=0.85, edgecolor='#9333ea')
rects2_cal = ax4.bar(x + width/2, [df_hmm.loc[f'HMM + {a}', 'OOS_Calmar'] for a in algos], width, label='HMM + Cluster Init', color='#06b6d4', alpha=0.85, edgecolor='#0891b2')

ax4.set_title('Out-of-Sample Calmar Ratio', fontsize=12, fontweight='bold', color=TEXT_WHITE, pad=10)
ax4.set_xticks(x)
ax4.set_xticklabels(algos, rotation=25, ha='right', fontsize=9, color=TEXT_MUTED)
ax4.set_ylabel('Calmar Ratio', fontsize=10, color=TEXT_MUTED)
ax4.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
ax4.legend(facecolor='#18181b', edgecolor='#374151', labelcolor=TEXT_WHITE, fontsize=8.5)
ax4.set_ylim(1.0, 3.8)

for rect in rects1_cal:
    h = rect.get_height()
    ax4.text(rect.get_x() + rect.get_width()/2., h + 0.05, f'{h:.2f}', ha='center', va='bottom', fontsize=8, color='#e9d5ff', fontweight='bold')
for rect in rects2_cal:
    h = rect.get_height()
    ax4.text(rect.get_x() + rect.get_width()/2., h + 0.05, f'{h:.2f}', ha='center', va='bottom', fontsize=8, color='#a5f3fc', fontweight='bold')

# ── 5. SUBPLOT 5: CLUSTERING GEOMETRY (SILHOUETTE & DAVIES-BOULDIN) ──
ax5 = fig.add_subplot(gs[1, 1], facecolor=BG_PANEL)
sil_vals = [df_direct.loc[a, 'Silhouette'] for a in algos]
db_vals = [df_direct.loc[a, 'Davies_Bouldin'] for a in algos]

ax5_twin = ax5.twinx()
ax5.bar(x - width/2, sil_vals, width, label='Silhouette Score (Higher Better)', color='#10b981', alpha=0.85, edgecolor='#059669')
ax5_twin.bar(x + width/2, db_vals, width, label='Davies-Bouldin (Lower Better)', color='#f97316', alpha=0.85, edgecolor='#ea580c')

ax5.set_title('Cluster Separation Metrics (Feature Space)', fontsize=12, fontweight='bold', color=TEXT_WHITE, pad=10)
ax5.set_xticks(x)
ax5.set_xticklabels(algos, rotation=25, ha='right', fontsize=9, color=TEXT_MUTED)
ax5.set_ylabel('Silhouette Score', fontsize=10, color='#10b981')
ax5_twin.set_ylabel('Davies-Bouldin Index', fontsize=10, color='#f97316')
ax5.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
ax5.tick_params(colors=TEXT_MUTED)
ax5_twin.tick_params(colors=TEXT_MUTED)

# ── 6. SUBPLOT 6: IN-SAMPLE TURNOVER & SWITCHES ──
ax6 = fig.add_subplot(gs[1, 2], facecolor=BG_PANEL)
switches_direct = [df_direct.loc[a, 'IS_Switches'] for a in algos]
switches_hmm = [df_hmm.loc[f'HMM + {a}', 'IS_Switches'] for a in algos]

rects1_sw = ax6.bar(x - width/2, switches_direct, width, label='Direct Clustering', color='#eab308', alpha=0.85, edgecolor='#ca8a04')
rects2_sw = ax6.bar(x + width/2, switches_hmm, width, label='HMM Regularized', color='#6366f1', alpha=0.85, edgecolor='#4f46e5')

ax6.set_title('In-Sample Regime Switches (Whipsaw / Turnover)', fontsize=12, fontweight='bold', color=TEXT_WHITE, pad=10)
ax6.set_xticks(x)
ax6.set_xticklabels(algos, rotation=25, ha='right', fontsize=9, color=TEXT_MUTED)
ax6.set_ylabel('Total Regime Switches', fontsize=10, color=TEXT_MUTED)
ax6.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
ax6.legend(facecolor='#18181b', edgecolor='#374151', labelcolor=TEXT_WHITE, fontsize=8.5)
ax6.set_ylim(0, 500)

for rect in rects1_sw:
    h = rect.get_height()
    ax6.text(rect.get_x() + rect.get_width()/2., h + 8, f'{int(h)}', ha='center', va='bottom', fontsize=8, color='#fef08a', fontweight='bold')
for rect in rects2_sw:
    h = rect.get_height()
    ax6.text(rect.get_x() + rect.get_width()/2., h + 8, f'{int(h)}', ha='center', va='bottom', fontsize=8, color='#c7d2fe', fontweight='bold')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)

# Master Title
fig.suptitle('CLUSTERING ALGORITHMS BENCHMARK IN PLACE OF K-MEANS\nWalk-Forward OOS (2018–2026, 27 Folds) & In-Sample Quantitative Scorecard',
             fontsize=14, fontweight='bold', color=TEXT_WHITE, y=0.98)

out_fig = 'figures/fig11_clustering_algorithms_benchmark.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'✓ Saved benchmark figure: {out_fig}')
