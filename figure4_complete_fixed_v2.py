#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 完整分析流程 - 基于探索结果
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import scipy.sparse
import requests
import time
from tqdm import tqdm
from scipy.stats import chi2_contingency
import gseapy as gp
import json

# 设置 matplotlib 字体为 Arial（投稿要求）
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置
sc.settings.verbosity = 3
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
class Config:
    BASE_DIR = Path.home() / "statics/GEO_data/GSE/figure4"
    RAW_DATA_DIR = BASE_DIR / "data/raw"
    PROCESSED_DIR = BASE_DIR / "data/processed"
    FIG_DIR = BASE_DIR / "figs"
    RESULTS_DIR = BASE_DIR / "results"
    
    RAW_H5AD = RAW_DATA_DIR / "4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad"
    TARGET_GENES_FILE = BASE_DIR / "imidazoline_SLE_intersection.csv"
    
    @classmethod
    def setup(cls):
        for d in [cls.PROCESSED_DIR, cls.FIG_DIR, cls.RESULTS_DIR,
                  cls.RESULTS_DIR / 'enrichr_results']:
            d.mkdir(parents=True, exist_ok=True)
        print("✅ 目录创建完成")

Config.setup()

# ==================== 1. 数据加载 ====================
print("\n" + "="*80)
print("1. 加载原始数据")
print("="*80)

adata = sc.read_h5ad(Config.RAW_H5AD)
print(f"细胞数: {adata.n_obs:,}")
print(f"基因数: {adata.n_vars:,}")

# ==================== 2. 基因名转换 ====================
print("\n" + "="*80)
print("2. 基因名转换 (ENSG → Gene Symbol)")
print("="*80)

# 直接使用 h5ad 内置的 feature_name 列，无需 API 查询
adata.var['Gene_Symbol'] = adata.var['feature_name'].values
adata.var['original_id'] = adata.var_names
adata.var_names = adata.var['feature_name'].values
adata.var_names_make_unique()

# 清除 raw 数据以避免基因名不一致问题
if adata.raw is not None:
    adata.raw = None
    print("⚠️  Cleared adata.raw to avoid gene name mismatch")

gene_symbols = list(adata.var_names)

print(f"\n转换结果:")
print(f"  - 总基因数: {len(gene_symbols)}")
print(f"  - 成功映射: {sum(1 for g in gene_symbols if not g.startswith('ENSG'))}")
print(f"  - 未映射: {sum(1 for g in gene_symbols if g.startswith('ENSG'))}")

# 检查关键基因
key_genes = ['CD14', 'CD19', 'CD3D', 'CD4', 'CD8A', 'MS4A1', 'NKG7', 'GNLY']
found = [g for g in key_genes if g in adata.var_names]
print(f"\n关键基因命中: {len(found)}/{len(key_genes)}")
print(f"命中基因: {found}")

# ==================== 2.5. 去除红细胞污染 ====================
print("\n" + "="*80)
print("2.5. 去除红细胞污染")
print("="*80)

rbc_genes = [g for g in ['HBB', 'HBA1', 'HBA2'] if g in adata.var_names]
print(f"使用红细胞标志基因: {rbc_genes}")

# 计算每个细胞的血红蛋白基因总表达量
rbc_expr = np.array(adata[:, rbc_genes].X.sum(axis=1)).flatten()
rbc_threshold = np.percentile(rbc_expr, 99)  # 去除表达量最高的 1%
keep_mask = rbc_expr <= rbc_threshold
n_before = adata.n_obs
adata = adata[keep_mask].copy()
print(f"过滤前细胞数: {n_before:,}")
print(f"过滤后细胞数: {adata.n_obs:,}  (去除 {n_before - adata.n_obs:,} 个红细胞污染细胞)")


print("\n" + "="*80)
print("3. 加载靶点基因列表")
print("="*80)

target_df = pd.read_csv(Config.TARGET_GENES_FILE)
target_genes = target_df.iloc[:, 0].tolist()
print(f"靶点基因总数: {len(target_genes)}")

# 找出在数据中存在的靶点
existing_targets = [g for g in target_genes if g in adata.var_names]
print(f"数据中存在的靶点: {len(existing_targets)}/{len(target_genes)}")

# ⚠️ 过滤靶点：只保留在免疫细胞中有实际表达的基因
print("\n诊断：靶点基因在免疫细胞中的表达情况...")
expr_matrix = adata[:, existing_targets].X
if hasattr(expr_matrix, 'toarray'):
    expr_matrix = expr_matrix.toarray()

pct_expressed = (expr_matrix > 0).mean(axis=0) * 100
mean_expr = np.array(expr_matrix.mean(axis=0)).flatten()

target_expr_df = pd.DataFrame({
    'gene': existing_targets,
    'pct_cells_expressed': pct_expressed,
    'mean_expression': mean_expr
}).sort_values('pct_cells_expressed', ascending=False)

print(f"  - 在 >10% 细胞中表达: {(target_expr_df['pct_cells_expressed'] > 10).sum()}")
print(f"  - 在 >5% 细胞中表达: {(target_expr_df['pct_cells_expressed'] > 5).sum()}")
print(f"  - 在 >1% 细胞中表达: {(target_expr_df['pct_cells_expressed'] > 1).sum()}")

# 使用阈值过滤（在至少5%细胞中表达）
MIN_PCT_CELLS = 5.0
filtered_targets = target_expr_df[target_expr_df['pct_cells_expressed'] > MIN_PCT_CELLS]['gene'].tolist()
print(f"\n✅ 过滤后靶点数: {len(filtered_targets)} (在 >{MIN_PCT_CELLS}% 细胞中表达)")
print(f"前10个表达最高的靶点: {filtered_targets[:10]}")

# 保存诊断信息
target_expr_df.to_csv(Config.RESULTS_DIR / 'target_gene_expression_diagnosis.csv', index=False)
print(f"诊断文件已保存: target_gene_expression_diagnosis.csv")

# 使用过滤后的靶点
existing_targets = filtered_targets

# ==================== 4. Figure 4A: 靶点活性评分 ====================
print("\n" + "="*80)
print("4. Figure 4A: 计算靶点活性评分")
print("="*80)

# 计算评分
sc.tl.score_genes(adata, gene_list=existing_targets, 
                  score_name='target_score', 
                  ctrl_size=50,
                  use_raw=False)

print(f"评分统计:")
print(f"  最小值: {adata.obs['target_score'].min():.4f}")
print(f"  最大值: {adata.obs['target_score'].max():.4f}")
print(f"  中位数: {adata.obs['target_score'].median():.4f}")

# 绘制Figure 4A
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 缩写 → 全名映射（用于所有图表）
celltype_names = {
    'T4':    'CD4+ T cell',
    'T8':    'CD8+ T cell',
    'B':     'B cell',
    'cM':    'Classical Monocyte',
    'NK':    'NK cell',
    'ncM':   'Non-classical Monocyte',
    'cDC':   'Classical DC',
    'Prolif':'Proliferating cells',
    'pDC':   'Plasmacytoid DC',
    'PB':    'Plasmablast',
    'Progen':'Progenitor'
}

# UMAP评分图
sc.pl.umap(adata, color='target_score', ax=axes[0],
           cmap='viridis', title='Target Activity Score',
           show=False)

# 箱线图 - 主要细胞类型（使用实际的缩写，x轴显示全名）
main_types = ['T4', 'T8', 'B', 'cM', 'NK', 'pDC']
available_types = [t for t in main_types if t in adata.obs['author_cell_type'].unique()]

data_for_violin = []
for t in available_types:
    mask = adata.obs['author_cell_type'] == t
    data_for_violin.append(adata.obs.loc[mask, 'target_score'].values)

bp = axes[1].boxplot(data_for_violin, patch_artist=True, showfliers=False)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(available_types)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

axes[1].set_xticklabels([celltype_names.get(t, t) for t in available_types], rotation=45, ha='right')
axes[1].set_ylabel('Target Activity Score')
axes[1].set_title('Score Distribution by Major Cell Types')

plt.tight_layout()
plt.savefig(Config.FIG_DIR / 'Figure4A_target_score.pdf', dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Figure 4A已保存")

# ==================== 5. Figure 4B: 高评分细胞亚群 ====================
print("\n" + "="*80)
print("5. Figure 4B: 高评分细胞亚群鉴定")
print("="*80)

threshold = np.percentile(adata.obs['target_score'], 80)
adata.obs['is_high_score'] = adata.obs['target_score'] > threshold
print(f"阈值 (前20%): {threshold:.4f}")
print(f"高评分细胞数: {adata.obs['is_high_score'].sum():,} ({adata.obs['is_high_score'].mean()*100:.1f}%)")

# 提取高评分细胞
high_score_adata = adata[adata.obs['is_high_score']].copy()

# 对高评分细胞重新聚类（adata.X 已是 log1p 归一化状态，无需重复处理）
sc.pp.filter_genes(high_score_adata, min_cells=3)
sc.pp.highly_variable_genes(high_score_adata, n_top_genes=2000)
sc.pp.scale(high_score_adata, max_value=10)
sc.tl.pca(high_score_adata, svd_solver='arpack')
sc.pp.neighbors(high_score_adata, n_pcs=30)
sc.tl.leiden(high_score_adata, resolution=0.5, key_added='high_score_cluster')
sc.tl.umap(high_score_adata)

# 绘制Figure 4B
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：高亮高评分细胞
adata.obs['highlight'] = adata.obs['is_high_score'].map({True: 'High Score', False: 'Other'})
colors = {'High Score': 'red', 'Other': 'lightgray'}
sc.pl.umap(adata, color='highlight', ax=axes[0],
           palette=colors, title='High-score Cells (Top 20%)',
           show=False)

# 右图：高评分细胞按 author_cell_type 着色（注明亚群来源）
high_score_adata.obs['Cell Type'] = high_score_adata.obs['author_cell_type'].astype(str).map(celltype_names).fillna(high_score_adata.obs['author_cell_type'].astype(str))
sc.pl.umap(high_score_adata, color='Cell Type', ax=axes[1],
           title='High-score Cells by Cell Type', show=False)

plt.tight_layout()
plt.savefig(Config.FIG_DIR / 'Figure4B_high_score_subset.pdf', dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Figure 4B已保存")

# ==================== 6. Figure 4C: 疾病特异性分析 ====================
print("\n" + "="*80)
print("6. Figure 4C: 疾病特异性分析")
print("="*80)

# 按 donor 统计（每个 donor 只属于一种 disease，不能做笛卡尔积联合分组）
sample_stats = adata.obs.groupby('donor_id').agg(
    disease=('disease', 'first'),
    total_cells=('is_high_score', 'count'),
    high_score_cells=('is_high_score', lambda x: (x == True).sum())
).reset_index()
sample_stats['high_score_percent'] = sample_stats['high_score_cells'] / sample_stats['total_cells'] * 100

print(f"样本统计 (n={len(sample_stats)} donors):")
print(sample_stats.head())

# 绘制箱线图
fig, ax = plt.subplots(figsize=(8, 6))

diseases = ['normal', 'systemic lupus erythematosus']
data_to_plot = []
for d in diseases:
    data = sample_stats[sample_stats['disease'] == d]['high_score_percent'].values
    data_to_plot.append(data)

bp = ax.boxplot(data_to_plot, patch_artist=True, showfliers=False)
colors = ['#7fbf7f', '#ff7f7f']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# 添加散点
for i, data in enumerate(data_to_plot):
    x = np.random.normal(i+1, 0.04, size=len(data))
    ax.scatter(x, data, alpha=0.6, s=30, color='black', zorder=3)

ax.set_xticklabels(['Healthy Control', 'SLE'])
ax.set_ylabel('High-score Cell Proportion (%)')
ax.set_title('High-score Cell Proportion in SLE vs HC')

# 以 donor 为单位做 Wilcoxon 秩和检验（正确的统计单元）
from scipy.stats import mannwhitneyu
hc_vals = sample_stats[sample_stats['disease'] == 'normal']['high_score_percent'].values
sle_vals = sample_stats[sample_stats['disease'] == 'systemic lupus erythematosus']['high_score_percent'].values
stat, p_value = mannwhitneyu(sle_vals, hc_vals, alternative='two-sided')
print(f"\nMann-Whitney U 检验 (donor level, n_HC={len(hc_vals)}, n_SLE={len(sle_vals)})")
print(f"p值: {p_value:.4e}")
ax.text(0.5, 0.95, f'p = {p_value:.2e} (Mann-Whitney, donor-level)', transform=ax.transAxes, ha='center')

plt.tight_layout()
plt.savefig(Config.FIG_DIR / 'Figure4C_disease_specificity.pdf', dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Figure 4C已保存")

# ==================== 7. Figure 4D: 特征基因分析 ====================
print("\n" + "="*80)
print("7. Figure 4D: 特征基因分析")
print("="*80)

# 正确方法：Pseudobulk差异分析（以donor为统计单元）
print("Computing pseudobulk differential expression (donor-level)...")

# 步骤1：计算cell-level的fold change用于预筛选
high_cells = adata[adata.obs['is_high_score'] == True]
low_cells = adata[adata.obs['is_high_score'] == False]

if scipy.sparse.issparse(adata.X):
    high_mean = np.array(high_cells.X.mean(axis=0)).flatten()
    low_mean = np.array(low_cells.X.mean(axis=0)).flatten()
else:
    high_mean = high_cells.X.mean(axis=0)
    low_mean = low_cells.X.mean(axis=0)

logFC_cell = high_mean - low_mean

# 预筛选：只测试logFC>0.15的基因（减少计算）
candidate_mask = logFC_cell > 0.15
candidate_genes = adata.var_names[candidate_mask].tolist()
print(f"Candidate genes (cell-level logFC > 0.15): {len(candidate_genes)}")

# 步骤2：构建pseudobulk矩阵（按donor聚合，用mean）
print(f"\nBuilding pseudobulk profiles...")
donors = adata.obs['donor_id'].unique()
n_donors = len(donors)

pseudobulk_high = np.zeros((n_donors, len(candidate_genes)))
pseudobulk_low = np.zeros((n_donors, len(candidate_genes)))
donor_has_high = []
donor_has_low = []

for i, donor in enumerate(donors):
    donor_cells = adata[adata.obs['donor_id'] == donor]

    # High-score cells in this donor
    high_mask = donor_cells.obs['is_high_score'] == True
    if high_mask.sum() > 0:
        donor_high = donor_cells[high_mask]
        if scipy.sparse.issparse(donor_high.X):
            pseudobulk_high[i] = np.array(donor_high[:, candidate_genes].X.mean(axis=0)).flatten()
        else:
            pseudobulk_high[i] = donor_high[:, candidate_genes].X.mean(axis=0)
        donor_has_high.append(True)
    else:
        donor_has_high.append(False)

    # Low-score cells in this donor
    low_mask = donor_cells.obs['is_high_score'] == False
    if low_mask.sum() > 0:
        donor_low = donor_cells[low_mask]
        if scipy.sparse.issparse(donor_low.X):
            pseudobulk_low[i] = np.array(donor_low[:, candidate_genes].X.mean(axis=0)).flatten()
        else:
            pseudobulk_low[i] = donor_low[:, candidate_genes].X.mean(axis=0)
        donor_has_low.append(True)
    else:
        donor_has_low.append(False)

# 只保留两组都有样本的donor
valid_donors = np.array(donor_has_high) & np.array(donor_has_low)
pseudobulk_high = pseudobulk_high[valid_donors]
pseudobulk_low = pseudobulk_low[valid_donors]

print(f"Valid donors (have both high and low cells): {valid_donors.sum()}/{n_donors}")

# 步骤3：Paired t-test（配对检验，同一个donor的high vs low）
from scipy.stats import ttest_rel
pvals = []
logFC_pseudobulk = []

for j in range(len(candidate_genes)):
    high_expr = pseudobulk_high[:, j]
    low_expr = pseudobulk_low[:, j]

    # Paired t-test
    try:
        stat, pval = ttest_rel(high_expr, low_expr, alternative='greater')
        pvals.append(pval)
    except:
        pvals.append(1.0)

    # LogFC in pseudobulk space
    logFC_pseudobulk.append(high_expr.mean() - low_expr.mean())

# FDR校正
from statsmodels.stats.multitest import fdrcorrection
_, p_adj = fdrcorrection(pvals)

# 构建结果表
markers = pd.DataFrame({
    'gene': candidate_genes,
    'logFC': logFC_pseudobulk,
    'p_val': pvals,
    'p_val_adj': p_adj
}).sort_values('p_val_adj')

print(f"\n诊断 - Markers数据分布:")
print(f"  Total genes tested: {len(markers)}")
print(f"  logFC范围: [{markers['logFC'].min():.3f}, {markers['logFC'].max():.3f}]")
print(f"  p_val_adj < 0.05: {(markers['p_val_adj'] < 0.05).sum()}")
print(f"  p_val_adj < 0.01: {(markers['p_val_adj'] < 0.01).sum()}")

# 过滤显著基因
significant_markers = markers[
    (markers['logFC'] > 0.15) &
    (markers['p_val_adj'] < 0.05)
].copy()

print(f"\n✅ 显著特征基因 (donor-level, logFC>0.15, p_adj<0.05): {len(significant_markers)}")
print(significant_markers.head(20)[['gene', 'logFC', 'p_val_adj']].to_string())

# 保存结果
significant_markers.to_csv(Config.RESULTS_DIR / 'marker_genes.csv', index=False)

# 绘制点图
if len(significant_markers) >= 10:
    top_genes = significant_markers.head(10)['gene'].tolist()

    # 添加全名列，dotplot y轴显示完整细胞类型名称
    adata.obs['Cell Type'] = adata.obs['author_cell_type'].astype(str).map(celltype_names).fillna(adata.obs['author_cell_type'].astype(str))

    fig, ax = plt.subplots(figsize=(12, 6))
    sc.pl.dotplot(adata, top_genes, groupby='Cell Type',
                 standard_scale='var', ax=ax, show=False,
                 use_raw=False,
                 title='Marker Gene Expression in High-score Cells')
    
    plt.tight_layout()
    plt.savefig(Config.FIG_DIR / 'Figure4D_marker_genes.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Figure 4D已保存")

# ==================== 8. Figure 4E: 功能富集分析 ====================
print("\n" + "="*80)
print("8. Figure 4E: 功能富集分析")
print("="*80)

if len(significant_markers) > 0:
    # 使用更多基因进行富集（放宽阈值）
    relaxed_markers = markers[
        (markers['logFC'] > 0.2) &
        (markers['p_val_adj'] < 0.05)
    ].head(100)  # 最多取100个基因

    gene_list = relaxed_markers['gene'].tolist()
    print(f"用于富集分析的基因数: {len(gene_list)}")

    try:
        # 分别运行GO和KEGG富集
        print("\n运行GO Biological Process富集...")
        go_enrich = gp.enrichr(gene_list=gene_list,
                              gene_sets=['GO_Biological_Process_2021'],
                              organism='Human',
                              outdir=Config.RESULTS_DIR / 'enrichr_results',
                              no_plot=True)

        print("运行KEGG通路富集...")
        kegg_enrich = gp.enrichr(gene_list=gene_list,
                                gene_sets=['KEGG_2021_Human'],
                                organism='Human',
                                outdir=Config.RESULTS_DIR / 'enrichr_results',
                                no_plot=True)

        # 优先展示GO结果
        if go_enrich.results is not None and len(go_enrich.results) > 0:
            go_results = go_enrich.results[go_enrich.results['Adjusted P-value'] < 0.05].copy()

            print(f"\n✅ GO Biological Process富集结果 (Top 10):")
            print(go_results[['Term', 'P-value', 'Adjusted P-value', 'Genes']].head(10).to_string())

            # 保存完整结果
            go_results.to_csv(Config.RESULTS_DIR / 'GO_enrichment_results.csv', index=False)

            if len(go_results) > 0:
                # 绘制GO气泡图
                top_go = go_results.head(15).copy()
                top_go['gene_count'] = top_go['Genes'].str.split(';').str.len()
                top_go['-log10_pval'] = -np.log10(top_go['Adjusted P-value'])

                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(top_go['gene_count'],
                                   range(len(top_go)),
                                   s=top_go['-log10_pval'] * 50,
                                   c=top_go['Adjusted P-value'],
                                   cmap='viridis_r',
                                   alpha=0.7,
                                   edgecolors='black',
                                   linewidth=1)

                ax.set_yticks(range(len(top_go)))
                ax.set_yticklabels([t[:60] + '...' if len(t) > 60 else t for t in top_go['Term']], fontsize=9)
                plt.colorbar(scatter, ax=ax, label='Adjusted P-value')
                ax.set_xlabel('Gene Count', fontsize=12)
                ax.set_ylabel('GO Biological Process', fontsize=12)
                ax.set_title('Functional Enrichment of Sensitive Cell Markers', fontsize=14)
                ax.invert_yaxis()

                plt.tight_layout()
                plt.savefig(Config.FIG_DIR / 'Figure4E_GO_enrichment.pdf', dpi=300, bbox_inches='tight')
                plt.show()
                print(f"✅ Figure 4E (GO) 已保存")

        # 保存KEGG结果（仅供参考）
        if kegg_enrich.results is not None and len(kegg_enrich.results) > 0:
            kegg_results = kegg_enrich.results[kegg_enrich.results['Adjusted P-value'] < 0.05].copy()
            kegg_results.to_csv(Config.RESULTS_DIR / 'KEGG_enrichment_results.csv', index=False)
            print(f"\n⚠️  KEGG富集结果已保存 (仅供参考，可能包含非免疫相关通路)")
            print(f"   文件: KEGG_enrichment_results.csv")

    except Exception as e:
        print(f"富集分析失败: {e}")

# ==================== 9. 保存所有结果 ====================
print("\n" + "="*80)
print("9. 保存处理后的数据")
print("="*80)

adata.write(Config.PROCESSED_DIR / 'adata_final.h5ad')
high_score_adata.write(Config.PROCESSED_DIR / 'adata_high_score.h5ad')

print(f"✅ 数据已保存")

# ==================== 9.5. 提取Figure 4关键输出（用于后续研究） ====================
print("\n" + "="*80)
print("9.5. 提取Figure 4关键输出")
print("="*80)

# 1. Figure 4D: 敏感细胞特征基因列表（Top 50）
top50_markers = significant_markers.head(50) if len(significant_markers) >= 50 else significant_markers
top50_markers.to_csv(Config.RESULTS_DIR / 'Figure4D_sensitive_cell_markers_top50.csv', index=False)
print(f"✅ Top 50 敏感细胞特征基因已保存: {len(top50_markers)} genes")
print(f"   文件: {Config.RESULTS_DIR / 'Figure4D_sensitive_cell_markers_top50.csv'}")

# 2. Figure 4A: 化合物靶点活性评分
target_score_df = pd.DataFrame({
    'cell_barcode': adata.obs_names,
    'target_activity_score': adata.obs['target_score'].values,
    'donor_id': adata.obs['donor_id'].values,
    'disease': adata.obs['disease'].values,
    'cell_type': adata.obs['author_cell_type'].values
})
target_score_df.to_csv(Config.RESULTS_DIR / 'Figure4A_target_activity_scores.csv', index=False)
print(f"✅ 化合物靶点活性评分已保存: {len(target_score_df)} cells")
print(f"   文件: {Config.RESULTS_DIR / 'Figure4A_target_activity_scores.csv'}")

# 3. Figure 4B: 敏感细胞标签（添加sensitive_cell列）
adata.obs['sensitive_cell'] = adata.obs['is_high_score']
sensitive_cells_df = pd.DataFrame({
    'cell_barcode': adata.obs_names,
    'sensitive_cell': adata.obs['sensitive_cell'].values,
    'target_score': adata.obs['target_score'].values,
    'donor_id': adata.obs['donor_id'].values,
    'disease': adata.obs['disease'].values,
    'cell_type': adata.obs['author_cell_type'].values
})
sensitive_cells_df.to_csv(Config.RESULTS_DIR / 'Figure4B_sensitive_cell_labels.csv', index=False)
print(f"✅ 敏感细胞标签已保存: {(adata.obs['sensitive_cell'] == True).sum()} sensitive / {len(adata.obs)} total")
print(f"   文件: {Config.RESULTS_DIR / 'Figure4B_sensitive_cell_labels.csv'}")

# 重新保存带有sensitive_cell标签的完整数据
adata.write(Config.PROCESSED_DIR / 'adata_final_with_sensitive_labels.h5ad')
print(f"✅ 带敏感细胞标签的完整数据已保存")
print(f"   文件: {Config.PROCESSED_DIR / 'adata_final_with_sensitive_labels.h5ad'}")

print("\n" + "="*80)
print("关键输出文件总结:")
print("="*80)
print(f"1. Top 50 敏感细胞特征基因: Figure4D_sensitive_cell_markers_top50.csv")
print(f"2. 化合物靶点活性评分: Figure4A_target_activity_scores.csv")
print(f"3. 敏感细胞标签: Figure4B_sensitive_cell_labels.csv")
print(f"4. 完整数据(含标签): adata_final_with_sensitive_labels.h5ad")
print("="*80)


report = {
    'total_cells': int(adata.n_obs),
    'total_genes': int(adata.n_vars),
    'high_score_cells': int((adata.obs['is_high_score'] == True).sum()),
    'high_score_percent': float((adata.obs['is_high_score'] == True).mean() * 100),
    'marker_genes': len(significant_markers) if len(significant_markers) > 0 else 0,
    'top_markers': significant_markers.head(10)['gene'].tolist() if len(significant_markers) > 0 else []
}

with open(Config.RESULTS_DIR / 'analysis_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*80)
print("✅ 分析完成！所有结果已保存")
print("="*80)
print(f"📁 图片目录: {Config.FIG_DIR}")
print(f"📁 结果目录: {Config.RESULTS_DIR}")
print("="*80)