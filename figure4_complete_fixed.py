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
print(f"前10个: {existing_targets[:10]}")

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

# 先计算高变基因以加速差异分析
print("Identifying highly variable genes...")
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
print(f"HVG count: {adata.var['highly_variable'].sum()}")

# 创建高变基因子集用于差异分析（大幅加速）
adata_hvg = adata[:, adata.var['highly_variable']].copy()
adata_hvg.obs['is_high_score'] = adata_hvg.obs['is_high_score'].astype('category')

# 差异表达分析（仅对高变基因）
print("Running differential expression analysis on HVGs...")
sc.tl.rank_genes_groups(adata_hvg, groupby='is_high_score',
                       method='wilcoxon',
                       reference='rest',
                       n_genes=200,
                       use_raw=False)

# 提取结果
result = adata_hvg.uns['rank_genes_groups']
markers = pd.DataFrame({
    'gene': result['names']['True'],
    'logFC': result['logfoldchanges']['True'],
    'p_val_adj': result['pvals_adj']['True']
})

# 过滤显著基因
significant_markers = markers[
    (markers['logFC'] > 0.25) &
    (markers['p_val_adj'] < 0.01)
].copy()

print(f"显著特征基因: {len(significant_markers)}")
print(significant_markers.head(10))

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
    gene_list = significant_markers.head(50)['gene'].tolist()
    
    try:
        # GO富集分析
        go_enrich = gp.enrichr(gene_list=gene_list,
                              gene_sets=['GO_Biological_Process_2021',
                                       'KEGG_2021_Human'],
                              organism='Human',
                              outdir=Config.RESULTS_DIR / 'enrichr_results',
                              no_plot=True)
        
        if go_enrich.results is not None and len(go_enrich.results) > 0:
            results = go_enrich.results
            print(f"\n富集结果:")
            print(results[['Term', 'P-value', 'Genes']].head(10))
            
            # 保存结果
            results.to_csv(Config.RESULTS_DIR / 'enrichment_results.csv', index=False)
            
            # 绘制气泡图
            top_results = results.head(15).copy()
            top_results['gene_count'] = top_results['Genes'].str.split(';').str.len()
            top_results['-log10_pval'] = -np.log10(top_results['P-value'])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(top_results['gene_count'], 
                               top_results['Term'],
                               s=top_results['-log10_pval'] * 50,
                               c=top_results['P-value'],
                               cmap='viridis_r',
                               alpha=0.7,
                               edgecolors='black',
                               linewidth=1)
            
            plt.colorbar(scatter, ax=ax, label='P-value')
            ax.set_xlabel('Gene Count')
            ax.set_ylabel('Pathway')
            ax.set_title('Functional Enrichment Analysis of Marker Genes')
            
            plt.tight_layout()
            plt.savefig(Config.FIG_DIR / 'Figure4E_enrichment_bubble.pdf', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Figure 4E已保存")
            
    except Exception as e:
        print(f"富集分析失败: {e}")

# ==================== 9. 保存所有结果 ====================
print("\n" + "="*80)
print("9. 保存处理后的数据")
print("="*80)

adata.write(Config.PROCESSED_DIR / 'adata_final.h5ad')
high_score_adata.write(Config.PROCESSED_DIR / 'adata_high_score.h5ad')

print(f"✅ 数据已保存")

# ==================== 10. 生成报告 ====================
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