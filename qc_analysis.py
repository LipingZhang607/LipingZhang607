#!/usr/bin/env python3
"""
单细胞数据质控分析脚本
针对 SLE vs Normal PBMC 数据 (.h5ad 文件)
作者：LipingZhang607
GitHub: https://github.com/LipingZhang607/LipingZhang607
项目：SLE PBMC 单细胞分析
对应 Figure 4 分析

第0步：数据质控
- 细胞质控
- 去除死细胞和双细胞
- 过滤线粒体基因>10%的细胞
- 生成质控报告
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
np.random.seed(42)
sc.settings.verbosity = 3  # 详细输出
sc.logging.print_header()

# 设置工作目录和路径
WORK_DIR = "~/statics/GEO_data/GSE/figure4"
DATA_DIR = os.path.join(WORK_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "QC")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables", "QC")

# 创建目录
for dir_path in [DATA_DIR, os.path.join(DATA_DIR, "raw"), os.path.join(DATA_DIR, "processed"),
                 RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def setup_plotting():
    """设置绘图参数"""
    sc.settings.figdir = FIGURES_DIR
    sc.settings.set_figure_params(
        dpi=300,
        dpi_save=300,
        facecolor='white',
        frameon=False,
        vector_friendly=True,
        fontsize=14
    )
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    plt.rcParams['figure.autolayout'] = True

def load_and_qc_data():
    """
    加载数据并进行质控分析
    """
    print("="*80)
    print("步骤1: 加载数据")
    print("="*80)
    
    # 加载.h5ad文件
    print(f"加载数据从: {RAW_DATA_PATH}")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"数据文件不存在: {RAW_DATA_PATH}")
    
    adata = sc.read_h5ad(RAW_DATA_PATH)
    print(f"原始数据维度: {adata.shape[0]} 细胞, {adata.shape[1]} 基因")
    
    # 检查数据格式
    print("\n数据基本信息:")
    print(f"AnnData 对象类型: {type(adata)}")
    print(f"稀疏矩阵: {sparse.issparse(adata.X)}")
    if hasattr(adata, 'obs') and adata.obs.shape[1] > 0:
        print(f"观察变量 (obs) 列: {list(adata.obs.columns)}")
    if hasattr(adata, 'var') and adata.var.shape[1] > 0:
        print(f"特征变量 (var) 列: {list(adata.var.columns)}")
    
    return adata

def calculate_qc_metrics(adata):
    """
    计算质控指标
    """
    print("\n" + "="*80)
    print("步骤2: 计算质控指标")
    print("="*80)
    
    # 计算基本的QC指标
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['mt'],  # 线粒体基因
        percent_top=None,
        log1p=False,
        inplace=True
    )
    
    # 如果没有mt列，手动计算线粒体基因百分比
    if 'mt' not in adata.var.columns:
        print("检测线粒体基因...")
        # 常见线粒体基因前缀
        mt_prefixes = ['MT-', 'mt-', 'Mt-', '^MT', '^mt']
        for prefix in mt_prefixes:
            adata.var['mt'] = adata.var_names.str.startswith(prefix.upper()) | \
                              adata.var_names.str.startswith(prefix.lower())
            if adata.var['mt'].sum() > 0:
                print(f"使用前缀 '{prefix}' 找到 {adata.var['mt'].sum()} 个线粒体基因")
                break
        
        if 'pct_counts_mt' not in adata.obs.columns:
            sc.pp.calculate_qc_metrics(
                adata,
                qc_vars=['mt'],
                percent_top=None,
                log1p=False,
                inplace=True
            )
    
    # 计算其他重要的QC指标
    print("\n计算其他QC指标...")
    adata.obs['log_n_counts'] = np.log1p(adata.obs['total_counts'])
    adata.obs['log_n_genes'] = np.log1p(adata.obs['n_genes_by_counts'])
    
    # 如果有样本信息，也计算每个样本的QC
    if 'sample' in adata.obs.columns or 'Sample' in adata.obs.columns:
        sample_col = 'sample' if 'sample' in adata.obs.columns else 'Sample'
        print(f"按样本 {sample_col} 分组分析")
    
    return adata

def visualize_qc_metrics(adata, save_prefix='QC_metrics'):
    """
    可视化质控指标
    """
    print("\n" + "="*80)
    print("步骤3: 可视化质控指标")
    print("="*80)
    
    # 设置颜色
    if 'condition' in adata.obs.columns:
        palette = {'SLE': '#E64B35', 'Normal': '#4DBBD5'}
    elif 'disease' in adata.obs.columns:
        palette = {'SLE': '#E64B35', 'Normal': '#4DBBD5'}
    else:
        palette = None
    
    # 1. 基本的QC指标分布
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # n_counts分布
    ax = axes[0]
    ax.hist(adata.obs['total_counts'], bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('总计数 (n_counts)')
    ax.set_ylabel('细胞数')
    ax.set_title('总计数分布')
    ax.axvline(x=np.percentile(adata.obs['total_counts'], 5), color='red', linestyle='--', label='5%分位数')
    ax.axvline(x=np.percentile(adata.obs['total_counts'], 95), color='red', linestyle='--', label='95%分位数')
    ax.legend()
    
    # n_genes分布
    ax = axes[1]
    ax.hist(adata.obs['n_genes_by_counts'], bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('检测到的基因数')
    ax.set_ylabel('细胞数')
    ax.set_title('检测基因数分布')
    ax.axvline(x=np.percentile(adata.obs['n_genes_by_counts'], 5), color='red', linestyle='--')
    ax.axvline(x=np.percentile(adata.obs['n_genes_by_counts'], 95), color='red', linestyle='--')
    
    # 线粒体基因百分比分布
    ax = axes[2]
    ax.hist(adata.obs['pct_counts_mt'], bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('线粒体基因百分比 (%)')
    ax.set_ylabel('细胞数')
    ax.set_title('线粒体基因百分比分布')
    ax.axvline(x=10, color='red', linestyle='--', label='10% 阈值')
    ax.legend()
    
    # 三个指标的关系图
    ax = axes[3]
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=ax, show=False, 
                  color='condition' if 'condition' in adata.obs.columns else None,
                  palette=palette)
    ax.axhline(y=10, color='red', linestyle='--')
    ax.set_title('总计数 vs 线粒体基因%')
    
    ax = axes[4]
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=ax, show=False,
                  color='condition' if 'condition' in adata.obs.columns else None,
                  palette=palette)
    ax.set_title('总计数 vs 检测基因数')
    
    ax = axes[5]
    sc.pl.scatter(adata, x='n_genes_by_counts', y='pct_counts_mt', ax=ax, show=False,
                  color='condition' if 'condition' in adata.obs.columns else None,
                  palette=palette)
    ax.axhline(y=10, color='red', linestyle='--')
    ax.set_title('检测基因数 vs 线粒体基因%')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'{save_prefix}_distributions.pdf'), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, f'{save_prefix}_distributions.png'), dpi=300)
    plt.show()
    
    # 2. 小提琴图 - 按条件分组
    if 'condition' in adata.obs.columns or 'disease' in adata.obs.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 确定分组列和标签
        if 'condition' in adata.obs.columns:
            group_col = 'condition'
            groups = ['Normal', 'SLE']
        else:
            group_col = 'disease'
            groups = ['Normal', 'SLE']
        
        # 总计数小提琴图
        ax = axes[0]
        sns.violinplot(x=group_col, y='total_counts', data=adata.obs, 
                      order=groups, ax=ax, palette=palette, cut=0)
        ax.set_yscale('log')
        ax.set_title('总计数分布 (按条件)')
        ax.set_ylabel('总计数 (log scale)')
        
        # 检测基因数小提琴图
        ax = axes[1]
        sns.violinplot(x=group_col, y='n_genes_by_counts', data=adata.obs,
                      order=groups, ax=ax, palette=palette, cut=0)
        ax.set_yscale('log')
        ax.set_title('检测基因数分布 (按条件)')
        ax.set_ylabel('基因数 (log scale)')
        
        # 线粒体百分比小提琴图
        ax = axes[2]
        sns.violinplot(x=group_col, y='pct_counts_mt', data=adata.obs,
                      order=groups, ax=ax, palette=palette, cut=0)
        ax.axhline(y=10, color='red', linestyle='--', label='10% 阈值')
        ax.set_title('线粒体基因百分比分布 (按条件)')
        ax.set_ylabel('线粒体基因%')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'{save_prefix}_by_condition.pdf'), dpi=300)
        plt.savefig(os.path.join(FIGURES_DIR, f'{save_prefix}_by_condition.png'), dpi=300)
        plt.show()

def filter_cells(adata):
    """
    过滤细胞：去除死细胞、双细胞、线粒体基因>10%的细胞
    """
    print("\n" + "="*80)
    print("步骤4: 过滤低质量细胞")
    print("="*80)
    
    # 记录过滤前的细胞数
    n_cells_before = adata.shape[0]
    n_genes_before = adata.shape[1]
    
    print(f"过滤前: {n_cells_before} 个细胞, {n_genes_before} 个基因")
    
    # 1. 首先过滤掉线粒体基因>10%的细胞 (防止审稿人bb的关键步骤)
    mt_threshold = 10  # 10% 阈值
    mt_filter = adata.obs['pct_counts_mt'] <= mt_threshold
    adata_mt_filtered = adata[mt_filter, :].copy()
    n_mt_removed = n_cells_before - adata_mt_filtered.shape[0]
    
    print(f"\n1. 线粒体基因过滤 (> {mt_threshold}%):")
    print(f"   移除了 {n_mt_removed} 个细胞")
    print(f"   保留 {adata_mt_filtered.shape[0]} 个细胞")
    print(f"   移除比例: {n_mt_removed/n_cells_before*100:.2f}%")
    
    # 2. 基于总计数过滤 (去除死细胞和低质量细胞)
    # 计算自动阈值: 低于5%分位数或高于95%分位数
    lower_count = np.percentile(adata_mt_filtered.obs['total_counts'], 5)
    upper_count = np.percentile(adata_mt_filtered.obs['total_counts'], 95)
    
    count_filter = (adata_mt_filtered.obs['total_counts'] >= lower_count) & \
                   (adata_mt_filtered.obs['total_counts'] <= upper_count)
    adata_count_filtered = adata_mt_filtered[count_filter, :].copy()
    n_count_removed = adata_mt_filtered.shape[0] - adata_count_filtered.shape[0]
    
    print(f"\n2. 总计数过滤 ({lower_count:.0f} - {upper_count:.0f}):")
    print(f"   移除了 {n_count_removed} 个细胞")
    print(f"   保留 {adata_count_filtered.shape[0]} 个细胞")
    
    # 3. 基于检测到的基因数过滤
    lower_genes = np.percentile(adata_count_filtered.obs['n_genes_by_counts'], 5)
    upper_genes = np.percentile(adata_count_filtered.obs['n_genes_by_counts'], 95)
    
    gene_filter = (adata_count_filtered.obs['n_genes_by_counts'] >= lower_genes) & \
                  (adata_count_filtered.obs['n_genes_by_counts'] <= upper_genes)
    adata_filtered = adata_count_filtered[gene_filter, :].copy()
    n_gene_removed = adata_count_filtered.shape[0] - adata_filtered.shape[0]
    
    print(f"\n3. 检测基因数过滤 ({lower_genes:.0f} - {upper_genes:.0f}):")
    print(f"   移除了 {n_gene_removed} 个细胞")
    print(f"   保留 {adata_filtered.shape[0]} 个细胞")
    
    # 总结
    total_removed = n_cells_before - adata_filtered.shape[0]
    print(f"\n过滤总结:")
    print(f"   总共移除了 {total_removed} 个细胞")
    print(f"   最终保留 {adata_filtered.shape[0]} 个细胞")
    print(f"   总体移除比例: {total_removed/n_cells_before*100:.2f}%")
    
    # 保存过滤记录
    filter_stats = {
        'n_cells_before': n_cells_before,
        'n_cells_after': adata_filtered.shape[0],
        'n_mt_removed': n_mt_removed,
        'n_count_removed': n_count_removed,
        'n_gene_removed': n_gene_removed,
        'total_removed': total_removed,
        'mt_threshold': mt_threshold,
        'count_threshold_lower': lower_count,
        'count_threshold_upper': upper_count,
        'gene_threshold_lower': lower_genes,
        'gene_threshold_upper': upper_genes
    }
    
    return adata_filtered, filter_stats

def visualize_filtering_effects(adata_before, adata_after, filter_stats):
    """
    可视化过滤效果
    """
    print("\n" + "="*80)
    print("步骤5: 可视化过滤效果")
    print("="*80)
    
    # 创建比较图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 总计数分布对比
    ax = axes[0, 0]
    ax.hist(adata_before.obs['total_counts'], bins=100, alpha=0.5, 
            label=f'过滤前 ({filter_stats["n_cells_before"]} cells)', 
            edgecolor='black')
    ax.hist(adata_after.obs['total_counts'], bins=100, alpha=0.5, 
            label=f'过滤后 ({filter_stats["n_cells_after"]} cells)',
            edgecolor='black', color='red')
    ax.set_xlabel('总计数')
    ax.set_ylabel('细胞数')
    ax.set_title('总计数分布对比')
    ax.legend()
    ax.set_yscale('log')
    
    # 添加阈值线
    ax.axvline(x=filter_stats['count_threshold_lower'], color='blue', 
               linestyle='--', alpha=0.7, label='下阈值')
    ax.axvline(x=filter_stats['count_threshold_upper'], color='blue', 
               linestyle='--', alpha=0.7, label='上阈值')
    
    # 2. 检测基因数分布对比
    ax = axes[0, 1]
    ax.hist(adata_before.obs['n_genes_by_counts'], bins=100, alpha=0.5, 
            label='过滤前', edgecolor='black')
    ax.hist(adata_after.obs['n_genes_by_counts'], bins=100, alpha=0.5, 
            label='过滤后', edgecolor='black', color='red')
    ax.set_xlabel('检测到的基因数')
    ax.set_ylabel('细胞数')
    ax.set_title('检测基因数分布对比')
    ax.legend()
    ax.set_yscale('log')
    ax.axvline(x=filter_stats['gene_threshold_lower'], color='blue', 
               linestyle='--', alpha=0.7)
    ax.axvline(x=filter_stats['gene_threshold_upper'], color='blue', 
               linestyle='--', alpha=0.7)
    
    # 3. 线粒体百分比分布对比
    ax = axes[0, 2]
    ax.hist(adata_before.obs['pct_counts_mt'], bins=100, alpha=0.5, 
            label='过滤前', edgecolor='black')
    ax.hist(adata_after.obs['pct_counts_mt'], bins=100, alpha=0.5, 
            label='过滤后', edgecolor='black', color='red')
    ax.set_xlabel('线粒体基因百分比 (%)')
    ax.set_ylabel('细胞数')
    ax.set_title('线粒体基因百分比分布对比')
    ax.legend()
    ax.set_yscale('log')
    ax.axvline(x=filter_stats['mt_threshold'], color='blue', 
               linestyle='--', alpha=0.7, label='10% 阈值')
    
    # 4. 散点图对比
    ax = axes[1, 0]
    ax.scatter(adata_before.obs['total_counts'], adata_before.obs['pct_counts_mt'],
               alpha=0.3, s=1, label='过滤前', c='gray')
    ax.scatter(adata_after.obs['total_counts'], adata_after.obs['pct_counts_mt'],
               alpha=0.5, s=1, label='过滤后', c='red')
    ax.set_xlabel('总计数')
    ax.set_ylabel('线粒体基因%')
    ax.set_title('总计数 vs 线粒体基因% (对比)')
    ax.legend(markerscale=5)
    ax.axhline(y=filter_stats['mt_threshold'], color='blue', 
               linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    
    # 5. 移除细胞统计
    ax = axes[1, 1]
    removal_causes = ['线粒体基因>10%', '总计数异常', '基因数异常']
    removal_counts = [filter_stats['n_mt_removed'], 
                      filter_stats['n_count_removed'], 
                      filter_stats['n_gene_removed']]
    
    bars = ax.bar(removal_causes, removal_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel('移除细胞数')
    ax.set_title('按原因移除细胞数')
    ax.set_xticklabels(removal_causes, rotation=45, ha='right')
    
    # 在柱子上添加数值
    for bar, count in zip(bars, removal_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(count)}', ha='center', va='bottom')
    
    # 6. 过滤前后细胞数对比
    ax = axes[1, 2]
    stages = ['过滤前', '过滤后']
    cell_counts = [filter_stats['n_cells_before'], filter_stats['n_cells_after']]
    colors = ['#95a5a6', '#2ecc71']
    
    bars = ax.bar(stages, cell_counts, color=colors)
    ax.set_ylabel('细胞数')
    ax.set_title('过滤前后细胞数对比')
    
    # 在柱子上添加数值和百分比
    for bar, count, stage in zip(bars, cell_counts, stages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(count):,}', ha='center', va='bottom')
    
    # 添加移除百分比
    removal_pct = filter_stats['total_removed'] / filter_stats['n_cells_before'] * 100
    ax.text(0.5, 0.95, f'移除比例: {removal_pct:.2f}%',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'QC_filtering_effects.pdf'), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'QC_filtering_effects.png'), dpi=300)
    plt.show()
    
    # 按条件分组展示过滤效果
    if 'condition' in adata_before.obs.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 按条件统计过滤前后
        condition_counts_before = adata_before.obs['condition'].value_counts()
        condition_counts_after = adata_after.obs['condition'].value_counts()
        
        ax = axes[0]
        x = np.arange(len(condition_counts_before))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, condition_counts_before.values, width, 
                      label='过滤前', color='#95a5a6')
        bars2 = ax.bar(x + width/2, condition_counts_after.values, width, 
                      label='过滤后', color='#2ecc71')
        
        ax.set_xlabel('条件')
        ax.set_ylabel('细胞数')
        ax.set_title('各条件下细胞数变化')
        ax.set_xticks(x)
        ax.set_xticklabels(condition_counts_before.index)
        ax.legend()
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=8)
        
        # 计算并显示保留比例
        ax = axes[1]
        retention_rates = []
        for condition in condition_counts_before.index:
            if condition in condition_counts_after.index:
                retention_rate = condition_counts_after[condition] / condition_counts_before[condition] * 100
            else:
                retention_rate = 0
            retention_rates.append(retention_rate)
        
        bars = ax.bar(condition_counts_before.index, retention_rates, color='#3498db')
        ax.set_xlabel('条件')
        ax.set_ylabel('保留比例 (%)')
        ax.set_title('各条件下细胞保留比例')
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100%')
        
        # 添加百分比标签
        for bar, rate in zip(bars, retention_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'QC_filtering_by_condition.pdf'), dpi=300)
        plt.savefig(os.path.join(FIGURES_DIR, 'QC_filtering_by_condition.png'), dpi=300)
        plt.show()

def save_qc_results(adata_before, adata_after, filter_stats):
    """
    保存质控结果
    """
    print("\n" + "="*80)
    print("步骤6: 保存质控结果")
    print("="*80)
    
    # 1. 保存过滤后的数据
    processed_path = os.path.join(DATA_DIR, "processed", "SLE_PBMC_filtered.h5ad")
    adata_after.write_h5ad(processed_path)
    print(f"✓ 过滤后的数据保存至: {processed_path}")
    
    # 2. 保存过滤统计
    stats_df = pd.DataFrame([filter_stats])
    stats_path = os.path.join(TABLES_DIR, "QC_filtering_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ 过滤统计保存至: {stats_path}")
    
    # 3. 保存详细的QC指标表
    qc_table_before = adata_before.obs[['total_counts', 'n_genes_by_counts', 'pct_counts_mt']].describe()
    qc_table_after = adata_after.obs[['total_counts', 'n_genes_by_counts', 'pct_counts_mt']].describe()
    
    qc_table_before_path = os.path.join(TABLES_DIR, "QC_metrics_before_filtering.csv")
    qc_table_after_path = os.path.join(TABLES_DIR, "QC_metrics_after_filtering.csv")
    
    qc_table_before.to_csv(qc_table_before_path)
    qc_table_after.to_csv(qc_table_after_path)
    
    print(f"✓ 过滤前QC指标保存至: {qc_table_before_path}")
    print(f"✓ 过滤后QC指标保存至: {qc_table_after_path}")
    
    # 4. 生成质控报告文本
    report_path = os.path.join(RESULTS_DIR, "QC_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("单细胞数据质控报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. 数据基本信息\n")
        f.write(f"   原始数据: {adata_before.shape[0]} 细胞, {adata_before.shape[1]} 基因\n")
        f.write(f"   过滤后数据: {adata_after.shape[0]} 细胞, {adata_after.shape[1]} 基因\n\n")
        
        f.write("2. 过滤标准\n")
        f.write(f"   线粒体基因阈值: > {filter_stats['mt_threshold']}%\n")
        f.write(f"   总计数阈值: {filter_stats['count_threshold_lower']:.0f} - {filter_stats['count_threshold_upper']:.0f}\n")
        f.write(f"   检测基因数阈值: {filter_stats['gene_threshold_lower']:.0f} - {filter_stats['gene_threshold_upper']:.0f}\n\n")
        
        f.write("3. 过滤结果\n")
        f.write(f"   移除线粒体基因>10%的细胞: {filter_stats['n_mt_removed']} 个\n")
        f.write(f"   移除总计数异常的细胞: {filter_stats['n_count_removed']} 个\n")
        f.write(f"   移除基因数异常的细胞: {filter_stats['n_gene_removed']} 个\n")
        f.write(f"   总共移除: {filter_stats['total_removed']} 个细胞\n")
        f.write(f"   保留: {filter_stats['n_cells_after']} 个细胞\n")
        f.write(f"   总体移除比例: {filter_stats['total_removed']/filter_stats['n_cells_before']*100:.2f}%\n\n")
        
        f.write("4. 质控指标统计\n")
        f.write("   过滤前:\n")
        f.write(f"      - 平均总计数: {adata_before.obs['total_counts'].mean():.0f}\n")
        f.write(f"      - 平均检测基因数: {adata_before.obs['n_genes_by_counts'].mean():.0f}\n")
        f.write(f"      - 平均线粒体基因%: {adata_before.obs['pct_counts_mt'].mean():.2f}%\n\n")
        
        f.write("   过滤后:\n")
        f.write(f"      - 平均总计数: {adata_after.obs['total_counts'].mean():.0f}\n")
        f.write(f"      - 平均检测基因数: {adata_after.obs['n_genes_by_counts'].mean():.0f}\n")
        f.write(f"      - 平均线粒体基因%: {adata_after.obs['pct_counts_mt'].mean():.2f}%\n\n")
        
        if 'condition' in adata_before.obs.columns:
            f.write("5. 按条件统计\n")
            for condition in adata_before.obs['condition'].unique():
                n_before = (adata_before.obs['condition'] == condition).sum()
                n_after = (adata_after.obs['condition'] == condition).sum()
                retention = n_after / n_before * 100 if n_before > 0 else 0
                f.write(f"   {condition}: {n_before} → {n_after} 细胞 (保留 {retention:.1f}%)\n")
    
    print(f"✓ 质控报告保存至: {report_path}")
    
    # 5. 打印关键信息
    print("\n关键质控指标:")
    print(f"   最终数据维度: {adata_after.shape[0]} 细胞, {adata_after.shape[1]} 基因")
    print(f"   平均线粒体基因%: {adata_after.obs['pct_counts_mt'].mean():.2f}%")
    print(f"   最大线粒体基因%: {adata_after.obs['pct_counts_mt'].max():.2f}%")
    print(f"   线粒体基因% >10%的细胞数: {(adata_after.obs['pct_counts_mt'] > 10).sum()}")

def main():
    """
    主函数：执行完整质控流程
    """
    print("="*80)
    print("单细胞数据质控分析 - 第0步")
    print("针对 SLE vs Normal PBMC 数据")
    print("GitHub: https://github.com/LipingZhang607/LipingZhang607")
    print("="*80)
    
    # 设置绘图
    setup_plotting()
    
    try:
        # 1. 加载数据
        adata = load_and_qc_data()
        
        # 2. 计算QC指标
        adata = calculate_qc_metrics(adata)
        
        # 3. 可视化原始QC指标
        visualize_qc_metrics(adata, save_prefix='QC_before_filtering')
        
        # 4. 过滤细胞
        adata_before = adata.copy()  # 保存原始数据用于对比
        adata_filtered, filter_stats = filter_cells(adata)
        
        # 5. 可视化过滤效果
        visualize_filtering_effects(adata_before, adata_filtered, filter_stats)
        
        # 6. 保存结果
        save_qc_results(adata_before, adata_filtered, filter_stats)
        
        print("\n" + "="*80)
        print("质控分析完成！")
        print("="*80)
        
        return adata_filtered
        
    except Exception as e:
        print(f"\n错误发生: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='单细胞数据质控分析')
    parser.add_argument('--data_path', type=str, default=RAW_DATA_PATH,
                       help='输入.h5ad文件路径')
    parser.add_argument('--mt_threshold', type=float, default=10.0,
                       help='线粒体基因百分比阈值')
    parser.add_argument('--output_dir', type=str, default=WORK_DIR,
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 更新路径
    if args.data_path != RAW_DATA_PATH:
        RAW_DATA_PATH = args.data_path
    
    # 运行主函数
    filtered_adata = main()