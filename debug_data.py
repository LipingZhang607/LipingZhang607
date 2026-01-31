# debug_data.py
import scanpy as sc
import pandas as pd

adata = sc.read_h5ad("data/raw/4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad")

print("数据形状:", adata.shape)
print("\nobs列:", list(adata.obs.columns))
print("\nvar列:", list(adata.var.columns))
print("\n基因名前5个:", list(adata.var_names[:5]))
print("\n是否稀疏矩阵:", hasattr(adata.X, 'A1'))