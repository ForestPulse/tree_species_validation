# Validation Pipeline Step 3:
# Modelling Ground-truth values from  BAZ forest inventory data and plot sampling

import pandas as pd
import geopandas as gpd
import numpy as np

# Input
input_gpkg = "" # Input from Step 2
output_gpkg = ""
baumarten_cols = ['buche', 'eiche', 'fichte', 'tanne', 'kiefer',
                  'ahorn', 'andere', 'erle', 'douglasie', 'laerche', 'birke']

boden_col           = "boden"
confidence_col      = "confidence"
coverage_col        = "coverage"

# Preprocessing
gdf = gpd.read_file(input_gpkg)
gdf.columns = [c.strip().lower() for c in gdf.columns]
gdf[confidence_col] = pd.to_numeric(gdf[confidence_col], errors="coerce")
gdf[coverage_col]   = pd.to_numeric(gdf[coverage_col],   errors="coerce")
for col in baumarten_cols:
    gdf[col]              = pd.to_numeric(gdf[col],              errors="coerce")
    gdf["prop_" + col]    = pd.to_numeric(gdf["prop_" + col],    errors="coerce")

# Calculate GT with confidence of plot sampling
gt_cols = []

for col in baumarten_cols:
    col_gt = f"gt_{col}"
    q1 = gdf[col].fillna(0)
    q2 = gdf["prop_" + col]

    gt = q1 + (q2.fillna(0) * gdf[confidence_col].fillna(0)) / 2
    gt[q2.isna()] = q1[q2.isna()]

    gdf[col_gt] = gt
    gt_cols.append(col_gt)

# Derive class "boden" from ground coverage
coverage        = gdf[coverage_col].clip(0, 1)
gdf["gt_boden"] = (1 - coverage) * 100
available       = coverage * 100

# Normalice tree species fractions to ground cover
gt_sum      = gdf[gt_cols].sum(axis=1).replace(0, np.nan)
normalized  = gdf[gt_cols].div(gt_sum, axis=0)
gdf[gt_cols] = normalized.multiply(available, axis=0).fillna(0)


# Stabilizing
total = gdf[gt_cols].sum(axis=1) + gdf["gt_boden"]
total = total.replace(0, np.nan)
gdf[gt_cols]     = gdf[gt_cols].div(total, axis=0) * 100
gdf["gt_boden"]  = gdf["gt_boden"].div(total) * 100
check = gdf[gt_cols].sum(axis=1) + gdf["gt_boden"]
print(f"– Min: {check.min():.2f}  Max: {check.max():.2f}  (Soll: 100)")

# Export
gdf.to_file(output_gpkg, driver="GPKG")
