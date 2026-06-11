# Validation Pipeline Step 1:
# Ratio Estimator for forest inventory attributes

import geopandas as gpd
import pandas as pd
import numpy as np

# Setup
polygons  = "" # Polygons on which the calculation should be based upon 
trees     = "" # sample point data 
output    = ""

polygon_id       = ""                # primary key for polygons
target_variable  = ""                # target for the ratio estimator
weight_a         = "GEWICHT_A"       # weight A from sampling data
grid_density     = 150               # distance between plot points [m]
outer_radius     = 12                # outer radious of sample plot [m]
height_col       = ""           # height attribute for layering

tree_species = ["Buche","Fichte","Kiefer","Eiche","Ahorn","Andere","Laerche","Birke","Erle","Douglasie","Tanne"]

auf_int = 10000 / (grid_density** 2) # Aufnahmeintensität - acquisition intensity

# Preprocessing
polygons = gpd.read_file(polygons)
trees    = gpd.read_file(trees).pipe(lambda d: d.to_crs(polygons.crs))
trees[height_col] = pd.to_numeric(trees[height_col], errors="coerce")
'''
_______________________________________________________________________________
# | Classify into height layers

      ≤ 1/3 · h_max  → lower layer
      ≤ 2/3 · h_max  → intermediate layer
        > 2/3 · h_max  → main layer
   Plots with < 3 trees --> all main layer
_______________________________________________________________________________
'''
trees["_h_max"]   = trees.groupby("STP_NR")[height_col].transform("max")
trees["_h_count"] = trees.groupby("STP_NR")[height_col].transform("count")
trees["_h_rel"]   = trees[height_col] / trees["_h_max"]

trees["Schicht"] = pd.cut(
    trees["_h_rel"],
    bins=[0, 1/3, 2/3, 1.0],
    labels=["Unterschicht", "Zwischenschicht", "Hauptschicht"],
    include_lowest=True
).astype(str)

# Plots with  < 3 trees
trees.loc[trees["_h_count"] < 3, "Schicht"] = "Hauptschicht"
trees = trees.drop(columns=["_h_max", "_h_count", "_h_rel"])

# Calculate tree object positions
trees["azimut_rad"] = trees["AZIMUT"] * np.pi / 200
trees["tree_x"]     = trees["IST_X"] + trees["ENTFERNUNG"] * np.sin(trees["azimut_rad"])
trees["tree_y"]     = trees["IST_Y"] + trees["ENTFERNUNG"] * np.cos(trees["azimut_rad"])
trees["geometry"]   = gpd.points_from_xy(trees.tree_x, trees.tree_y)
trees = gpd.GeoDataFrame(trees, geometry="geometry", crs=polygons.crs)

# Create plots and assign trees
plots = (trees[["STP_NR","IST_X","IST_Y"]].drop_duplicates()
         .assign(geometry=lambda d: gpd.points_from_xy(d.IST_X, d.IST_Y)))
plots = gpd.GeoDataFrame(plots, geometry="geometry", crs=polygons.crs)

plots = plots.drop(columns=[c for c in plots.columns if "index_right" in c], errors="ignore")
trees = trees.drop(columns=[c for c in trees.columns if "index_right" in c], errors="ignore")

plots = gpd.sjoin(plots, polygons[[polygon_id,"geometry"]], predicate="within")
trees = gpd.sjoin(trees, polygons[[polygon_id,"geometry"]], predicate="within")

trees = (trees
         .merge(plots[["STP_NR", polygon_id]], on="STP_NR", suffixes=("_tree","_plot"))
         .query(f"{polygon_id}_tree == {polygon_id}_plot"))

# Filtering and parsing
trees["Baumart"] = trees["BA_parsed"].str.capitalize()
trees = trees[trees["Baumart"].isin(tree_species)].copy()
trees = trees[trees["Schicht"] == "Hauptschicht"].copy()

'''
_______________________________________________________________________________
Local density of target variable per plot per tree species --> Eq. 2.5 / Sampling techniques for forest inventories (Mandalaz, 2007)

    w_i    = weight_a_i · auf_int
    Y_i    = target_variable_i · w_i
    Y_s(x) = Σ_{i ∈ s₂(x), art=s}  Y_i
    Y(x)   = Σ_{i ∈ s₂(x)}         Y_i
_______________________________________________________________________________
'''
trees["w_i"] = trees[weight_a] * auf_int
trees["Y_i"] = trees[target_variable] * trees["w_i"]

Y_by_species = (trees
                .groupby(["STP_NR","Baumart"])["Y_i"]
                .sum()
                .unstack(fill_value=0.0)
                .reindex(columns=tree_species, fill_value=0.0))
Y_by_species["Y_total"] = Y_by_species[tree_species].sum(axis=1)
print(Y_by_species.head())

'''
_______________________________________________________________________________
Boundary correction b(x) R-Package forestinventory Eq. 3 --> optional use later

    b(x) = λ(circle(x, outer_radius) ∩ polygon) / λ(circle)
_______________________________________________________________________________


OUTER_CIRCLE_AREA = np.pi * outer_radius ** 2
poly_geom = polygons.set_index(polygon_id)["geometry"]

plots["circle"] = plots.geometry.buffer(outer_radius)
plots["b_x"] = plots.apply(
    lambda r: r["circle"].intersection(poly_geom[r[polygon_id]]).area / OUTER_CIRCLE_AREA,
    axis=1
)

'''

plots["b_x"] = 1.0

Y_wx = Y_by_species.join(plots.set_index("STP_NR")[[polygon_id, "b_x"]])

print(plots.head())

'''
_______________________________________________________________________________
 Mean local density per polygon

    Ȳ_s(G) = (1/n_G) · Σ_{x∈G} b(x)·Y_s(x)   ← mean density for species s
     Ȳ(G)   = (1/n_G) · Σ_{x∈G} b(x)·Y(x)      ← mean sum of densities
________________________________________________________________________________
'''

for s in tree_species:
    Y_wx["num_" + s] =  Y_wx[s]
Y_wx["denom"] = Y_wx["Y_total"]

print(Y_wx.head(20))

num_cols = ["num_" + s for s in tree_species]
cols_to_agg = num_cols + ["denom"]

Y_wx[cols_to_agg] = Y_wx[cols_to_agg].apply(pd.to_numeric, errors="coerce")

agg = Y_wx.groupby(polygon_id)[cols_to_agg].mean()
agg_plots = plots.groupby("weflbkz")[["b_x"]].sum()

# Normalisation
prop_cols = ["prop_" + s for s in tree_species]
denom_arr = agg["denom"].values

for num_col, prop_col in zip(num_cols, prop_cols):
    agg[prop_col] = np.where(denom_arr > 0, agg[num_col].values / denom_arr, np.nan)

row_sum = agg[prop_cols].sum(axis=1).values
for prop_col in prop_cols:
    agg[prop_col] = np.where(row_sum > 0, agg[prop_col].values / row_sum * 100, np.nan)

# Export with fractions and confidence per poly
result = polygons.merge(agg[prop_cols], on=polygon_id, how="left")
result = result.merge(agg_plots, on=polygon_id, how="left")
result["confidence"] = result["b_x"] / (result["geometry"].area / 10000)
fact_nom = np.max(result["confidence"])
result["confidence"] = result["confidence"] / fact_nom
print(result.head(50)[[polygon_id]+ ["confidence"] + prop_cols].round(1).to_string())
result.to_file(output, driver="GPKG")

