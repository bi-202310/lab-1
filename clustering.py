
# Lib Import
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Data Understanding

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
print(os.getcwd())

df_roads = pd.read_csv(f"{os.getcwd()}/data/BiciAlpes.csv", index_col=False, encoding="latin1", sep=";")
df_roads = df_roads.iloc[:, :-1]

print(df_roads.shape)
print(df_roads.columns)

# Completitud
# Porcentaje de campos de que estan vacios
# Se hace prueba de completitud a nivel de tabla


print((df_roads.isna().sum() / df_roads.shape[0]).sort_values(ascending=False))

# Unicidad
# Duplicados en la tabla


print("Repeated registries:\n",
	df_roads.loc[df_roads.duplicated(subset=df_roads.columns[0:], keep=False)].sum()
)