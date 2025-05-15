import pandas as pd

# python flights_expand.py

scale_1G = 385

path1 = "nsl-kdd-test.csv"

outpath1 = "../data-extension/1G/"
outpath2 = "../data-extension/10G/"
outpath3 = "../data-extension/"

data = pd.read_csv(path1)

X = data.drop("label", axis=1)
X_expanded = pd.concat([X] * scale_1G, ignore_index=True)
X_expanded.to_csv(outpath1 + "nsl-kdd.csv", index=False)

# expand to 10G
# X_expanded_1G = pd.read_csv(outpath1 + path1)
# X_expanded_10G = pd.concat([X_expanded_1G] * 10, ignore_index=True)
# X_expanded_10G.to_csv(outpath2 + path1, index=False)
