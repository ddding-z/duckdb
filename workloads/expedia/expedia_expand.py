import pandas as pd

# python expedia_expand.py

scale_1G = 8

path1 = "S_listings.csv"
path2 = "R1_hotels.csv"
path3 = "R2_searches.csv"

outpath1 = "./data-extension/1G/"
outpath2 = "./data-extension/10G/"
outpath3 = "./data-extension/"

S_listings = pd.read_csv(path1)
R1_hotels = pd.read_csv(path2)
R2_searches = pd.read_csv(path3)

S_listings["srch_id"] = S_listings["srch_id"].str.replace("'", "")
S_listings["prop_id"] = S_listings["prop_id"].str.replace("'", "")
R1_hotels["prop_id"] = R1_hotels["prop_id"].str.replace("'", "")
R2_searches["srch_id"] = R2_searches["srch_id"].str.replace("'", "")

R1_hotels.to_csv(outpath3 + path2, index=False)
R2_searches.to_csv(outpath3 + path3, index=False)

X = S_listings.drop("position", axis=1)
X_expanded_1G = pd.concat([X] * scale_1G, ignore_index=True)

data = pd.merge(
    pd.merge(X_expanded_1G, R1_hotels, how="inner"), R2_searches, how="inner"
)
data.to_csv("expedia_1G.csv", index=False)
X_expanded_1G.to_csv(outpath1 + path1, index=False)

# expand to 10G
# X_expanded_1G = pd.read_csv(outpath1 + path1)
# X_expanded_10G = pd.concat([X_expanded_1G] * 10, ignore_index=True)
# X_expanded_10G.to_csv(outpath2 + path1, index=False)
