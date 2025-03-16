import pandas as pd

# python walmart_expand.py

scale_1G = 17

path1 = "S_sales.csv"
path2 = "R1_indicators.csv"
path3 = "R2_stores.csv"

outpath1 = "./data-extension/1G/"
outpath2 = "./data-extension/10G/"
outpath3 = "./data-extension/"

S_sales = pd.read_csv(path1)
R1_indicators = pd.read_csv(path2)
R2_stores = pd.read_csv(path3)

R1_indicators.to_csv(outpath3 + path2, index=False)
R2_stores.to_csv(outpath3 + path3, index=False)

S_sales['purchaseid'] = S_sales['purchaseid'].str.replace("'", "")

X = S_sales.drop('weekly_sales', axis=1)
X_expanded = pd.concat([X] * scale_1G, ignore_index=True)

data = pd.merge(pd.merge(X_expanded , R1_indicators, how = 'inner'), R2_stores, how = 'inner')
data.to_csv('walmarts_1G.csv', index=False)

X_expanded.to_csv(outpath1 + path1, index=False)

# expand to 10G
# X_expanded_1G = pd.read_csv(outpath1 + path1)
# X_expanded_10G = pd.concat([X_expanded_1G] * 10, ignore_index=True)
# X_expanded_10G.to_csv(outpath2 + path1, index=False)

