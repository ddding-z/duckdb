CREATE TABLE wine_quality AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/wine_quality/data-extension/?/wine_quality.csv', header=True, columns={
    'fixed_acidity': 'FLOAT',
    'volatile_acidity': 'FLOAT',
    'citric_acid': 'FLOAT',
    'residual_sugar': 'FLOAT',
    'chlorides': 'FLOAT',
    'free_sulfur_dioxide': 'FLOAT',
    'total_sulfur_dioxide': 'FLOAT',
    'density': 'FLOAT',
    'pH': 'FLOAT',
    'sulphates': 'FLOAT',
    'alcohol': 'FLOAT'
});