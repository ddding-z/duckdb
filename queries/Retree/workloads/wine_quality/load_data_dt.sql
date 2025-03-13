CREATE TABLE table1 AS
SELECT * FROM read_csv('/volumn/duckdb/data/dataset/expanded_data/wine_quality_?.csv', header=True, columns={
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