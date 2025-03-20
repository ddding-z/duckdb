CREATE TABLE sales AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/walmart_sales/data-extension/?/test.csv', header=True, columns={
    'Store': 'FLOAT',
    'Dept': 'FLOAT',
    'Date': 'DATE',
    'IsHoliday': 'BOOLEAN'
});

CREATE TABLE features AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/walmart_sales/data-extension/features.csv', header=True, columns={
    'Store': 'FLOAT',
    'Date': 'DATE',
    'Temperature': 'FLOAT',
    'Fuel_Price': 'FLOAT',
    'MarkDown1': 'FLOAT',
    'MarkDown2': 'FLOAT',
    'MarkDown3': 'FLOAT',
    'MarkDown4': 'FLOAT',
    'MarkDown5': 'FLOAT',
    'CPI': 'FLOAT',
    'Unemployment': 'FLOAT',
    'IsHoliday': 'BOOLEAN'
});

CREATE TABLE stores AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/walmart_sales/data-extension/stores.csv', header=True, columns={
    'Store': 'FLOAT',
    'Type': 'VARCHAR',
    'Size': 'FLOAT'
});