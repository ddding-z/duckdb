CREATE TABLE S_sales AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/walmart/data-extension/?/S_sales.csv', header=True, columns={
    'sid': 'INT64',
    'dept': 'INT64',
    'store': 'INT64',
    'purchaseid': 'VARCHAR'
});

CREATE TABLE R1_indicators AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/walmart/data-extension/R1_indicators.csv', header=True, columns={
    -- 'id': 'VARCHAR',
    'purchaseid': 'VARCHAR',
    'temperature_avg': 'FLOAT',
    'temperature_stdev': 'FLOAT',
    'fuel_price_avg': 'FLOAT',
    'fuel_price_stdev': 'FLOAT',
    'cpi_avg': 'FLOAT',
    'cpi_stdev': 'FLOAT',
    'unemployment_avg': 'FLOAT',
    'unemployment_stdev': 'FLOAT',
    'holidayfreq': 'FLOAT'
});

CREATE TABLE R2_stores AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/walmart/data-extension/R2_stores.csv', header=True, columns={
    'store': 'INT64',
    'type': 'INT64',
    'size': 'FLOAT'
});