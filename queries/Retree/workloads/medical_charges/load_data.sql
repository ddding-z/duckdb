CREATE TABLE table1 AS
SELECT * FROM read_csv('/volumn/duckdb/data/dataset/expanded_data/medical_charges_?.csv', header=True, columns={
    'Total_Discharges': 'INT64',
    'Average_Covered_Charges': 'FLOAT',
    'Average_Medicare_Payments': 'FLOAT'
});