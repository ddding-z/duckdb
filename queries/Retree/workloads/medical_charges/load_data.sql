CREATE TABLE medical_charges AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/medical_charges/data-extension/?/medical_charges.csv', header=True, columns={
    'Total_Discharges': 'INT64',
    'Average_Covered_Charges': 'FLOAT',
    'Average_Medicare_Payments': 'FLOAT'
});