CREATE TABLE S_routes AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/flights/data-extension/?/S_routes.csv', header=True, columns={
    'airlineid': 'INT64',
    'sairportid': 'INT64',
    'dairportid': 'INT64'
});

CREATE TABLE R1_airlines AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/flights/data-extension/?/R1_airlines.csv', header=True, columns={
    'airlineid': 'INT64',
    'name1': 'INT64',
    'name2': 'VARCHAR',
    'name4': 'VARCHAR',
    'acountry': 'VARCHAR',
    'active': 'VARCHAR'
});

CREATE TABLE R2_sairports AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/flights/data-extension/?/R2_sairports.csv', header=True, columns={
    'sairportid': 'INT64',
    'scity': 'VARCHAR',
    'scountry': 'VARCHAR',
    'slatitude':'FLOAT',
    'slongitude':'FLOAT',
    'stimezone':'INT64',
    'sdst':'VARCHAR'
});

CREATE TABLE R3_dairports AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/flights/data-extension/?/R3_dairports.csv', header=True, columns={
    'dairportid': 'INT64',
    'dcity': 'VARCHAR',
    'dcountry': 'VARCHAR',
    'dlatitude':'FLOAT',
    'dlongitude':'FLOAT',
    'dtimezone':'INT64',
    'ddst':'VARCHAR'
});