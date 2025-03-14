CREATE TABLE bike_sharing_demand AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/bike_sharing_demand/data-extension/?/bike_sharing_demand.csv', header=True, columns={
    'datetime': 'DATETIME',
    'season': 'INT64',
    'holiday': 'INT64',
    'workingday': 'INT64',
    'weather': 'INT64',
    'temp': 'FLOAT',
    'atemp': 'FLOAT',
    'humidity': 'FLOAT',
    'windspeed': 'FLOAT',
    'casual': 'FLOAT',
    'registered': 'FLOAT'
});