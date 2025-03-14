CREATE TABLE bike_sharing_demand AS
SELECT * FROM read_csv('/volumn/duckdb/data/dataset/expanded_data/bike_sharing_demand_?.csv', header=True, columns={
    'datetime': 'DATETIME',
    'season': 'INT64',
    'holiday': 'INT64',
    'workingday': 'INT64',
    'weather': 'INT64',
    'temp': 'FLOAT',
    'atemp': 'FLOAT',
    'humidity': 'INT64',
    'windspeed': 'FLOAT',
    'casual': 'INT64',
    'registered': 'INT64'
});