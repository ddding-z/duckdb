CREATE TABLE table1 AS
SELECT * FROM read_csv('/volumn/duckdb/data/dataset/expanded_data/nyc-taxi-green-dec-2016_?.csv', header=True, columns={
    'passenger_count': 'FLOAT',
    'tolls_amount': 'FLOAT',
    'total_amount': 'FLOAT',
    'lpep_pickup_datetime_day': 'FLOAT',
    'lpep_pickup_datetime_hour': 'FLOAT',
    'lpep_pickup_datetime_minute': 'FLOAT',
    'lpep_dropoff_datetime_day': 'FLOAT',
    'lpep_dropoff_datetime_hour': 'FLOAT',
    'lpep_dropoff_datetime_minute': 'FLOAT'
});