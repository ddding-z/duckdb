CREATE TABLE nyc-taxi-green-dec-2016 AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/nyc-taxi-green-dec-2016/data-extension/?/nyc-taxi-green-dec-2016.csv', header=True, columns={
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