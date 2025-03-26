EXPLAIN ANALYZE
SELECT
    count(*)
FROM
    nyc_taxi_green_dec_2016
WHERE
    predict (
        '/volumn/Retree_exp/workloads/nyc-taxi-green-dec-2016/model/?.onnx',
        passenger_count,
        tolls_amount,
        total_amount,
        lpep_pickup_datetime_day,
        lpep_pickup_datetime_hour,
        lpep_pickup_datetime_minute,
        lpep_dropoff_datetime_day,
        lpep_dropoff_datetime_hour,
        lpep_dropoff_datetime_minute
    ) > ?;