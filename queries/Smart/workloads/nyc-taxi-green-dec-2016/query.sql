EXPLAIN ANALYZE
    SELECT
        *
    FROM
        nyc_taxi_green_dec_2016
    WHERE
        total_amount >= 18.239999771118164
        and predict (
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
        ) > 2.131