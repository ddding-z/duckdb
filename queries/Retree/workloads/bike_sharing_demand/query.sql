EXPLAIN ANALYZE
SELECT
        count(*)
FROM
        bike_sharing_demand
WHERE
        predict (
                '/volumn/Retree_exp/workloads/bike_sharing_demand/model/?.onnx',
                CAST (EXTRACT(
                        HOUR
                        FROM
                                bike_sharing_demand.datetime
                )) AS FLOAT,
                atemp,
                humidity,
                windspeed,
                season,
                holiday,
                workingday,
                weather
        ) > ?;