EXPLAIN ANALYZE
SELECT
        count(*)
FROM
        bike_sharing_demand
WHERE
        EXTRACT(
                HOUR
                FROM
                        bike_sharing_demand.datetime
        ) > 11.5
        AND EXTRACT(
                HOUR
                FROM
                        bike_sharing_demand.datetime
        ) < 18.5
        AND humidity < 77.5
        AND predict (
                '/volumn/Retree_exp/workloads/bike_sharing_demand/model/?.onnx',
                EXTRACT(
                        HOUR
                        FROM
                                bike_sharing_demand.datetime
                ),
                atemp,
                humidity,
                windspeed,
                season,
                holiday,
                workingday,
                weather
        ) > ?;