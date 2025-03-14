EXPLAIN ANALYZE
SELECT
        count(*)
FROM
        bike_sharing_demand
WHERE
        predict (
                '/volumn/duckdb/data/model/?.onnx',
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