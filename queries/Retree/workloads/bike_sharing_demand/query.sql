EXPLAIN ANALYZE
SELECT
        count(*)
FROM
        table1
WHERE
        predict (
                '/volumn/duckdb/data/model/?.onnx',
                EXTRACT(
                        HOUR
                        FROM
                                table1.datetime
                ),
                atemp,
                humidity,
                windspeed,
                season,
                holiday,
                workingday,
                weather
        ) > ?;