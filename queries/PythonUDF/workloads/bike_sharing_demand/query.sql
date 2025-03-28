EXPLAIN ANALYZE
SELECT
        *
FROM
        bike_sharing_demand
WHERE
        predict (
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