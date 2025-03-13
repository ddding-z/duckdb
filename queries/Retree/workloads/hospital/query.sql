EXPLAIN ANALYZE
SELECT
    COUNT(*)
FROM
    table1
WHERE
    predict (
        '/volumn/duckdb/data/model/?.onnx',
        hematocrit,
        neutrophils,
        -- sodium,
        glucose,
        bloodureanitro,
        -- creatinine,
        bmi,
        pulse,
        respiration,
        -- secondarydiagnosisnonicd9,
        rcount,
        -- gender,
        -- dialysisrenalendstage,
        asthma,
        -- irondef,
        pneum,
        -- substancedependence,
        -- psychologicaldisordermajor,
        -- depress,
        -- psychother,
        -- fibrosisandother,
        -- malnutrition,
        hemo
    ) > ?;