EXPLAIN ANALYZE
SELECT
    count(*)
FROM
    table1
WHERE
    predict (
        '/volumn/duckdb/data/model/?.onnx',
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        -- residual_sugar,
        chlorides,
        -- free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        -- pH,
        sulphates,
        alcohol
    ) = ?;