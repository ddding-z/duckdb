EXPLAIN ANALYZE
SELECT
    *
FROM
    wine_quality
WHERE
    predict (
        '/volumn/Retree_exp/workloads/wine_quality/model/?.onnx',
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