EXPLAIN ANALYZE
SELECT
    *
FROM
    wine_quality
WHERE
    fixed_acidity >= 6.899999618530273
    and citric_acid >= 0.429999977350235
    and chlorides <= 0.0369999967515468
    and sulphates >= 0.3949999809265136
    and sulphates <= 0.4249999821186065
    and alcohol >= 12.774999618530272
    and predict (
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
    ) = 9;