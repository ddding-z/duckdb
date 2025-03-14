EXPLAIN ANALYZE
SELECT
    COUNT(*)
FROM
    hospital
WHERE
    predict (
        '/volumn/Retree_exp/workloads/hospital/model/?.onnx',
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