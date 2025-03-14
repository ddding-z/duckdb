EXPLAIN ANALYZE
SELECT
    count(*)
FROM
    medical_charges
WHERE
    predict (
        '/volumn/Retree_exp/workloads/medical_charges/model/?.onnx',
        Total_Discharges,
        Average_Covered_Charges,
        Average_Medicare_Payments
    ) > ?;