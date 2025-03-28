EXPLAIN ANALYZE
SELECT
    *
FROM
    medical_charges
WHERE
    predict (
        Total_Discharges,
        Average_Covered_Charges,
        Average_Medicare_Payments
    ) > ?;