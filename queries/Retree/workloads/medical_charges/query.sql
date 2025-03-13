EXPLAIN ANALYZE
SELECT
    count(*)
FROM
    table1
WHERE
    predict (
        '/volumn/duckdb/data/model/?.onnx',
        Total_Discharges,
        Average_Covered_Charges,
        Average_Medicare_Payments
    ) > ?;