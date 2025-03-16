EXPLAIN ANALYZE
SELECT
    count(*)
FROM
    S_sales,
    R1_indicators,
    R2_stores
WHERE
    S_sales.purchaseid = R1_indicators.purchaseid
    and S_sales.store = R2_stores.store
    and predict (
        '/volumn/Retree_exp/workloads/walmart/model/?.onnx',
        temperature_avg,
        -- temperature_stdev,
        fuel_price_avg,
        -- fuel_price_stdev,
        cpi_avg,
        -- cpi_stdev,
        unemployment_avg,
        -- unemployment_stdev,
        holidayfreq,
        size,
        dept,
        type
    ) = ?;

    