EXPLAIN ANALYZE
SELECT
    *
FROM
    S_routes,
    R1_airlines,
    R2_sairports,
    R3_dairports
WHERE
    S_routes.airlineid = R1_airlines.airlineid
    and S_routes.sairportid = R2_sairports.sairportid
    and S_routes.dairportid = R3_dairports.dairportid
    and predict (
        '/volumn/Retree_exp/workloads/flights/model/?.onnx',
        slatitude,
        slongitude,
        dlatitude,
        dlongitude,
        -- name1,
        -- name2,
        -- name4,
        -- acountry,
        active,
        -- scity,
        -- scountry,
        -- stimezone,
        sdst,
        -- dcity,
        -- dcountry,
        -- dtimezone,
        ddst
    ) = ?;