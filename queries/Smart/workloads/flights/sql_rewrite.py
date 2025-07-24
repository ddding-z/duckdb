import pandas as pd
df = pd.read_csv("predicates.csv")
predicate= int(df.iloc[0]["predicate"])

conditions = []
for _, row in df.iterrows():
    feature = row["feature_name"]
    lvalue_str = str(row["lvalue"]).strip().lower()
    rvalue_str = str(row["rvalue"]).strip().lower()
    
    lvalue = None if lvalue_str == "inf" else float(row["lvalue"])
    rvalue = None if rvalue_str == "-inf" else float(row["rvalue"])
    
    if rvalue is not None:
        conditions.append(f"{feature} >= {rvalue}")
    if lvalue is not None:
        conditions.append(f"{feature} <= {lvalue}")

where_conditions = "\n    and ".join(conditions)

sql = f"""EXPLAIN ANALYZE
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
    ) = {predicate};"""

if len(conditions):
    sql = f"""EXPLAIN ANALYZE
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
        and {where_conditions}
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
        ) = {predicate};"""

with open(f"query.sql", "w", encoding="utf-8") as f:
    f.write(sql)