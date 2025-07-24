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
    ) = {predicate};"""

if len(conditions):
    sql = f"""EXPLAIN ANALYZE
    SELECT
        *
    FROM
        wine_quality
    WHERE
        {where_conditions}
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
        ) = {predicate};"""

with open(f"query.sql", "w", encoding="utf-8") as f:
    f.write(sql)
