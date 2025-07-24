import pandas as pd
df = pd.read_csv("predicates.csv")
predicate= df.iloc[0]["predicate"]

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
WITH
    merged_data AS (
        SELECT
            s.Store,
            s.Dept,
            s.Date,
            CASE
                WHEN s.IsHoliday THEN 1
                ELSE 0
            END AS IsHoliday,
            f.Temperature,
            f.Fuel_Price,
            f.MarkDown1,
            f.MarkDown2,
            f.MarkDown3,
            f.MarkDown4,
            f.MarkDown5,
            f.CPI,
            f.Unemployment,
            st.Size,
            CASE
                WHEN s.Date IN ('2010-02-12', '2011-02-11', '2012-02-10') THEN 1
                ELSE 0
            END AS Super_Bowl,
            CASE
                WHEN s.Date IN ('2010-09-10', '2011-09-09', '2012-09-07') THEN 1
                ELSE 0
            END AS Labor_Day,
            CASE
                WHEN s.Date IN ('2010-11-26', '2011-11-25') THEN 1
                ELSE 0
            END AS Thanksgiving,
            CASE
                WHEN s.Date IN ('2010-12-31', '2011-12-30') THEN 1
                ELSE 0
            END AS Christmas,
            EXTRACT(
                WEEK
                FROM
                    s.Date
            ) AS week,
            EXTRACT(
                MONTH
                FROM
                    s.Date
            ) AS month,
            EXTRACT(
                YEAR
                FROM
                    s.Date
            ) AS year,
            CASE
                WHEN st.Type = 'A' THEN 1
                WHEN st.Type = 'B' THEN 2
                WHEN st.Type = 'C' THEN 3
                ELSE NULL
            END AS Type
        FROM
            sales s
            INNER JOIN features f ON s.Store = f.Store
            AND s.Date = f.Date
            INNER JOIN stores st ON s.Store = st.Store
    )

SELECT
    *
FROM
    merged_data
WHERE
    predict (
        '/volumn/Retree_exp/workloads/walmart_sales/model/?.onnx',
        Store,
        Dept,
        -- Date,
        CAST(IsHoliday AS FLOAT),
        -- Temperature,
        Fuel_Price,
        MarkDown1,
        MarkDown2,
        MarkDown3,
        -- MarkDown4,
        -- MarkDown5,
        -- CPI,
        -- Unemployment,
        CAST(Type AS FLOAT),
        Size,
        CAST(Super_Bowl AS FLOAT),
        CAST(Labor_Day AS FLOAT),
        CAST(Thanksgiving AS FLOAT),
        CAST(Christmas AS FLOAT),
        CAST(week AS FLOAT),
        CAST(month AS FLOAT),
        CAST(year AS FLOAT)
    ) > {predicate};"""

if len(conditions):
    sql = f"""EXPLAIN ANALYZE
    WITH
        merged_data AS (
            SELECT
                s.Store,
                s.Dept,
                s.Date,
                CASE
                    WHEN s.IsHoliday THEN 1
                    ELSE 0
                END AS IsHoliday,
                f.Temperature,
                f.Fuel_Price,
                f.MarkDown1,
                f.MarkDown2,
                f.MarkDown3,
                f.MarkDown4,
                f.MarkDown5,
                f.CPI,
                f.Unemployment,
                st.Size,
                CASE
                    WHEN s.Date IN ('2010-02-12', '2011-02-11', '2012-02-10') THEN 1
                    ELSE 0
                END AS Super_Bowl,
                CASE
                    WHEN s.Date IN ('2010-09-10', '2011-09-09', '2012-09-07') THEN 1
                    ELSE 0
                END AS Labor_Day,
                CASE
                    WHEN s.Date IN ('2010-11-26', '2011-11-25') THEN 1
                    ELSE 0
                END AS Thanksgiving,
                CASE
                    WHEN s.Date IN ('2010-12-31', '2011-12-30') THEN 1
                    ELSE 0
                END AS Christmas,
                EXTRACT(
                    WEEK
                    FROM
                        s.Date
                ) AS week,
                EXTRACT(
                    MONTH
                    FROM
                        s.Date
                ) AS month,
                EXTRACT(
                    YEAR
                    FROM
                        s.Date
                ) AS year,
                CASE
                    WHEN st.Type = 'A' THEN 1
                    WHEN st.Type = 'B' THEN 2
                    WHEN st.Type = 'C' THEN 3
                    ELSE NULL
                END AS Type
            FROM
                sales s
                INNER JOIN features f ON s.Store = f.Store
                AND s.Date = f.Date
                AND s.IsHoliday = f.IsHoliday
                INNER JOIN stores st ON s.Store = st.Store
        )

    SELECT
        *
    FROM
        merged_data
    WHERE
        {where_conditions}
        and predict (
            '/volumn/Retree_exp/workloads/walmart_sales/model/?.onnx',
            Store,
            Dept,
            -- Date,
            CAST(IsHoliday AS FLOAT),
            -- Temperature,
            Fuel_Price,
            MarkDown1,
            MarkDown2,
            MarkDown3,
            -- MarkDown4,
            -- MarkDown5,
            -- CPI,
            -- Unemployment,
            CAST(Type AS FLOAT),
            Size,
            CAST(Super_Bowl AS FLOAT),
            CAST(Labor_Day AS FLOAT),
            CAST(Thanksgiving AS FLOAT),
            CAST(Christmas AS FLOAT),
            CAST(week AS FLOAT),
            CAST(month AS FLOAT),
            CAST(year AS FLOAT)
        ) > {predicate};"""

with open(f"query.sql", "w", encoding="utf-8") as f:
    f.write(sql)
