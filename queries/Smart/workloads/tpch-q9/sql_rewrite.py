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
select
    nation,
    o_year,
    count(*)
from
    (
        select
            n_name as nation,
            extract(
                year
                from
                    o_orderdate
            ) as o_year,
            -- l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
        from
            tpch_db.part,
            tpch_db.supplier,
            tpch_db.lineitem,
            tpch_db.partsupp,
            tpch_db.orders,
            tpch_db.nation
        where
            s_suppkey = l_suppkey
            and ps_suppkey = l_suppkey
            and ps_partkey = l_partkey
            and p_partkey = l_partkey
            and o_orderkey = l_orderkey
            and s_nationkey = n_nationkey
            and predict(
                '/volumn/Retree_exp/workloads/tpch-q9/model/?.onnx',
                CAST(l_extendedprice AS FLOAT),
                CAST(l_discount AS FLOAT),
                CAST(ps_supplycost AS FLOAT),
                CAST(l_quantity AS FLOAT)
            ) > {predicate}
    ) as profit
group by
    nation,
    o_year
order by
    nation,
    o_year desc;"""

if len(conditions):
    sql = f"""EXPLAIN ANALYZE
    select
        nation,
        o_year,
        count(*)
    from
        (
            select
                n_name as nation,
                extract(
                    year
                    from
                        o_orderdate
                ) as o_year,
                -- l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
            from
                tpch_db.part,
                tpch_db.supplier,
                tpch_db.lineitem,
                tpch_db.partsupp,
                tpch_db.orders,
                tpch_db.nation
            where
                s_suppkey = l_suppkey
                and ps_suppkey = l_suppkey
                and ps_partkey = l_partkey
                and p_partkey = l_partkey
                and o_orderkey = l_orderkey
                and s_nationkey = n_nationkey
                and {where_conditions}
                and predict(
                    '/volumn/Retree_exp/workloads/tpch-q9/model/?.onnx',
                    CAST(l_extendedprice AS FLOAT),
                    CAST(l_discount AS FLOAT),
                    CAST(ps_supplycost AS FLOAT),
                    CAST(l_quantity AS FLOAT)
                ) > {predicate}
        ) as profit
    group by
        nation,
        o_year
    order by
        nation,
        o_year desc;"""

with open(f"query.sql", "w", encoding="utf-8") as f:
    f.write(sql)
