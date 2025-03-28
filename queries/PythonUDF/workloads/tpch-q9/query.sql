EXPLAIN ANALYZE
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
                CAST(l_extendedprice AS FLOAT),
                CAST(l_discount AS FLOAT),
                CAST(ps_supplycost AS FLOAT),
                CAST(l_quantity AS FLOAT)
            ) > ?
    ) as profit
group by
    nation,
    o_year
order by
    nation,
    o_year desc;