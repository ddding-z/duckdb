EXPLAIN ANALYZE
select
  n_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue
from
  tpch_db.customer,
  tpch_db.orders,
  tpch_db.lineitem,
  tpch_db.supplier,
  tpch_db.nation,
  tpch_db.region
where
  c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and l_suppkey = s_suppkey
  and c_nationkey = s_nationkey
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
--   and r_name = 'ASIA'
--   and o_orderdate >= date '1994-01-01'
--   and o_orderdate < date '1994-01-01' + interval '1' year
  and predict (
    '/volumn/Retree_exp/workloads/tpch-q5/model/?.onnx',
    CAST(c_acctbal AS FLOAT),
    CAST(o_totalprice AS FLOAT),
    c_mktsegment,
    CAST(o_shippriority AS INT64)
    -- CAST(l_quantity AS int64),
    -- CAST(l_extendedprice AS FLOAT),
    -- CAST(l_discount AS FLOAT),
    -- CAST(l_tax AS FLOAT),
    -- CAST(s_acctbal AS FLOAT),
    -- o_orderstatus,
    -- l_returnflag,
    -- l_linestatus,
    -- l_shipinstruct,
    -- l_shipmode,
    -- CAST(n_nationkey AS int64),
    -- CAST(n_regionkey AS int64)
  ) = ?
group by
  n_name
order by
  revenue desc;