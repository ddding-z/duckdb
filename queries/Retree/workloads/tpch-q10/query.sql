EXPLAIN ANALYZE
SELECT
      c_custkey,
      c_name,
      sum(l_extendedprice * (1 - l_discount)) as revenue,
      c_acctbal,
      n_name,
      c_address,
      c_phone,
      c_comment
from
      tpch_db.customer,
      tpch_db.orders,
      tpch_db.lineitem,
      tpch_db.nation
where
      c_custkey = o_custkey
      and l_orderkey = o_orderkey
    --   and o_orderdate >= DATE '1993-10-01'
    --   and o_orderdate < DATE '1993-10-01' + interval '3' month
      and c_nationkey = n_nationkey
      and predict (
            '/volumn/Retree_exp/workloads/tpch-q10/model/?.onnx',
            CAST(c_acctbal AS FLOAT),
            CAST(o_totalprice AS FLOAT),
            -- CAST(l_quantity AS int64),
            -- CAST(l_extendedprice AS FLOAT),
            -- CAST(l_discount AS FLOAT),
            -- CAST(l_tax AS FLOAT)
            o_orderstatus,
            o_orderpriority,
            l_linestatus,
            l_shipinstruct,
            l_shipmode
            -- CAST(n_nationkey AS int64),
            -- CAST(n_regionkey AS int64)
      ) = ?
group by
      c_custkey,
      c_name,
      c_acctbal,
      c_phone,
      n_name,
      c_address,
      c_comment
order by
      revenue desc;