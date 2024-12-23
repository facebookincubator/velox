

select
	n_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue
from
	region,
	nation,
	supplier,
	customer,
	lineitem,
	orders
where
	c_custkey = o_custkey
	and o_orderkey = l_orderkey
	and l_suppkey = s_suppkey
	and c_nationkey = s_nationkey
	and s_nationkey = n_nationkey
	and n_regionkey = r_regionkey
	and r_name = 'ASIA'
	and o_orderdate >= {d '1994-01-01'}
	and o_orderdate < {fn timestampadd (SQL_TSI_YEAR, 1, {d '1994-01-01'})}
group by
	n_name
order by
	revenue desc;

