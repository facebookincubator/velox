================
Window functions
================

Window functions perform calculations across rows of the query result. They
run after the HAVING clause but before the ORDER BY clause.

Invoking a window function requires special syntax using the OVER clause to specify
the window as follows:

.. code-block::

   function(args) OVER (
        [PARTITION BY expression]
        [ORDER BY expression [ASC|DESC]]
        [frame]
   )

A frame is one of:

.. code-block::

   {RANGE|ROWS} frame_start
   {RANGE|ROWS} BETWEEN frame_start AND frame_end

frame_start and frame_end can be any of:

.. code-block::

   UNBOUNDED PRECEDING
   expression PRECEDING  -- only allowed in ROWS mode
   CURRENT ROW
   expression FOLLOWING  -- only allowed in ROWS mode
   UNBOUNDED FOLLOWING

The window definition has 3 components:

* The PARTITION BY clause separates the input rows into different partitions.

  This is analogous to how the GROUP BY clause separates rows into different groups for aggregate functions.
  If PARTITION BY is not specified, the entire input is treated as a single partition.

* The ORDER BY clause determines the order in which input rows will be processed by the window function.

  If ORDER BY is not specified, the ordering is undefined.
  Note that the ORDER BY clause within window functions does not support ordinals. You need to use actual expressions.

* The frame clause specifies the sliding window of rows to be processed by the function for a given input row.

  A frame can be ROWS type or RANGE type, and it runs from frame_start to frame_end.
  If frame_end is not specified, a default value of CURRENT ROW is used.

  In ROWS mode, CURRENT ROW refers specifically to the current row.

  In RANGE mode, CURRENT ROW refers to any peer row of the current row for the purpose of the ORDER BY.
  A frame start of CURRENT ROW refers to the first peer row of the current row,
  while a frame end of CURRENT ROW refers to the last peer row of the current row.
  If no ORDER BY is specified, all rows are considered peers of the current row.

  If no frame is specified, a default frame of RANGE UNBOUNDED PRECEDING is used.

Examples
________

The following query ranks orders for each clerk by price:

.. code-block:: sql

   SELECT orderkey, clerk, totalprice,
          rank() OVER (PARTITION BY clerk ORDER BY totalprice DESC) AS rnk
   FROM orders ORDER BY clerk, rnk;

=================
Ranking functions
=================

.. function:: row_number() -> bigint

Returns a unique, sequential number for each row, starting with one, according to the ordering of rows
within the window partition.
