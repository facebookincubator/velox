================
Window functions
================

Spark window functions can be used to compute SQL window functions.
More details about window functions can be found at :doc:`/develop/window`.

Value functions
---------------

.. function:: lag(x[, offset[, default_value]]) -> [same as input]

Returns the value at ``offset`` rows before the current row in the window partition.
The input ``x`` can be a constant value or a column reference. Offsets start at ``0``,
which is the current row. The default ``offset`` is ``1``.  The offset can be a constant
value or a column reference. If the offset is ``null``, ``null`` is returned. If the
offset refers to a row that is not within the partition, the ``default_value`` is returned,
or if ``default_value`` is not specified ``null`` is returned.

If ``IGNORE NULLS`` is specified, ``null`` values are ignored during offset counting.
If not enough non-null values are found during offset counting, ``default_value``
is returned.

.. function:: lead(x[, offset[, default_value]]) -> [same as input]

Returns the value at ``offset`` rows after the current row in the window partition.
The input ``x`` can be a constant value or a column reference. Offsets start at ``0``,
which is the current row. The default ``offset`` is ``1``. The offset can be a constant
value or a column reference. If the offset is ``null``, ``null`` is returned. If the
offset refers to a row that is not within the partition, the ``default_value`` is returned,
or if ``default_value`` is not specified ``null`` is returned.

If ``IGNORE NULLS`` is specified, ``null`` values are ignored during offset counting.
If not enough non-null values are found during offset counting, ``default_value``
is returned.

.. function:: nth_value(x, offset) -> [same as input]
   :noindex:

Returns the value at the specified offset from the beginning of the window. Offsets start at 1.
The offset should be a positive int literal. If the offset is greater than the number of values
in the window, null is returned. It is an error for the offset to be zero or negative.

Rank functions
---------------

.. spark:function:: dense_rank() -> integer

Returns the rank of a value in a group of values. This is similar to rank(), except that tie values do not produce gaps in the sequence.

.. spark:function:: ntile(n) -> integer

Divides the rows for each window partition into n buckets ranging from 1 to at most ``n``. Bucket values will differ by at most 1. If the number of rows in the partition does not divide evenly into the number of buckets, then the remainder values are distributed one per bucket, starting with the first bucket.

For example, with 6 rows and 4 buckets, the bucket values would be as follows: ``1 1 2 2 3 4``

.. spark:function:: rank() -> integer

Returns the rank of a value in a group of values. The rank is one plus the number of rows preceding the row that are not peer with the row. Thus, the values in the ordering will produce gaps in the sequence. The ranking is performed for each window partition.

.. spark:function:: row_number() -> integer

Returns a unique, sequential number to each row, starting with one, according to the ordering of rows within the window partition.
