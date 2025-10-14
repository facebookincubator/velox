*****************
Iceberg Functions
*****************

Here is a list of all scalar Iceberg functions available in Velox.
Function names link to function description.

These functions are used in partition transform.
Refer to `Iceberg documenation <https://iceberg.apache.org/spec/#partition-transforms>`_ for details.

.. iceberg:function:: bucket(numBuckets, input) -> integer

   Returns an integer between 0 and numBuckets - 1, indicating the assigned bucket.
   Bucket partitioning is based on a 32-bit hash of the input, specifically using the x86
   variant of the Murmur3 hash function with a seed of 0.

   The function can be expressed in pseudo-code as below. ::

       def bucket(numBuckets, input)= (murmur3_x86_32_hash(input) & Integer.MAX_VALUE) % numBuckets

   The ``numBuckets`` is of type INTEGER and must be greater than 0. Otherwise, an exception is thrown.
   Supported types for ``input`` are INTEGER, BIGINT, DECIMAL, DATE, TIMESTAMP, VARCHAR, VARBINARY. ::
       SELECT bucket(128, 'abcd'); -- 4
       SELECT bucket(100, 34L); -- 79

.. iceberg:function:: days(input) -> date

   Returns the date. ::

       SELECT days(DATE '2017-12-01'); -- 2017-12-01
       SELECT days(TIMESTAMP '2017-12-01 10:12:55.038194'); -- 2017-12-01
       SELECT days(DATE '1969-12-31'); -- 1969-12-31

.. iceberg:function:: hours(input) -> integer

   Returns the number of hours since epoch (1970-01-01 00:00:00). Returns 0 for '1970-01-01 00:00:00' timestamps.
   Returns negative value for timestamps before '1970-01-01 00:00:00'. ::

       SELECT hours(TIMESTAMP '2017-12-01 10:12:55.038194'); -- 420034
       SELECT hours(TIMESTAMP '1969-12-31 23:59:58.999999'); -- -1

.. iceberg:function:: months(input) -> integer

   Returns the number of months since epoch (1970-01-01). Returns 0 for '1970-01-01' date and timestamps.
   Returns negative value for dates and timestamps before '1970-01-01'.  ::

       SELECT months(DATE '2017-12-01'); -- 575
       SELECT months(TIMESTAMP '2017-12-01 10:12:55.038194'); -- 575
       SELECT months(DATE '1960-01-01'); -- -120

.. iceberg:function:: truncate(width, input) -> same type as input

   Returns the truncated value of the input based on the specified width.
   For numeric values, truncate to the nearest lower multiple of ``width``, the truncate function is: input - (((input % width) + width) % width).
   The ``width`` is used to truncate decimal values is applied using unscaled value to avoid additional (and potentially conflicting) parameters.
   For string values, it truncates a valid UTF-8 string with no more than ``width`` code points.
   In contrast to strings, binary values do not have an assumed encoding and are truncated to ``width`` bytes.

   Argument ``width`` must be a positive integer.
   Supported types for ``input`` are: SHORTINT, TYNYINT, SMALLINT, INTEGER, BIGINT, DECIMAL, VARCHAR, VARBINARY. ::

       SELECT truncate(10, 11); -- 10
       SELECT truncate(10, -11); -- -20
       SELECT truncate(7, 22); -- 21
       SELECT truncate(0, 11); -- error: Reason: (0 vs. 0) Invalid truncate width\nExpression: width <= 0
       SELECT truncate(-3, 11); -- error: Reason: (-3 vs. 0) Invalid truncate width\nExpression: width <= 0
       SELECT truncate(4, 'iceberg'); -- 'iceb'
       SELECT truncate(1, '测试'); -- 测
       SELECT truncate(6, '测试'); -- 测试
       SELECT truncate(6, cast('测试' as binary)); -- 测试_

.. iceberg:function:: years(input) -> integer

   Returns the number of years since epoch (1970-01-01). Returns 0 for '1970-01-01' date and timestamps.
   Returns negative value for dates and timestamps before '1970-01-01'.  ::

       SELECT years(DATE '2017-12-01'); -- 47
       SELECT years(TIMESTAMP '2017-12-01 10:12:55.038194'); -- 47
       SELECT years(DATE '1960-01-01'); -- -10
