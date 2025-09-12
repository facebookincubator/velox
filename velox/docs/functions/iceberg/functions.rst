*****************
Iceberg Functions
*****************

Here is a list of all scalar Iceberg functions available in Velox.
Function names link to function description.

These functions are used in partition transform.
Refer to `Iceberg documenation <https://iceberg.apache.org/spec/#partition-transforms>`_ for details.

.. iceberg:function:: days(input) -> integer
   Returns the day of the month from a date or timestamp input, as days from 1970-01-01.

   Supported types for ``input`` are: ``DATE``, ``TIMESTAMP``. ::

       SELECT days(DATE '2017-12-01'); -- 17501
       SELECT days(TIMESTAMP '2017-12-01 10:12:55.038194'); -- 17501

.. iceberg:function:: hours(input) -> integer
   Returns the hour from a timestamp input, as hours from 1970-01-01 00:00:00.

   Supported types for ``input`` are: ``TIMESTAMP``. ::

       SELECT hours(TIMESTAMP '2017-12-01 10:12:55.038194'); -- 420034

.. iceberg:function:: months(input) -> integer
   Returns the month from a date or timestamp input, as months from 1970-01-01.

   Supported types for ``input`` are: ``DATE``, ``TIMESTAMP``. ::

       SELECT months(DATE '2017-12-01'); -- 575
       SELECT months(TIMESTAMP '2017-12-01 10:12:55.038194'); -- 575

.. iceberg:function:: years(input) -> integer
   Returns the year from a date or timestamp input, as years from 1970.

   Supported types for ``input`` are: ``DATE``, ``TIMESTAMP``. ::

       SELECT years(DATE '2017-12-01'); -- 47
       SELECT years(TIMESTAMP '2017-12-01 10:12:55.038194'); -- 47

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
