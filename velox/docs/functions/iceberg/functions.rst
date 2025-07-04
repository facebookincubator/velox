*****************
Iceberg Functions
*****************

Here is a list of all scalar Iceberg functions available in Velox.
Function names link to function description.

These functions are used in partition transform.
Refer to `Iceberg documenation <https://iceberg.apache.org/spec/#partition-transforms>`_ for details.

.. iceberg:function:: bucket(numBuckets, input) -> integer

   Returns an integer between 0 and ``numBuckets - 1`` representing the bucket assignment.
   Bucket partition transforms use a 32-bit hash of the ``input``. The 32-bit hash implementation is the 32-bit Murmur3 hash, x86 variant, seeded with 0.
   The hash mod ``numBuckets`` must produce a positive value by first discarding the sign bit of the hash value.

   In pseudo-code, the function is showing as following. ::

       def bucket_N(x) = (murmur3_x86_32_hash(x) & Integer.MAX_VALUE) % N

   Argument ``numBuckets`` is of type INTEGER, the ``numBuckets`` must be more than 0, otherwise, throws.
   Supported types for ``input`` are INTEGER, BIGINT, DECIMAL, DATE, TIMESTAMP, VARCHAR, VARBINARY. ::
       SELECT bucket(128, 'abcd'); -- 4
       SELECT bucket(100, 34L); -- 79
