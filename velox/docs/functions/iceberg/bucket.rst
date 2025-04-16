=============================
Bucket Function
=============================
.. iceberg:function:: bucket(numBuckets, input) -> integer

   Returns an integer between 0 and ``numBuckets - 1`` representing the bucket assignment. ::
       SELECT system.bucket(128, 'abcd'); -- 4
       SELECT system.bucket(100, 34L); -- 79
