===================
Aggregate Functions
===================

Aggregate functions operate on a set of values to compute a single result.

General Aggregate Functions
---------------------------

.. spark:function:: bit_xor(x) -> bigint

    Returns the bitwise XOR of all non-null input values, or null if none.

.. spark:function:: bloom_filter_agg(hash, estimatedNumItems, numBits) -> varbinary

    Creates bloom filter from input hashes and returns it serialized into VARBINARY.
    The caller is expected to apply xxhash64 function to input data before calling bloom_filter_agg.
    For example, 
        bloom_filter_agg(xxhash64(x), 100, 1024)   
    In Spark implementation, ``estimatedNumItems`` and ``numBits`` are used to decide the number of hash functions and bloom filter capacity.
    In Velox implementation, ``estimatedNumItems`` is not used.

    ``hash`` cannot be null.
    ``numBits`` specifies max capacity of the bloom filter, which allows to trade accuracy for memory.
    In Spark,  the value of``numBits`` is automatically capped at 67,108,864.
    In Velxo, the value of``numBits`` is automatically capped at 716,800.

    ``x``, ``estimatedNumItems`` and ``numBits`` must be ``BIGINT``.

.. spark:function:: bloom_filter_agg(hash, estimatedNumItems) -> varbinary

    As ``bloom_filter_agg``.
    ``hash`` cannot be null.
    ``estimatedNumItems`` provides an estimate of the number of values of ``y``.
    Value of estimatedNumItems is capped at 4,000,000.
    Default numBits = estimatedNumItems * 8. 

.. spark:function:: bloom_filter_agg(hash) -> varbinary
    
    As ``bloom_filter_agg``.
    ``hash`` cannot be null.
    Default estimatedNumItems = 1,000,000.
    Default numBits in spark is estimatedNumItems * 8 = 8,000,000, here is max value 716,800.

.. spark:function:: first(x) -> x

    Returns the first value of `x`.

.. spark:function:: first_ignore_null(x) -> x

    Returns the first non-null value of `x`.

.. spark:function:: last(x) -> x

    Returns the last value of `x`.

.. spark:function:: last_ignore_null(x) -> x

    Returns the last non-null value of `x`.
