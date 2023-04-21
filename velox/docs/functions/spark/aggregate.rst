===================
Aggregate Functions
===================

Aggregate functions operate on a set of values to compute a single result.

General Aggregate Functions
---------------------------

.. spark:function:: bit_xor(x) -> bigint

    Returns the bitwise XOR of all non-null input values, or null if none.

.. spark:function:: bloom_filter_agg(x, estimatedNumItems, numBits) -> varbinary

    Creates bloom filter from values of hashed value 'x' and returns it serialized into VARBINARY.
    ``estimatedNumItems`` and ``numBits`` decides the number of hash functions and bloom filter capacity in Spark.
    Current bloom filter implementation is different with Spark, if specified ``numBits``, ``estimatedNumItems``
    will not be used.

    ``x`` should be xxhash64(``y``).
    ``estimatedNumItems`` provides an estimate of the number of values of ``y``, which takes no effect here.
    ``numBits`` specifies max capacity of the bloom filter, which allows to trade accuracy for memory.
    Value of numBits in Spark is capped at 67,108,864, actually is capped at 716,800 in case of class memory limit .

    ``x``, ``estimatedNumItems`` and ``numBits`` must be ``bigint``.

.. spark:function:: bloom_filter_agg(x, estimatedNumItems) -> varbinary

    As ``bloom_filter_agg``.

    ``x`` should be xxhash64(``y``).
    ``estimatedNumItems`` provides an estimate of the number of values of ``y``.
    Value of estimatedNumItems is capped at 4,000,000.
    Default numBits = estimatedNumItems * 8. 

.. spark:function:: bloom_filter_agg(x) -> varbinary
    
    As ``bloom_filter_agg``.
    ``x`` should be xxhash64(``y``).
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
