===================
Aggregate Functions
===================

Aggregate functions operate on a set of values to compute a single result.

Except for :func:`count`, :func:`count_if`, :func:`max_by`, :func:`min_by` and
:func:`approx_distinct`, all of these aggregate functions ignore null values
and return null for no input rows or when all values are null. For example,
:func:`sum` returns null rather than zero and :func:`avg` does not include null
values in the count. The ``coalesce`` function can be used to convert null into
zero.

Some aggregate functions such as :func:`array_agg` produce different results
depending on the order of input values.

General Aggregate Functions
---------------------------

.. prestofunction:: arbitrary(x) -> [same as input]

Returns an arbitrary non-null value of ``x``, if one exists.

.. prestofunction:: array_agg(x) -> array<[same as input]>

Returns an array created from the input ``x`` elements.

.. prestofunction:: avg(x) -> double|real

Returns the average (arithmetic mean) of all input values.
When x is of type REAL, the result type is REAL.
For all other input types, the result type is DOUBLE.

.. prestofunction:: bool_and(boolean) -> boolean

Returns ``TRUE`` if every input value is ``TRUE``, otherwise ``FALSE``.

.. prestofunction:: bool_or(boolean) -> boolean

Returns ``TRUE`` if any input value is ``TRUE``, otherwise ``FALSE``.

.. prestofunction:: checksum(x) -> varbinary

Returns an order-insensitive checksum of the given values.

.. prestofunction:: count(*) -> bigint

Returns the number of input rows.

.. prestofunction:: count(x) -> bigint

Returns the number of non-null input values.

.. prestofunction:: count_if(x) -> bigint

Returns the number of ``TRUE`` input values.
This function is equivalent to ``count(CASE WHEN x THEN 1 END)``.

.. prestofunction:: every(boolean) -> boolean

This is an alias for `bool_and`.

.. prestofunction:: histogram(x)

Returns a map containing the count of the number of times
each input value occurs. Supports integral, floating-point,
boolean, timestamp, and date input types.

.. prestofunction:: max_by(x, y) -> [same as x]

Returns the value of ``x`` associated with the maximum value of ``y`` over all input values.

.. prestofunction:: min_by(x, y) -> [same as x]

Returns the value of ``x`` associated with the minimum value of ``y`` over all input values.

.. prestofunction:: max(x) -> [same as input]

Returns the maximum value of all input values.

.. prestofunction:: min(x) -> [same as input]

Returns the minimum value of all input values.

.. prestofunction:: sum(x) -> [same as input]

Returns the sum of all input values.

Bitwise Aggregate Functions
---------------------------

.. prestofunction:: bitwise_and_agg(x) -> bigint

Returns the bitwise AND of all input values in 2's complement representation.

.. prestofunction:: bitwise_or_agg(x) -> bigint

Returns the bitwise OR of all input values in 2's complement representation.

Map Aggregate Functions
-----------------------

.. prestofunction:: map_agg(key, value) -> map(K,V)

Returns a map created from the input ``key`` / ``value`` pairs.

.. prestofunction:: map_union(map(K,V)) -> map(K,V)

Returns the union of all the input ``maps``.
If a ``key`` is found in multiple input ``maps``,
that ``key’s`` ``value`` in the resulting ``map`` comes from an arbitrary input ``map``.

Approximate Aggregate Functions
-------------------------------

.. prestofunction:: approx_distinct(x) -> bigint

Returns the approximate number of distinct input values.
This function provides an approximation of ``count(DISTINCT x)``.
Zero is returned if all input values are null.

This function should produce a standard error of 2.3%, which is the
standard deviation of the (approximately normal) error distribution over
all possible sets. It does not guarantee an upper bound on the error for
any specific input set.

.. prestofunction:: approx_distinct(x, e) -> bigint

Returns the approximate number of distinct input values.
This function provides an approximation of ``count(DISTINCT x)``.
Zero is returned if all input values are null.

This function should produce a standard error of no more than ``e``, which
is the standard deviation of the (approximately normal) error distribution
over all possible sets. It does not guarantee an upper bound on the error
for any specific input set. The current implementation of this function
requires that ``e`` be in the range of ``[0.0040625, 0.26000]``.

.. prestofunction:: approx_most_frequent(buckets, value, capacity) -> map<[same as value], bigint>

Computes the top frequent values up to ``buckets`` elements approximately.
Approximate estimation of the function enables us to pick up the frequent
values with less memory.  Larger ``capacity`` improves the accuracy of
underlying algorithm with sacrificing the memory capacity.  The returned
value is a map containing the top elements with corresponding estimated
frequency.

The error of the function depends on the permutation of the values and its
cardinality.  We can set the capacity same as the cardinality of the
underlying data to achieve the least error.

``buckets`` and ``capacity`` must be ``bigint``.  ``value`` can be numeric
or string type.

The function uses the stream summary data structure proposed in the paper
`Efficient computation of frequent and top-k elements in data streams`__
by A. Metwally, D. Agrawal and A. Abbadi.

__ https://www.cse.ust.hk/~raywong/comp5331/References/EfficientComputationOfFrequentAndTop-kElementsInDataStreams.pdf

.. prestofunction:: approx_percentile(x, percentage) -> [same as x]

Returns the approximate percentile for all input values of ``x`` at the
given ``percentage``. The value of ``percentage`` must be between zero and
one and must be constant for all input rows.

.. prestofunction:: approx_percentile(x, percentage, accuracy) -> [same as x]

As ``approx_percentile(x, percentage)``, but with a maximum rank
error of ``accuracy``. The value of ``accuracy`` must be between
zero and one (exclusive) and must be constant for all input rows.
Note that a lower "accuracy" is really a lower error threshold,
and thus more accurate.  The default accuracy is 0.0133.  The
underlying implementation is KLL sketch thus has a stronger
guarantee for accuracy than T-Digest.

.. prestofunction:: approx_percentile(x, percentages) -> array<[same as x]>

Returns the approximate percentile for all input values of ``x`` at each of
the specified percentages. Each element of the ``percentages`` array must be
between zero and one, and the array must be constant for all input rows.

.. prestofunction:: approx_percentile(x, percentages, accuracy) -> array<[same as x]>

As ``approx_percentile(x, percentages)``, but with a maximum rank error of
``accuracy``.

.. prestofunction:: approx_percentile(x, w, percentage) -> [same as x]

Returns the approximate weighed percentile for all input values of ``x``
using the per-item weight ``w`` at the percentage ``p``. The weight must be
an integer value of at least one. It is effectively a replication count for
the value ``x`` in the percentile set. The value of ``p`` must be between
zero and one and must be constant for all input rows.

.. prestofunction:: approx_percentile(x, w, percentage, accuracy) -> [same as x]

As ``approx_percentile(x, w, percentage)``, but with a maximum
rank error of ``accuracy``.

.. prestofunction:: approx_percentile(x, w, percentages) -> array<[same as x]>

Returns the approximate weighed percentile for all input values of ``x``
using the per-item weight ``w`` at each of the given percentages specified
in the array. The weight must be an integer value of at least one. It is
effectively a replication count for the value ``x`` in the percentile
set. Each element of the array must be between zero and one, and the array
must be constant for all input rows.

.. prestofunction:: approx_percentile(x, w, percentages, accuracy) -> array<[same as x]>

As ``approx_percentile(x, w, percentages)``, but with a maximum rank error
of ``accuracy``.

Statistical Aggregate Functions
-------------------------------

.. prestofunction:: corr(y, x) -> double

Returns correlation coefficient of input values.

.. prestofunction:: covar_pop(y, x) -> double

Returns the population covariance of input values.

.. prestofunction:: covar_samp(y, x) -> double

Returns the sample covariance of input values.

.. prestofunction:: stddev(x) -> double

This is an alias for stddev_samp().

.. prestofunction:: stddev_pop(x) -> double

Returns the population standard deviation of all input values.

.. prestofunction:: stddev_samp(x) -> double

Returns the sample standard deviation of all input values.

.. prestofunction:: variance(x) -> double

This is an alias for var_samp().

.. prestofunction:: var_pop(x) -> double

Returns the population variance of all input values.

.. prestofunction:: var_samp(x) -> double

Returns the sample variance of all input values.

Miscellaneous
-------------

.. prestofunction:: max_data_size_for_stats(x) -> bigint

Returns an estimate of the the maximum in-memory size in bytes of ``x``.
