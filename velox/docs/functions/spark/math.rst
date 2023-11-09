====================================
Mathematical Functions
====================================

.. spark:function:: abs(x) -> [same as x]

    Returns the absolute value of ``x``.

.. spark:function:: acos(x) -> double

    Returns the inverse cosine (a.k.a. arc cosine) of ``x``.

.. spark:function:: acosh(x) -> double

    Returns inverse hyperbolic cosine of ``x``.

.. spark:function:: asinh(x) -> double

    Returns inverse hyperbolic sine of ``x``.

.. spark:function:: atanh(x) -> double

    Returns inverse hyperbolic tangent of ``x``.

.. spark:function:: add(x, y) -> [same as x]

    Returns the result of adding x to y. The types of x and y must be the same.
    For integral types, overflow results in an error. Corresponds to sparks's operator ``+``.

.. spark:function:: bin(x) -> varchar

    Returns the string representation of the long value ``x`` represented in binary.

.. spark:function:: ceil(x) -> [same as x]

    Returns ``x`` rounded up to the nearest integer.  
    Supported types are: BIGINT and DOUBLE.

.. spark:function:: cosh(x) -> double

    Returns the hyperbolic cosine of ``x``.

.. spark:function:: cot(x) -> double

    Returns the cotangent of ``x``(measured in radians). Supported type is DOUBLE.

.. spark:function:: csc(x) -> double

    Returns the cosecant of ``x``.

.. spark:function:: decimal_round(decimal, scale) -> [decimal]

    Returns ``decimal`` rounded to a new scale using HALF_UP rounding mode. In HALF_UP rounding, the digit 5 is rounded up. ``scale`` is the new scale to be rounded to.
    The result precision and scale are decided with the precision and scale of input ``decimal`` and ``scale``.
    After rounding we may need one more digit in the integral part.
    When ``scale`` is negative, we need to adjust ``-scale`` number of digits before the decimal point,
    which means we need at least ``-scale + 1`` digits after rounding, and the result scale is 0.

        SELECT round(cast (0.856 as DECIMAL(3, 3)), -1); -- decimal 0
        SELECT round(cast (85.6 as DECIMAL(3, 1)), -1); -- decimal 90
        SELECT round(cast (85.6 as DECIMAL(3, 1)), -2); -- decimal 100

    When ``scale`` is 0, the result scale will be 0.

        SELECT round(cast (0.856 as DECIMAL(3, 3)), 0); -- decimal 1

    When ``scale`` is positive, the result scale will be the minor one of input scale and ``scale``.
    The result precision is decided with the number of integral digits and the result scale, but cannot exceed the max precision of decimal.

        SELECT round(cast (85.681 as DECIMAL(3, 1)), 1); -- decimal 85.7

.. spark:function:: decimal_round(decimal) -> [decimal]

    A version of ``decimal_round`` that rounds ``decimal`` to scale 0.

        SELECT round(cast (85.6 as DECIMAL(3, 1))); -- decimal 86

.. spark:function:: divide(x, y) -> double

    Returns the results of dividing x by y. Performs floating point division.
    Supported type is DOUBLE.
    Corresponds to Spark's operator ``/``. ::

        SELECT 3 / 2; -- 1.5
        SELECT 2L / 2L; -- 1.0
        SELECT 3 / 0; -- NULL

.. spark:function:: divide(x, y) -> decimal

    Returns the results of dividing x by y.
    Supported type is DECIMAL which can be different precision and scale.
    Performs floating point division.
    The result type depends on the precision and scale of x and y.
    Overflow results return null. Corresponds to Spark's operator ``/``. ::

        SELECT CAST(1 as DECIMAL(17, 3)) / CAST(2 as DECIMAL(17, 3)); -- decimal 0.500000000000000000000
        SELECT CAST(1 as DECIMAL(20, 3)) / CAST(20 as DECIMAL(20, 2)); -- decimal 0.0500000000000000000
        SELECT CAST(1 as DECIMAL(20, 3)) / CAST(0 as DECIMAL(20, 3)); -- NULL

.. spark:function:: exp(x) -> double

    Returns Euler's number raised to the power of ``x``.

.. spark:function:: floor(x) -> [same as x]

    Returns ``x`` rounded down to the nearest integer.
    Supported types are: BIGINT and DOUBLE.

.. spark:function:: hypot(a, b) -> double

    Returns the square root of `a` squared plus `b` squared.


.. function:: log1p(x) -> double

    Returns the natural logarithm of the “given value ``x`` plus one”.
    Return NULL if x is less than or equal to -1.

.. spark:function:: log2(x) -> double

    Returns the logarithm of ``x`` with base 2. Return null for zero and non-positive input.

.. spark:function:: log10(x) -> double

    Returns the logarithm of ``x`` with base 10. Return null for zero and non-positive input.

.. spark:function:: multiply(x, y) -> [same as x]

    Returns the result of multiplying x by y. The types of x and y must be the same.
    For integral types, overflow results in an error. Corresponds to Spark's operator ``*``.

.. spark:function:: multiply(x, y) -> [decimal]

    Returns the result of multiplying x by y. The types of x and y must be decimal which can be different precision and scale.
    The result type depends on the precision and scale of x and y.
    Overflow results return null. Corresponds to Spark's operator ``*``. ::

        SELECT CAST(1 as DECIMAL(17, 3)) * CAST(2 as DECIMAL(17, 3)); -- decimal 2.000000
        SELECT CAST(1 as DECIMAL(20, 3)) * CAST(20 as DECIMAL(20, 2)); -- decimal 20.00000
        SELECT CAST(1 as DECIMAL(20, 3)) * CAST(0 as DECIMAL(20, 3)); -- decimal 0.000000
        SELECT CAST(201e-38 as DECIMAL(38, 38)) * CAST(301e-38 as DECIMAL(38, 38)); -- decimal 0.0000000000000000000000000000000000000

.. spark:function:: not(x) -> boolean

    Logical not. ::

        SELECT not true; -- false
        SELECT not false; -- true
        SELECT not NULL; -- NULL

.. spark:function:: pmod(n, m) -> [same as n]

    Returns the positive remainder of n divided by m.
    Supported types are: TINYINT, SMALLINT, INTEGER, BIGINT, FLOAT and DOUBLE.

.. spark:function:: power(x, p) -> double

    Returns ``x`` raised to the power of ``p``.

.. spark:function:: rand() -> double

    Returns a random value with uniformly distributed values in [0, 1). ::

        SELECT rand(); -- 0.9629742951434543

.. spark:function:: rand(seed, partitionIndex) -> double

    Returns a random value with uniformly distributed values in [0, 1) using a seed formed
    by combining user-specified ``seed`` and framework provided ``partitionIndex``. The
    framework is responsible for deterministic partitioning of the data and assigning unique
    ``partitionIndex`` to each thread (in a deterministic way).
    ``seed`` must be constant. NULL ``seed`` is identical to zero ``seed``. ``partitionIndex``
    cannot be NULL. ::

        SELECT rand(0);    -- 0.5488135024422883
        SELECT rand(NULL); -- 0.5488135024422883

.. spark:function:: random() -> double

    An alias for ``rand()``.

.. spark:function:: random(seed, partitionIndex) -> double

    An alias for ``rand(seed, partitionIndex)``.

.. spark:function:: remainder(n, m) -> [same as n]

    Returns the modulus (remainder) of ``n`` divided by ``m``. Corresponds to Spark's operator ``%``.

.. spark:function:: round(x, d) -> [same as x]

    Returns ``x`` rounded to ``d`` decimal places using HALF_UP rounding mode. 
    In HALF_UP rounding, the digit 5 is rounded up.
    Supported types for ``x`` are integral and floating point types.

.. spark:function:: sec(x) -> double

    Returns the secant of ``x``.

.. spark:function:: sinh(x) -> double

    Returns hyperbolic sine of ``x``.

.. spark:function:: subtract(x, y) -> [same as x]

    Returns the result of subtracting y from x. The types of x and y must be the same.
    For integral types, overflow results in an error. Corresponds to Spark's operator ``-``.

.. spark:function:: unaryminus(x) -> [same as x]

    Returns the negative of `x`.  Corresponds to Spark's operator ``-``.
