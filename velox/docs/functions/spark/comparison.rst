=====================================
Comparison Functions
=====================================

.. sparkfunction:: between(x, min, max) -> boolean

Returns true if x is within the specified [min, max] range
inclusive. The types of all arguments must be the same.
Supported types are: TINYINT, SMALLINT, INTEGER, BIGINT, DOUBLE, REAL.

.. sparkfunction:: equalnullsafe(x, y) -> boolean

Returns true if x is equal to y. Supports all scalar types. The
types of x and y must be the same. this differs from EqualTo in that
it returns true (rather than NULL) if both inputs are NULL,
and false (rather than NULL) if one of the input is NULL.
Corresponds to scala's operator ``<=>``.

.. sparkfunction:: equalto(x, y) -> boolean

Returns true if x is equal to y. Supports all scalar types. The
types of x and y must be the same. Corresponds to scala's operator ``=`` and ``==``.

.. sparkfunction:: greaterthan(x, y) -> boolean

Returns true if x is greater than y. Supports all scalar types. The
types of x and y must be the same. Corresponds to scala's operator ``>``.

.. sparkfunction:: greaterthanorequal(x, y) -> boolean

Returns true if x is greater than y or x is equal to y. Supports all scalar types. The
types of x and y must be the same. Corresponds to scala's operator ``>=``.

.. sparkfunction:: greatest(value1, value2, ..., valueN) -> [same as input]

Returns the largest of the provided values. skipping null values. Supports all scalar types. 
The types of all arguments must be the same. ::

    SELECT greatest(10, 9, 2, 4, 3); -- 10
    SELECT greatest(10, 9, 2, 4, 3, null); -- 10
    SELECT greatest(null ,null) - null

.. sparkfunction:: least(value1, value2, ..., valueN) -> [same as input]

Returns the smallest of the provided values. Skipping null values. Supports all scalar types.
The types of all arguments must be the same. ::

    SELECT least(10, 9, 2, 4, 3); -- 2
    SELECT least(10, 9, 2, 4, 3, null); -- 2
    SELECT least(null ,null) - null

.. sparkfunction:: lessthan(x, y) -> boolean

Returns true if x is less than y. Supports all scalar types. The types
of x and y must be the same. Corresponds to scala's operator ``<``.

.. sparkfunction:: lessthanorequal(x, y) -> boolean

Returns true if x is less than y or x is equal to y. Supports all scalar types. The
types of x and y must be the same. Corresponds to scala's operator ``<=``.

.. sparkfunction:: notequalto(x, y) -> boolean

Returns true if x is not equal to y. Supports all scalar types. The types
of x and y must be the same. Corresponds to scala's operator ``!=``.



