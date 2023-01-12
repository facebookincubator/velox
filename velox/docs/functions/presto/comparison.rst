=====================================
Comparison Functions
=====================================

.. prestofunction:: between(x, min, max) -> boolean

Returns true if x is within the specified [min, max] range
inclusive. Supports TINYINT, SMALLINT, INTEGER, BIGINT, DOUBLE,
REAL, VARCHAR, DATE types. The types of all arguments must be
the same.

.. prestofunction:: distinct_from(x, y) -> boolean

In SQL a ``NULL`` value signifies an unknown value, so any comparison
involving a ``NULL`` will produce NULL. The ``distinct_from`` treats
NULL as a known value and guarantees either a ``true`` or ``false``
outcome even in the presence of ``NULL`` input.
So ``distinct_from(NULL, NULL)`` returns ``false``, since a ``NULL``
value is not considered distinct from ``NULL``.

.. prestofunction:: eq(x, y) -> boolean

Returns true if x is equal to y. Supports all scalar types. The
types of x and y must be the same.

.. prestofunction:: greatest(value1, value2, ..., valueN) -> [same as input]

Returns the largest of the provided values. Supports DOUBLE, BIGINT,
VARCHAR, TIMESTAMP, DATE input types. The types of all arguments must
be the same.

.. prestofunction:: gt(x, y) -> boolean

Returns true if x is greater than y. Supports all scalar types. The
types of x and y must be the same.

.. prestofunction:: gte(x, y) -> boolean

Returns true if x is greater than or equal to y. Supports all scalar
types. The types of x and y must be the same.

.. prestofunction:: is_null(x) -> boolean

Returns true if x is a null. Supports all types.

.. prestofunction:: least(value1, value2, ..., valueN) -> [same as input]

Returns the smallest of the provided values. Supports DOUBLE, BIGINT,
VARCHAR, TIMESTAMP, DATE input types. The types of all arguments must
be the same.

.. prestofunction:: lt(x, y) -> boolean

Returns true if x is less than y. Supports all scalar types. The types
of x and y must be the same.

.. prestofunction:: lte(x, y) -> boolean

Returns true if x is less than or equal to y. Supports all scalar types.
The types of x and y must be the same.

.. prestofunction:: neq(x, y) -> boolean

Returns true if x is not equal to y. Supports all scalar types. The types
of x and y must be the same.
