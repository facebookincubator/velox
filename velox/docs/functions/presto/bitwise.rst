=================
Bitwise Functions
=================

.. prestofunction:: bit_count(x, bits) -> bigint

Count the number of bits set in ``x`` (treated as ``bits``-bit signed
integer) in 2's complement representation::

    SELECT bit_count(9, 64); -- 2
    SELECT bit_count(9, 8); -- 2
    SELECT bit_count(-7, 64); -- 62
    SELECT bit_count(-7, 8); -- 6

.. prestofunction:: bitwise_and(x, y) -> [bigint]

Returns the bitwise AND of ``x`` and ``y`` in 2's complement representation.

.. prestofunction:: bitwise_arithmetic_shift_right(x, shift) -> [bigint]``

Returns the arithmetic right shift operation on ``x`` shifted by ``shift`` in 2’s complement representation.

.. prestofunction:: bitwise_left_shift(x, shift) -> [bigint]``

Returns the left shifted value of ``x``. Here x can be of type ``TINYINT`` , ``SMALLINT``, ``INTEGER`` and ``BIGINT``.

.. prestofunction:: bitwise_logical_shift_right(x, shift, bits) -> [bigint]``

Returns the logical right shift operation on ``x`` (treated as ``bits``-bit integer) shifted by ``shift``.

.. prestofunction:: bitwise_not(x) -> [bigint]

Returns the bitwise NOT of ``x`` in 2's complement representation.

.. prestofunction:: bitwise_or(x, y) -> [bigint]

Returns the bitwise OR of ``x`` and ``y`` in 2's complement representation.

.. prestofunction:: bitwise_right_shift(x, shift) -> [bigint]``

Returns the logical right shifted value of ``x``. Here x can be of type ``TINYINT``, ``SMALLINT``, ``INTEGER`` and ``BIGINT``.

.. prestofunction:: bitwise_right_shift_arithmetic(x, shift) -> [bigint]``

Returns the arithmetic right shift value of ``x``.

.. prestofunction:: bitwise_shift_left(x, shift, bits) -> [bigint]``

Returns the left shift operation on ``x`` (treated as ``bits``-bit integer) shifted by ``shift``.

.. prestofunction:: bitwise_xor(x, y) -> [bigint]``

Returns the bitwise XOR of ``x`` and ``y`` in 2's complement representation.
