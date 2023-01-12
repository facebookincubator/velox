=================
Bitwise Functions
=================

.. sparkfunction:: bitwise_and(x, y) -> [same as input]

Returns the bitwise AND of ``x`` and ``y`` in 2's complement representation. 
Corresponds to scala's operator ``&``.

.. sparkfunction:: bitwise_or(x, y) -> [same as input]

Returns the bitwise OR of ``x`` and ``y`` in 2's complement representation.
Corresponds to scala's operator ``^``.

.. sparkfunction:: shiftleft(x, shift) -> [same as x]

Returns the left shift operation on ``x`` (treated as ``bits``-bit integer) shifted by ``shift``.
Supported types for 'x' are INTEGER and BIGINT.

.. sparkfunction:: shiftright(x, shift) -> [same as x]

Returns the logical right shifted value of ``x``. Supported types for 'x' are INTEGER and BIGINT.