=================
Bitwise Functions
=================

.. spark:function:: bitwise_and(x, y) -> [same as input]

    Returns the bitwise AND of ``x`` and ``y`` in 2's complement representation. 
    Corresponds to Spark's operator ``&``.

.. spark:function:: bitwise_or(x, y) -> [same as input]

    Returns the bitwise OR of ``x`` and ``y`` in 2's complement representation.
    Corresponds to Spark's operator ``^``.

.. spark:function:: bit_count(x) -> integer

    Returns the number of bits that are set in the argument ``x`` as an unsigned 64-bit integer,
    or NULL if the argument is NULL.

.. spark:function:: bit_get(x, pos) -> byte

    Returns the value of the bit (0 or 1) at the specified position.
    The positions are numbered from right to left, starting at zero.
    The valid position argument should in range [0, bits of x), otherwise throw exception.

.. spark:function:: shiftleft(x, n) -> [same as x]

    Returns x bitwise left shifted by n bits. Supported types for 'x' are INTEGER and BIGINT.

.. spark:function:: shiftright(x, n) -> [same as x]

    Returns x bitwise right shifted by n bits. Supported types for 'x' are INTEGER and BIGINT.