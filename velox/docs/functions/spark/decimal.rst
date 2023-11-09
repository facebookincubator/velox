===================================
Decimal functions and special forms
===================================

Decimal Functions
-----------------

.. spark:function:: unscaled_value(x) -> bigint

    Return the unscaled bigint value of a short decimal ``x``.
    Supported type is: SHORT_DECIMAL.

Decimal Special Forms
---------------------

.. spark:function:: make_decimal(x, nullOnOverflow) -> decimal

    Create ``decimal`` of requsted precision and scale from an unscaled bigint value ``x``.
    If overflows, return null when ``nullOnOverflow`` is true, otherwise throw exception.
