====================================
Mathematical Functions
====================================

.. sparkfunction:: abs(x) -> [same as x]

Returns the absolute value of ``x``.

.. sparkfunction:: add(x, y) -> [same as x]

Returns the result of adding x to y. The types of x and y must be the same.
For integral types, overflow results in an error. Corresponds to sparks's operator ``+``.

.. sparkfunction:: ceil(x) -> [same as x]

Returns ``x`` rounded up to the nearest integer.  
In spark only long, double, and decimal have ceil.

.. sparkfunction:: divide(x, y) -> [same as x]

Returns the results of dividing x by y. The types of x and y must be the same.
It always performs floating point division. Corresponds to spark's operator ``/``.

.. sparkfunction:: exp(x) -> double

Returns Euler's number raised to the power of ``x``.

.. sparkfunction:: floor(x) -> [same as x]

Returns ``x`` rounded down to the nearest integer.
In spark only long, double, and decimal have floor.

.. sparkfunction:: multiply(x, y) -> [same as x]

Returns the result of multiplying x by y. The types of x and y must be the same.
For integral types, overflow results in an error. Corresponds to spark's operator ``*``.

.. sparkfunction:: pmod(n, m) -> [same as n]

Returns the positive value of ``n`` divided by ``m``.

.. sparkfunction:: power(x, p) -> double

Returns ``x`` raised to the power of ``p``.

.. sparkfunction:: remainder(n, m) -> [same as n]

Returns the modulus (remainder) of ``n`` divided by ``m``. Corresponds to spark's operator ``%``.

.. sparkfunction:: round(x, d) -> [same as x]

Returns ``x`` rounded to ``d`` decimal places using HALF_UP rounding mode.

.. sparkfunction:: subtract(x, y) -> [same as x]

Returns the result of subtracting y from x. The types of x and y must be the same.
For integral types, overflow results in an error. Corresponds to scala's operator ``-``.
