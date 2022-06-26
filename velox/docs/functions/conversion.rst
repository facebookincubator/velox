=====================
Conversion Functions
=====================

.. function:: typeof(x) -> varchar

    Returns the name of the type of the provided expression. ::

        SELECT typeof(123); -- INTEGER
        SELECT typeof('cat'); -- VARCHAR
        SELECT typeof(cos(2) + 1.5); -- DOUBLE