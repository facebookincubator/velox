=============================
Array Functions
=============================

.. function:: array_intersect(array(E) x, array(E) y) -> array(E)

    Returns an array of the elements in the intersection of array ``x`` and array ``y``, without duplicates. ::

        SELECT array_intersect(ARRAY [1, 2, 3], ARRAY[4, 5, 6]); -- []
        SELECT array_intersect(ARRAY [1, 2, 2], ARRAY[1, 1, 2]); -- [1, 2]
        SELECT array_intersect(ARRAY [1, NULL, NULL], ARRAY[1, 1, NULL]); -- [1, NULL]

.. function:: array_except(array(E) x, array(E) y) -> array(E)

    Returns an array of the elements in array ``x`` but not in array ``y``, without duplicates. ::

        SELECT array_except(ARRAY [1, 2, 3], ARRAY [4, 5, 6]); -- [1, 2, 3]
        SELECT array_except(ARRAY [1, 2, 3], ARRAY [1, 2]); -- [3]
        SELECT array_except(ARRAY [1, 2, 2], ARRAY [1, 1, 2]); -- []
        SELECT array_except(ARRAY [1, 2, 2], ARRAY [1, 3, 4]); -- [2]
        SELECT array_except(ARRAY [1, NULL, NULL], ARRAY [1, 1, NULL]); -- []

.. function:: array_max(array(E)) -> E

    Returns the maximum value of input array. ::

        SELECT array_max(ARRAY [1, 2, 3]); -- 3
        SELECT array_max(ARRAY [-1, -2, -2]); -- -1
        SELECT array_max(ARRAY [-1, -2, NULL]); -- NULL
        SELECT array_max(ARRAY []); -- NULL

.. function:: array_min(array(E)) -> E

    Returns the minimum value of input array. ::

        SELECT array_min(ARRAY [1, 2, 3]); -- 1
        SELECT array_min(ARRAY [-1, -2, -2]); -- -2
        SELECT array_min(ARRAY [-1, -2, NULL]); -- NULL
        SELECT array_min(ARRAY []); -- NULL

.. function:: cardinality(x) -> bigint

    Returns the cardinality (size) of the array ``x``.

.. function:: contains(x, element) -> boolean

    Returns true if the array ``x`` contains the ``element``.

.. function:: element_at(array(E), index) -> E

    Returns element of ``array`` at given ``index``.
    If ``index`` > 0, this function provides the same functionality as the SQL-standard subscript operator (``[]``).
    If ``index`` < 0, ``element_at`` accesses elements from the last to the first.

.. function:: filter(array(T), function(T,boolean)) -> array(T)

    Constructs an array from those elements of ``array`` for which ``function`` returns true::

        SELECT filter(ARRAY [], x -> true); -- []
        SELECT filter(ARRAY [5, -6, NULL, 7], x -> x > 0); -- [5, 7]
        SELECT filter(ARRAY [5, NULL, 7, NULL], x -> x IS NOT NULL); -- [5, 7]

.. function:: reduce(array(T), initialState S, inputFunction(S,T,S), outputFunction(S,R)) -> R

    Returns a single value reduced from ``array``. ``inputFunction`` will
    be invoked for each element in ``array`` in order. In addition to taking
    the element, ``inputFunction`` takes the current state, initially
    ``initialState``, and returns the new state. ``outputFunction`` will be
    invoked to turn the final state into the result value. It may be the
    identity function (``i -> i``). ::

        SELECT reduce(ARRAY [], 0, (s, x) -> s + x, s -> s); -- 0
        SELECT reduce(ARRAY [5, 20, 50], 0, (s, x) -> s + x, s -> s); -- 75
        SELECT reduce(ARRAY [5, 20, NULL, 50], 0, (s, x) -> s + x, s -> s); -- NULL
        SELECT reduce(ARRAY [5, 20, NULL, 50], 0, (s, x) -> s + COALESCE(x, 0), s -> s); -- 75
        SELECT reduce(ARRAY [5, 20, NULL, 50], 0, (s, x) -> IF(x IS NULL, s, s + x), s -> s); -- 75
        SELECT reduce(ARRAY [2147483647, 1], CAST (0 AS BIGINT), (s, x) -> s + x, s -> s); -- 2147483648
        SELECT reduce(ARRAY [5, 6, 10, 20], -- calculates arithmetic average: 10.25
                      CAST(ROW(0.0, 0) AS ROW(sum DOUBLE, count INTEGER)),
                      (s, x) -> CAST(ROW(x + s.sum, s.count + 1) AS ROW(sum DOUBLE, count INTEGER)),
                      s -> IF(s.count = 0, NULL, s.sum / s.count));

.. function:: subscript(array(E), index) -> E

    Returns element of ``array`` at given ``index``. The index starts from one.
    Throws if the element is not present in the array. Corresponds to SQL subscript operator [].

    SELECT my_array[1] AS first_element

.. function:: transform(array(T), function(T,U)) -> array(U)

    Returns an array that is the result of applying ``function`` to each element of ``array``::

        SELECT transform(ARRAY [], x -> x + 1); -- []
        SELECT transform(ARRAY [5, 6], x -> x + 1); -- [6, 7]
        SELECT transform(ARRAY [5, NULL, 6], x -> COALESCE(x, 0) + 1); -- [6, 1, 7]
        SELECT transform(ARRAY ['x', 'abc', 'z'], x -> x || '0'); -- ['x0', 'abc0', 'z0']
        SELECT transform(ARRAY [ARRAY [1, NULL, 2], ARRAY[3, NULL]], a -> filter(a, x -> x IS NOT NULL)); -- [[1, 2], [3]]
