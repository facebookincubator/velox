=============================
Array Functions
=============================

.. spark:function:: aggregate(array(E), start, merge, finish) -> array(E)
    
    Applies a binary operator to an initial state and all elements in the array, and reduces this to a single state.
    The final state is converted into the final result by applying a finish function. ::
    
        SELECT aggregate(array(1, 2, 3), 0, (acc, x) -> acc + x, acc -> acc * 10); -- 60

.. spark:function:: array(E, E1, ..., En) -> array(E, E1, ..., En)

    Returns an array with the given elements. ::

        SELECT array(1, 2, 3); -- [1,2,3]

.. spark:function:: array_contains(array(E), value) -> boolean

    Returns true if the array contains the value. ::

        SELECT array_contains(array(1, 2, 3), 2); -- true

.. spark:function:: array_intersect(array(E), array(E1)) -> array(E2)

    Returns an array of the elements in the intersection of array1 and array2, without duplicates. ::

        SELECT array_intersect(array(1, 2, 3), array(1, 3, 5)); -- [1,3]

.. spark:function:: array_sort(array(E)) -> array(E)

    Returns an array which has the sorted order of the input array(E). The elements of array(E) must
    be orderable. Null elements will be placed at the end of the returned array. ::

        SELECT array_sort(ARRAY [1, 2, 3]); -- [1, 2, 3]
        SELECT array_sort(ARRAY [3, 2, 1]); -- [1, 2, 3]
        SELECT array_sort(ARRAY [2, 1, NULL]; -- [1, 2, NULL]
        SELECT array_sort(ARRAY [NULL, 1, NULL]); -- [1, NULL, NULL]
        SELECT array_sort(ARRAY [NULL, 2, 1]); -- [1, 2, NULL]

.. spark:function:: concat(array(E), array(E1), ..., array(En)) -> array(E, E1, ..., En)
    
    Returns the concatenation of array(E), array(E1), ..., array(En). ::

        SELECT concat(array(1, 2, 3), array(4, 5), array(6)); -- [1, 2, 3, 4, 5, 6]

.. spark:function:: filter(array(E), func) -> array(E)

    Filters the input array using the given predicate. ::

        SELECT filter(array(1, 2, 3), x -> x % 2 == 1); -- [1, 3]
        SELECT filter(array(0, 2, 3), (x, i) -> x > i); -- [2, 3]
        SELECT filter(array(0, null, 2, 3, null), x -> x IS NOT NULL); -- [0, 2, 3]

.. spark:function:: in(value, array(E)) -> boolean

    Returns true if value matches at least one of the elements of the array.
    Supports BOOLEAN, REAL, DOUBLE, BIGINT, VARCHAR, TIMESTAMP, DATE input types.

.. spark:function:: size(array(E)) -> bigint

    Returns the size of the array. Returns null for null input
    if :doc:`spark.legacy-size-of-null <../../configs>` is set to false. 
    Otherwise, returns -1 for null input.

.. spark:function:: sort_array(array(E)) -> array(E)

    Returns an array which has the sorted order of the input array. The elements of array must
    be orderable. Null elements will be placed at the beginning of the returned array. ::

        SELECT sort_array(ARRAY [1, 2, 3]); -- [1, 2, 3]
        SELECT sort_array(ARRAY [NULL, 2, 1]); -- [NULL, 1, 2]

.. spark:function:: sort_array(array(E), ascendingOrder) -> array(E)

    Returns an array which has the sorted order of the input array. The elements of array must
    be orderable. Null elements will be placed at the beginning of the returned array in ascending
    order or at the end of the returned array in descending order. ::

        SELECT sort_array(ARRAY [3, 2, 1], true); -- [1, 2, 3]
        SELECT sort_array(ARRAY [2, 1, NULL, true]; -- [NULL, 1, 2]
        SELECT sort_array(ARRAY [NULL, 1, NULL], false); -- [1, NULL, NULL]

.. spark:function:: transform(array(E), function) -> array(E)

    Transforms elements in an array using the function. ::

        SELECT transform(array(1, 2, 3), x -> x + 1); -- [2,3,4]
        SELECT transform(array(1, 2, 3), (x, i) -> x + i); -- [1,3,5]
