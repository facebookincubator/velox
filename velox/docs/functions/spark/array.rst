=============================
Array Functions
=============================

.. sparkfunction:: array_sort(array(E)) -> array(E)

Returns an array which has the sorted order of the input array(E). The elements of array(E) must
be orderable. Null elements will be placed at the end of the returned array. ::

    SELECT array_sort(ARRAY [1, 2, 3]); -- [1, 2, 3]
    SELECT array_sort(ARRAY [3, 2, 1]); -- [1, 2, 3]
    SELECT array_sort(ARRAY [2, 1, NULL]; -- [1, 2, NULL]
    SELECT array_sort(ARRAY [NULL, 1, NULL]); -- [1, NULL, NULL]
    SELECT array_sort(ARRAY [NULL, 2, 1]); -- [1, 2, NULL]

.. sparkfunction:: in(x, element) -> boolean

Returns true if the array ``x`` equals to any value of ``element``.
Supports BOOLEAN, REAL, DOUBLE, BIGINT, VARCHAR, TIMESTAMP, DATE input types.

.. sparkfunction:: size(array(E)) -> bigint

Returns the size of the input array. The function returns null for null input
if legacySizeOfNull is set to false. Otherwise, the function returns -1 for null input.
With the default settings, the function returns -1 for null input.

.. sparkfunction:: sort_array(array(E)) -> array(E)

Returns an array which has the sorted order of the input array. The elements of array must
be orderable. Null elements will be placed at the beginning of the returned array. ::

    SELECT sort_array(ARRAY [1, 2, 3]); -- [1, 2, 3]
    SELECT sort_array(ARRAY [NULL, 2, 1]); -- [null, 1, 2]

.. sparkfunction:: sort_array(array(E), ascendingOrder) -> array(E)

Returns an array which has the sorted order of the input array. The elements of array must
be orderable. Null elements will be placed at the beginning of the returned array in ascending
order or at the end of the returned array in descending order. ::

    SELECT sort_array(ARRAY [3, 2, 1], true); -- [1, 2, 3]
    SELECT sort_array(ARRAY [2, 1, NULL, true]; -- [NULL, 1, 2]
    SELECT sort_array(ARRAY [NULL, 1, NULL], false); -- [1, NULL, NULL]

