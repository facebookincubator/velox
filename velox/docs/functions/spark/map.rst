===========================
Map Functions
===========================

.. spark:function:: element_at(map(K,V), key) -> V

    Returns value for given ``key``, or ``NULL`` if the key is not contained in the map.

.. spark:function:: map(K, V, K, V, ...) -> map(K,V)

    Returns a map created using the given key/value pairs. Keys are not allowed to be null. ::

        SELECT map(1, 2, 3, 4); -- {1 -> 2, 3 -> 4}

        SELECT map(array(1, 2), array(3, 4)); -- {[1, 2] -> [3, 4]}

.. spark:function:: map_entries(map(K,V)) -> array(row(K,V))

    Returns an array of all entries in the given map. ::

        SELECT map_entries(MAP(ARRAY[1, 2], ARRAY['x', 'y'])); -- [ROW(1, 'x'), ROW(2, 'y')]

.. spark:function:: map_filter(map(K,V), func) -> map(K,V)

    Filters entries in a map using the function. ::

        SELECT map_filter(map(1, 0, 2, 2, 3, -1), (k, v) -> k > v); -- {1 -> 0, 3 -> -1}

.. spark:function:: map_from_arrays(array(K), array(V)) -> map(K,V)

    Creates a map with a pair of the given key/value arrays. All elements in keys should not be null.
    If key size != value size will throw exception that key and value must have the same length.::

        SELECT map_from_arrays(array(1.0, 3.0), array('2', '4')); -- {1.0 -> 2, 3.0 -> 4}

.. spark:function:: map_keys(x(K,V)) -> array(K)

    Returns all the keys in the map ``x``.

.. spark:function:: map_values(x(K,V)) -> array(V)

    Returns all the values in the map ``x``.

.. spark:function:: size(map(K,V)) -> bigint
   :noindex:

    Returns the size of the input map. Returns null for null input
    if :doc:`spark.legacy_size_of_null <../../configs>` is set to false.
    Otherwise, returns -1 for null input.

.. spark:function:: transform_keys(map(K1,V), function(K1,V,K2)) -> map(K2,V)

    Returns a map that applies ``function`` to each entry of ``map`` and transforms the keys.::

        SELECT transform_keys(MAP(ARRAY[], ARRAY[]), (k, v) -> k + 1); -- {}
        SELECT transform_keys(MAP(ARRAY [1, 2, 3], ARRAY ['a', 'b', 'c']), (k, v) -> k + 1); -- {2 -> a, 3 -> b, 4 -> c}
        SELECT transform_keys(MAP(ARRAY ['a', 'b', 'c'], ARRAY [1, 2, 3]), (k, v) -> v * v); -- {1 -> 1, 4 -> 2, 9 -> 3}
        SELECT transform_keys(MAP(ARRAY ['a', 'b'], ARRAY [1, 2]), (k, v) -> k || CAST(v as VARCHAR)); -- {a1 -> 1, b2 -> 2}
        SELECT transform_keys(MAP(ARRAY [1, 2], ARRAY [1.0, 1.4]), -- {one -> 1.0, two -> 1.4}
                              (k, v) -> MAP(ARRAY[1, 2], ARRAY['one', 'two'])[k]);

.. spark:function:: transform_values(map(K,V1), function(K,V1,V2)) -> map(K,V2)

    Returns a map that applies ``function`` to each entry of ``map`` and transforms the values.::

        SELECT transform_values(MAP(ARRAY[], ARRAY[]), (k, v) -> v + 1); -- {}
        SELECT transform_values(MAP(ARRAY [1, 2, 3], ARRAY [10, 20, 30]), (k, v) -> v + k); -- {1 -> 11, 2 -> 22, 3 -> 33}
        SELECT transform_values(MAP(ARRAY [1, 2, 3], ARRAY ['a', 'b', 'c']), (k, v) -> k * k); -- {1 -> 1, 2 -> 4, 3 -> 9}
        SELECT transform_values(MAP(ARRAY ['a', 'b'], ARRAY [1, 2]), (k, v) -> k || CAST(v as VARCHAR)); -- {a -> a1, b -> b2}
        SELECT transform_values(MAP(ARRAY [1, 2], ARRAY [1.0, 1.4]), -- {1 -> one_1.0, 2 -> two_1.4}
                                (k, v) -> MAP(ARRAY[1, 2], ARRAY['one', 'two'])[k] || '_' || CAST(v AS VARCHAR));
