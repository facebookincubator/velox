====================================
Struct Functions
====================================

.. spark:function:: named_struct(name1, val1, name2, val2, ..., namen, valn) -> struct

    Creates a struct with the given field names and values. ::

        SELECT named_struct('a', 1, 'b', 2, 'c', 3); -- {"a" : 1, 'b' : 2, 'c' : 3}
