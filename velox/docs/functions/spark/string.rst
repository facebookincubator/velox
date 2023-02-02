====================================
String Functions
====================================

For functions contains, endsWith, startsWith, returns NULL if either left or right is NULL. 
Both left or right must be of STRING. 

.. spark:function:: ascii(string) -> integer

    Returns the numeric value of the first character of ``string``.

.. spark:function:: chr(n) -> varchar

    Returns the Unicode code point ``n`` as a single character string.

.. spark:function:: contains(left, right) -> boolean

    Returns true if 'right' is found in 'left'. Otherwise, returns False. ::
        
        SELECT contains('Spark SQL', 'Spark'); -- true
        SELECT contains('Spark SQL', 'SPARK'); -- false
        SELECT contains('Spark SQL', null); -- NULL
        SELECT contains(x'537061726b2053514c', x'537061726b'); -- true

.. spark:function:: endsWith(left, right) -> boolean

    Returns true if 'left' ends with 'right'. Otherwise, returns False. ::

        SELECT endswith('js SQL', 'SQL'); -- true
        SELECT endswith('js SQL', 'js'); -- false
        SELECT endswith('js SQL', NULL); -- NULL

.. spark:function:: instr(string, substring) -> integer

    Returns the starting position of the first instance of ``substring`` in
    ``string``. Positions start with ``1``. Returns 0 if 'substring' is not found.

.. spark:function:: length(string) -> integer

    Returns the length of ``string`` in characters.

.. spark:function:: split(string, delimiter) -> array(string)

    Splits ``string`` on ``delimiter`` and returns an array. ::

        SELECT split('oneAtwoBthreeC', '[ABC]'); -- ["one","two","three",""]
        SELECT split('one', ''); -- ["o", "n", "e", ""]
        SELECT split('one', '1'); -- ["one"]

.. spark:function:: split(string, delimiter, limit) -> array(string)

    Splits ``string`` on ``delimiter`` and returns an array of size at most ``limit``. ::

        SELECT split('oneAtwoBthreeC', '[ABC]', -1); -- ["one","two","three",""]
        SELECT split('oneAtwoBthreeC', '[ABC]', 0); -- ["one", "two", "three", ""]
        SELECT split('oneAtwoBthreeC', '[ABC]', 2); -- ["one","twoBthreeC"]

.. spark:function:: startsWith(left, right) -> boolean

    Returns true if 'left' start with 'right'. Otherwise, returns False. ::

        SELECT startswith('js SQL', 'js'); -- true
        SELECT startswith('js SQL', 'SQL'); -- false
        SELECT startswith('js SQL', null); -- NULL

.. spark:function:: substring(string, start) -> varchar

    Returns the rest of ``string`` from the starting position ``start``.
    Positions start with ``1``. A negative starting position is interpreted
    as being relative to the end of the string. Supported types for ``Start`` is INTEGER. 

.. spark:function:: substring(string, start, length) -> varchar

    Returns a substring from ``string`` of length ``length`` from the starting
    position ``start``. Positions start with ``1``. A negative starting
    position is interpreted as being relative to the end of the string.
    Supported types for ``Start`` is INTEGER. ::
        SELECT substring('Spark SQL', 5, 1); -- k
        SELECT substring('Spark SQL', 5, 0); -- ""
        SELECT substring('Spark SQL', 5, -1); -- ""
        SELECT substring('Spark SQL', 5, 10000); -- "k SQL"