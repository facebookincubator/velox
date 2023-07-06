====================================
String Functions
====================================

Unless specified otherwise, all functions return NULL if at least one of the arguments is NULL.

.. spark:function:: ascii(string) -> integer

    Returns the numeric value of the first character of ``string``.

.. spark:function:: chr(n) -> varchar

    Returns the Unicode code point ``n`` as a single character string.

.. spark:function:: contains(left, right) -> boolean

    Returns true if 'right' is found in 'left'. Otherwise, returns false. ::
        
        SELECT contains('Spark SQL', 'Spark'); -- true
        SELECT contains('Spark SQL', 'SPARK'); -- false
        SELECT contains('Spark SQL', null); -- NULL
        SELECT contains(x'537061726b2053514c', x'537061726b'); -- true

.. spark:function:: endswith(left, right) -> boolean

    Returns true if 'left' ends with 'right'. Otherwise, returns false. ::

        SELECT endswith('js SQL', 'SQL'); -- true
        SELECT endswith('js SQL', 'js'); -- false
        SELECT endswith('js SQL', NULL); -- NULL

.. spark:function:: instr(string, substring) -> integer

    Returns the starting position of the first instance of ``substring`` in
    ``string``. Positions start with ``1``. Returns 0 if 'substring' is not found.

.. spark:function:: left(string, length) -> string

    Returns the leftmost length characters from the ``string``.
    If ``length`` is less or equal than 0 the result is an empty string.

.. spark:function:: length(string) -> integer

    Returns the length of ``string`` in characters.

.. spark:function:: lower(string) -> string

    Returns string with all characters changed to lowercase. ::

        SELECT lower('SparkSql'); -- sparksql

.. spark:function:: ltrim(string) -> varchar

    Removes leading 0x20(space) characters from ``string``. ::

        SELECT ltrim('  data  '); -- "data  "

.. spark:function:: ltrim(trimCharacters, string) -> varchar
   :noindex:

    Removes specified leading characters from ``string``. The specified character
    is any character contained in ``trimCharacters``.
    ``trimCharacters`` can be empty and may contain duplicate characters. ::

        SELECT ltrim('ps', 'spark'); -- "ark"

.. spark:function:: overlay(input, replace, pos, len) -> same as input

    Replace a substring of ``input`` starting at ``pos`` character with ``replace`` and
    going for rest ``len`` characters of ``input``.
    Types of ``input`` and ``replace`` must be the same. Supported types are: VARCHAR and VARBINARY.
    When ``input`` types are VARCHAR, ``len`` and ``pos`` are specified in characters, otherwise, bytes.
    Result is constructed from three parts.
    First part is first pos - 1 characters of ``input`` when ``pos`` if greater then zero, otherwise, empty string.
    Second part is ``replace``.
    Third part is rest of ``input`` from indices pos + len which is 1-based,
    if ``len`` is negative, it will be set to size of ``replace``,
    if pos + len is negative, it refers to -(pos + len)th element before the end of ``input``.
    ::

        SELECT overlay('Spark SQL', '_', 6, -1); -- "Spark_SQL"
        SELECT overlay('Spark SQL', 'CORE', 7, -1); -- "Spark CORE"
        SELECT overlay('Spark SQL', 'ANSI ', 7, 0); -- "Spark ANSI SQL"
        SELECT overlay('Spark SQL', 'tructured', 2, 4); -- "Structured SQL"
        SELECT overlay('Spark SQL', '_', -6, 3); -- "_Sql"

.. spark:function:: replace(string, search, replace) -> string

    Replaces all occurrences of `search` with `replace`. ::

        SELECT replace('ABCabc', 'abc', 'DEF'); -- ABCDEF

.. spark:function:: rtrim(string) -> varchar

    Removes trailing 0x20(space) characters from ``string``. ::

        SELECT rtrim('  data  '); -- "  data"

.. spark:function:: rtrim(trimCharacters, string) -> varchar
   :noindex:

    Removes specified trailing characters from ``string``. The specified character
    is any character contained in ``trimCharacters``.
    ``trimCharacters`` can be empty and may contain duplicate characters. ::

        SELECT rtrim('kr', 'spark'); -- "spa"

.. spark:function:: split(string, delimiter) -> array(string)

    Splits ``string`` on ``delimiter`` and returns an array. ::

        SELECT split('oneAtwoBthreeC', '[ABC]'); -- ["one","two","three",""]
        SELECT split('one', ''); -- ["o", "n", "e", ""]
        SELECT split('one', '1'); -- ["one"]

.. spark:function:: split(string, delimiter, limit) -> array(string)
   :noindex:

    Splits ``string`` on ``delimiter`` and returns an array of size at most ``limit``. ::

        SELECT split('oneAtwoBthreeC', '[ABC]', -1); -- ["one","two","three",""]
        SELECT split('oneAtwoBthreeC', '[ABC]', 0); -- ["one", "two", "three", ""]
        SELECT split('oneAtwoBthreeC', '[ABC]', 2); -- ["one","twoBthreeC"]

.. spark:function:: startswith(left, right) -> boolean

    Returns true if 'left' starts with 'right'. Otherwise, returns false. ::

        SELECT startswith('js SQL', 'js'); -- true
        SELECT startswith('js SQL', 'SQL'); -- false
        SELECT startswith('js SQL', null); -- NULL

.. spark:function:: substring(string, start) -> varchar

    Returns the rest of ``string`` from the starting position ``start``.
    Positions start with ``1``. A negative starting position is interpreted
    as being relative to the end of the string. When the starting position is 0,
    the meaning is to refer to the first character.Type of 'start' must be an INTEGER. 

.. spark:function:: substring(string, start, length) -> varchar
   :noindex:

    Returns a substring from ``string`` of length ``length`` from the starting
    position ``start``. Positions start with ``1``. A negative starting
    position is interpreted as being relative to the end of the string.
    When the starting position is 0, the meaning is to refer to the first character.
    Type of 'start' must be an INTEGER. ::

        SELECT substring('Spark SQL', 0, 2); -- Sp
        SELECT substring('Spark SQL', 1, 2); -- Sp
        SELECT substring('Spark SQL', 5, 0); -- ""
        SELECT substring('Spark SQL', 5, -1); -- ""
        SELECT substring('Spark SQL', 5, 10000); -- "k SQL"
        SELECT substring('Spark SQL', -9, 3); -- "Spa"
        SELECT substring('Spark SQL', -10, 3); -- "Sp"
        SELECT substring('Spark SQL', -20, 3); -- ""

.. spark:function:: trim(string) -> varchar

    Removes leading and trailing 0x20(space) characters from ``string``. ::

        SELECT trim('  data  '); -- "data"

.. spark:function:: trim(trimCharacters, string) -> varchar
   :noindex:

    Removes specified leading and trailing characters from ``string``.
    The specified character is any character contained in ``trimCharacters``.
    ``trimCharacters`` can be empty and may contain duplicate characters. ::

        SELECT trim('sprk', 'spark'); -- "a"

.. spark:function:: upper(string) -> string

    Returns string with all characters changed to uppercase. ::

        SELECT upper('SparkSql'); -- SPARKSQL