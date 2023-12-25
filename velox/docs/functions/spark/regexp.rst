============================
Regular Expression Functions
============================

Regular expression functions use RE2 as the regex engine. RE2 is fast, but
supports only a subset of PCRE syntax and in particular does not support
backtracking and associated features (e.g. back references).
See https://github.com/google/re2/wiki/Syntax for more information.

.. spark:function:: regexp_extract(string, pattern) -> varchar

    Returns the first substring matched by the regular expression ``pattern``
    in ``string``. ::

        SELECT regexp_extract('1a 2b 14m', '\d+'); -- 1

.. spark:function:: regexp_extract(string, pattern, group) -> varchar
   :noindex:

    Finds the first occurrence of the regular expression ``pattern`` in
    ``string`` and returns the capturing group number ``group``. ::

        SELECT regexp_extract('1a 2b 14m', '(\d+)([a-z]+)', 2); -- 'a'

.. spark:function:: rlike(string, pattern) -> boolean

    Evaluates the regular expression ``pattern`` and determines if it is
    contained within ``string``.

    This function is similar to the ``LIKE`` operator, except that the
    pattern only needs to be contained within ``string``, rather than
    needing to match all of ``string``. In other words, this performs a
    *contains* operation rather than a *match* operation. You can match
    the entire string by anchoring the pattern using ``^`` and ``$``. ::

        SELECT rlike('1a 2b 14m', '\d+b'); -- true

.. spark:function:: regex_replace(string, pattern, overwrite) -> varchar

    Replaces all substrings in ``string`` that match the regular expression ``pattern`` with the string ``overwrite``. If no match is found, the original string is returned as is.
    There is a limit to the number of unique regexes to be compiled per function call, which is 20.

    Parameters:

    - **string**: The string to be searched.
    - **pattern**: The regular expression pattern that is searched for in the string.
    - **overwrite**: The string that replaces the substrings in ``string`` that match the ``pattern``.

    Examples:

    ::

        SELECT regex_replace('Hello, World!', 'l', 'L'); -- 'HeLLo, WorLd!'
        SELECT regex_replace('300-300', '(\\d+)-(\\d+)', '400'); -- '400'
        SELECT regex_replace('300-300', '(\\d+)', '400'); -- '400-400'

.. spark:function:: regex_replace(string, pattern, overwrite, position) -> varchar
    :noindex:

    Replaces all substrings in ``string`` that match the regular expression ``pattern`` with the string ``overwrite`` starting from the specified ``position``. If the ``position`` is less than one, the function returns an error. If ``position`` is greater than the length of ``string``, the function returns the original ``string`` without any modifications.
    There is a limit to the number of unique regexes to be compiled per function call, which is 20.

    This function is 1-indexed, meaning the position of the first character is 1.
    Parameters:

    - **string**: The string to be searched.
    - **pattern**: The regular expression pattern that is searched for in the string.
    - **overwrite**: The string that replaces the substrings in ``string`` that match the ``pattern``.
    - **position**: The position to start from in terms of number of characters. 1 means to start from the beginning of the string. 3 means to start from the 3rd character. Positions less than one, the function will throw an error. If ``position`` is greater than the length of ``string``, the function returns the original ``string`` without any modifications.

    Examples:

    ::

        SELECT regex_replace('Hello, World!', 'l', 'L', 6); -- 'Hello, WorLd!'

        SELECT regex_replace('Hello, World!', 'l', 'L', -5); -- 'Hello, World!'

        SELECT regex_replace('Hello, World!', 'l', 'L', 100); -- ERROR: Position exceeds string length.
