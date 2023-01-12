====================================
String Functions
====================================

.. note::

    These functions assume that the input strings contain valid UTF-8 encoded
    Unicode code points. There are no explicit checks for valid UTF-8 and
    the functions may return incorrect results on invalid UTF-8.

    Additionally, the functions operate on Unicode code points and not user
    visible *characters* (or *grapheme clusters*).  Some languages combine
    multiple code points into a single user-perceived *character*, the basic
    unit of a writing system for a language, but the functions will treat each
    code point as a separate unit.

.. sparkfunction:: ascii(string) -> integer

Returns the numeric value of the first character of ``string``.

.. sparkfunction:: chr(n) -> varchar

Returns the Unicode code point ``n`` as a single character string.

.. sparkfunction:: contains(left, right) -> boolean

Returns a boolean. The value is True if right is found inside left.
Returns NULL if either input expression is NULL. Otherwise, returns False.
Both left or right must be of STRING. ::

SELECT contains('js SQL', 'js'); -- true
SELECT contains('js SQL', 'js'); -- false
SELECT contains('js SQL', null); -- NULL

.. sparkfunction:: endsWith(left, right) -> boolean

Returns a boolean. The value is True if left ends with right.
Returns NULL if either input expression is NULL. Otherwise, returns False.
Both left or right must be of STRING. ::

SELECT endswith('js SQL', 'SQL'); -- true
SELECT endswith('js SQL', 'js'); -- false
SELECT endswith('js SQL', null); -- NULL

.. sparkfunction:: instr(string, substring) -> integer

Returns the starting position of the first instance of ``substring`` in
``string``. Positions start with ``1``. If not found, ``0`` is returned.

.. sparkfunction:: length(string) -> integer

Returns the length of ``string`` in characters.

.. sparkfunction:: split(string, delimiter) -> array(string)

Splits ``string`` on ``delimiter`` and returns an array. ::

SELECT split('oneAtwoBthreeC', '[ABC]'); -- ["one","two","three",""]

.. sparkfunction:: split(string, delimiter, limit) -> array(string)

Splits ``string`` on ``delimiter`` and returns an array of size at most ``limit``. ::

SELECT split('oneAtwoBthreeC', '[ABC]', -1); -- ["one","two","three",""]
SELECT split('oneAtwoBthreeC', '[ABC]', 2); -- ["one","twoBthreeC"]

.. sparkfunction:: startsWith(left, right) -> boolean

Returns a boolean. The value is True if left starts with right.
Returns NULL if either input expression is NULL. Otherwise, returns False.
Both left or right must be of STRING. ::

SELECT startswith('js SQL', 'js'); -- true
SELECT startswith('js SQL', 'SQL'); -- false
SELECT startswith('js SQL', null); -- NULL

.. sparkfunction:: substring(string, start) -> varchar

Returns the rest of ``string`` from the starting position ``start``.
Positions start with ``1``. A negative starting position is interpreted
as being relative to the end of the string.

.. sparkfunction:: substring(string, start, length) -> varchar

Returns a substring from ``string`` of length ``length`` from the starting
position ``start``. Positions start with ``1``. A negative starting
position is interpreted as being relative to the end of the string.