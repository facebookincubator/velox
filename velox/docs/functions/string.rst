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

    The :func:`lower` and :func:`upper` functions do not perform
    locale-sensitive, context-sensitive, or one-to-many mappings required for
    some languages. Specifically, this will return incorrect results for
    Lithuanian, Turkish and Azeri.

.. function:: chr(n) -> varchar

    Returns the Unicode code point ``n`` as a single character string.

.. function:: codepoint(string) -> integer

    Returns the Unicode code point of the only character of ``string``.

.. function:: concat(string1, ..., stringN) -> varchar

    Returns the concatenation of ``string1``, ``string2``, ``...``, ``stringN``.
    This function provides the same functionality as the
    SQL-standard concatenation operator (``||``).

.. function:: length(string) -> bigint

    Returns the length of ``string`` in characters.

.. function:: lower(string) -> varchar

    Converts ``string`` to lowercase.

.. function:: ltrim(string) -> varchar

    Removes leading whitespace from string.

.. function:: replace(string, search) -> varchar

    Removes all instances of ``search`` from ``string``.

.. function:: replace(string, search, replace) -> varchar

    Replaces all instances of ``search`` with ``replace`` in ``string``.

    If ``search`` is an empty string, inserts ``replace`` in front of every
    character and at the end of the ``string``.

.. function:: rtrim(string) -> varchar

    Removes trailing whitespace from string.

.. function:: strpos(string, substring) -> bigint

    Returns the starting position of the first instance of ``substring`` in
    ``string``. Positions start with ``1``. If not found, ``0`` is returned.

.. function:: strpos(string, substring, instance) -> bigint

    Returns the position of the N-th ``instance`` of ``substring`` in ``string``.
    ``instance`` must be a positive number.
    Positions start with ``1``. If not found, ``0`` is returned.

.. function:: substr(string, start) -> varchar

    Returns the rest of ``string`` from the starting position ``start``.
    Positions start with ``1``. A negative starting position is interpreted
    as being relative to the end of the string.

.. function:: substr(string, start, length) -> varchar

    Returns a substring from ``string`` of length ``length`` from the starting
    position ``start``. Positions start with ``1``. A negative starting
    position is interpreted as being relative to the end of the string.

.. function:: trim(string) -> varchar

    Removes starting and ending whitespaces from ``string``.

.. function:: upper(string) -> varchar

    Converts ``string`` to uppercase.


Unicode Functions
-----------------

.. function:: to_utf8(string) -> varbinary

    Encodes ``string`` into a UTF-8 varbinary representation.
