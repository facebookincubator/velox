==============
JSON Functions
==============

JSON Format
-----------

JSON is a language-independent data format that represents data as
human-readable text. A JSON text can represent a number, a boolean, a
string, an array, an object, or a null. A JSON text representing a string
must escape all characters and enclose the string in double quotes, e.g.,
``"123\n"``, whereas a JSON text representing a number does not need to,
e.g., ``123``. A JSON text representing an array must enclose the array
elements in square brackets, e.g., ``[1,2,3]``. More detailed grammar can
be found in `this JSON introduction`_.

.. _this JSON introduction: https://www.json.org

JSON Functions
--------------

.. spark:function:: from_json(jsonString [, option...]) -> array / map / row

    Casts ``jsonString`` to an ARRAY, MAP, or ROW type, with the output type
    determined by the expression. In the default ``PERMISSIVE`` mode,
    unparsable input returns NULL for ARRAY and MAP outputs and a ROW whose
    fields are NULL for ROW outputs, except for an optional corrupt-record
    column.
    Supported element types include BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT,
    REAL, DOUBLE, DECIMAL, DATE, TIMESTAMP, VARCHAR, ARRAY, MAP and ROW. When
    casting to ARRAY or MAP, the element type of the array or the value type of
    the map must be one of these supported types, and for maps, the key type must
    be VARCHAR. Casting to ROW supports only JSON objects.
    Velox typed expressions carry the output type, so the schema is not a
    runtime argument as it is in Spark. Conceptual examples are shown with the
    expression output type on the left. ::

        ROW(a BOOLEAN):             from_json('{"a": true}')              -- {'a'=true}
        ROW(a INTEGER):             from_json('{"a": 1}')                 -- {'a'=1}
        ROW(a DOUBLE):              from_json('{"a": 1.0}')               -- {'a'=1.0}
        ROW(a DECIMAL(7, 2)):       from_json('{"a": 5.321E2}')           -- {'a'=532.10}
        ROW(a DATE):                from_json('{"a":"2021-7-1T"}')         -- {'a'="2021-07-01"}
        ROW(a TIMESTAMP):           from_json('{"a":"2021-07-01 12:34:56"}')
        ARRAY(VARCHAR):             from_json('["name", "age", "id"]')     -- ['name', 'age', 'id']
        MAP(VARCHAR, INTEGER):      from_json('{"a": 1, "b": 2}')         -- {'a'=1, 'b'=2}
        ROW(a ROW(b INTEGER)):      from_json('{"a": {"b": 1}}')          -- {'a'={b=1}}

    Implemented Spark JSON options are ``allowNonNumericNumbers``, ``mode``
    (``PERMISSIVE`` or ``FAILFAST``), ``columnNameOfCorruptRecord``,
    ``dateFormat``, ``timestampFormat``, and ``timeZone``. Velox also accepts
    ``sessionTimezone`` as an alias for ``timeZone`` and exposes
    ``enablePartialResults`` and ``caseSensitiveFieldMatch``. The latter
    enables exact-case field matching and last-value-wins duplicate-key
    behavior. Integrations pass each option as a constant ``key=value`` string
    argument. Null option arguments are rejected. Option names are
    case-insensitive. Boolean option values must be ``true`` or ``false``
    (case-insensitive).

    See `Spark's from_json documentation
    <https://spark.apache.org/docs/latest/api/sql/index.html#from_json>`_ and
    `Spark's JSON option documentation
    <https://spark.apache.org/docs/latest/sql-data-sources-json.html#data-source-option>`_
    for the canonical Spark interface and option definitions.

    ``allowNonNumericNumbers`` controls the special NaN and infinity tokens,
    including their quoted spellings. It does not reject overflow of ordinary
    JSON numbers: for example, ``1e400`` produces infinity even when the option
    is false.

    ``DROPMALFORMED`` mode is not supported, matching Spark's ``from_json``
    behavior. Mode names are case-insensitive. Unrecognized mode names use
    ``PERMISSIVE`` behavior.

    JSON integer values are accepted for TIMESTAMP fields as epoch seconds.
    As in Spark, conversion to microseconds uses wrapping 64-bit arithmetic.
    Timestamp strings without an explicit offset use the query session
    timezone unless overridden by ``timeZone``. Local times in a daylight
    saving gap shift forward by the gap length, and ambiguous local times use
    the earlier offset. Unlike integer epoch seconds, timestamp strings must
    fit in signed 64-bit microseconds after timezone conversion. Out-of-range
    strings are field conversion failures.

    Custom timestamp formats preserve up to six fractional-second digits under
    the default formatter policy. When a pattern accepts more than six digits,
    additional digits are truncated. The legacy formatter retains its
    millisecond-based interpretation of fractional fields. Custom date and
    timestamp patterns use the existing Velox formatter syntax, not the full
    Spark ``java.time`` pattern grammar.
    An explicitly empty format does not fall back to default string parsing.
    Out-of-range numeric fields in custom patterns are rejected rather than
    wrapping.

    Empty input and input containing only JSON whitespace (space, tab, line
    feed, or carriage return) return NULL in every mode. Other whitespace
    characters are malformed JSON. A JSON ``null`` root follows malformed-record
    handling, unlike a SQL NULL input, which returns NULL without an error.

    ``columnNameOfCorruptRecord`` is ignored for non-ROW outputs and when the
    named column is absent. When present, the column must be VARCHAR and JSON
    input fields with the same name are ignored. An empty column name is
    supported when explicitly configured. An empty ``timeZone`` option is
    invalid.

    Case-insensitive matching remains the default for backward compatibility.
    In this mode, duplicate fields use first-non-null-wins behavior. Set
    ``caseSensitiveFieldMatch=true`` for Spark-compatible exact-case and
    last-value-wins behavior.
    ``enablePartialResults`` defaults to true and preserves successfully
    converted fields inside nested ROW values. When false, a conversion failure
    collapses the containing nested ROW to NULL; successful siblings in the
    root ROW remain available.

    Except for the supported NaN and infinity tokens, parsing follows simdjson's
    strict JSON syntax. Spark options that relax JSON syntax,
    including ``allowSingleQuotes``, are accepted but cannot change parser
    behavior. Inputs using unsupported syntax return the configured permissive
    result or fail in ``FAILFAST`` mode.
    Options that affect only file-source behavior or schema inference, such as
    ``multiLine`` and ``samplingRatio``, are accepted as no-ops.

.. spark:function:: get_json_object(jsonString, path) -> varchar

    Returns a json object, represented by VARCHAR, from ``jsonString`` by searching ``path``.
    Returns NULL if ``jsonString`` or ``path`` is malformed or ``path`` does not exist. ::

        SELECT get_json_object('{"a":"b"}', '$.a'); -- 'b'
        SELECT get_json_object('{"a":{"b":"c"}}', '$.a'); -- '{"b":"c"}'
        SELECT get_json_object('{"a":3}', '$.b'); -- NULL (unexisting field)
        SELECT get_json_object('{"a"-3}'', '$.a'); -- NULL (malformed JSON string)
        SELECT get_json_object('{"a":3}'', '.a'); -- NULL (malformed JSON path)

    Valid ``path`` syntax:
        * Must start with '$'.
        * Using "[index]", "['field']" or ".field" to navigate to the desired JSON object.
        * Whitespace is allowed **after the dot** and **before the field name**, e.g., "$.  field".
        * Trailing whitespace after '$' is allowed, e.g., "$   ".

    When ``path`` resolves to an object or array, characters in the Unicode
    supplementary planes (code points >= U+10000, e.g. emoji) are escaped as
    ``\uXXXX\uXXXX`` UTF-16 surrogate pairs, while Basic Multilingual Plane
    (BMP) characters (e.g. Chinese/Japanese/Korean, CJK) are left literal.
    Scalar string results are returned literal. ::

        SELECT get_json_object('{"a":{"n":"🧧会员"}}', '$.a'); -- '{"n":"\uD83E\uDDE7会员"}'
        SELECT get_json_object('{"a":["🥇"]}', '$.a'); -- '["\uD83E\uDD47"]'
        SELECT get_json_object('{"a":"🥇"}', '$.a'); -- '🥇'

.. spark:function:: json_array_length(jsonString) -> integer

    Returns the number of elements in the outermost JSON array from ``jsonString``.
    If ``jsonString`` is not a valid JSON array or NULL, the function returns NULL. ::

        SELECT json_array_length('[1,2,3,4]'); -- 4
        SELECT json_array_length('[1,2,3,{"f1":1,"f2":[5,6]},4]'); -- 5
        SELECT json_array_length('[1,2'); -- NULL

.. spark:function:: json_object_keys(jsonString) -> array(string)

    Returns all the keys of the outermost JSON object as an array if a valid JSON object is given.
    If it is any other valid JSON string, an invalid JSON string or an empty string, the function
    returns null. ::

        SELECT json_object_keys('{}'); -- []
        SELECT json_object_keys('{"name": "Alice", "age": 5, "id": "001"}'); -- ['name', 'age', 'id']
        SELECT json_object_keys(''); -- NULL
        SELECT json_object_keys(1); -- NULL
        SELECT json_object_keys('"hello"'); -- NULL
        SELECT json_object_keys("invalid json"); -- NULL

.. spark:function:: to_json(jsonObject) -> jsonString

    Converts a JSON object (ROW, ARRAY, or MAP) into a JSON string.

    Supported primitive types are: BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT,
    REAL, DOUBLE, DECIMAL, DATE, TIMESTAMP, VARCHAR, and VARBINARY. ROW, ARRAY,
    and MAP can be nested. ::

        SELECT to_json(named_struct('c0', 1, 'c1', 'a')); -- {"c0":1,"c1":"a"}
        SELECT to_json(ARRAY(1, 2, 3)); -- [1,2,3]
        SELECT to_json(MAP('x', 1, 'y', 2)); -- {"x":1,"y":2}

    The current implementation has following limitations.

    * Does not support user provided options. ::

        to_json(MAP(1, 'a'), map('option', 'value'))

    * MAP key type cannot be/contain MAP. ::

        to_json(MAP(MAP('a', 1), 10))
