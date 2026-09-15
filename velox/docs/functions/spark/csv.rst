=============
CSV Functions
=============

.. spark:function:: from_csv(csvString, schema) -> row

    Parses ``csvString`` into a ROW using the target ``schema``. Fields are
    matched by position (first CSV field → first struct field, etc.).

    Returns NULL if the entire input is NULL. Returns NULL for individual
    fields that cannot be parsed into the target type (PERMISSIVE mode).

    Supported field types: BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT,
    REAL, DOUBLE, DECIMAL, DATE, TIMESTAMP, VARBINARY, and VARCHAR.

    **Parsing behavior:**

    * RFC 4180 quoting: fields may be enclosed in double quotes. A literal
      double quote inside a quoted field is escaped as ``""``.
    * Backslash escape character (Univocity default): ``\"`` inside a quoted
      field is treated as a literal double quote.
    * Delimiter: comma (``,``) by default.
    * Whitespace is trimmed for REAL/DOUBLE fields before parsing (matching
      Java's parseFloat/parseDouble which accept surrounding whitespace),
      but preserved for all other types including VARCHAR.
    * Unquoted empty fields (e.g. the two fields in ``,``) map to NULL for
      all types, matching Spark's ``nullValue`` default of ``""``.
    * Quoted empty fields (``""``) yield a literal empty string for VARCHAR
      and VARBINARY, and NULL for other types (empty strings are not valid
      numeric, boolean, date, timestamp, or decimal values). This matches
      Spark's ``emptyValue`` default of ``""``.
    * Extra fields beyond the schema are silently ignored.
    * Missing fields (fewer CSV fields than schema columns) yield NULL for
      the missing positions.

    **Type-specific rules:**

    * BOOLEAN: accepts ``true``/``false`` (case-insensitive).
    * Integer types: reject decimal points, hex notation, and leading ``+``
      followed by non-digit. Overflow yields NULL.
    * REAL/DOUBLE: accepts ``NaN``, ``Inf``/``Infinity``,
      ``-Inf``/``-Infinity``. Rejects hex float notation (``0x...``).
      Overflow yields ±Infinity; underflow yields ±0 (sign preserved from
      the input).
    * DECIMAL: parsed with exact precision; overflow yields NULL.
      Whitespace is not trimmed (matching Java's BigDecimal constructor).
    * DATE: parsed as ``yyyy-[M]M-[d]d``.
    * TIMESTAMP: parsed as ``yyyy-[M]M-[d]d[T| ][H]H:[m]m:[s]s[.SSS]``,
      interpreted in the session time zone.
    * VARBINARY: treated as raw UTF-8 bytes (no decoding).

    **Input size implementation limit:** This implementation caps individual
    CSV records at 10 MB. Inputs exceeding this limit yield a non-null row
    with every field set to NULL (equivalent to Spark PERMISSIVE mode for a
    row that failed to parse). Apache Spark itself has no such cap; this is
    a Velox-specific safeguard against unbounded per-row allocation.

    **Unsupported options:** Only the 1-argument form ``from_csv(csvString,
    schema)`` is currently supported. Spark 3.0+ also accepts a 3-argument
    ``from_csv(csvString, schema, options)`` overload; passing an options map
    fails at plan time with ``"from_csv expects one argument."`` until the
    3-argument form is implemented.

    **Unsupported Spark features:**

    * ``columnNameOfCorruptRecord`` — a schema field named
      ``_corrupt_record`` (or whatever ``spark.sql.columnNameOfCorruptRecord``
      is configured to) is treated as a regular column here and receives NULL
      on parse failure. Spark PERMISSIVE mode would instead populate it with
      the raw input line.
    * Nested types — ``ARRAY``, ``MAP``, and nested ``ROW`` columns are
      rejected at plan time. Spark 3.4+ supports delimiter-encoded arrays;
      that path is not implemented.
    * ``multiLine`` option is not applicable (single-string input).
    * ``mode`` is pinned to ``PERMISSIVE``. Spark's ``DROPMALFORMED`` and
      ``FAILFAST`` modes are not exposed; parse failures always produce
      NULL fields (never dropped, never thrown).

    Examples::

        SELECT from_csv('1,hello,true', 'a INT, b STRING, c BOOLEAN');
        -- {a=1, b='hello', c=true}

        SELECT from_csv('10.5,abc', 'x DOUBLE, y STRING');
        -- {x=10.5, y='abc'}

        SELECT from_csv('"quoted,field",plain', 'a STRING, b STRING');
        -- {a='quoted,field', b='plain'}

        SELECT from_csv('bad,123', 'a INT, b INT');
        -- {a=NULL, b=123}  (PERMISSIVE: 'bad' fails INT parse → NULL)

        SELECT from_csv('1,2,3', 'a INT, b INT');
        -- {a=1, b=2}  (extra field '3' silently ignored)

        SELECT from_csv('1', 'a INT, b INT');
        -- {a=1, b=NULL}  (missing field → NULL)
