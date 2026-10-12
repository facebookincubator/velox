=============
CSV Functions
=============

.. spark:function:: from_csv(csvString, schema) -> row

    Parses ``csvString`` into a ROW using the target ``schema``. Fields are
    matched by position (first CSV field → first struct field, etc.). The
    behavior follows `Spark's from_csv function
    <https://spark.apache.org/docs/latest/api/sql/index.html#from_csv>`_ except
    for the limitations documented below.

    A NULL input produces a NULL row. Every non-null input produces a non-null
    row; individual fields that cannot be parsed into the target type are NULL
    (PERMISSIVE mode).

    Supported field types: BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT,
    REAL, DOUBLE, DECIMAL, DATE, TIMESTAMP, VARBINARY, and VARCHAR.

    **Parsing behavior:**

    * Fields may be enclosed in double quotes. Inside quoted fields, backslash
      decodes only ``\"`` and ``\\``; before any other byte it is preserved.
      In unquoted fields, backslash escapes the following byte, so ``\,`` is a
      literal comma. A trailing backslash is preserved.
    * Delimiter: fixed at comma (``,``); the supported two-argument overload
      exposes no option to change it.
    * Empty fields map to NULL for all types, whether quoted (``""``) or
      unquoted (e.g. the two fields in ``,``). Spark's default ``emptyValue``
      and ``nullValue`` are both ``""``.
    * Extra fields beyond the schema are silently ignored.
    * Missing fields (fewer CSV fields than schema columns) yield NULL for
      the missing positions.

    **Type-specific rules:**

    * BOOLEAN: accepts ``true``/``false`` (case-insensitive).
    * Integer types: use the same conversion as TextReader. A leading ``+`` is
      rejected. A decimal continuation is accepted and truncated toward zero
      (for example, ``123.45`` becomes ``123``). Other trailing text, hex
      notation, and overflow yield NULL.
    * REAL/DOUBLE: use the same conversion as TextReader. Bytes 0x00 through
      0x20 are trimmed, matching Java ``String.trim``. ``NaN``, ``+NaN``,
      ``Inf``, and ``Infinity`` are accepted case-insensitively, while
      ``-NaN`` yields NULL. Hexadecimal floating-point syntax is accepted.
      Conversion uses the C numeric locale regardless of the process locale;
      the platform C conversion determines overflow and underflow results.
    * DECIMAL: uses Velox's canonical decimal conversion, including exponent
      notation. Precision overflow yields NULL. Grouping commas are not
      removed.
    * DATE: uses Velox's Presto-cast date parser. The accepted form is
      ``[+-]YYYY-MM-DD`` with single-digit month/day and surrounding ASCII
      whitespace also accepted. Year-only and year-month forms yield NULL.
    * TIMESTAMP: uses Velox's Presto-cast timestamp parser. Naive values are
      interpreted as ``America/Los_Angeles`` local time, matching TextReader,
      and converted to UTC. Local times skipped by a daylight-saving
      transition yield NULL; repeated local times use the earlier instant. A
      space separates date and time; date-only input and fractional seconds
      are supported. ISO ``T`` separators and explicit zone suffixes yield
      NULL. The query session time zone is not used.
    * VARBINARY: valid base64 is decoded. Invalid base64 is copied unchanged
      for TextReader compatibility.

    **Unsupported options:** The two-argument SQL overload
    ``from_csv(csvString, schema)`` is supported. Velox resolves the constant
    schema into the result type, leaving the CSV value as the special form's
    single expression child. Spark 3.0+ also accepts a three-argument
    ``from_csv(csvString, schema, options)`` overload; options maps are not
    currently supported.

    Nested field types (``ARRAY``, ``MAP``, and nested ``ROW``) are rejected at
    plan time, matching Spark's ``UNSUPPORTED_DATATYPE`` behavior.

    **Notable Spark compatibility differences:**

    * ``columnNameOfCorruptRecord`` — a schema field named
      ``_corrupt_record`` (or whatever ``spark.sql.columnNameOfCorruptRecord``
      is configured to) is treated as an ordinary positional column here, so
      it consumes a CSV field and shifts later columns. Spark excludes this
      field from positional matching and populates it with the raw input only
      for malformed records. Avoid corrupt-record fields in Velox schemas.
    * Integer conversion rejects leading ``+`` and accepts a trailing decimal
      continuation, unlike Spark's Java integer parser.
    * Floating-point conversion is case-insensitive for special values,
      accepts hexadecimal syntax, and rejects Java type suffixes such as
      ``1.0f``. Unlike Spark, ``-NaN`` yields NULL.
    * Decimal grouping commas are rejected instead of removed.
    * DATE and TIMESTAMP use the same Presto-cast conversion as TextReader
      rather than Spark's CSV-specific conversion. In particular, TIMESTAMP
      interprets naive values in ``America/Los_Angeles`` rather than the query
      session time zone, does not accept explicit zone suffixes, and has a
      wider supported year range than Spark. Values such as year 294248 are
      accepted; values outside Velox's timestamp and time-zone conversion
      range yield NULL.
    * VARBINARY base64-decodes valid input instead of always returning raw
      bytes.
    * Backslash escapes the following byte in unquoted fields, and a trailing
      backslash is preserved, matching TextReader rather than Spark's
      Univocity parser.
    * Non-ASCII decimal digits accepted by Java's integer and decimal parsers
      are rejected.
    * RFC 4180 doubled-quote escaping is not enabled because Spark's default
      escape character is backslash and the options overload is unsupported.
    * Malformed quoted fields can differ from Univocity's recovery in two
      narrow cases: a closing quote immediately followed by the backslash
      escape, and an end-of-record quote reached after whitespace-plus-garbage
      recovery. Velox applies ``STOP_AT_DELIMITER`` literal fallback without
      consuming later columns into the malformed field or appending
      Univocity's synthetic trailing quote.
    * ``multiLine`` option is not applicable (single-string input).
    * ``mode`` is pinned to ``PERMISSIVE``. Spark's ``from_csv`` also accepts
      ``FAILFAST`` (and rejects ``DROPMALFORMED``), but FAILFAST cannot be
      selected because options maps are unsupported. Parse failures always
      produce NULL fields.

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
