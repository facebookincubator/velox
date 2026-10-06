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

    * Fields may be enclosed in double quotes. Spark's default escape is
      backslash rather than RFC 4180 doubled quotes; doubled quotes can
      therefore be preserved literally under the fixed defaults.
    * Backslash escape character (Spark default): ``\"`` inside a quoted
      field is treated as a literal double quote, and ``\\`` as one literal
      backslash. Fixed at ``\``; the supported two-argument overload exposes
      no option to change it.
    * Delimiter: fixed at comma (``,``); the supported two-argument overload
      exposes no option to change it.
    * REAL/DOUBLE trim code units from U+0000 through U+0020 before parsing,
      matching Java's ``String.trim``. DATE/TIMESTAMP remove leading and
      trailing bytes through 0x20 plus 0x7F before parsing. BOOLEAN,
      integer types, DECIMAL, VARCHAR, and VARBINARY preserve whitespace.
      Non-ASCII Unicode whitespace is not trimmed.
    * Empty fields map to NULL for all types, whether quoted (``""``) or
      unquoted (e.g. the two fields in ``,``). Spark's default ``emptyValue``
      and ``nullValue`` are both ``""``.
    * Extra fields beyond the schema are silently ignored.
    * Missing fields (fewer CSV fields than schema columns) yield NULL for
      the missing positions.

    **Type-specific rules:**

    * BOOLEAN: accepts ``true``/``false`` (case-insensitive).
    * Integer types: reject decimal points, hex notation, and leading ``+``
      followed by non-digit. Overflow yields NULL.
    * REAL/DOUBLE: accepts ``NaN`` with an optional sign, full
      ``Infinity``/``-Infinity`` with an optional leading ``+``, and the exact
      untrimmed Spark CSV sentinels ``Inf``/``-Inf``. Surrounding Java
      whitespace is accepted for the full forms, but not for the short
      sentinels. Rejects hex float notation (``0x...``). Overflow yields
      ±Infinity; underflow yields ±0 (sign preserved from the input).
    * DECIMAL: parsed with exact precision; overflow yields NULL.
      Comma grouping separators are removed using Spark's fixed en-US locale.
      Whitespace is not trimmed (matching Java's BigDecimal constructor).
    * DATE: uses Velox's Spark CAST string-to-date parser. Supported forms
      include single-digit month/day, year-month, year-only, and surrounding
      ASCII whitespace/control bytes.
    * TIMESTAMP: uses Velox's Spark CAST string-to-timestamp parser. Naive
      timestamps use the session time zone. Supported forms include a space
      date/time separator, date-only input, extended fractional seconds,
      hour-only offsets, and surrounding ASCII whitespace/control bytes.
      Values outside Spark's signed 64-bit microsecond range (approximately
      years -290308 through 294247) produce NULL, as in Spark.
    * VARBINARY: treated as raw UTF-8 bytes (no decoding).

    **Input size implementation limit:** This implementation caps individual
    CSV records at 10 MB. Inputs exceeding this limit yield a non-null row
    with every field set to NULL (equivalent to Spark PERMISSIVE mode for a
    row that failed to parse). Apache Spark itself has no such cap; this is
    a Velox-specific safeguard against unbounded per-row allocation.

    **Unsupported options:** The two-argument SQL overload
    ``from_csv(csvString, schema)`` is supported. Velox resolves the constant
    schema into the result type, leaving the CSV value as the special form's
    single expression child. Spark 3.0+ also accepts a three-argument
    ``from_csv(csvString, schema, options)`` overload; options maps are not
    currently supported.

    Nested field types (``ARRAY``, ``MAP``, and nested ``ROW``) are rejected at
    plan time, matching Spark's ``UNSUPPORTED_DATATYPE`` behavior.

    **Unsupported Spark features:**

    * ``columnNameOfCorruptRecord`` — a schema field named
      ``_corrupt_record`` (or whatever ``spark.sql.columnNameOfCorruptRecord``
      is configured to) is treated as an ordinary positional column here, so
      it consumes a CSV field and shifts later columns. Spark excludes this
      field from positional matching and populates it with the raw input only
      for malformed records. Avoid corrupt-record fields in Velox schemas.
    * Hexadecimal floating-point literals accepted by Spark's Java parser are
      rejected.
    * Non-ASCII decimal digits accepted by Java's integer and decimal parsers
      are rejected.
    * DATE parsing inherits differences between Velox's Spark CAST parser and
      Spark's string-to-date parser; the following list is not exhaustive.
      Velox accepts years wider than seven digits and accepts year-only or
      year-month values followed by ``T`` (for example ``2024T`` or
      ``2024-01T10``), while Spark returns NULL.
    * Spark's from_csv-specific legacy fallback that removes literal ``GMT``
      substrings after its primary date/time formatter fails is not
      implemented. A bare ``GMT``/``UTC`` suffix and whole-hour
      ``GMT±h[h]`` offsets parse normally. ``GMT``/``UTC``/``UT``-prefixed
      offsets with non-zero minutes are truncated to the whole hour
      (``GMT+05:30`` is read as ``+05:00``), whereas Spark keeps the minutes.
    * TIMESTAMP parsing inherits differences between Velox's Spark CAST parser
      and Spark's ``stringToTimestamp``; the following list is not exhaustive.
      Velox rejects time-only values, bare hours, trailing decimal points, and
      Java short zone IDs such as ``PST``/``EST`` that Spark accepts. Velox
      accepts leap-second rollover, a zone directly after ``HH:mm``, and
      case-insensitive zone names that Spark rejects. Zone IDs resolve through
      Velox's case-insensitive time-zone table; fixed offsets support
      ``±HH``, ``±HH:MM``, or ``±HHMM`` within ±14:00. Spellings accepted only
      by Java ``ZoneId`` (including one-digit components, offset seconds,
      ±18:00, and some aliases) produce NULL, while some case variants or
      aliases rejected by Spark are accepted. Spark also limits timestamp
      years to 4–6 digits; Velox accepts wider years within its range.
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
