=============
CSV Functions
=============

.. spark:function:: to_csv(row [, option...]) -> varchar

    Serializes a ROW value as a CSV record. A null ROW produces null. Null
    fields use the configured ``nullValue``, which is empty by default. See
    Spark's `to_csv function
    <https://spark.apache.org/docs/latest/api/sql/index.html#to_csv>`_ for the
    canonical semantics.

    Supported field types are BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT,
    REAL, DOUBLE, and VARCHAR. DATE, TIMESTAMP, DECIMAL, VARBINARY, ARRAY, MAP,
    and nested ROW fields are not supported. REAL and DOUBLE use Java-compatible
    shortest-round-trip formatting.

    Options are supplied as trailing ``key=value`` VARCHAR arguments. Option
    names are case-insensitive. Supported options are ``sep`` (or
    ``delimiter``), ``quote``, ``escape``, and ``nullValue``. The separator,
    quote, and escape values must each contain exactly one byte. Later options
    override earlier options. Null option arguments are ignored. ::

        SELECT to_csv(named_struct('a', 1, 'b', 'hello')); -- 1,hello
        SELECT to_csv(named_struct('a', 1, 'b', NULL), 'nullValue=NULL');
        -- 1,NULL
        SELECT to_csv(named_struct('a', 1, 'b', 2), 'sep=|'); -- 1|2

    VARCHAR fields follow Spark's default CSV write behavior. Leading and
    trailing ASCII control and whitespace bytes are removed. Empty strings are
    written as ``""``. Fields containing the separator, quote, carriage
    return, or line feed are quoted. Within quoted fields, quote and escape
    characters are escaped using the configured escape character.
