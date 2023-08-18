=====================================
Date and Time Functions
=====================================

Convenience Extraction Functions
--------------------------------

These functions support TIMESTAMP and DATE input types.

.. spark:function:: last_day(date) -> date

    Returns the last day of the month which the date belongs to.

.. spark:function:: make_date(year, month, day) -> date

    Returns the date from year, month and day fields.
    ``year``, ``month`` and ``day`` must be ``INTEGER``.
    Throws an error if inputs are not valid.

    The valid inputs need to meet the following conditions,
    ``month`` need to be from 1 (January) to 12 (December).
    ``day`` need to be from 1 to 31, and matches the number of days in each month.
    days of ``year-month-day - 1970-01-01`` need to be in the range of INTEGER type.

.. spark:function:: to_unix_timestamp(string) -> integer

    Alias for ``unix_timestamp(string) -> integer``.

.. spark:function:: to_unix_timestamp(string, format) -> integer
   :noindex:

    Alias for ``unix_timestamp(string, format) -> integer``.

.. spark:function:: unix_timestamp() -> integer

    Returns the current UNIX timestamp in seconds.

.. spark:function:: unix_timestamp(string) -> integer
   :noindex:

    Returns the UNIX timestamp of time specified by ``string``. Assumes the 
    format ``yyyy-MM-dd HH:mm:ss``. Returns null if ``string`` does not match
    ``format``.

.. spark:function:: unix_timestamp(string, format) -> integer
   :noindex:

    Returns the UNIX timestamp of time specified by ``string`` using the
    format described in the ``format`` string. The format follows Spark's
    `Datetime patterns for formatting and parsing
    <https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html>`_.
    Returns null if ``string`` does not match ``format`` or if ``format``
    is invalid.

.. function:: week_of_year(x) -> integer

    Returns the `ISO-Week`_ of the year from x. The value ranges from ``1`` to ``53``.
    A week is considered to start on a Monday and week 1 is the first week with >3 days.

.. _ISO-Week: https://en.wikipedia.org/wiki/ISO_week_date

.. spark:function:: year(x) -> integer

    Returns the year from ``x``.
