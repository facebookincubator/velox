=============
URL Functions
=============

Introduction
------------

The URL extraction function ``parse_url`` parses URLs using Java's
``java.net.URI`` grammar rather than strict RFC 3986. Relative references
without a scheme, the empty string, unencoded non-ASCII characters (other than
Unicode spaces and ISO controls), and ``[`` / ``]`` in queries are accepted.
The general syntax is:

.. code-block:: bash

    [protocol:][//[userinfo@]host[:port]][path][?query][#ref]

Extracted components are returned in their raw, still percent-encoded form and
do not include URI syntax separators such as ``:``, ``?`` and ``#`` (except
``FILE``, which includes ``?`` when a query is present).

Consider for example the below URL:

.. code-block::

    http://user:pass@www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related

    PROTOCOL  = http
    USERINFO  = user:pass
    HOST      = www.ics.uci.edu
    AUTHORITY = user:pass@www.ics.uci.edu
    PATH      = /pub/ietf/uri/
    QUERY     = k1=v1
    FILE      = /pub/ietf/uri/?k1=v1
    REF       = Related

Invalid URIs
------------

Well-formed URIs cannot contain ASCII whitespace or ISO control characters, and
percent-escapes ``%XX`` (outside IPv6 scope IDs) must be followed by two
hexadecimal digits. When ANSI mode is disabled (``spark.sql.ansi.enabled`` is
``false``), ``parse_url`` returns ``NULL`` for an invalid URL. When ANSI mode
is enabled, an invalid URL fails the query with an error.

.. code-block::

    # Examples of parse_url with invalid URIs (with ANSI mode disabled).

    # Invalid URI due to whitespace
    SELECT parse_url('foo ', 'PATH'); -- NULL
    SELECT parse_url('http://www.foo.com ', 'HOST'); -- NULL

    # Invalid URI due to improper escaping of '%'
    SELECT parse_url('https://www.ucu.edu.uy/agenda/evento/%%UCUrlCompartir%%', 'PATH'); -- NULL

Extraction Functions
--------------------

.. spark:function:: parse_url(url, part) -> varchar

    Extracts a part from ``url``. ``part`` must be one of (case-sensitive):
    ``PROTOCOL``, ``HOST``, ``PATH``, ``REF``, ``AUTHORITY``, ``FILE``,
    ``USERINFO``, ``QUERY``. Returns ``NULL`` if the requested part is absent
    or if ``part`` is not one of the recognized names (even in ANSI mode).
    An invalid ``url`` returns ``NULL`` when ANSI mode is disabled and fails
    the query with an error when ANSI mode is enabled
    (``spark.sql.ansi.enabled``).

    .. code-block::

        SELECT parse_url('http://facebook.com/path1/p.php?k1=v1&k2=v2#Ref1', 'HOST');
        -- 'facebook.com'

.. spark:function:: parse_url(url, part, key) -> varchar

    Extracts the value of ``key`` from the query string of ``url``. Only
    extracts a value when ``part`` is ``'QUERY'``; passing any other ``part``
    returns ``NULL`` even in ANSI mode. ``key`` is interpreted as a regular
    expression fragment matching ``(&|^)<key>=([^&]*)``. As with the
    two-argument form, an invalid ``url`` returns ``NULL`` unless ANSI mode is
    enabled, in which case the query fails with an error.

    Note: ``key`` is compiled with RE2. Non-empty keys without regex
    metacharacters are matched directly without compiling a regular expression.
    The compiled-regex limit applies only to non-constant keys that contain
    regex metacharacters (or are empty): each distinct pattern is compiled at
    most once per function instance, and the number of compiled patterns is
    capped by ``expression.max_compiled_regexes`` (default 100); exceeding the
    cap fails the query.

    Note: Differences from Spark:

    * **Regular expression syntax**: Velox compiles ``key`` using RE2 rather
      than Java's ``java.util.regex.Pattern``. Keys that use Java-only regex
      features (such as possessive quantifiers like ``a++`` or backreferences)
      work in Spark but fail in Velox. See
      https://github.com/google/re2/wiki/Syntax for supported syntax.
    * **Malformed UTF-8**: Malformed UTF-8 byte sequences in ``url`` are
      accepted and returned unchanged, whereas Spark decodes them through Java
      ``String`` and replaces them with ``U+FFFD``.

    .. code-block::

        SELECT parse_url('http://facebook.com/path1/p.php?k1=v1&k2=v2#Ref1', 'QUERY', 'k1');
        -- 'v1'

Encoding Functions
------------------

.. spark:function:: url_decode(value) -> varchar

    Unescapes the URL encoded ``value``.
    This function is the inverse of :spark:func:`url_encode`.

.. spark:function:: url_encode(value) -> varchar

    Escapes ``value`` by encoding it so that it can be safely included in
    URL query parameter names and values:

    * Alphanumeric characters are not encoded.
    * The characters ``.``, ``-``, ``*`` and ``_`` are not encoded.
    * The ASCII space character is encoded as ``+``.
    * All other characters are converted to UTF-8 and the bytes are encoded
      as the string ``%XX`` where ``XX`` is the uppercase hexadecimal
      value of the UTF-8 byte.
