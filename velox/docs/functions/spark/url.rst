============================
URL Functions
============================

Introduction
------------

The URL extraction function extracts components from HTTP URLs (or any valid URIs conforming to `RFC 3986 <https://tools.ietf.org/html/rfc3986.html>`_). The following syntax is supported:

.. code-block:: bash

    [protocol:][//host[:port]][path][?query][#fragment]


Consider for example the below URI:

.. code-block::

    http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related

    scheme    = http
    authority = www.ics.uci.edu
    path      = /pub/ietf/uri/
    query     = k1=v1
    fragment  = Related


Invalid URI's
-------------

Well formed URI's should not contain ascii whitespace. `Percent-encoded URI's <https://www.rfc-editor.org/rfc/rfc3986#section-2.1>`_ should be followed by two hexadecimal
digits after the percent character "%". All the url extract functions will return null when passed an invalid uri.

.. code-block::

    # Examples of url functions with Invalid URI's.

    # Invalid URI due to whitespace
    SELECT url_extract_path('foo '); -- NULL (1 row)
    SELECT url_extract_host('http://www.foo.com '); -- NULL (1 row)

    # Invalid URI due to improper escaping of '%'
    SELECT url_extract_path('https://www.ucu.edu.uy/agenda/evento/%%UCUrlCompartir%%'); -- NULL (1 row)
    SELECT url_extract_host('https://www.ucu.edu.uy/agenda/evento/%%UCUrlCompartir%%'); -- NULL (1 row)

.. spark:function:: parse_url(string, partToExtract) -> varchar

    Extracts a part from a URL. The part to extract can be one of the following:

    * `HOST`: The host name.
    * `PATH`: The path.
    * `QUERY`: The query.
    * `FRAGMENT`: The fragment.
    * `PROTOCOL`: The protocol.

    :param string: The URL to extract the part from.
    :param partToExtract: The part to extract from the URL.
    :return: The extracted part of the URL.

    .. code-block:: sql

        SELECT parse_url('http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related', 'HOST');
        -- www.ics.uci.edu

        SELECT parse_url('http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related', 'PATH');
        -- /pub/ietf/uri/

        SELECT parse_url('http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related', 'QUERY');
        -- k1=v1

        SELECT parse_url('http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related', 'FRAGMENT');
        -- Related

        SELECT parse_url('http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related', 'PROTOCOL');
        -- http

.. spark:function:: parse_url(string, partToExtract, key) -> varchar

    Extracts a part from a URL. The part to extract must be QUERY, otherwise function will return null:

    :param string: The URL to extract the part from.
    :param partToExtract: The part to extract from the URL.
    :param key: The key to extract from the query part of the URL.
    :return: The extracted part of the URL.

    .. code-block:: sql

        SELECT parse_url('http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related', 'QUERY', 'k1');
        -- v1

        SELECT parse_url('http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related', 'PROTOCOL', 'k2');
        -- NULL