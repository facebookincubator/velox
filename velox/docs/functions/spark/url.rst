============================
URL Functions
============================

Introduction
------------

The URL extraction functions extract components from HTTP URLs (or any valid URIs conforming to `RFC 3986 <https://tools.ietf.org/html/rfc3986.html>`_). The following syntax is supported:

.. code-block:: bash

    [protocol]://[[userinfo@]host[:port]][[path][?query]][#ref]


Consider for example the below URI:

.. code-block::

    http://user:pass@example.com:8080/path1/p.php?k1=v1&k2=v2#Ref1

    protocol  = http
    host      = example.com
    path      = /path1/p.php
    userinfo  = user:pass
    authority = user:pass@example.com:8080
    file      = /path1/p.php?k1=v1&k2=v2
    query     = k1=v1&k2=v2
    ref       = Ref1


Invalid URI's
-------------

Well formed URI's should not contain ascii whitespace. `Percent-encoded URI's <https://www.rfc-editor.org/rfc/rfc3986#section-2.1>`_ should be followed by two hexadecimal
digits after the percent character "%". All the url extract functions will return null when passed an invalid uri.

.. code-block::

    # Examples of url functions with Invalid URI's.

    # Invalid URI due to whitespace
    SELECT parse_url('foo ', 'FILE'); -- NULL (1 row)
    SELECT parse_url('http://www.foo.com ', 'FILE'); -- NULL (1 row)

    # Invalid URI due to improper escaping of '%'
    SELECT parse_url('https://www.ucu.edu.uy/agenda/evento/%%UCUrlCompartir%%', 'FILE'); -- NULL (1 row)
    SELECT parse_url('https://www.ucu.edu.uy/agenda/evento/%%UCUrlCompartir%%', 'FILE'); -- NULL (1 row)

.. spark:function:: parse_url(string, partToExtract) -> varchar

    Extracts a part from a URL. The part to extract can be one of the following:

    * `PROTOCOL`: The protocol.
    * `HOST`: The host name.
    * `PATH`: The path.
    * `USERINFO` : The username and/or password.
    * `AUTHORITY` : The host and optionally userinfo and/or port.
    * `FILE` : The file.
    * `QUERY`: The query.
    * `REF` : The reference.


    :param string: The URL to extract the part from.
    :param partToExtract: The part to extract from the URL. Must be uppercase, lowercase values will return null.
    :return: The extracted part of the URL.

    .. code-block:: sql

        SELECT parse_url('http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related', 'HOST');
        -- www.ics.uci.edu

        SELECT parse_url('http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related', 'PATH');
        -- /pub/ietf/uri/

        SELECT parse_url('http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related', 'QUERY');
        -- k1=v1

        SELECT parse_url('http://www.ics.uci.edu/pub/ietf/uri/?k1=v1#Related', 'REF');
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