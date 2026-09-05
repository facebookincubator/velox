=============
XML Functions
=============

.. spark:function:: xpath_boolean(xml, path) -> boolean

    Evaluates the XPath expression ``path`` against the XML document ``xml`` and
    returns its boolean value. A node-set result is ``true`` when it is
    non-empty (a matching node exists); a boolean expression returns its own
    value. Returns NULL if ``xml`` or ``path`` is NULL or empty. Throws an
    error if ``xml`` is not a valid XML document or ``path`` is not a valid
    XPath expression. ::

        SELECT xpath_boolean('<a><b>1</b></a>', 'a/b'); -- true
        SELECT xpath_boolean('<a><b>1</b></a>', 'a/c'); -- false
        SELECT xpath_boolean('<a><b>1</b></a>', 'a/b = "1"'); -- true

.. spark:function:: xpath_string(xml, path) -> varchar

    Returns the text contents of the first node in the XML document ``xml`` that
    matches the XPath expression ``path``. Returns an empty string if no node
    matches. Returns NULL if ``xml`` or ``path`` is NULL or empty. Throws an
    error if ``xml`` is not a valid XML document or ``path`` is not a valid
    XPath expression. ::

        SELECT xpath_string('<a><b>bee</b></a>', 'a/b'); -- 'bee'
        SELECT xpath_string('<a><b>b1</b><b>b2</b></a>', 'a/b[2]'); -- 'b2'
        SELECT xpath_string('<a><b>bee</b></a>', 'a/c'); -- ''
