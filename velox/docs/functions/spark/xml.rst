=============
XML Functions
=============

These functions evaluate XPath 1.0 expressions from the XML document node.
Inputs are UTF-8 strings; an XML encoding declaration does not change their
encoding. A leading byte-order mark is rejected, matching Spark's character
Reader input.

Spark XPath functions require libxml2 earlier than 2.14. The bundled libxml2
2.13.5 dependency is used by default.

Compatibility notes
-------------------

* XML parsing is namespace-aware, unlike Spark's namespace-unaware parser.
  Unprefixed paths do not match elements in a default namespace, producing
  the usual no-match result. An evaluated node step that uses an unregistered
  namespace prefix returns NULL, as do function calls and variable references
  with such a prefix. The predefined ``xml`` prefix is supported. Namespace
  declarations (``xmlns`` and ``xmlns:prefix``) are not attribute nodes, so
  ``@*`` does not select them.
* External DTD loading and entity substitution are disabled. Documents with
  retained user-defined entity references return NULL rather than expanding
  those references for XPath evaluation, including a reference left in place
  when the document has only an external subset. System identifiers are not
  fetched. Predefined entities such as ``&amp;`` and numeric character
  references are supported. Comments and processing instructions are omitted
  from an element's string value. Adjacent text and CDATA sections form a
  single XPath text node. Attribute default values declared in a DOCTYPE are
  not applied, so such an attribute is absent unless the element specifies it.
* Documents that exceed libxml2's default limits are rejected, and the HUGE
  parser option is not enabled. With the bundled libxml2 2.13, the limits are
  256 nested elements, a text node of 10,000,000 bytes, and an entity
  amplification factor of 5; other versions can differ. Before libxml2 2.13,
  an oversized text node can be reported as an allocation failure, which
  raises an internal error rather than an invalid document error.
* An invalid document error does not include document text or a system
  identifier. A parser failure includes a libxml2 code, except when the parser
  reports no diagnostic or an allocation failure. Inputs rejected before
  parsing, including an embedded NUL, a leading byte-order mark, or a document
  larger than the parser can accept, are rejected without echoing the input.
  An invalid path error includes only a short UTF-8 prefix of the path:
  control characters, DEL, C1 controls, line and paragraph separators, bidi and
  zero-width format controls, and invalid UTF-8 bytes are replaced. The prefix
  is not split inside a code point. Context-associated parse and XPath
  diagnostics use local handlers; libxml2 may still use process-wide handlers
  for out-of-context allocation failures.
* Velox accepts dynamic ``path`` arguments; Spark requires foldable arguments.
  XML or path values containing embedded NUL characters are rejected.
* Finite numeric results use shortest round-trip decimal notation without an
  exponent, including numbers converted inside XPath string functions. As with
  other floating-point operations, the final decimal digits can differ between
  libxml2 and JVM versions.

.. spark:function:: xpath_boolean(xml, path) -> boolean

    Evaluates the XPath expression ``path`` against the XML document ``xml`` and
    returns its boolean value. A node-set result is ``true`` when it is
    non-empty (a matching node exists); a boolean expression returns its own
    value. Returns NULL if ``xml`` or ``path`` is NULL or empty. Throws an
    error if ``xml`` is not a valid XML document or ``path`` is not a valid
    XPath expression, except for the compatibility cases above. See Spark's
    `xpath_boolean documentation
    <https://spark.apache.org/docs/latest/api/sql/index.html#xpath_boolean>`_.
    ::

        SELECT xpath_boolean('<a><b>1</b></a>', 'a/b'); -- true
        SELECT xpath_boolean('<a><b>1</b></a>', 'a/c'); -- false
        SELECT xpath_boolean('<a><b>1</b></a>', 'a/b = "1"'); -- true

.. spark:function:: xpath_string(xml, path) -> varchar

    Returns the XPath string value of ``path`` evaluated against ``xml``.
    For a node-set, this is the string value of the first node in document
    order, or an empty string if no node matches. Boolean and numeric
    expressions are converted to strings. Returns NULL if ``xml`` or ``path``
    is NULL or empty. Throws an error if ``xml`` is not a valid XML document or
    ``path`` is not a valid XPath expression, except for the compatibility
    cases above. See Spark's `xpath_string documentation
    <https://spark.apache.org/docs/latest/api/sql/index.html#xpath_string>`_.
    ::

        SELECT xpath_string('<a><b>bee</b></a>', 'a/b'); -- 'bee'
        SELECT xpath_string('<a><b>b1</b><b>b2</b></a>', 'a/b[2]'); -- 'b2'
        SELECT xpath_string('<a><b>bee</b></a>', 'a/c'); -- ''
        SELECT xpath_string('<a/>', 'true()'); -- 'true'
