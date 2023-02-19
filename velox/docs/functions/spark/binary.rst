================
Binary Functions
================

.. spark:function:: hash(x) -> integer

    Computes the hash of x.

.. spark:function:: md5(x) -> varbinary

    Computes the md5 of x.

.. spark:function:: sha1(x) -> varchar

    Computes SHA-1 digest of x and convert the result to a hex string.
    Note: x can only be varbinary type.

.. spark:function:: sha2(x, bitLength) -> varchar

    Calculate SHA-2 family of functions (SHA-224, SHA-256, SHA-384, and SHA-512) and convert the result to a hex string.
    If asking for an unsupported SHA function, the return value is NULL.
    Note: x can only be varbinary type.

.. spark:function:: xxhash64(x) -> integer

    Computes the xxhash64 of x.
