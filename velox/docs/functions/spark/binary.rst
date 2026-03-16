================
Binary Functions
================

.. spark:function:: aes_encrypt(input, key, mode, padding, iv, aad) -> varbinary

    Encrypts ``input`` using AES with the given ``key``. ``mode`` can be
    ``ECB``, ``CBC``, or ``GCM``. ``padding`` can be ``PKCS``,
    ``NONE``, or ``DEFAULT``. ``iv`` is the initialization vector.
    ``aad`` is additional authenticated data (GCM only).
    Key must be 16, 24, or 32 bytes (AES-128/192/256).
    The output is ``[IV || ciphertext]`` for CBC/GCM modes.

    .. warning::
       Never reuse the same IV with the same key in GCM mode. IV reuse
       completely breaks GCM's authentication guarantees.

.. spark:function:: aes_decrypt(input, key, mode, padding, iv, aad) -> varbinary

    Decrypts ``input`` using AES with the given ``key``. Parameters match
    :spark:func:`aes_encrypt`. Expects input format ``[IV || ciphertext]``
    for CBC/GCM modes. For GCM, verifies the authentication tag.

.. spark:function:: crc32(binary) -> bigint

    Computes the crc32 checksum of ``binary``.

.. spark:function:: hash(x, ...) -> integer

    Computes the hash of one or more input values using seed value of 42. For
    multiple arguments, their types can be different.

.. spark:function:: hash_with_seed(seed, x, ...) -> integer

    Computes the hash of one or more input values using specified seed. For
    multiple arguments, their types can be different.

.. spark:function:: md5(x) -> varbinary

    Computes the md5 of x.

.. spark:function:: might_contain(bloomFilter, value) -> boolean

   Returns TRUE if ``bloomFilter`` might contain ``value``.

   ``bloomFilter`` is a VARBINARY computed using ::spark:function::`bloom_filter_agg` aggregate function.
   ``value`` is a BIGINT.

.. spark:function:: sha1(x) -> varchar

    Computes SHA-1 digest of x and convert the result to a hex string.
    Note: x can only be varbinary type.

.. spark:function:: sha2(x, bitLength) -> varchar

    Calculate SHA-2 family of functions (SHA-224, SHA-256, SHA-384, and SHA-512) and
    convert the result to a hex string.
    The second argument indicates the desired bit length of the result, which must
    have a value of 224, 256, 384, 512, or 0 (which is equivalent to 256). If asking
    for an unsupported bitLength, the return value is NULL.
    Note: x can only be varbinary type.

.. spark:function:: xxhash64(x, ...) -> bigint

    Computes the xxhash64 of one or more input values using seed value of 42.
    For multiple arguments, their types can be different.

.. spark:function:: xxhash64_with_seed(seed, x, ...) -> bigint

    Computes the xxhash64 of one or more input values using specified seed. For
    multiple arguments, their types can be different.
