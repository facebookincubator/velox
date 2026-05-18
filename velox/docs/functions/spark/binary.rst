================
Binary Functions
================

.. spark:function:: aes_encrypt(input, key, mode, padding, iv, aad) -> varbinary

    Encrypts ``input`` (varbinary) with AES (Advanced Encryption Standard)
    symmetric-key encryption using ``key`` (16/24/32 bytes for AES-128/192/256).
    ``iv`` is the initialization vector (16 bytes for CBC, 12 bytes for GCM,
    empty for ECB; if empty/NULL for CBC/GCM a random IV is generated and
    prepended to the output). ``aad`` is additional authenticated data,
    GCM only; it is authenticated but not encrypted. ``mode`` is the
    block-cipher mode (``ECB``, ``CBC``, or ``GCM``); ``padding`` is
    ``PKCS``, ``NONE``, or ``DEFAULT`` (PKCS for ECB/CBC, none for GCM).
    Output is ``ciphertext`` for ECB, ``IV || ciphertext`` for CBC, and
    ``IV || ciphertext || tag`` (16-byte GCM tag) for GCM. ::

        SELECT base64(aes_encrypt(CAST('Spark' AS VARBINARY),
                                  CAST('abcdefghijklmnop' AS VARBINARY),
                                  'ECB', 'PKCS',
                                  CAST('' AS VARBINARY),
                                  CAST('' AS VARBINARY))); -- 4Hv0UKCx6nfUeAoPZo1z+w==

    Never reuse an IV with the same key in GCM mode; IV reuse breaks GCM's
    authentication guarantees and may leak plaintext.

.. spark:function:: aes_decrypt(input, key, mode, padding, iv, aad) -> varbinary

    Decrypts ``input`` (varbinary) with AES using ``key``. Parameters match
    :spark:func:`aes_encrypt`. For CBC/GCM with empty ``iv``, the IV is read
    from the first bytes of ``input`` (12 bytes for GCM, 16 for CBC) and
    the rest is treated as ciphertext. For GCM, the trailing 16 bytes are
    the authentication tag and are verified against ``aad``; a mismatch
    (wrong key, tampered ciphertext, or wrong AAD) throws. ::

        SELECT CAST(aes_decrypt(unbase64('4Hv0UKCx6nfUeAoPZo1z+w=='),
                                CAST('abcdefghijklmnop' AS VARBINARY),
                                'ECB', 'PKCS',
                                CAST('' AS VARBINARY),
                                CAST('' AS VARBINARY)) AS VARCHAR); -- Spark

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
