================
Binary Functions
================

.. spark:function:: aes_encrypt(input, key, mode, padding, iv, aad) -> varbinary

    Encrypts ``input`` (varbinary) using AES (Advanced Encryption Standard)
    symmetric-key encryption with the given ``key``.

    Parameters:

    * ``input`` — varbinary plaintext to encrypt.
    * ``key`` — varbinary AES key. Must be 16, 24, or 32 bytes, selecting
      AES-128, AES-192, or AES-256 respectively.
    * ``mode`` — varchar block-cipher mode of operation. One of:

      * ``ECB`` — Electronic Code Book. No IV; identical plaintext blocks
        produce identical ciphertext blocks. Not recommended for general
        use.
      * ``CBC`` — Cipher Block Chaining. Requires a 16-byte IV
        (initialization vector); each plaintext block is XORed with the
        previous ciphertext block.
      * ``GCM`` — Galois/Counter Mode. Authenticated encryption with an
        associated data tag. Requires a 12-byte IV. Verifies integrity on
        decryption.

    * ``padding`` — varchar block-padding scheme. One of:

      * ``PKCS`` — PKCS#7 padding (only valid for ECB and CBC).
      * ``NONE`` — no padding; input length must be a multiple of 16 bytes
        (only valid for ECB and CBC; GCM never needs padding).
      * ``DEFAULT`` — PKCS for ECB/CBC, none for GCM.

    * ``iv`` — varbinary initialization vector. Empty for ECB. For CBC
      and GCM, if empty/NULL, a random IV is generated and prepended to
      the output.
    * ``aad`` — varbinary additional authenticated data, for GCM only.
      AAD is authenticated but not encrypted; passing different AAD on
      decryption fails the tag check. Must be empty for ECB and CBC.

    Output format:

    * ECB: just the ciphertext.
    * CBC: ``[IV || ciphertext]``.
    * GCM: ``[IV || ciphertext || tag]`` where ``tag`` is 16 bytes.

    Example::

        SELECT aes_decrypt(
                 aes_encrypt(CAST('hello' AS VARBINARY),
                             CAST('0123456789abcdef' AS VARBINARY),
                             'GCM', 'DEFAULT', CAST('' AS VARBINARY),
                             CAST('' AS VARBINARY)),
                 CAST('0123456789abcdef' AS VARBINARY),
                 'GCM', 'DEFAULT', CAST('' AS VARBINARY),
                 CAST('' AS VARBINARY))
          = CAST('hello' AS VARBINARY);  -- true

    .. warning::
       Never reuse the same IV with the same key in GCM mode. IV reuse
       completely breaks GCM's authentication guarantees and may also
       leak plaintext.

.. spark:function:: aes_decrypt(input, key, mode, padding, iv, aad) -> varbinary

    Decrypts ``input`` (varbinary) using AES with the given ``key``.
    Parameters match :spark:func:`aes_encrypt`.

    For CBC and GCM with empty ``iv``, the IV is read from the first
    bytes of ``input`` (12 bytes for GCM, 16 for CBC), and the rest is
    treated as ciphertext. For GCM, the trailing 16 bytes of the
    ciphertext portion are the authentication tag and are verified
    against ``aad``; a mismatch (wrong key, tampered ciphertext, or
    wrong AAD) causes the function to throw.

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
