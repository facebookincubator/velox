================
Binary Functions
================

.. prestofunction:: crc32(binary) -> bigint

Computes the crc32 checksum of ``binary``.

.. prestofunction:: from_base64(string) -> varbinary

Decodes binary data from the base64 encoded ``string``.

.. prestofunction:: from_hex(string) -> varbinary

Decodes binary data from the hex encoded ``string``.

.. prestofunction:: hmac_sha1(binary, key) -> varbinary

Computes the HMAC with sha1 of ``binary`` with the given ``key``.

.. prestofunction:: hmac_sha256(binary, key) -> varbinary

Computes the HMAC with sha256 of ``binary`` with the given ``key``.

.. prestofunction:: hmac_sha512(binary, key) -> varbinary

Computes the HMAC with sha512 of ``binary`` with the given ``key``.

.. prestofunction:: md5(binary) -> varbinary

Computes the md5 hash of ``binary``.

.. prestofunction:: sha1(binary) -> varbinary

Computes the SHA-1 hash of ``binary``.

.. prestofunction:: sha256(binary) -> varbinary

Computes the SHA-256 hash of ``binary``.

.. prestofunction:: sha512(binary) -> varbinary

Computes the SHA-512 hash of ``binary``.

.. prestofunction:: spooky_hash_v2_32(binary) -> varbinary

Computes the SpookyHashV2 32-bit hash of ``binary``.

.. prestofunction:: spooky_hash_v2_64(binary) -> varbinary

Computes the 64-bit SpookyHashV2 hash of ``binary``.

.. prestofunction:: to_base64(binary) -> varchar

Encodes ``binary`` into a base64 string representation.

.. prestofunction:: to_hex(binary) -> varchar

Encodes ``binary`` into a hex string representation.

.. prestofunction:: xxhash64(binary) -> varbinary

Computes the xxhash64 hash of ``binary``.
