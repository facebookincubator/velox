====================
Set Digest Functions
====================

MinHash, or the min-wise independent permutations locality-sensitive hashing scheme,
is a technique used to quickly estimate how similar two sets are.
MinHash serves as a probabilistic data structure that estimates the Jaccard similarity
coefficient - the measure of the overlap between two sets as a percentage of the
total unique elements in both sets. Velox offers several functions that deal with the
`MinHash <https://wikipedia.org/wiki/MinHash>`_ technique.

MinHash is used to quickly estimate the
`Jaccard similarity coefficient <https://wikipedia.org/wiki/Jaccard_index>`_
between two sets. It is commonly used in data mining to detect near-duplicate
web pages at scale. By using this information, the search engines efficiently
avoid showing two pages within the search results two pagees that are nearly identical.

Data Structures
---------------

Velox implements Set Digest data sketches by encapsulating the following components:

- `HyperLogLog <https://wikipedia.org/wiki/HyperLogLog>`_
- `MinHash with a single hash function <http://wikipedia.org/wiki/MinHash#Variant_with_a_single_hash_function>`_

``HyperLogLog (HLL)``: HyperLogLog is an algorithm used to estimate the cardinality
of a set — that is, the number of distinct elements in a large data set.

``MinHash``: MinHash is used to estimate the similarity between two or more sets,
commonly known as Jaccard similarity. It is particularly effective when dealing
with large data sets and is generally used in data clustering and near-duplicate
detection.

The Velox type for this data structure is called ``setdigest``.
Velox offers the ability to merge multiple Set Digest data sketches.

Serialization
-------------

Data sketches such as those created via the use of MinHash or HyperLogLog can be
serialized to and from ``varbinary``. Serializing these data structures allows
them to be efficiently stored and transferred between different systems or sessions.
The serialization format is compatible with Presto's format.

Functions
---------

.. function:: make_set_digest(x) -> setdigest

    Composes all input values of ``x`` into a ``setdigest``.

    Supported input types include: ``boolean``, ``tinyint``, ``smallint``,
    ``integer``, ``bigint``, ``real``, ``double``, ``date``, ``varchar``,
    and ``varbinary``.

    Examples:

    Create a ``setdigest`` corresponding to a ``bigint`` array::

        SELECT make_set_digest(value)
        FROM (VALUES 1, 2, 3) T(value);

    Create a ``setdigest`` corresponding to a ``varchar`` array::

        SELECT make_set_digest(value)
        FROM (VALUES 'Presto', 'SQL', 'on', 'everything') T(value);

.. function:: merge_set_digest(setdigest) -> setdigest

    Returns the ``setdigest`` of the aggregate union of the individual
    ``setdigest`` structures.

    Example::

        SELECT merge_set_digest(a)
        FROM (SELECT make_set_digest(value) AS a
              FROM (VALUES 4, 3, 2, 1) T(value));

.. function:: cardinality(setdigest) -> bigint
    :noindex:

    Returns the cardinality of the set digest from its internal
    ``HyperLogLog`` component.

    Example::

        SELECT cardinality(make_set_digest(value))
        FROM (VALUES 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5) T(value);
        -- 5

.. function:: intersection_cardinality(x, y) -> bigint

    Returns the estimation for the cardinality of the intersection of the
    two set digests.

    ``x`` and ``y`` must be of type ``setdigest``.

    If both digests are exact (cardinality < maxHashes), returns the exact
    intersection count. Otherwise, returns an estimated intersection using
    the Jaccard index.

    Example::

        SELECT intersection_cardinality(make_set_digest(v1), make_set_digest(v2))
        FROM (VALUES (1, 1), (NULL, 2), (2, 3), (3, 4)) T(v1, v2);
        -- 3

.. function:: jaccard_index(x, y) -> double

    Returns the estimation of the `Jaccard index <https://wikipedia.org/wiki/Jaccard_index>`_
    for the two set digests.

    The Jaccard index is a value in [0, 1] where:

    - 1.0 means the sets are identical
    - 0.0 means the sets are disjoint (no overlap)

    ``x`` and ``y`` must be of type ``setdigest``.

    Uses MinHash estimation for efficient computation.

    Example::

        SELECT jaccard_index(make_set_digest(v1), make_set_digest(v2))
        FROM (VALUES (1, 1), (NULL, 2), (2, 3), (NULL, 4)) T(v1, v2);
        -- 0.5

.. function:: hash_counts(x) -> map(bigint, smallint)

    Returns a map containing the `Murmur3Hash128 <https://wikipedia.org/wiki/MurmurHash#MurmurHash3>`_
    hashed values and the count of their occurrences within the internal
    ``MinHash`` structure belonging to ``x``.

    ``x`` must be of type ``setdigest``.

    Example::

        SELECT hash_counts(make_set_digest(value))
        FROM (VALUES 1, 1, 1, 2, 2) T(value);
        -- {19144387141682250=3, -2447670524089286488=2}
